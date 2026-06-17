import json
import argparse
import networkx as nx
import matplotlib.pyplot as plt


def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def load_nodes(stage):
    path = f"../output/{stage}.jsonl"
    print(f"[INFO] Loading nodes from {path} ...")
    return load_jsonl(path)


def load_edges():
    path = "../output/edges.jsonl"
    print(f"[INFO] Loading edges from {path} ...")
    return load_jsonl(path)


def build_graph(nodes, edges, filter_type=None):
    G = nx.DiGraph()

    # add nodes
    for n in nodes:
        ntype = n.get("type", "Unknown")
        subtype = n.get("subtype", None)

        if filter_type and ntype != filter_type:
            continue

        label = n.get("meta", {}).get("name") or n.get("meta", {}).get("title") or n["node_id"]

        G.add_node(
            n["node_id"],
            label=label,
            type=ntype,
            subtype=subtype,
        )

    # add edges
    for e in edges:
        src = e.get("src")
        dst = e.get("dst")
        etype = e.get("type", "rel")
        subtype = e.get("subtype", None)

        if src in G.nodes and dst in G.nodes:
            G.add_edge(src, dst, type=etype, subtype=subtype)

    return G


def draw_graph(G):
    color_map = {
        "Paper": "skyblue",
        "Author": "lightgreen",
        "Institution": "orange",
    }

    node_colors = []
    labels = {}

    for n, data in G.nodes(data=True):
        ntype = data.get("type")
        node_colors.append(color_map.get(ntype, "grey"))

        if data.get("subtype"):
            labels[n] = f"{data.get('label')} ({data['subtype']})"
        else:
            labels[n] = data.get("label")

    plt.figure(figsize=(16, 12))
    pos = nx.spring_layout(G, seed=42, k=0.3)  # 调整 k 控制紧凑度

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=500)
    nx.draw_networkx_edges(G, pos, arrows=True, alpha=0.4)
    nx.draw_networkx_labels(G, pos, labels, font_size=8)

    # edge type
    edge_labels = {}
    for u, v, data in G.edges(data=True):
        subtype = data.get("subtype")
        etype = data.get("type", "")
        if subtype:
            edge_labels[(u, v)] = subtype
        else:
            edge_labels[(u, v)] = etype

    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)
    plt.title("Graph Visualization")
    plt.axis("off")
    plt.show()




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        type=str,
        default="stage3_validated",
        choices=["stage1_nodes", "stage2_refined", "stage3_validated"],
        help="Which node file to visualize"
    )
    parser.add_argument(
        "--filter",
        type=str,
        default=None,
        choices=["Paper", "Author", "Institution", None],
        help="Optionally visualize only one node type"
    )
    args = parser.parse_args()

    nodes = load_nodes(args.stage)
    edges = load_edges()

    G = build_graph(nodes, edges, filter_type=args.filter)
    draw_graph(G)


if __name__ == "__main__":
    main()
