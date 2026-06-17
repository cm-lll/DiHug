import json
import networkx as nx
import matplotlib.pyplot as plt


# Node-type-based shape
NODE_SHAPES = {
    "Paper": "o",
    "Author": "s",
    "Organization": "^"
}

# Palette for subtype colors
COLOR_PALETTE = [
    "#ff4d4d", "#4da6ff", "#85e085", "#ffd480",
    "#d966ff", "#ff99c2", "#99ffcc", "#ffcc99",
    "#80b3ff", "#ffb380"
]


def load_stage_nodes(path):
    nodes = {}
    with open(path, "r", encoding="utf8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            nid = obj["node_id"]
            sub = obj.get("raw_labels", ["unknown"])
            nodes[nid] = {
                "type": obj["type"],
                "subtype": sub[0] if sub else "unknown",
                "meta": obj.get("meta", {})
            }
    return nodes


def load_edges(path):
    edges = []
    with open(path, "r", encoding="utf8") as f:
        for line in f:
            if not line.strip():
                continue
            edges.append(json.loads(line))
    return edges


def visualize_png(
    nodes_file,
    edges_file,
    output_png="hetero_graph.png",
    max_nodes=500
):
    nodes = load_stage_nodes(nodes_file)
    edges = load_edges(edges_file)

    # Reduce size for plotting
    if len(nodes) > max_nodes:
        print(f"[INFO] Too many nodes: {len(nodes)}. Sampling {max_nodes} for visualization.")
        sampled_nodes = set(list(nodes.keys())[:max_nodes])
    else:
        sampled_nodes = set(nodes.keys())

    G = nx.Graph()

    # Assign colors by subtype
    subtype_to_color = {}
    color_idx = 0

    for nid, info in nodes.items():
        if nid not in sampled_nodes:
            continue

        subtype = info["subtype"]
        if subtype not in subtype_to_color:
            subtype_to_color[subtype] = COLOR_PALETTE[color_idx % len(COLOR_PALETTE)]
            color_idx += 1

        G.add_node(
            nid,
            node_type=info["type"],
            subtype=subtype,
            color=subtype_to_color[subtype],
            shape=NODE_SHAPES.get(info["type"], "o")
        )

    # Add edges
    for e in edges:
        src, tgt = e.get("src"), e.get("dst")
        if src in G and tgt in G:
            G.add_edge(src, tgt)

    # Spring layout
    pos = nx.spring_layout(G, seed=42, k=0.3)

    # Draw each node type separately
    for ntype, shape in NODE_SHAPES.items():
        nodes_of_type = [n for n, attr in G.nodes(data=True) if attr["node_type"] == ntype]
        colors = [G.nodes[n]["color"] for n in nodes_of_type]

        nx.draw_networkx_nodes(
            G, pos,
            nodelist=nodes_of_type,
            node_color=colors,
            node_shape=shape,
            node_size=80,
            alpha=0.9
        )

    nx.draw_networkx_edges(G, pos, alpha=0.4, width=0.8)

    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    print(f"[INFO] Graph PNG saved to {output_png}")


if __name__ == "__main__":
    visualize_png(
        nodes_file="preprocess/output/stage1_nodes.jsonl",
        edges_file="preprocess/output/edges.jsonl",
        output_png="preprocess/output/hetero_graph.png",
        max_nodes=400
    )
