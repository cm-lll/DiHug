#!/usr/bin/env python3
"""
简单重测 Degree Assortativity：从 IMDB 子图加载，构建 nx 图后计算。
用法: python scripts/check_degree_assortativity.py [subgraph_path]
"""
import sys
import torch
import networkx as nx
import numpy as np

def load_subgraph_edges(subdir):
    """从 subgraph_XXX 目录加载边（与 ACMSubgraphsDataset 一致）"""
    import json
    nodes_t = torch.load(f"{subdir}/nodes.pt", weights_only=True)
    edges_t = torch.load(f"{subdir}/edges.pt", weights_only=True)
    with open(f"{subdir}/meta.json") as f:
        meta = json.load(f)

    fam_endpoints = meta["fam_endpoints"]
    fam_data = edges_t["families"]
    offsets = {}
    cur = 0
    for nt in meta["node_types"]:
        if nt in nodes_t:
            offsets[nt] = cur
            cur += nodes_t[nt]["subtype"].shape[0]
    n = cur

    all_edges = []
    for fam, fd in fam_data.items():
        ep = fam_endpoints[fam]
        src_type, dst_type = ep["src_type"], ep["dst_type"]
        src = (fd["src_local"].long() + offsets[src_type]).tolist()
        dst = (fd["dst_local"].long() + offsets[dst_type]).tolist()
        for u, v in zip(src, dst):
            all_edges.append((u, v))

    return all_edges, n


def main():
    subdir = sys.argv[1] if len(sys.argv) > 1 else "data/IMDB_subgraphs/subgraph_000"
    edges, n = load_subgraph_edges(subdir)

    # 无向图（与 sampling_metrics 一致）
    undir_edges = set()
    for u, v in edges:
        if u != v:
            undir_edges.add((min(u, v), max(u, v)))
    g_undir = nx.Graph()
    g_undir.add_nodes_from(range(n))
    g_undir.add_edges_from(undir_edges)

    # 有向图
    g_dir = nx.DiGraph()
    g_dir.add_nodes_from(range(n))
    g_dir.add_edges_from(edges)

    a_undir = nx.degree_assortativity_coefficient(g_undir)
    a_dir = nx.degree_assortativity_coefficient(g_dir)

    print(f"图: {subdir}")
    print(f"  节点: {n}, 边(有向): {len(edges)}, 边(无向): {len(undir_edges)}")
    print(f"  Degree Assortativity (无向): {a_undir:.6f}")
    print(f"  Degree Assortativity (有向): {a_dir:.6f}")


if __name__ == "__main__":
    main()
