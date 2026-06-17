#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从完整 DBLP 图构建 k-hop 子图（确定性，非随机）。

- 输入: 与原始数据集相同格式的 nodes.jsonl + edges.jsonl（如 data/DBLP_four_area_processed/）
- 输出: 同格式的 nodes.jsonl + edges.jsonl 到新目录，可直接给 dblp_single 或其它模型用
- 规模: 脚本结束会打印子图规模（节点数、边数、按类型分布等）
"""

import argparse
import json
from collections import deque, defaultdict
from pathlib import Path


def load_nodes(nodes_path: Path):
    """读取 nodes.jsonl，返回 (nodes_list, id2node_idx)。"""
    nodes = []
    with nodes_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            nodes.append(json.loads(line))
    id2idx = {str(n["id"]): i for i, n in enumerate(nodes)}
    return nodes, id2idx


def load_edges(edges_path: Path):
    """读取 edges.jsonl，返回 list[dict]（保留原始字段），以及邻接表 adj[id] -> [id, ...]（无向）。"""
    edges = []
    adj = defaultdict(list)
    with edges_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            e = json.loads(line)
            u, v = str(e["src"]), str(e["dst"])
            edges.append(e)
            adj[u].append(v)
            adj[v].append(u)
    return edges, adj


def khop_bfs(adj: dict, seed_id: str, num_hops: int) -> set:
    """从 seed_id 出发做 num_hops 跳 BFS，返回保留的节点 id 集合（字符串）。"""
    keep = set()
    dist = {seed_id: 0}
    q = deque([seed_id])
    while q:
        u = q.popleft()
        keep.add(u)
        if dist[u] >= num_hops:
            continue
        for v in adj.get(u, []):
            if v not in dist:
                dist[v] = dist[u] + 1
                q.append(v)
    return keep


def write_jsonl(path: Path, rows: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Build k-hop subgraph from DBLP nodes.jsonl + edges.jsonl. Output same format for reuse in other models."
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=None,
        help="Directory containing nodes.jsonl and edges.jsonl (default: data/DBLP_four_area_processed)",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Output directory for subgraph nodes.jsonl and edges.jsonl (default: data/DBLP_subgraph_5hop)",
    )
    parser.add_argument(
        "--num_hops",
        type=int,
        default=5,
        help="Number of hops from seed node (default: 5)",
    )
    parser.add_argument(
        "--seed_id",
        type=str,
        default=None,
        help="Seed node id (e.g. a Paper id). If not set, use first Paper node id.",
    )
    parser.add_argument(
        "--seed_type",
        type=str,
        default="Paper",
        choices=("Author", "Paper", "Conference", "Term"),
        help="Type of seed node when --seed_id is not set: pick first node of this type (default: Paper)",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    input_dir = args.input_dir or (repo_root / "data" / "DBLP_four_area_processed")
    output_dir = args.output_dir or (repo_root / "data" / "DBLP_subgraph_5hop")

    nodes_path = input_dir / "nodes.jsonl"
    edges_path = input_dir / "edges.jsonl"
    if not nodes_path.exists() or not edges_path.exists():
        raise SystemExit(f"Missing input files: {nodes_path} or {edges_path}")

    print(f"[build_dblp_subgraph] Loading from {input_dir}")
    nodes, id2idx = load_nodes(nodes_path)
    edges, adj = load_edges(edges_path)
    n_full, m_full = len(nodes), len(edges)
    print(f"[build_dblp_subgraph] Full graph: nodes={n_full}, edges={m_full}")

    # Resolve seed
    seed_id = args.seed_id
    if seed_id is None:
        for n in nodes:
            if n.get("type") == args.seed_type:
                seed_id = str(n["id"])
                break
        if seed_id is None:
            raise SystemExit(f"No node of type '{args.seed_type}' found; set --seed_id explicitly.")
        print(f"[build_dblp_subgraph] Seed: first {args.seed_type} id = {seed_id}")
    else:
        seed_id = str(seed_id)
        if seed_id not in id2idx:
            raise SystemExit(f"Seed id '{seed_id}' not found in nodes.")
        print(f"[build_dblp_subgraph] Seed: {seed_id} (type={nodes[id2idx[seed_id]].get('type', '?')})")

    keep_ids = khop_bfs(adj, seed_id, args.num_hops)
    print(f"[build_dblp_subgraph] k-hop (k={args.num_hops}) subgraph: {len(keep_ids)} nodes")

    sub_nodes = [n for n in nodes if str(n["id"]) in keep_ids]
    sub_edges = [
        e for e in edges
        if str(e["src"]) in keep_ids and str(e["dst"]) in keep_ids
    ]

    write_jsonl(output_dir / "nodes.jsonl", sub_nodes)
    write_jsonl(output_dir / "edges.jsonl", sub_edges)
    print(f"[build_dblp_subgraph] Written to {output_dir}")

    # --- 子图规模统计 ---
    by_type = defaultdict(int)
    for n in sub_nodes:
        by_type[n.get("type", "?")] += 1
    by_etype = defaultdict(int)
    for e in sub_edges:
        by_etype[e.get("type", "generic")] += 1

    n_author = by_type.get("Author", 0)
    n_conf = by_type.get("Conference", 0)
    n_paper = by_type.get("Paper", 0)
    n_term = by_type.get("Term", 0)

    print("\n--- 子图规模 (Subgraph scale) ---")
    print(f"  节点总数 (nodes):     {len(sub_nodes)}")
    print(f"  边总数 (edges):       {len(sub_edges)}")
    print(f"  相对全图:             nodes {len(sub_nodes)/max(1,n_full)*100:.2f}%, edges {len(sub_edges)/max(1,m_full)*100:.2f}%")
    print("  ---")
    print(f"  作者 (Author):        {n_author}")
    print(f"  机构 (Conference):    {n_conf}")
    print(f"  论文 (Paper):         {n_paper}")
    print(f"  词 (Term):            {n_term}")
    print("  节点按类型 (by type):")
    for t in sorted(by_type.keys()):
        print(f"    {t}: {by_type[t]}")
    print("  边按关系类型 (by edge type):")
    for t in sorted(by_etype.keys()):
        print(f"    {t}: {by_etype[t]}")
    print("\n使用方式: 在 config 中设置 datadir 为该输出目录，例如:")
    print(f"  datadir: '{output_dir}'")
    print("  (或使用相对路径，如 'data/DBLP_subgraph_5hop')")


if __name__ == "__main__":
    main()
