#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
规则方式预处理 DBLP_four_area：
- 输入: data/raw/DBLP_four_area/node.dat, link.dat
- 输出: data/DBLP_four_area_processed/nodes.jsonl, edges.jsonl
每个节点一个子类别(subtype)，完全不用 LLM / 嵌入。
"""

import json
from pathlib import Path

# --- 映射表：可以按需要调整 ---

# 节点类型编码 -> 可读类型名
NODE_TYPE_MAP = {
    "A": "Author",
    "P": "Paper",
    "C": "Conference",
    "T": "Term",
}

# 领域标签 -> 子类别名称
AREA_LABEL_MAP = {
    "0": "Database",
    "1": "DataMining",
    "2": "AI",
    "3": "InformationRetrieval",
}


def read_nodes(node_path: Path):
    """
    读取 node.dat: 'id type label'
    返回:
      nodes: list[dict]
      id2type: dict[id -> type_name]
    """
    nodes = []
    id2type = {}

    with node_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            nid, tcode, label = parts[0], parts[1], parts[2]

            node_type = NODE_TYPE_MAP.get(tcode, tcode)
            subtype = AREA_LABEL_MAP.get(label, label)  # 找不到就用原始数字

            node = {
                "id": nid,                 # 全局节点 id（字符串也可以）
                "type": node_type,         # Author / Paper / Conference / Term
                "label_id": int(label),    # 0/1/2/3
                "subtype": subtype,        # Database / DataMining / AI / IR
            }
            nodes.append(node)
            id2type[nid] = node_type

    return nodes, id2type


def infer_edge_type(t_src: str, t_dst: str) -> str:
    """
    根据端点类型推一个简单的关系名，仅用于可读性。
    若不匹配任何规则，则返回 'generic'。
    """
    pair = (t_src, t_dst)
    if pair == ("Author", "Paper") or pair == ("Paper", "Author"):
        return "author_paper"
    if pair == ("Paper", "Conference") or pair == ("Conference", "Paper"):
        return "paper_conference"
    if pair == ("Paper", "Term") or pair == ("Term", "Paper"):
        return "paper_term"
    if pair == ("Author", "Author"):
        return "author_author"
    if pair == ("Paper", "Paper"):
        return "paper_paper"
    if pair == ("Conference", "Conference"):
        return "conference_conference"
    if pair == ("Term", "Term"):
        return "term_term"
    return "generic"


def read_edges(link_path: Path, id2type: dict):
    """
    读取 link.dat: 假定格式为 'src dst'（若有多列，只取前两列）
    返回 edges: list[dict]，形如:
      {"src": "u", "dst": "v", "type": "author_paper"}
    """
    edges = []
    with link_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            u, v = parts[0], parts[1]
            t_u = id2type.get(u, "Unknown")
            t_v = id2type.get(v, "Unknown")
            etype = infer_edge_type(t_u, t_v)
            edges.append({
                "src": u,
                "dst": v,
                "type": etype,
            })
    return edges


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    repo_root = Path(__file__).resolve().parents[1]

    # Raw DBLP_four_area directory under the DiHuG data root.
    raw_dir = repo_root / "data" / "raw" / "DBLP_four_area"
    node_path = raw_dir / "node.dat"
    link_path = raw_dir / "link.dat"

    if not node_path.exists() or not link_path.exists():
        raise SystemExit(f"Missing DBLP_four_area files: {node_path} or {link_path}")

    print(f"[DBLP] Reading nodes from {node_path}")
    nodes, id2type = read_nodes(node_path)
    print(f"[DBLP] Loaded {len(nodes)} nodes")

    print(f"[DBLP] Reading edges from {link_path}")
    edges = read_edges(link_path, id2type)
    print(f"[DBLP] Loaded {len(edges)} edges")

    out_dir = repo_root / "data" / "DBLP_four_area_processed"
    nodes_out = out_dir / "nodes.jsonl"
    edges_out = out_dir / "edges.jsonl"

    print(f"[DBLP] Writing nodes to {nodes_out}")
    write_jsonl(nodes_out, nodes)
    print(f"[DBLP] Writing edges to {edges_out}")
    write_jsonl(edges_out, edges)

    print("\n[DBLP] Done. Summary:")
    print(f"  nodes: {len(nodes)}  -> {nodes_out}")
    print(f"  edges: {len(edges)}  -> {edges_out}")


if __name__ == "__main__":
    main()
