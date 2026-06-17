#!/usr/bin/env python3
"""
PubMed 预处理：PyTorch Geometric Planetoid 数据集
 -> nodes.jsonl + edges.jsonl（含子类别）

Planetoid PubMed：仅有 Paper 节点和 paper_paper 引用边，无 Author。
子类别：将 3 类标签映射为 Theory / Application / Other。
"""
import json
from pathlib import Path

# 依赖 torch_geometric
try:
    from torch_geometric.datasets import Planetoid
except ImportError:
    raise ImportError("需要安装 torch_geometric: pip install torch_geometric")

# 3 类标签 -> 子类别（医学文献通常偏应用，可后续根据领域调整）
CLASS_TO_SUBTYPE = {0: "Theory", 1: "Application", 2: "Other"}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="data/raw/PubMed")
    parser.add_argument("--output-dir", type=str, default="data/PubMed_processed")
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[1/3] 加载 Planetoid PubMed ...")
    dataset = Planetoid(root=str(root), name="PubMed")
    data = dataset[0]
    n = data.num_nodes
    y = data.y  # [N] 0/1/2
    edge_index = data.edge_index  # [2, E]

    print(f"      共 {n} 节点, {edge_index.size(1)} 条边")

    print("[2/3] 划分子类别 ...")
    nodes = []
    for i in range(n):
        c = int(y[i].item()) if y.dim() > 0 else 0
        subtype = CLASS_TO_SUBTYPE.get(c, "Other")
        nodes.append({"id": str(i), "type": "Paper", "subtype": subtype})

    edges = []
    for j in range(edge_index.size(1)):
        src = int(edge_index[0, j].item())
        dst = int(edge_index[1, j].item())
        edges.append({"src": str(src), "dst": str(dst), "type": "paper_paper"})

    print("[3/3] 写入 nodes.jsonl, edges.jsonl ...")
    with open(out_dir / "nodes.jsonl", "w", encoding="utf-8") as f:
        for n in nodes:
            f.write(json.dumps(n, ensure_ascii=False) + "\n")

    with open(out_dir / "edges.jsonl", "w", encoding="utf-8") as f:
        for e in edges:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    from collections import Counter
    cnt = Counter(n["subtype"] for n in nodes)
    print(f"\n[Paper 子类别分布] {dict(cnt)}")
    print(f"[DONE] 输出: {out_dir}/nodes.jsonl, edges.jsonl")


if __name__ == "__main__":
    main()
