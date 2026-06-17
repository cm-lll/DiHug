#!/usr/bin/env python3
"""
通用子图提取：基于 BFS k-hop ego 扩展，保证连通性，节点数 ≤ max_nodes。
适用于 DBLP、IMDB、PubMed、Aminer 等异质图。
不直接切割，从种子节点 BFS 扩展得到连通子图。
"""
import argparse
import json
import random
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any

import torch

# -------------------- 数据集配置 --------------------
DATASET_CONFIGS = {
    "dblp": {
        "node_types": ["Author", "Paper", "Conference", "Term"],
        "schema_by_type": {
            "Author": ["Database", "DataMining", "AI", "InformationRetrieval", "Other"],
            "Paper": ["Database", "DataMining", "AI", "InformationRetrieval", "Other"],
            "Conference": ["Database", "DataMining", "AI", "InformationRetrieval", "Other"],
            "Term": ["Database", "DataMining", "AI", "InformationRetrieval", "Other"],
        },
        "fam_endpoints": {
            "author_paper": {"src_type": "Author", "dst_type": "Paper"},
            "paper_conference": {"src_type": "Paper", "dst_type": "Conference"},
            "paper_term": {"src_type": "Paper", "dst_type": "Term"},
            "author_author": {"src_type": "Author", "dst_type": "Author"},
        },
        "seed_types": ["Paper", "Author"],  # 优先从这些类型选 seed
    },
    "imdb": {
        "node_types": ["Director", "Actor", "Movie"],
        "schema_by_type": {
            "Director": ["High", "Medium", "Low"],
            "Actor": ["High", "Medium", "Low"],
            "Movie": ["Action", "Comedy", "Drama", "Adventure", "Crime", "Biography", "Horror", "Documentary", "Animation", "Fantasy", "Romance", "Sci-Fi", "Thriller", "Western", "Other"],
        },
        "fam_endpoints": {
            "director_movie": {"src_type": "Director", "dst_type": "Movie"},
            "actor_movie": {"src_type": "Actor", "dst_type": "Movie"},
        },
        "seed_types": ["Movie"],
    },
    "pubmed": {
        "node_types": ["Paper"],
        "schema_by_type": {
            "Paper": ["Theory", "Application", "Other"],
        },
        "fam_endpoints": {
            "paper_paper": {"src_type": "Paper", "dst_type": "Paper"},
        },
        "seed_types": ["Paper"],
    },
    "pubmed_pgb": {
        "node_types": ["Paper", "Author"],
        "schema_by_type": {
            "Paper": ["Review", "Clinical", "Article", "CaseReport", "Study", "Other"],
            "Author": ["High", "Medium", "Low"],
        },
        "fam_endpoints": {
            "author_paper": {"src_type": "Author", "dst_type": "Paper"},
            "paper_paper": {"src_type": "Paper", "dst_type": "Paper"},
        },
        "seed_types": ["Paper", "Author"],
    },
    "aminer": {
        "node_types": ["Paper", "Author", "Venue"],
        "schema_by_type": {
            "Paper": ["Theory", "Application", "Other"],
            "Author": ["Other"],
            "Venue": ["Journal", "Conference", "Other"],
        },
        "fam_endpoints": {
            "author_paper": {"src_type": "Author", "dst_type": "Paper"},
            "paper_venue": {"src_type": "Paper", "dst_type": "Venue"},
            "paper_paper": {"src_type": "Paper", "dst_type": "Paper"},
        },
        "seed_types": ["Paper"],
    },
}


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_graph(nodes_path: Path, edges_path: Path) -> Tuple[Dict[str, Any], List[Tuple[str, str, str]], Dict[str, str]]:
    """加载 nodes.jsonl 和 edges.jsonl，返回 (节点字典 id->{type,subtype}, 边列表 [(src,dst,etype)], id2type)"""
    nodes = {}
    id2type = {}

    # label_id 映射（DBLP 等用数字表示子类型）
    LABEL_ID_MAP = {0: "Database", 1: "DataMining", 2: "AI", 3: "InformationRetrieval"}

    for obj in read_jsonl(nodes_path):
        nid = str(obj.get("id", ""))
        ntype = obj.get("type", "Other")
        subtype = obj.get("subtype") or obj.get("label")
        if subtype is None and "label_id" in obj:
            subtype = LABEL_ID_MAP.get(obj["label_id"], "Other")
        if isinstance(subtype, list) and subtype:
            subtype = str(subtype[0])
        elif subtype is not None and hasattr(subtype, "__iter__") and not isinstance(subtype, str):
            subtype = str(list(subtype)[0]) if subtype else "Other"
        else:
            subtype = str(subtype) if subtype else "Other"
        nodes[nid] = {"type": ntype, "subtype": subtype}
        id2type[nid] = ntype

    edges = []
    for obj in read_jsonl(edges_path):
        src = str(obj.get("src"))
        dst = str(obj.get("dst"))
        etype = obj.get("type", "generic")
        if src in nodes and dst in nodes:
            edges.append((src, dst, etype))

    return nodes, edges, id2type


def build_adjacency(edges: List[Tuple[str, str, str]]) -> Dict[str, Set[str]]:
    """构建邻接表（无向，用于 BFS）"""
    adj = defaultdict(set)
    for src, dst, _ in edges:
        adj[src].add(dst)
        adj[dst].add(src)
    return dict(adj)


def bfs_ego(adj: Dict[str, Set[str]], seed: str, max_nodes: int) -> Set[str]:
    """从 seed 出发 BFS，最多扩展 max_nodes 个节点，保证连通性"""
    visited = set()
    q = deque([seed])
    visited.add(seed)

    while q and len(visited) < max_nodes:
        u = q.popleft()
        for v in adj.get(u, []):
            if v not in visited:
                visited.add(v)
                q.append(v)
                if len(visited) >= max_nodes:
                    return visited
    return visited


def pick_seeds(
    nodes: Dict[str, Any],
    adj: Dict[str, Set[str]],
    config: Dict,
    n_seeds: int,
    min_nodes: int,
    max_nodes: int,
    seed_random: int = 42,
) -> List[str]:
    """挑选种子节点：优先从 seed_types 中选，且 BFS 扩展后节点数在 [min_nodes, max_nodes] 之间"""
    seed_types = set(config.get("seed_types", []))
    if not seed_types:
        seed_types = set(nodes[n]["type"] for n in nodes)

    candidates = []
    for nid, info in nodes.items():
        if info["type"] not in seed_types:
            continue
        deg = len(adj.get(nid, set()))
        if deg < 2:
            continue
        # BFS 扩展规模（最多 max_nodes，保证子图在范围内）
        size = len(bfs_ego(adj, nid, max_nodes))
        if min_nodes <= size <= max_nodes:
            candidates.append((nid, size))

    candidates.sort(key=lambda x: x[1])
    random.seed(seed_random)
    if len(candidates) > n_seeds:
        indices = random.sample(range(len(candidates)), min(n_seeds, len(candidates)))
        return [candidates[i][0] for i in sorted(indices)]
    return [c[0] for c in candidates[:n_seeds]]


def extract_subgraph(
    nodes: Dict[str, Any],
    edges: List[Tuple[str, str, str]],
    node_set: Set[str],
    config: Dict,
) -> Tuple[Dict, Dict, Dict]:
    """从全图中提取子图，转为 nodes_t, fam_data, meta 格式（与 ACM 一致）"""
    node_types = config["node_types"]
    schema_by_type = config["schema_by_type"]
    fam_endpoints = config["fam_endpoints"]

    # 按类型组织节点
    per_type_ids = {t: [] for t in node_types}
    per_type_sub = {t: [] for t in node_types}
    subtype2id = {t: {s: i for i, s in enumerate(schema_by_type.get(t, ["Other"]))} for t in node_types}
    gid2info = {}
    for nid in node_set:
        info = nodes.get(nid)
        if not info:
            continue
        nt = info["type"]
        if nt not in node_types:
            nt = node_types[0]
        st = info.get("subtype", "Other")
        st_names = schema_by_type.get(nt, ["Other"])
        if st not in st_names:
            st = "Other" if "Other" in st_names else (st_names[0] if st_names else "Other")
        st_id = subtype2id[nt].get(st, 0)
        gid2info[nid] = (nt, len(per_type_ids[nt]))
        per_type_ids[nt].append(nid)
        per_type_sub[nt].append(st_id)

    # 边
    fam2edges = defaultdict(list)
    for src, dst, etype in edges:
        if src not in node_set or dst not in node_set:
            continue
        if etype not in fam_endpoints:
            continue
        fam2edges[etype].append((src, dst))

    fam_data = {}
    fam_label2id = {}
    fam_id2label = {}
    for fam, elist in fam2edges.items():
        if not elist:
            continue
        vocab = {f"{fam}:__none__": 1}
        src_local = []
        dst_local = []
        y_list = []
        ep = fam_endpoints[fam]
        src_type = ep["src_type"]
        dst_type = ep["dst_type"]
        for s, d in elist:
            st, si = gid2info[s]
            tt, di = gid2info[d]
            src_local.append(si)
            dst_local.append(di)
            y_list.append(1)
        fam_data[fam] = {
            "src_local": torch.tensor(src_local, dtype=torch.long),
            "dst_local": torch.tensor(dst_local, dtype=torch.long),
            "y": torch.tensor(y_list, dtype=torch.long),
        }
        fam_label2id[fam] = {f"{fam}:__none__": 1}
        fam_id2label[fam] = {"1": f"{fam}:__none__"}

    nodes_t = {
        nt: {
            "ids": per_type_ids[nt],
            "subtype": torch.tensor(per_type_sub[nt], dtype=torch.long),
            "A": len(schema_by_type.get(nt, ["Other"])),
        }
        for nt in node_types
        if per_type_ids[nt]
    }

    meta = {
        "node_types": node_types,
        "schema_by_type": schema_by_type,
        "fam_endpoints": {k: v for k, v in fam_endpoints.items() if k in fam_data},
        "fam_label2id": fam_label2id,
        "fam_id2label": fam_id2label,
        "unknown_edges_dropped": 0,
        "num_nodes_by_type": {nt: len(per_type_ids[nt]) for nt in node_types if per_type_ids[nt]},
        "num_edges_by_family": {f: int(fam_data[f]["y"].numel()) for f in fam_data},
    }

    return nodes_t, fam_data, meta


def main():
    parser = argparse.ArgumentParser(description="BFS ego 子图提取，保证连通性，节点≤max_nodes")
    parser.add_argument("nodes", type=str, help="nodes.jsonl 路径")
    parser.add_argument("edges", type=str, help="edges.jsonl 路径")
    parser.add_argument("out_dir", type=str, help="输出目录，将创建 subgraph_000/, subgraph_001/ ...")
    parser.add_argument("--dataset", type=str, default="dblp", choices=list(DATASET_CONFIGS.keys()),
                        help="数据集类型，用于 schema 和边类型映射")
    parser.add_argument("--min-nodes", type=int, default=50, help="每个子图最少节点数")
    parser.add_argument("--max-nodes", type=int, default=500, help="每个子图最多节点数")
    parser.add_argument("--n-subgraphs", type=int, default=100, help="目标子图数量")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    nodes_path = Path(args.nodes)
    edges_path = Path(args.edges)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not nodes_path.exists() or not edges_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {nodes_path}, {edges_path}")

    config = DATASET_CONFIGS[args.dataset].copy()
    # 若某些边类型不在 fam_endpoints，从数据中补充
    nodes_dict, edges_list, id2type = load_graph(nodes_path, edges_path)
    edge_types = set(e[2] for e in edges_list)
    for et in edge_types:
        if et not in config["fam_endpoints"]:
            # 根据边推断 src/dst 类型
            src_types = set()
            dst_types = set()
            for src, dst, t in edges_list:
                if t == et:
                    src_types.add(nodes_dict.get(src, {}).get("type", "Other"))
                    dst_types.add(nodes_dict.get(dst, {}).get("type", "Other"))
            if src_types and dst_types:
                config["fam_endpoints"][et] = {
                    "src_type": list(src_types)[0],
                    "dst_type": list(dst_types)[0],
                }

    adj = build_adjacency(edges_list)
    print(f"[INFO] 加载完成: {len(nodes_dict)} 节点, {len(edges_list)} 边")

    seeds = pick_seeds(
        nodes_dict, adj, config,
        n_seeds=args.n_subgraphs,
        min_nodes=args.min_nodes,
        max_nodes=args.max_nodes,
        seed_random=args.seed,
    )
    print(f"[INFO] 选出 {len(seeds)} 个种子")

    written = 0
    for i, seed in enumerate(seeds):
        node_set = bfs_ego(adj, seed, args.max_nodes)
        if len(node_set) < args.min_nodes:
            continue
        nodes_t, fam_data, meta = extract_subgraph(nodes_dict, edges_list, node_set, config)
        if sum(meta["num_nodes_by_type"].values()) < args.min_nodes:
            continue

        subdir = out_dir / f"subgraph_{written:03d}"
        subdir.mkdir(parents=True, exist_ok=True)
        torch.save(nodes_t, subdir / "nodes.pt")
        torch.save({"families": fam_data}, subdir / "edges.pt")
        with (subdir / "meta.json").open("w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        written += 1
        if written % 10 == 0:
            print(f"[INFO] 已写入 {written} 个子图, 当前节点数={len(node_set)}")

    print(f"[DONE] 共 {written} 个子图, 输出目录: {out_dir}")


if __name__ == "__main__":
    main()
