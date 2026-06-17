#!/usr/bin/env python3
"""
Generate HGHG-style synthetic graphs in ACMSubgraphsDataset-compatible format.

Output directory layout:
  <out_dir>/
    subgraph_000/
      nodes.pt
      edges.pt
      meta.json
      metapaths.json
      metapath_emb.pt
    subgraph_001/
      ...

Design choices for this generator:
  - 3 random node types: T0/T1/T2
  - Graph scales: N in {100, 200, 500} by default
  - Undirected edge count: M = 8 * N
  - Heterogeneous relation families by type-pair:
      link_T0_T0, link_T0_T1, ..., link_T2_T2  (9 directed families)
  - No fine-grained edge subtype:
      each family has only "__none__" label (existence-only within family)
  - Node subtype is a single default subtype per node type
  - Enumerate metapaths of length 2/3/4 and write random 32-d embeddings
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import torch


NODE_TYPES = ["T0", "T1", "T2"]


def _build_all_families(node_types: List[str]) -> Dict[str, Dict[str, str]]:
    fam_endpoints = {}
    for src_t in node_types:
        for dst_t in node_types:
            fam = f"link_{src_t}_{dst_t}"
            fam_endpoints[fam] = {"src_type": src_t, "dst_type": dst_t}
    return fam_endpoints


def _sample_undirected_edges(num_nodes: int, num_edges: int, rng: random.Random) -> List[Tuple[int, int]]:
    max_edges = num_nodes * (num_nodes - 1) // 2
    if num_edges > max_edges:
        raise ValueError(
            f"Requested {num_edges} undirected edges for N={num_nodes}, "
            f"but maximum is {max_edges}."
        )
    edges: Set[Tuple[int, int]] = set()
    while len(edges) < num_edges:
        u = rng.randrange(num_nodes)
        v = rng.randrange(num_nodes)
        if u == v:
            continue
        a, b = (u, v) if u < v else (v, u)
        edges.add((a, b))
    return list(edges)


def _build_nodes_pt(node_type_ids: List[int]) -> Tuple[Dict, Dict[int, int], Dict[str, List[int]]]:
    per_type_nodes: Dict[str, List[int]] = {t: [] for t in NODE_TYPES}
    for nid, tid in enumerate(node_type_ids):
        per_type_nodes[NODE_TYPES[tid]].append(nid)

    global_to_local: Dict[int, int] = {}
    for t in NODE_TYPES:
        for local_idx, gid in enumerate(per_type_nodes[t]):
            global_to_local[gid] = local_idx

    nodes_pt = {}
    for t in NODE_TYPES:
        ids = per_type_nodes[t]
        nodes_pt[t] = {
            "ids": ids,
            "subtype": torch.zeros(len(ids), dtype=torch.long),  # single default subtype
            "A": 1,
        }
    return nodes_pt, global_to_local, per_type_nodes


def _build_edges_pt(
    undirected_edges: List[Tuple[int, int]],
    node_type_ids: List[int],
    global_to_local: Dict[int, int],
) -> Dict:
    buckets = defaultdict(lambda: {"src_local": [], "dst_local": [], "y": []})

    for u, v in undirected_edges:
        tu = NODE_TYPES[node_type_ids[u]]
        tv = NODE_TYPES[node_type_ids[v]]
        fam_uv = f"link_{tu}_{tv}"
        fam_vu = f"link_{tv}_{tu}"

        # directed u -> v
        buckets[fam_uv]["src_local"].append(global_to_local[u])
        buckets[fam_uv]["dst_local"].append(global_to_local[v])
        buckets[fam_uv]["y"].append(1)

        # directed v -> u
        buckets[fam_vu]["src_local"].append(global_to_local[v])
        buckets[fam_vu]["dst_local"].append(global_to_local[u])
        buckets[fam_vu]["y"].append(1)

    families = {}
    for fam, content in buckets.items():
        families[fam] = {
            "src_local": torch.tensor(content["src_local"], dtype=torch.long),
            "dst_local": torch.tensor(content["dst_local"], dtype=torch.long),
            "y": torch.tensor(content["y"], dtype=torch.long),
        }
    return {"families": families}


def _build_meta(
    per_type_nodes: Dict[str, List[int]],
    edges_pt: Dict,
    fam_endpoints: Dict[str, Dict[str, str]],
) -> Dict:
    schema_by_type = {t: ["default"] for t in NODE_TYPES}

    # Keep all 9 families in vocabulary for stable offset layout.
    fam_label2id = {fam: {f"{fam}:__none__": 1} for fam in fam_endpoints}
    fam_id2label = {fam: {"1": f"{fam}:__none__"} for fam in fam_endpoints}

    num_edges_by_family = {}
    for fam in fam_endpoints:
        fd = edges_pt["families"].get(fam, None)
        num_edges_by_family[fam] = int(fd["y"].numel()) if fd is not None else 0

    return {
        "node_types": NODE_TYPES,
        "schema_by_type": schema_by_type,
        "fam_endpoints": fam_endpoints,
        "fam_label2id": fam_label2id,
        "fam_id2label": fam_id2label,
        "unknown_edges_dropped": 0,
        "num_nodes_by_type": {t: len(per_type_nodes[t]) for t in NODE_TYPES},
        "num_edges_by_family": num_edges_by_family,
    }


def _build_adj(num_nodes: int, undirected_edges: List[Tuple[int, int]]) -> List[List[int]]:
    adj = [[] for _ in range(num_nodes)]
    for u, v in undirected_edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj


def _enumerate_metapaths(
    node_type_ids: List[int],
    undirected_edges: List[Tuple[int, int]],
    lengths: List[int],
) -> Dict[int, Dict[str, int]]:
    max_len = max(lengths)
    n = len(node_type_ids)
    adj = _build_adj(n, undirected_edges)
    type_names = [NODE_TYPES[t] for t in node_type_ids]

    counts: Dict[int, Dict[str, int]] = {l: defaultdict(int) for l in lengths}

    def dfs(start: int, cur: int, depth: int, seq: List[str]) -> None:
        if depth in counts:
            key = "->".join(seq)
            counts[depth][key] += 1
        if depth == max_len:
            return
        for nb in adj[cur]:
            seq.append(type_names[nb])
            dfs(start, nb, depth + 1, seq)
            seq.pop()

    for s in range(n):
        dfs(s, s, 0, [type_names[s]])

    # convert defaultdict to normal dict
    out = {}
    for l in lengths:
        out[l] = dict(sorted(counts[l].items(), key=lambda kv: kv[0]))
    return out


def _make_metapath_embeddings(
    metapath_counts: Dict[int, Dict[str, int]],
    emb_dim: int,
    seed: int,
) -> Dict[str, torch.Tensor]:
    uniq_paths = set()
    for _, d in metapath_counts.items():
        uniq_paths.update(d.keys())

    g = torch.Generator()
    g.manual_seed(seed)
    emb = {}
    for p in sorted(uniq_paths):
        emb[p] = torch.randn(emb_dim, generator=g)
    return emb


def generate_dataset(
    out_dir: str,
    scales: List[int],
    graphs_per_scale: int,
    edge_multiplier: int,
    metapath_lengths: List[int],
    emb_dim: int,
    seed: int,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    rng = random.Random(seed)
    fam_endpoints = _build_all_families(NODE_TYPES)

    graph_idx = 0
    for n in scales:
        m_undirected = edge_multiplier * n
        for k in range(graphs_per_scale):
            graph_seed = seed + graph_idx * 9973 + n * 17 + k
            local_rng = random.Random(graph_seed)

            # 3 random node types
            node_type_ids = [local_rng.randrange(len(NODE_TYPES)) for _ in range(n)]
            undirected_edges = _sample_undirected_edges(n, m_undirected, local_rng)

            nodes_pt, global_to_local, per_type_nodes = _build_nodes_pt(node_type_ids)
            edges_pt = _build_edges_pt(undirected_edges, node_type_ids, global_to_local)
            meta = _build_meta(per_type_nodes, edges_pt, fam_endpoints)

            metapath_counts = _enumerate_metapaths(
                node_type_ids=node_type_ids,
                undirected_edges=undirected_edges,
                lengths=metapath_lengths,
            )
            metapath_emb = _make_metapath_embeddings(
                metapath_counts=metapath_counts,
                emb_dim=emb_dim,
                seed=graph_seed + 123,
            )

            subdir = os.path.join(out_dir, f"subgraph_{graph_idx:03d}")
            os.makedirs(subdir, exist_ok=True)

            torch.save(nodes_pt, os.path.join(subdir, "nodes.pt"))
            torch.save(edges_pt, os.path.join(subdir, "edges.pt"))
            with open(os.path.join(subdir, "meta.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

            # Auxiliary files (not required by loader, useful for HGHG experiments)
            mp_json = {
                "lengths": metapath_lengths,
                "counts": {str(l): metapath_counts[l] for l in metapath_lengths},
                "embedding_dim": emb_dim,
            }
            with open(os.path.join(subdir, "metapaths.json"), "w", encoding="utf-8") as f:
                json.dump(mp_json, f, ensure_ascii=False, indent=2)
            torch.save(metapath_emb, os.path.join(subdir, "metapath_emb.pt"))

            print(
                f"[OK] {subdir} | N={n}, undirected_E={m_undirected}, "
                f"typed_families_with_edges={len(edges_pt['families'])}, "
                f"metapaths={sum(len(v) for v in metapath_counts.values())}"
            )
            graph_idx += 1

    print(f"\n[DONE] Generated {graph_idx} subgraphs under: {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate HGHG synthetic dataset in ACMSubgraphs-compatible format."
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="data/HGHG_subgraphs",
        help="Output root directory containing subgraph_*/",
    )
    parser.add_argument(
        "--scales",
        type=str,
        default="100,200,500",
        help="Comma-separated node counts.",
    )
    parser.add_argument(
        "--graphs-per-scale",
        type=int,
        default=1,
        help="Number of graphs generated for each scale.",
    )
    parser.add_argument(
        "--edge-multiplier",
        type=int,
        default=8,
        help="Undirected edge count is edge_multiplier * N.",
    )
    parser.add_argument(
        "--metapath-lengths",
        type=str,
        default="2,3,4",
        help="Comma-separated metapath lengths to enumerate.",
    )
    parser.add_argument(
        "--emb-dim",
        type=int,
        default=32,
        help="Random embedding dimension for each metapath.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    scales = [int(x) for x in args.scales.split(",") if x.strip()]
    metapath_lengths = [int(x) for x in args.metapath_lengths.split(",") if x.strip()]

    generate_dataset(
        out_dir=args.out_dir,
        scales=scales,
        graphs_per_scale=args.graphs_per_scale,
        edge_multiplier=args.edge_multiplier,
        metapath_lengths=metapath_lengths,
        emb_dim=args.emb_dim,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
