#!/usr/bin/env python3
"""
Convert HGEN-style PubMed baseline pickles into DiHuG subgraph_* layout.

Ground truth is **undirected**: one simple edge per node pair.

Each undirected edge is stored as ONE canonical directed edge (SparseDiff-style state):
  - link_{T(u)}_{T(v)} with u -> v where u < v by global node id

Reverse message passing for HGT is NOT stored in the diffusion state; it is added
only in the computational graph via get_computational_graph(..., for_message_passing=True)
which calls to_undirected on the comp graph.

Evaluation collapses directed predictions back to undirected (canonical direction only).

Input:
  <input_dir>/adj_matrix.p, new_node_dict.p

Output:
  <out_dir>/subgraph_000/  (full graph, train/val/test share)
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
from collections import defaultdict
from typing import Dict, List, Tuple

import torch


NODE_TYPES = ["Gene", "Disease", "Chemical", "Species"]
HGEN_TYPE_TO_NAME = {0: "Gene", 1: "Disease", 2: "Chemical", 3: "Species"}


def _build_fam_endpoints() -> Dict[str, Dict[str, str]]:
    """16 type-pair families (one canonical directed slot per undirected heterogeneous edge)."""
    fam_endpoints: Dict[str, Dict[str, str]] = {}
    for src_t in NODE_TYPES:
        for dst_t in NODE_TYPES:
            base = f"link_{src_t}_{dst_t}"
            fam_endpoints[base] = {"src_type": src_t, "dst_type": dst_t}
    return fam_endpoints


def _load_csr(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def _undirected_edges_from_csr(adj) -> List[Tuple[int, int]]:
    rows, cols = adj.nonzero()
    edges = set()
    for u, v in zip(rows.tolist(), cols.tolist()):
        if u == v:
            continue
        a, b = (u, v) if u < v else (v, u)
        edges.add((a, b))
    return sorted(edges)


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
            "subtype": torch.zeros(len(ids), dtype=torch.long),
            "A": 1,
        }
    return nodes_pt, global_to_local, per_type_nodes


def _build_edges_pt(
    undirected_edges: List[Tuple[int, int]],
    node_type_ids: List[int],
    global_to_local: Dict[int, int],
) -> Dict:
    """
    Store each undirected edge (a,b), a<b, as one canonical directed edge:
      a -> b in link_{T(a)}_{T(b)}
    """
    buckets = defaultdict(lambda: {"src_local": [], "dst_local": [], "y": []})

    for a, b in undirected_edges:
        tu = NODE_TYPES[node_type_ids[a]]
        tv = NODE_TYPES[node_type_ids[b]]
        base_fam = f"link_{tu}_{tv}"

        buckets[base_fam]["src_local"].append(global_to_local[a])
        buckets[base_fam]["dst_local"].append(global_to_local[b])
        buckets[base_fam]["y"].append(1)

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
    num_undirected_edges: int,
) -> Dict:
    schema_by_type = {t: ["default"] for t in NODE_TYPES}
    fam_label2id = {fam: {f"{fam}:__none__": 1} for fam in fam_endpoints}
    fam_id2label = {fam: {"1": f"{fam}:__none__"} for fam in fam_endpoints}

    num_edges_by_family = {}
    for fam in fam_endpoints:
        fd = edges_pt["families"].get(fam)
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
        "num_undirected_edges": num_undirected_edges,
        "undirected_ground_truth": True,
        "bidirectional_mode": "canonical_single",
        "source": "HGEN PubMed_baseline (full graph, undirected GT + canonical single directed edge)",
    }


def _write_subgraph(
    out_subdir: str,
    node_type_ids: List[int],
    undirected_edges: List[Tuple[int, int]],
    fam_endpoints: Dict[str, Dict[str, str]],
) -> None:
    os.makedirs(out_subdir, exist_ok=True)
    nodes_pt, global_to_local, per_type_nodes = _build_nodes_pt(node_type_ids)
    edges_pt = _build_edges_pt(undirected_edges, node_type_ids, global_to_local)
    meta = _build_meta(per_type_nodes, edges_pt, fam_endpoints, len(undirected_edges))

    torch.save(nodes_pt, os.path.join(out_subdir, "nodes.pt"))
    torch.save(edges_pt, os.path.join(out_subdir, "edges.pt"))
    with open(os.path.join(out_subdir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def convert(input_dir: str, out_dir: str) -> None:
    adj = _load_csr(os.path.join(input_dir, "adj_matrix.p"))
    with open(os.path.join(input_dir, "new_node_dict.p"), "rb") as f:
        node_dict = pickle.load(f)

    n = adj.shape[0]
    node_type_ids = [int(node_dict[i]) for i in range(n)]
    if set(node_type_ids) - set(HGEN_TYPE_TO_NAME):
        raise ValueError(f"Unexpected node types in new_node_dict: {set(node_type_ids)}")

    full_edges = _undirected_edges_from_csr(adj)
    fam_endpoints = _build_fam_endpoints()

    os.makedirs(out_dir, exist_ok=True)
    full_dir = os.path.join(out_dir, "subgraph_000")
    _write_subgraph(full_dir, node_type_ids, full_edges, fam_endpoints)

    stale = os.path.join(out_dir, "subgraph_001")
    if os.path.isdir(stale):
        shutil.rmtree(stale)

    processed_dir = os.path.join(out_dir, "processed")
    if os.path.isdir(processed_dir):
        shutil.rmtree(processed_dir)
    os.makedirs(processed_dir, exist_ok=True)
    splits = {
        "train": torch.tensor([0], dtype=torch.long),
        "val": torch.tensor([0], dtype=torch.long),
        "test": torch.tensor([0], dtype=torch.long),
    }
    torch.save(splits, os.path.join(processed_dir, "splits.pt"))

    marker = os.path.join(out_dir, "raw", "_acm_subgraphs_ready.txt")
    os.makedirs(os.path.dirname(marker), exist_ok=True)
    with open(marker, "a", encoding="utf-8"):
        pass

    edges_pt_written = torch.load(os.path.join(full_dir, "edges.pt"), weights_only=False)
    directed_edges = sum(int(fd["y"].numel()) for fd in edges_pt_written["families"].values())
    n_families = len(edges_pt_written["families"])

    print(f"[OK] undirected GT: nodes={n}, edges={len(full_edges)}")
    print(f"[OK] canonical directed edges: {directed_edges} (should equal undirected), active_families={n_families}/16")
    print(f"[OK] mode: canonical_single (MP reverse via get_computational_graph to_undirected)")
    print(f"[OK] output -> {full_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert HGEN PubMed baseline to DiHuG subgraph format.")
    parser.add_argument("--input-dir", type=str, default="data/raw/PubMed_baseline")
    parser.add_argument("--out-dir", type=str, default="data/PubMed_baseline_subgraphs")
    args = parser.parse_args()
    convert(args.input_dir, args.out_dir)


if __name__ == "__main__":
    main()
