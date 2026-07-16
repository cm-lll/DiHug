#!/usr/bin/env python3
"""
Convert HGEN-style single-graph datasets into DiHuG subgraph_* layout.

Expected input:
  <input-dir>/adj_matrix.p
  <input-dir>/new_node_dict.p

The output follows the PubMed_baseline_subgraphs convention used by
ACMSubgraphsDataset:
  <out-dir>/subgraph_000/nodes.pt
  <out-dir>/subgraph_000/edges.pt
  <out-dir>/subgraph_000/meta.json
  <out-dir>/processed/splits.pt

Ground truth is treated as undirected. Each simple undirected edge is stored once
in canonical global-id direction; reverse message-passing edges are added later
when the computational graph is built.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
from collections import defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple

import torch


PRESET_NODE_TYPES = {
    "dblp_four_area": ["Paper", "Author", "Term", "Conference"],
    "imdb_movie": ["Director", "Actor", "Movie", "Genre"],
}


def _load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def _normalize_name(name: str) -> str:
    return name.strip().replace(" ", "_").replace("-", "_")


def _parse_node_types(raw: str | None, preset: str | None, observed_type_ids: Sequence[int]) -> List[str]:
    if raw:
        node_types = [_normalize_name(x) for x in raw.split(",") if x.strip()]
    elif preset:
        key = preset.lower()
        if key not in PRESET_NODE_TYPES:
            raise ValueError(f"Unknown preset {preset!r}; choices: {sorted(PRESET_NODE_TYPES)}")
        node_types = PRESET_NODE_TYPES[key]
    else:
        node_types = [f"Type{tid}" for tid in sorted(observed_type_ids)]

    expected = len(set(observed_type_ids))
    if len(node_types) != expected:
        raise ValueError(
            f"Node type name count mismatch: got {len(node_types)} names for {expected} observed ids "
            f"{sorted(set(observed_type_ids))}"
        )
    return node_types


def _node_type_ids_from_mapping(node_dict, n_nodes: int) -> List[int]:
    if isinstance(node_dict, dict):
        missing = [i for i in range(n_nodes) if i not in node_dict]
        if missing:
            raise ValueError(f"new_node_dict is missing node ids, first missing={missing[:5]}")
        return [int(node_dict[i]) for i in range(n_nodes)]
    if isinstance(node_dict, (list, tuple)) and len(node_dict) == n_nodes:
        return [int(x) for x in node_dict]
    raise TypeError(f"Unsupported new_node_dict type/shape: {type(node_dict)}")


def _undirected_edges_from_sparse(adj) -> List[Tuple[int, int]]:
    rows, cols = adj.nonzero()
    edges = set()
    for u, v in zip(rows.tolist(), cols.tolist()):
        if u == v:
            continue
        a, b = (u, v) if u < v else (v, u)
        edges.add((int(a), int(b)))
    return sorted(edges)


def _build_fam_endpoints(node_types: Sequence[str]) -> Dict[str, Dict[str, str]]:
    return {
        f"link_{src_t}_{dst_t}": {"src_type": src_t, "dst_type": dst_t}
        for src_index, src_t in enumerate(node_types)
        for dst_t in node_types[src_index:]
    }


def _build_nodes_pt(
    node_types: Sequence[str],
    node_type_ids: Sequence[int],
) -> Tuple[Dict[str, Dict], Dict[int, int], Dict[str, List[int]]]:
    id_to_name = {tid: node_types[pos] for pos, tid in enumerate(sorted(set(node_type_ids)))}
    per_type_nodes: Dict[str, List[int]] = {t: [] for t in node_types}
    for nid, tid in enumerate(node_type_ids):
        per_type_nodes[id_to_name[int(tid)]].append(nid)

    global_to_local: Dict[int, int] = {}
    for t in node_types:
        for local_idx, gid in enumerate(per_type_nodes[t]):
            global_to_local[gid] = local_idx

    nodes_pt = {}
    for t in node_types:
        ids = per_type_nodes[t]
        nodes_pt[t] = {
            "ids": ids,
            "subtype": torch.zeros(len(ids), dtype=torch.long),
            "A": 1,
        }
    return nodes_pt, global_to_local, per_type_nodes


def _build_edges_pt(
    node_types: Sequence[str],
    node_type_ids: Sequence[int],
    undirected_edges: Iterable[Tuple[int, int]],
    global_to_local: Dict[int, int],
) -> Dict:
    id_to_name = {tid: node_types[pos] for pos, tid in enumerate(sorted(set(node_type_ids)))}
    type_rank = {name: rank for rank, name in enumerate(node_types)}
    buckets = defaultdict(lambda: {"src_local": [], "dst_local": [], "y": []})

    for a, b in undirected_edges:
        src_t = id_to_name[int(node_type_ids[a])]
        dst_t = id_to_name[int(node_type_ids[b])]
        if type_rank[src_t] > type_rank[dst_t]:
            a, b = b, a
            src_t, dst_t = dst_t, src_t
        fam = f"link_{src_t}_{dst_t}"
        buckets[fam]["src_local"].append(global_to_local[a])
        buckets[fam]["dst_local"].append(global_to_local[b])
        buckets[fam]["y"].append(1)

    families = {}
    for fam, values in sorted(buckets.items()):
        families[fam] = {
            "src_local": torch.tensor(values["src_local"], dtype=torch.long),
            "dst_local": torch.tensor(values["dst_local"], dtype=torch.long),
            "y": torch.tensor(values["y"], dtype=torch.long),
        }
    return {"families": families}


def _build_meta(
    dataset_name: str,
    node_types: Sequence[str],
    per_type_nodes: Dict[str, List[int]],
    edges_pt: Dict,
    fam_endpoints: Dict[str, Dict[str, str]],
    num_undirected_edges: int,
) -> Dict:
    fam_label2id = {fam: {f"{fam}:__none__": 1} for fam in fam_endpoints}
    fam_id2label = {fam: {"1": f"{fam}:__none__"} for fam in fam_endpoints}
    num_edges_by_family = {
        fam: int(edges_pt["families"].get(fam, {"y": torch.empty(0, dtype=torch.long)})["y"].numel())
        for fam in fam_endpoints
    }

    return {
        "node_types": list(node_types),
        "schema_by_type": {t: ["default"] for t in node_types},
        "fam_endpoints": fam_endpoints,
        "fam_label2id": fam_label2id,
        "fam_id2label": fam_id2label,
        "unknown_edges_dropped": 0,
        "num_nodes_by_type": {t: len(per_type_nodes[t]) for t in node_types},
        "num_edges_by_family": num_edges_by_family,
        "num_undirected_edges": int(num_undirected_edges),
        "undirected_ground_truth": True,
        "bidirectional_mode": "canonical_single",
        "source": f"HGEN {dataset_name} (full graph, undirected GT + canonical single directed edge)",
    }


def _write_ready_marker(out_dir: str) -> None:
    marker = os.path.join(out_dir, "raw", "_acm_subgraphs_ready.txt")
    os.makedirs(os.path.dirname(marker), exist_ok=True)
    with open(marker, "a", encoding="utf-8"):
        pass


def _write_splits(out_dir: str) -> None:
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


def convert(input_dir: str, out_dir: str, dataset_name: str, preset: str | None, node_types_raw: str | None) -> None:
    adj = _load_pickle(os.path.join(input_dir, "adj_matrix.p"))
    node_dict = _load_pickle(os.path.join(input_dir, "new_node_dict.p"))

    n_nodes = int(adj.shape[0])
    node_type_ids = _node_type_ids_from_mapping(node_dict, n_nodes)
    observed_type_ids = sorted(set(node_type_ids))
    if observed_type_ids != list(range(len(observed_type_ids))):
        raise ValueError(f"Expected contiguous type ids from 0, got {observed_type_ids}")

    node_types = _parse_node_types(node_types_raw, preset, observed_type_ids)
    undirected_edges = _undirected_edges_from_sparse(adj)
    fam_endpoints = _build_fam_endpoints(node_types)

    full_dir = os.path.join(out_dir, "subgraph_000")
    os.makedirs(full_dir, exist_ok=True)

    nodes_pt, global_to_local, per_type_nodes = _build_nodes_pt(node_types, node_type_ids)
    edges_pt = _build_edges_pt(node_types, node_type_ids, undirected_edges, global_to_local)
    meta = _build_meta(dataset_name, node_types, per_type_nodes, edges_pt, fam_endpoints, len(undirected_edges))

    torch.save(nodes_pt, os.path.join(full_dir, "nodes.pt"))
    torch.save(edges_pt, os.path.join(full_dir, "edges.pt"))
    with open(os.path.join(full_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    stale = os.path.join(out_dir, "subgraph_001")
    if os.path.isdir(stale):
        shutil.rmtree(stale)
    _write_splits(out_dir)
    _write_ready_marker(out_dir)

    active_families = {fam: int(fd["y"].numel()) for fam, fd in edges_pt["families"].items()}
    print(f"[OK] {dataset_name}: nodes={n_nodes}, undirected_edges={len(undirected_edges)}")
    print(f"[OK] node_types={node_types}")
    print(f"[OK] num_nodes_by_type={meta['num_nodes_by_type']}")
    print(f"[OK] active_families={len(active_families)}/{len(fam_endpoints)}")
    for fam, count in sorted(active_families.items(), key=lambda item: (-item[1], item[0]))[:12]:
        print(f"  {fam}: {count}")
    print(f"[OK] output -> {full_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert HGEN single-graph data to DiHuG subgraph format.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--preset", default=None, choices=sorted(PRESET_NODE_TYPES))
    parser.add_argument("--node-types", default=None, help="Comma-separated type names ordered by numeric type id.")
    args = parser.parse_args()
    convert(args.input_dir, args.out_dir, args.dataset_name, args.preset, args.node_types)


if __name__ == "__main__":
    main()
