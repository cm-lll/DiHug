#!/usr/bin/env python3
"""
Convert raw synthetic folders (syn_100/syn_200/syn_500) into
ACMSubgraphs-compatible layout for the `hghg_subgraphs` dataset loader.

Expected raw layout:
  <raw_root>/
    syn_100/
      node.dat
      link.dat
    syn_200/
      node.dat
      link.dat
    syn_500/
      node.dat
      link.dat

Output layout:
  <out_root>/
    subgraph_000/
      nodes.pt
      edges.pt
      meta.json
    subgraph_001/
      ...
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from itertools import product
from typing import Dict, List, Set, Tuple

import torch


def _read_nodes(node_path: str) -> Tuple[Dict[int, str], List[str]]:
    """
    node.dat line format:
      <node_id> <node_type> <type_id_or_aux>
    Example:
      0 A 0
      30 B 1
    """
    node_type_map: Dict[int, str] = {}
    seen_types: Set[str] = set()
    with open(node_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = s.split()
            if len(parts) < 2:
                continue
            nid = int(parts[0])
            ntype = str(parts[1])
            node_type_map[nid] = ntype
            seen_types.add(ntype)
    node_types = sorted(seen_types)
    return node_type_map, node_types


def _read_undirected_edges(link_path: str, num_nodes: int) -> List[Tuple[int, int]]:
    """
    link.dat line format:
      <src> <dst> <aux>
    We ignore aux and keep unique undirected simple edges.
    """
    edges: Set[Tuple[int, int]] = set()
    with open(link_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = s.split()
            if len(parts) < 2:
                continue
            u = int(parts[0])
            v = int(parts[1])
            if u == v:
                continue  # drop self-loops
            if not (0 <= u < num_nodes and 0 <= v < num_nodes):
                continue
            a, b = (u, v) if u < v else (v, u)
            edges.add((a, b))
    return sorted(edges)


def _build_nodes_pt(
    node_type_map: Dict[int, str],
    node_types: List[str],
) -> Tuple[Dict, Dict[int, int], Dict[str, int], Dict[str, List[int]]]:
    """
    Build nodes.pt with single subtype per node type (subtype=0).
    Returns:
      nodes_pt, global2local, type_offsets, per_type_ids
    """
    per_type_ids: Dict[str, List[int]] = {t: [] for t in node_types}
    for nid, t in sorted(node_type_map.items()):
        per_type_ids[t].append(nid)

    global2local: Dict[int, int] = {}
    for t in node_types:
        for li, gid in enumerate(per_type_ids[t]):
            global2local[gid] = li

    nodes_pt = {}
    for t in node_types:
        ids = per_type_ids[t]
        nodes_pt[t] = {
            "ids": ids,
            "subtype": torch.zeros(len(ids), dtype=torch.long),  # single subtype
            "A": 1,
        }

    # kept for completeness/debug (not required by nodes.pt itself)
    type_offsets = {}
    cur = 0
    for t in node_types:
        type_offsets[t] = cur
        cur += 1

    return nodes_pt, global2local, type_offsets, per_type_ids


def _build_families(node_types: List[str]) -> Dict[str, Dict[str, str]]:
    fam_endpoints = {}
    for src_t, dst_t in product(node_types, node_types):
        fam = f"link_{src_t}_{dst_t}"
        fam_endpoints[fam] = {"src_type": src_t, "dst_type": dst_t}
    return fam_endpoints


def _build_edges_pt(
    undirected_edges: List[Tuple[int, int]],
    node_type_map: Dict[int, str],
    global2local: Dict[int, int],
) -> Dict:
    """
    Build directed family edges from undirected edges.
    Each directed edge has y=1 (existence-only).
    """
    buckets = defaultdict(lambda: {"src_local": [], "dst_local": [], "y": []})

    for u, v in undirected_edges:
        tu = node_type_map[u]
        tv = node_type_map[v]
        fam_uv = f"link_{tu}_{tv}"
        fam_vu = f"link_{tv}_{tu}"

        buckets[fam_uv]["src_local"].append(global2local[u])
        buckets[fam_uv]["dst_local"].append(global2local[v])
        buckets[fam_uv]["y"].append(1)

        buckets[fam_vu]["src_local"].append(global2local[v])
        buckets[fam_vu]["dst_local"].append(global2local[u])
        buckets[fam_vu]["y"].append(1)

    families = {}
    for fam, d in buckets.items():
        families[fam] = {
            "src_local": torch.tensor(d["src_local"], dtype=torch.long),
            "dst_local": torch.tensor(d["dst_local"], dtype=torch.long),
            "y": torch.tensor(d["y"], dtype=torch.long),
        }
    return {"families": families}


def _build_meta(
    node_types: List[str],
    fam_endpoints: Dict[str, Dict[str, str]],
    per_type_ids: Dict[str, List[int]],
    edges_pt: Dict,
) -> Dict:
    schema_by_type = {t: ["default"] for t in node_types}
    fam_label2id = {fam: {f"{fam}:__none__": 1} for fam in fam_endpoints}
    fam_id2label = {fam: {"1": f"{fam}:__none__"} for fam in fam_endpoints}

    num_edges_by_family = {}
    for fam in fam_endpoints:
        fd = edges_pt["families"].get(fam)
        num_edges_by_family[fam] = int(fd["y"].numel()) if fd is not None else 0

    return {
        "node_types": node_types,
        "schema_by_type": schema_by_type,
        "fam_endpoints": fam_endpoints,
        "fam_label2id": fam_label2id,
        "fam_id2label": fam_id2label,
        "unknown_edges_dropped": 0,
        "num_nodes_by_type": {t: len(per_type_ids[t]) for t in node_types},
        "num_edges_by_family": num_edges_by_family,
    }


def convert_one(raw_dir: str, out_subdir: str) -> None:
    node_path = os.path.join(raw_dir, "node.dat")
    link_path = os.path.join(raw_dir, "link.dat")
    if not (os.path.exists(node_path) and os.path.exists(link_path)):
        raise FileNotFoundError(f"Missing node.dat/link.dat in {raw_dir}")

    node_type_map, node_types = _read_nodes(node_path)
    num_nodes = len(node_type_map)
    undirected_edges = _read_undirected_edges(link_path, num_nodes=num_nodes)

    nodes_pt, global2local, _, per_type_ids = _build_nodes_pt(node_type_map, node_types)
    fam_endpoints = _build_families(node_types)
    edges_pt = _build_edges_pt(undirected_edges, node_type_map, global2local)
    meta = _build_meta(node_types, fam_endpoints, per_type_ids, edges_pt)

    os.makedirs(out_subdir, exist_ok=True)
    torch.save(nodes_pt, os.path.join(out_subdir, "nodes.pt"))
    torch.save(edges_pt, os.path.join(out_subdir, "edges.pt"))
    with open(os.path.join(out_subdir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(
        f"[OK] {out_subdir} | N={num_nodes}, undirected_E={len(undirected_edges)}, "
        f"directed_E={2*len(undirected_edges)}, node_types={node_types}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert syn_* raw folders into hghg_subgraphs format."
    )
    parser.add_argument(
        "--raw-root",
        type=str,
        default="data/raw",
        help="Root containing syn_100/syn_200/syn_500 folders.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="data/HGHG_subgraphs",
        help="Output directory with subgraph_*/",
    )
    parser.add_argument(
        "--folders",
        type=str,
        default="syn_100,syn_200,syn_500",
        help="Comma-separated raw folder names under raw-root.",
    )
    args = parser.parse_args()

    folders = [x.strip() for x in args.folders.split(",") if x.strip()]
    os.makedirs(args.out_dir, exist_ok=True)

    idx = 0
    for fd in folders:
        raw_dir = os.path.join(args.raw_root, fd)
        out_subdir = os.path.join(args.out_dir, f"subgraph_{idx:03d}")
        convert_one(raw_dir, out_subdir)
        idx += 1

    print(f"[DONE] Converted {idx} graphs -> {args.out_dir}")


if __name__ == "__main__":
    main()
