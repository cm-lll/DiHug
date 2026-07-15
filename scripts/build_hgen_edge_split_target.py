#!/usr/bin/env python3
"""Convert HGEN's held-out edge graph into the DiHuG typed-graph schema.

HGEN's default real-world protocol keeps every node and evaluates against 40%
held-out edges (seed 481516234).  It is not a 40% node-induced subgraph.  This
adapter consumes HGEN's already materialized ``test_graph.p`` so the comparison
uses exactly the same target edges and node ordering as the baseline.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import networkx as nx
import scipy.sparse as sp
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from sparse_diffusion.structure_energy.graph import TypedGraph, load_typed_graph
from sparse_diffusion.structure_energy.switch_chain import graph_metrics


def _raw_to_dense(dataset_dir: Path) -> dict[int, int]:
    nodes = torch.load(dataset_dir / "nodes.pt", map_location="cpu", weights_only=False)
    meta = json.loads((dataset_dir / "meta.json").read_text(encoding="utf-8"))
    type_names = tuple(meta.get("node_types") or nodes.keys())
    ordered_raw_ids = [
        int(raw_id)
        for type_name in type_names
        for raw_id in nodes[type_name]["ids"]
    ]
    if len(set(ordered_raw_ids)) != len(ordered_raw_ids):
        raise ValueError("dataset node ids are not unique")
    return {raw_id: dense_id for dense_id, raw_id in enumerate(ordered_raw_ids)}


def build_target(dataset_dir: Path, test_graph_path: Path) -> TypedGraph:
    full = load_typed_graph(dataset_dir, active_only=True)
    raw_to_dense = _raw_to_dense(dataset_dir)
    with test_graph_path.open("rb") as handle:
        held_out = pickle.load(handle).tocsr()
    held_out.eliminate_zeros()
    if held_out.shape != (len(raw_to_dense), len(raw_to_dense)):
        raise ValueError("HGEN test graph and DiHuG dataset node counts disagree")

    family_edges = {family: [] for family in full.family_specs}
    rows, cols = held_out.nonzero()
    seen = set()
    for raw_u, raw_v in zip(rows.tolist(), cols.tolist()):
        if raw_u == raw_v:
            continue
        raw_edge = (min(int(raw_u), int(raw_v)), max(int(raw_u), int(raw_v)))
        if raw_edge in seen:
            continue
        seen.add(raw_edge)
        u, v = raw_to_dense[raw_edge[0]], raw_to_dense[raw_edge[1]]
        family = full.family_of(u, v)
        if family is None:
            raise ValueError(f"held-out edge {raw_edge} is absent from the full graph")
        family_edges[family].append(full.orient_edge(family, (u, v)))

    target = TypedGraph(
        full.node_types.copy(),
        full.type_names,
        full.family_specs,
        family_edges,
    )
    target.validate()
    return target


def _full_raw_adjacency(dataset_dir: Path) -> tuple[TypedGraph, dict[int, int], sp.csr_matrix]:
    full = load_typed_graph(dataset_dir, active_only=True)
    raw_to_dense = _raw_to_dense(dataset_dir)
    dense_to_raw = {dense: raw for raw, dense in raw_to_dense.items()}
    rows = []
    cols = []
    for edges in full.family_edges.values():
        for u, v in edges:
            raw_u, raw_v = dense_to_raw[int(u)], dense_to_raw[int(v)]
            rows.extend((raw_u, raw_v))
            cols.extend((raw_v, raw_u))
    values = np.ones(len(rows), dtype=np.int8)
    adjacency = sp.csr_matrix(
        (values, (rows, cols)), shape=(len(raw_to_dense), len(raw_to_dense))
    )
    adjacency.eliminate_zeros()
    return full, raw_to_dense, adjacency


def _hgen_test_edges(adjacency: sp.csr_matrix, test_share: float, seed: int) -> np.ndarray:
    """Reproduce HGEN's every-node edge-cover split for positive test edges."""

    if not 0.0 < float(test_share) < 1.0:
        raise ValueError("test share must be in (0, 1)")
    adjacency = sp.tril(adjacency).tocsr()
    adjacency.eliminate_zeros()
    rng = np.random.RandomState(int(seed))
    edge_count = int(adjacency.nnz)
    node_count = int(adjacency.shape[0])
    train_size = int(edge_count * (1.0 - float(test_share)))

    matching = list(nx.maximal_matching(nx.Graph(adjacency)))
    hold_edges = np.asarray(
        [
            (int(u), int(v))
            if adjacency[int(u), int(v)]
            else (int(v), int(u))
            for u, v in matching
        ],
        dtype=np.int64,
    )
    covered = set(hold_edges.reshape(-1).tolist()) if hold_edges.size else set()
    uncovered = np.asarray(
        sorted(set(range(node_count)).difference(covered)), dtype=np.int64
    )
    minimum_train_size = int(hold_edges.shape[0] + len(uncovered))
    if minimum_train_size > train_size:
        raise ValueError("training share is too small for HGEN's node-cover split")

    degree = adjacency.sum(1).A1
    uncovered_degree = degree[uncovered]
    extra = []
    for node in uncovered[uncovered_degree > 0]:
        neighbors = adjacency[int(node)].indices
        extra.append((int(node), int(rng.choice(neighbors))))
    for node in uncovered[uncovered_degree == 0]:
        neighbors = adjacency[:, int(node)].T.tocsr().indices
        # HGEN's observed graphs have no dangling nodes. Learned prior samples
        # may contain a few true isolates, for which no cover edge exists.
        if len(neighbors):
            extra.append((int(rng.choice(neighbors)), int(node)))
    if extra:
        extra_array = np.asarray(extra, dtype=np.int64)
        hold_edges = (
            np.row_stack((hold_edges, extra_array)) if hold_edges.size else extra_array
        )

    hold = sp.csr_matrix(
        (np.ones(len(hold_edges)), (hold_edges[:, 0], hold_edges[:, 1])),
        shape=adjacency.shape,
    )
    hold[hold > 1] = 1
    hold.eliminate_zeros()
    sample = adjacency - hold
    sample.eliminate_zeros()
    remaining_train = train_size - int(hold.nnz)
    ones = np.column_stack(sample.nonzero())
    permutation = rng.permutation(sample.nnz)
    return ones[permutation[remaining_train:]]


def build_generated_split_target(
    dataset_dir: Path, test_share: float, split_seed: int
) -> TypedGraph:
    full, raw_to_dense, adjacency = _full_raw_adjacency(dataset_dir)
    test_edges = _hgen_test_edges(adjacency, test_share, split_seed)
    family_edges = {family: [] for family in full.family_specs}
    for raw_u, raw_v in test_edges.tolist():
        u, v = raw_to_dense[int(raw_u)], raw_to_dense[int(raw_v)]
        family = full.family_of(u, v)
        if family is None:
            raise AssertionError("generated HGEN split contains a non-edge")
        family_edges[family].append(full.orient_edge(family, (u, v)))
    target = TypedGraph(
        full.node_types.copy(), full.type_names, full.family_specs, family_edges
    )
    target.validate()
    return target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--hgen-test-graph", type=Path)
    parser.add_argument("--generate-split", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--test-share", type=float, default=0.4)
    parser.add_argument("--split-seed", type=int, default=481516234)
    args = parser.parse_args()

    if args.generate_split == (args.hgen_test_graph is not None):
        raise ValueError(
            "choose exactly one of --hgen-test-graph or --generate-split"
        )
    target = (
        build_generated_split_target(args.dataset_dir, args.test_share, args.split_seed)
        if args.generate_split
        else build_target(args.dataset_dir, args.hgen_test_graph)
    )
    full = load_typed_graph(args.dataset_dir, active_only=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "target_graph.json").write_text(
        json.dumps(target.to_json_dict(), indent=2, sort_keys=True), encoding="utf-8"
    )
    report = {
        "format": "dihug-hgen-edge-split-target-v1",
        "protocol": "full_node_set_held_out_edges",
        "split_source": "generated_hgen_algorithm" if args.generate_split else "materialized_hgen_graph",
        "test_share": float(args.test_share),
        "split_seed": int(args.split_seed),
        "full_metrics": graph_metrics(full),
        "target_metrics": graph_metrics(target),
        "target_family_counts": {
            family: len(target.family_edges[family])
            for family in sorted(target.family_edges)
        },
        "generation_budget": {
            "num_nodes": target.num_nodes,
            "num_edges": target.num_edges,
            "reveals_target_family_counts": False,
            "reveals_target_degrees": False,
            "reveals_target_adjacency": False,
        },
    }
    (args.output_dir / "split_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        "[HGEN-EDGE-SPLIT] "
        f"nodes={target.num_nodes} edges={target.num_edges} "
        f"triangles={report['target_metrics']['triangles']:.0f} "
        f"clustering={report['target_metrics']['clustering']:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
