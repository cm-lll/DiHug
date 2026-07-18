#!/usr/bin/env python3
"""Diagnose how PubMed block strategies preserve hub-core structure.

This script is read-only: it compares partition strategies on the same clean
PubMed graph and reports whether high-high family edges and triangles stay
inside blocks or are cut across blocks.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, pstdev

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sparse_diffusion.graph_partition.connected_blocks import (  # noqa: E402
    hetero_metis_blocks_from_graph,
)


ROLE_NAMES = ("low", "mid", "high")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/PubMed_baseline_subgraphs/processed"),
    )
    parser.add_argument("--edge-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/block_core_preservation"),
    )
    parser.add_argument(
        "--include-random-template",
        action="store_true",
        help="Also compare a random partition with the current block type-count template.",
    )
    return parser.parse_args()


def load_inputs(processed_dir: Path):
    graph = torch.load(
        processed_dir / "train.pt", map_location="cpu", weights_only=False
    )[0]
    with open(processed_dir / "vocab.json", "r", encoding="utf-8") as handle:
        vocab = json.load(handle)
    return graph, vocab


def canonical_edges(edge_index: torch.Tensor):
    u = torch.minimum(edge_index[0].long(), edge_index[1].long())
    v = torch.maximum(edge_index[0].long(), edge_index[1].long())
    keep = u != v
    return torch.stack([u[keep], v[keep]], dim=0), keep


def coefficient_of_variation(values):
    values = [float(x) for x in values]
    if not values or abs(mean(values)) < 1e-12:
        return 0.0
    return float(pstdev(values) / max(abs(mean(values)), 1e-12))


def gini(values):
    values = sorted(float(x) for x in values)
    if not values or sum(values) <= 0:
        return 0.0
    n = len(values)
    total = sum(values)
    return float(sum((2 * i - n - 1) * x for i, x in enumerate(values, 1)) / (n * total))


def build_blocks(
    graph,
    edge_index,
    edge_family,
    strategy: dict,
    rho: float,
):
    if strategy["kind"] == "metis":
        return hetero_metis_blocks_from_graph(
            edge_index,
            edge_family,
            int(graph.num_nodes),
            float(rho),
            relation_balance_power=float(strategy["relation_balance_power"]),
            node_type_local=graph.node_type.long(),
            refine_degree_balance=bool(strategy["refine_degree_balance"]),
            refine_max_iter=int(strategy["refine_max_iter"]),
            refine_preserve_high_high=bool(
                strategy.get("refine_preserve_high_high", False)
            ),
            refine_preserve_high_quantile=float(
                strategy.get("refine_preserve_high_quantile", 0.8)
            ),
            refine_preserve_penalty_weight=float(
                strategy.get("refine_preserve_penalty_weight", 0.0)
            ),
        )
    raise ValueError(f"Unknown strategy kind: {strategy['kind']}")


def random_template_blocks(node_types, template_blocks, seed: int):
    generator = torch.Generator().manual_seed(int(seed))
    type_ids = sorted(int(x) for x in torch.unique(node_types).tolist())
    template_counts = []
    for block in template_blocks:
        block_tensor = torch.tensor(block, dtype=torch.long)
        template_counts.append(
            {
                type_id: int((node_types[block_tensor] == type_id).sum())
                for type_id in type_ids
            }
        )

    output = [[] for _ in template_blocks]
    for type_id in type_ids:
        nodes = torch.where(node_types == type_id)[0]
        nodes = nodes[torch.randperm(nodes.numel(), generator=generator)]
        cursor = 0
        for block_idx, counts in enumerate(template_counts):
            take = int(counts[type_id])
            output[block_idx].extend(nodes[cursor : cursor + take].tolist())
            cursor += take
        assert cursor == nodes.numel()
    return [sorted(block) for block in output]


def block_of_nodes(blocks, num_nodes: int):
    block_of = torch.full((int(num_nodes),), -1, dtype=torch.long)
    for block_id, block in enumerate(blocks):
        if block:
            block_of[torch.tensor(block, dtype=torch.long)] = int(block_id)
    return block_of


def degree_roles(edge_index, node_type, num_nodes: int):
    degree = torch.zeros(int(num_nodes), dtype=torch.float)
    if edge_index.numel():
        ones = torch.ones(edge_index.shape[1], dtype=torch.float)
        degree.scatter_add_(0, edge_index[0].long(), ones)
        degree.scatter_add_(0, edge_index[1].long(), ones)
    roles = torch.ones(int(num_nodes), dtype=torch.long)
    thresholds = {}
    for type_id in sorted(int(x) for x in torch.unique(node_type).tolist()):
        mask = node_type.long() == int(type_id)
        vals = degree[mask]
        if vals.numel() == 0:
            continue
        p50 = float(torch.quantile(vals, 0.5).item())
        p80 = float(torch.quantile(vals, 0.8).item())
        local_roles = torch.ones_like(vals, dtype=torch.long)
        local_roles[vals <= p50] = 0
        local_roles[vals > p80] = 2
        roles[mask] = local_roles
        thresholds[int(type_id)] = {
            "p50": p50,
            "p80": p80,
            "max": float(vals.max().item()),
            "count": int(vals.numel()),
        }
    return degree, roles, thresholds


def family_names(vocab):
    mapping = vocab.get("edge_family2id", {})
    return {int(v): str(k) for k, v in mapping.items()}


def role_pair_name(src_role: int, dst_role: int, same_type: bool):
    a = int(src_role)
    b = int(dst_role)
    if same_type and a > b:
        a, b = b, a
    return f"{ROLE_NAMES[a]}-{ROLE_NAMES[b]}"


def edge_split_rows(
    strategy_name,
    edge_index,
    edge_family,
    node_type,
    roles,
    block_of,
    id2family,
):
    rows = []
    counters = defaultdict(lambda: Counter())
    u = edge_index[0].long()
    v = edge_index[1].long()
    intra = block_of[u] == block_of[v]
    for idx in range(edge_index.shape[1]):
        fam_id = int(edge_family[idx].item())
        same_type = int(node_type[u[idx]].item()) == int(node_type[v[idx]].item())
        pair = role_pair_name(int(roles[u[idx]].item()), int(roles[v[idx]].item()), same_type)
        key = (fam_id, pair)
        counters[key]["total"] += 1
        counters[key]["intra" if bool(intra[idx]) else "cross"] += 1
    for (fam_id, pair), c in sorted(counters.items()):
        total = int(c["total"])
        rows.append(
            {
                "strategy": strategy_name,
                "family_id": fam_id,
                "family": id2family.get(fam_id, str(fam_id)),
                "role_pair": pair,
                "total_edges": total,
                "intra_edges": int(c["intra"]),
                "cross_edges": int(c["cross"]),
                "intra_ratio": float(c["intra"] / max(total, 1)),
            }
        )
    return rows


def block_summary_rows(strategy_name, blocks, edge_index, edge_family, node_type, degree, id2family):
    block_of = block_of_nodes(blocks, int(node_type.numel()))
    rows = []
    for block_id, block in enumerate(blocks):
        nodes = torch.tensor(block, dtype=torch.long)
        type_counts = Counter(int(x) for x in node_type[nodes].tolist())
        edge_mask = (block_of[edge_index[0]] == block_id) & (block_of[edge_index[1]] == block_id)
        fam_counts = Counter(int(x) for x in edge_family[edge_mask].tolist())
        chem_nodes = nodes[node_type[nodes] == 2]
        chem_high = int(((node_type[nodes] == 2) & (degree[nodes] > torch.quantile(degree[node_type == 2], 0.8))).sum().item()) if (node_type == 2).any() else 0
        rows.append(
            {
                "strategy": strategy_name,
                "block_id": int(block_id),
                "nodes": int(nodes.numel()),
                "degree_load": float(degree[nodes].sum().item()),
                "intra_edges": int(edge_mask.sum().item()),
                "chemical_nodes": int(chem_nodes.numel()),
                "chemical_high_nodes": chem_high,
                "type_counts_json": json.dumps(dict(type_counts), sort_keys=True),
                "family_counts_json": json.dumps(
                    {id2family.get(k, str(k)): int(v) for k, v in fam_counts.items()},
                    sort_keys=True,
                ),
            }
        )
    return rows


def enumerate_triangles(edge_index, edge_family, node_type, block_of, id2family):
    adjacency = defaultdict(set)
    edge_family_by_key = {}
    for idx in range(edge_index.shape[1]):
        u = int(edge_index[0, idx].item())
        v = int(edge_index[1, idx].item())
        if u == v:
            continue
        if u > v:
            u, v = v, u
        adjacency[u].add(v)
        adjacency[v].add(u)
        edge_family_by_key[(u, v)] = int(edge_family[idx].item())

    counters = Counter()
    family_triangle = defaultdict(Counter)
    nodes_sorted = sorted(adjacency)
    for u in nodes_sorted:
        nbrs_u = [v for v in adjacency[u] if v > u]
        for v in nbrs_u:
            common = adjacency[u].intersection(adjacency[v])
            for w in common:
                if w <= v:
                    continue
                fams = [
                    edge_family_by_key.get(tuple(sorted((u, v))), -1),
                    edge_family_by_key.get(tuple(sorted((u, w))), -1),
                    edge_family_by_key.get(tuple(sorted((v, w))), -1),
                ]
                all_intra = (
                    int(block_of[u].item())
                    == int(block_of[v].item())
                    == int(block_of[w].item())
                )
                counters["all_total"] += 1
                counters["all_intra"] += int(all_intra)
                if all(int(node_type[x].item()) == 2 for x in (u, v, w)):
                    counters["chemical_total"] += 1
                    counters["chemical_intra"] += int(all_intra)
                for fam_id in set(f for f in fams if f >= 0):
                    family_triangle[fam_id]["total"] += 1
                    family_triangle[fam_id]["intra"] += int(all_intra)
    rows = [
        {
            "triangle_group": "all",
            "family_id": "",
            "family": "all",
            "total": int(counters["all_total"]),
            "intra": int(counters["all_intra"]),
            "intra_ratio": float(counters["all_intra"] / max(counters["all_total"], 1)),
        },
        {
            "triangle_group": "all_chemical_nodes",
            "family_id": "",
            "family": "all_chemical_nodes",
            "total": int(counters["chemical_total"]),
            "intra": int(counters["chemical_intra"]),
            "intra_ratio": float(counters["chemical_intra"] / max(counters["chemical_total"], 1)),
        },
    ]
    for fam_id, c in sorted(family_triangle.items()):
        rows.append(
            {
                "triangle_group": "contains_family_edge",
                "family_id": int(fam_id),
                "family": id2family.get(int(fam_id), str(fam_id)),
                "total": int(c["total"]),
                "intra": int(c["intra"]),
                "intra_ratio": float(c["intra"] / max(int(c["total"]), 1)),
            }
        )
    return rows


def strategy_summary(strategy_name, blocks, edge_index, edge_family, node_type, degree, roles, id2family):
    block_of = block_of_nodes(blocks, int(node_type.numel()))
    u = edge_index[0].long()
    v = edge_index[1].long()
    intra = block_of[u] == block_of[v]
    block_sizes = [len(b) for b in blocks]
    degree_loads = [
        float(degree[torch.tensor(b, dtype=torch.long)].sum().item()) if b else 0.0
        for b in blocks
    ]
    edge_loads = []
    chemical_high_counts = []
    chemical_mask = node_type == 2
    chemical_high_threshold = (
        torch.quantile(degree[chemical_mask], 0.8) if chemical_mask.any() else torch.tensor(float("inf"))
    )
    for block_id, block in enumerate(blocks):
        nodes = torch.tensor(block, dtype=torch.long)
        edge_loads.append(int(((block_of[u] == block_id) & (block_of[v] == block_id)).sum().item()))
        chemical_high_counts.append(
            int(((node_type[nodes] == 2) & (degree[nodes] > chemical_high_threshold)).sum().item())
            if nodes.numel()
            else 0
        )

    summary = {
        "strategy": strategy_name,
        "num_blocks": int(len(blocks)),
        "size_min": int(min(block_sizes)),
        "size_max": int(max(block_sizes)),
        "size_cv": coefficient_of_variation(block_sizes),
        "degree_load_min": float(min(degree_loads)),
        "degree_load_max": float(max(degree_loads)),
        "degree_load_cv": coefficient_of_variation(degree_loads),
        "edge_load_cv": coefficient_of_variation(edge_loads),
        "chemical_high_count_cv": coefficient_of_variation(chemical_high_counts),
        "chemical_high_count_gini": gini(chemical_high_counts),
        "edges_total": int(edge_index.shape[1]),
        "edges_intra": int(intra.sum().item()),
        "edge_intra_ratio": float(intra.float().mean().item()),
    }

    for fam_name in ("link_Chemical_Chemical", "link_Disease_Chemical", "link_Gene_Chemical"):
        fam_id = next((k for k, v_name in id2family.items() if v_name == fam_name), None)
        if fam_id is None:
            continue
        fam_mask = edge_family == int(fam_id)
        high_high = fam_mask & (roles[u] == 2) & (roles[v] == 2)
        summary[f"{fam_name}_edges"] = int(fam_mask.sum().item())
        summary[f"{fam_name}_intra_ratio"] = (
            float(intra[fam_mask].float().mean().item()) if fam_mask.any() else 0.0
        )
        summary[f"{fam_name}_high_high_edges"] = int(high_high.sum().item())
        summary[f"{fam_name}_high_high_intra_ratio"] = (
            float(intra[high_high].float().mean().item()) if high_high.any() else 0.0
        )
    return summary


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    graph, vocab = load_inputs(args.processed_dir)
    edge_index, keep = canonical_edges(graph.edge_index)
    edge_family = graph.edge_family.long()[keep]
    node_type = graph.node_type.long()
    id2family = family_names(vocab)
    degree, roles, thresholds = degree_roles(edge_index, node_type, int(graph.num_nodes))

    strategies = [
        {
            "name": "metis_unweighted_no_refine",
            "kind": "metis",
            "relation_balance_power": 0.0,
            "refine_degree_balance": False,
            "refine_max_iter": 0,
        },
        {
            "name": "metis_relation_balanced_no_refine",
            "kind": "metis",
            "relation_balance_power": 0.5,
            "refine_degree_balance": False,
            "refine_max_iter": 0,
        },
        {
            "name": "metis_mild_refine20",
            "kind": "metis",
            "relation_balance_power": 0.5,
            "refine_degree_balance": True,
            "refine_max_iter": 20,
        },
        {
            "name": "metis_preserve_refine50_w2",
            "kind": "metis",
            "relation_balance_power": 0.5,
            "refine_degree_balance": True,
            "refine_max_iter": 50,
            "refine_preserve_high_high": True,
            "refine_preserve_high_quantile": 0.8,
            "refine_preserve_penalty_weight": 2.0,
        },
        {
            "name": "metis_preserve_refine200_w2",
            "kind": "metis",
            "relation_balance_power": 0.5,
            "refine_degree_balance": True,
            "refine_max_iter": 200,
            "refine_preserve_high_high": True,
            "refine_preserve_high_quantile": 0.8,
            "refine_preserve_penalty_weight": 2.0,
        },
        {
            "name": "metis_current_refine200",
            "kind": "metis",
            "relation_balance_power": 0.5,
            "refine_degree_balance": True,
            "refine_max_iter": 200,
        },
    ]

    built = {}
    for strategy in strategies:
        built[strategy["name"]] = build_blocks(
            graph, edge_index, edge_family, strategy, args.edge_fraction
        )
    if args.include_random_template:
        built["random_type_template_current"] = random_template_blocks(
            node_type,
            built["metis_current_refine200"],
            seed=int(args.seed),
        )

    summary_rows = []
    edge_rows = []
    triangle_rows = []
    block_rows = []
    for strategy_name, blocks in built.items():
        block_of = block_of_nodes(blocks, int(graph.num_nodes))
        summary_rows.append(
            strategy_summary(
                strategy_name,
                blocks,
                edge_index,
                edge_family,
                node_type,
                degree,
                roles,
                id2family,
            )
        )
        edge_rows.extend(
            edge_split_rows(
                strategy_name,
                edge_index,
                edge_family,
                node_type,
                roles,
                block_of,
                id2family,
            )
        )
        for row in enumerate_triangles(edge_index, edge_family, node_type, block_of, id2family):
            triangle_rows.append({"strategy": strategy_name, **row})
        block_rows.extend(
            block_summary_rows(
                strategy_name,
                blocks,
                edge_index,
                edge_family,
                node_type,
                degree,
                id2family,
            )
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "strategy_summary.csv", summary_rows)
    write_csv(args.output_dir / "family_role_edge_split.csv", edge_rows)
    write_csv(args.output_dir / "triangle_preservation.csv", triangle_rows)
    write_csv(args.output_dir / "block_summary.csv", block_rows)
    with open(args.output_dir / "degree_role_thresholds.json", "w", encoding="utf-8") as handle:
        json.dump(thresholds, handle, indent=2, ensure_ascii=False)
    with open(args.output_dir / "blocks.json", "w", encoding="utf-8") as handle:
        json.dump({k: [list(map(int, b)) for b in v] for k, v in built.items()}, handle)

    print(f"Saved diagnostics to {args.output_dir}")
    important = [
        "strategy",
        "degree_load_cv",
        "edge_load_cv",
        "chemical_high_count_cv",
        "edge_intra_ratio",
        "link_Chemical_Chemical_intra_ratio",
        "link_Chemical_Chemical_high_high_intra_ratio",
    ]
    print("\t".join(important))
    for row in summary_rows:
        print(
            "\t".join(
                f"{row[k]:.4f}" if isinstance(row.get(k), float) else str(row.get(k, ""))
                for k in important
            )
        )
    print("\nTriangle preservation:")
    for row in triangle_rows:
        if row["triangle_group"] in {"all", "all_chemical_nodes"} or row["family"] == "link_Chemical_Chemical":
            print(
                f"{row['strategy']:34s} {row['triangle_group']:22s} "
                f"{row['family']:28s} total={row['total']} intra_ratio={row['intra_ratio']:.4f}"
            )


if __name__ == "__main__":
    main()
