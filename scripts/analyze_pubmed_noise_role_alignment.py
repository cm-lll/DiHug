#!/usr/bin/env python3
"""Compare forward-noised PubMed graphs with sampling-style randomized graphs."""

import argparse
import csv
import json
import math
import pickle
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sparse_diffusion.graph_partition.connected_blocks import hetero_metis_blocks_from_graph


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/PubMed_baseline_subgraphs/processed"),
    )
    parser.add_argument("--edge-fraction", type=float, default=0.5)
    parser.add_argument("--diffusion-steps", type=int, default=100)
    parser.add_argument("--times", type=float, nargs="+", default=[0, 0.25, 0.5, 0.75, 0.95, 1.0])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/noise_role_alignment"),
    )
    return parser.parse_args()


def cosine_betas(timesteps, s=0.008):
    steps = int(timesteps) + 2
    x = np.linspace(0, steps, steps)
    alpha_bar = np.cos(0.5 * np.pi * ((x / steps) + s) / (1 + s)) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]
    alpha = alpha_bar[1:] / alpha_bar[:-1]
    return np.clip(1.0 - alpha, a_min=0.0, a_max=0.999)


def load_inputs(processed_dir):
    graph = torch.load(processed_dir / "train.pt", map_location="cpu", weights_only=False)[0]
    with open(processed_dir / "vocab.json", "r", encoding="utf-8") as handle:
        vocab = json.load(handle)
    with open(processed_dir / "train_edge_family_avg_counts.pickle", "rb") as handle:
        avg_counts = pickle.load(handle)
    return graph, vocab, avg_counts


def canonical_edges(edge_index):
    u = torch.minimum(edge_index[0].long(), edge_index[1].long())
    v = torch.maximum(edge_index[0].long(), edge_index[1].long())
    keep = u != v
    return torch.stack([u[keep], v[keep]], dim=0)


def family_endpoints(family_name, node_type2id):
    _, src_name, dst_name = family_name.split("_", 2)
    return int(node_type2id[src_name]), int(node_type2id[dst_name])


def candidate_edges(nodes_src, nodes_dst, same_type):
    if same_type:
        pairs = torch.triu_indices(nodes_src.numel(), nodes_src.numel(), offset=1)
        return torch.stack([nodes_src[pairs[0]], nodes_src[pairs[1]]], dim=0)
    flat = torch.arange(nodes_src.numel() * nodes_dst.numel())
    return torch.stack(
        [nodes_src[flat // nodes_dst.numel()], nodes_dst[flat % nodes_dst.numel()]],
        dim=0,
    )


def build_family_candidates(node_types, vocab, avg_counts):
    result = {}
    node_type2id = vocab["node_type2id"]
    for family_name, family_id in vocab["edge_family2id"].items():
        if float(avg_counts.get(family_name, 0.0)) <= 0:
            continue
        src_type, dst_type = family_endpoints(family_name, node_type2id)
        src = torch.where(node_types == src_type)[0]
        dst = torch.where(node_types == dst_type)[0]
        result[int(family_id)] = {
            "name": family_name,
            "edges": candidate_edges(src, dst, src_type == dst_type),
            "density": float(avg_counts[family_name])
            / max(int(candidate_edges(src, dst, src_type == dst_type).shape[1]), 1),
        }
    return result


def clean_family_keys(graph, num_nodes):
    edge_index = canonical_edges(graph.edge_index)
    keys_by_family = {}
    for family_id in torch.unique(graph.edge_family.long()).tolist():
        mask = graph.edge_family.long() == int(family_id)
        fam_edges = canonical_edges(graph.edge_index[:, mask])
        keys_by_family[int(family_id)] = set(
            (fam_edges[0] * int(num_nodes) + fam_edges[1]).tolist()
        )
    return keys_by_family


def metis_blocks(graph, edge_fraction):
    return hetero_metis_blocks_from_graph(
        canonical_edges(graph.edge_index),
        graph.edge_family.long(),
        int(graph.num_nodes),
        float(edge_fraction),
        relation_balance_power=0.5,
        node_type_local=graph.node_type.long(),
        refine_degree_balance=True,
        refine_max_iter=200,
    )


def random_template_blocks(node_types, real_blocks, generator):
    block_sets = [set(block) for block in real_blocks]
    type_ids = sorted(torch.unique(node_types).tolist())
    template_counts = []
    for block in real_blocks:
        block_tensor = torch.tensor(block, dtype=torch.long)
        template_counts.append(
            {int(type_id): int((node_types[block_tensor] == int(type_id)).sum()) for type_id in type_ids}
        )

    output = [[] for _ in real_blocks]
    for type_id in type_ids:
        nodes = torch.where(node_types == int(type_id))[0]
        nodes = nodes[torch.randperm(nodes.numel(), generator=generator)]
        cursor = 0
        for block_idx, counts in enumerate(template_counts):
            take = int(counts[type_id])
            output[block_idx].extend(nodes[cursor : cursor + take].tolist())
            cursor += take
        assert cursor == nodes.numel()
    assert sum(len(block) for block in output) == node_types.numel()
    assert len(set().union(*(set(block) for block in output))) == node_types.numel()
    del block_sets
    return [sorted(block) for block in output]


def block_marginal_graph(family_candidates, blocks, generator):
    block_of = {}
    for block_idx, block in enumerate(blocks):
        for node in block:
            block_of[int(node)] = int(block_idx)

    selected = []
    selected_family = []
    for family_id, info in family_candidates.items():
        edges = info["edges"]
        density = float(info["density"])
        bu = torch.tensor([block_of[int(x)] for x in edges[0].tolist()])
        bv = torch.tensor([block_of[int(x)] for x in edges[1].tolist()])
        pair_a = torch.minimum(bu, bv)
        pair_b = torch.maximum(bu, bv)
        for pair in torch.stack([pair_a, pair_b], dim=1).unique(dim=0):
            mask = (pair_a == pair[0]) & (pair_b == pair[1])
            local = mask.nonzero(as_tuple=True)[0]
            quota = min(int(round(density * local.numel())), int(local.numel()))
            if quota <= 0:
                continue
            chosen = local[torch.randperm(local.numel(), generator=generator)[:quota]]
            selected.append(edges[:, chosen])
            selected_family.append(torch.full((quota,), int(family_id), dtype=torch.long))
    return torch.cat(selected, dim=1), torch.cat(selected_family)


def forward_noisy_graph(
    family_candidates,
    clean_keys,
    num_nodes,
    alpha_bar,
    generator,
):
    selected = []
    selected_family = []
    for family_id, info in family_candidates.items():
        edges = info["edges"]
        keys = edges[0] * int(num_nodes) + edges[1]
        clean = torch.tensor(
            [int(key) in clean_keys.get(int(family_id), set()) for key in keys.tolist()],
            dtype=torch.float32,
        )
        density = float(info["density"])
        prob = float(alpha_bar) * clean + (1.0 - float(alpha_bar)) * density
        keep = torch.rand(prob.numel(), generator=generator) < prob
        selected.append(edges[:, keep])
        selected_family.append(
            torch.full((int(keep.sum()),), int(family_id), dtype=torch.long)
        )
    return torch.cat(selected, dim=1), torch.cat(selected_family)


def shuffled_graph(family_candidates, family_counts, generator):
    selected = []
    selected_family = []
    for family_id, info in family_candidates.items():
        edges = info["edges"]
        quota = min(int(family_counts.get(int(family_id), 0)), int(edges.shape[1]))
        if quota <= 0:
            continue
        chosen = torch.randperm(edges.shape[1], generator=generator)[:quota]
        selected.append(edges[:, chosen])
        selected_family.append(torch.full((quota,), int(family_id), dtype=torch.long))
    return torch.cat(selected, dim=1), torch.cat(selected_family)


def gini(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0 or values.sum() <= 0:
        return 0.0
    values = np.sort(values)
    index = np.arange(1, values.size + 1)
    return float((np.sum((2 * index - values.size - 1) * values)) / (values.size * values.sum()))


def graph_metrics(edge_index, edge_family, num_nodes, blocks):
    graph = nx.Graph()
    graph.add_nodes_from(range(int(num_nodes)))
    graph.add_edges_from(edge_index.t().tolist())
    degrees = np.asarray([degree for _, degree in graph.degree()], dtype=np.float64)
    components = list(nx.connected_components(graph))
    triangles = int(sum(nx.triangles(graph).values()) // 3)
    block_of = np.full(int(num_nodes), -1, dtype=np.int64)
    for block_idx, block in enumerate(blocks):
        block_of[np.asarray(block, dtype=np.int64)] = int(block_idx)
    u = edge_index[0].numpy()
    v = edge_index[1].numpy()
    intra = int(np.sum(block_of[u] == block_of[v])) if edge_index.numel() else 0
    family_counts = torch.bincount(edge_family.long(), minlength=16).tolist()
    return {
        "edges": int(edge_index.shape[1]),
        "lcc": max((len(component) for component in components), default=0),
        "components": len(components),
        "clustering": float(nx.average_clustering(graph)),
        "triangles": triangles,
        "assortativity": float(nx.degree_assortativity_coefficient(graph)),
        "degree_std": float(degrees.std()),
        "degree_cv": float(degrees.std() / max(degrees.mean(), 1e-12)),
        "degree_gini": gini(degrees),
        "degree_max": int(degrees.max(initial=0)),
        "degree_p90": float(np.quantile(degrees, 0.90)),
        "degree_p99": float(np.quantile(degrees, 0.99)),
        "isolated": int(np.sum(degrees == 0)),
        "intra_ratio": float(intra / max(int(edge_index.shape[1]), 1)),
        "family_counts": family_counts,
    }


def flatten_record(kind, time_value, repeat, alpha_bar, metrics):
    record = {
        "kind": kind,
        "time": float(time_value),
        "repeat": int(repeat),
        "alpha_bar": float(alpha_bar),
    }
    for key, value in metrics.items():
        if key != "family_counts":
            record[key] = value
    return record


def main():
    args = parse_args()
    graph, vocab, avg_counts = load_inputs(args.processed_dir)
    node_types = graph.node_type.long()
    num_nodes = int(graph.num_nodes)
    family_candidates = build_family_candidates(node_types, vocab, avg_counts)
    clean_keys = clean_family_keys(graph, num_nodes)
    blocks = metis_blocks(graph, args.edge_fraction)
    betas = cosine_betas(args.diffusion_steps)
    alpha_bars = np.cumprod(1.0 - betas)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    raw = []
    for repeat in range(args.repeats):
        template_generator = torch.Generator().manual_seed(args.seed + 10000 * repeat)
        template_blocks = random_template_blocks(node_types, blocks, template_generator)
        block_edges, block_family = block_marginal_graph(
            family_candidates,
            template_blocks,
            torch.Generator().manual_seed(args.seed + 10000 * repeat + 1),
        )
        block_metrics = graph_metrics(block_edges, block_family, num_nodes, template_blocks)

        for time_value in args.times:
            step = int(round(float(time_value) * args.diffusion_steps))
            alpha_bar = 1.0 if step <= 0 else float(alpha_bars[min(step, args.diffusion_steps) - 1])
            noisy_edges, noisy_family = forward_noisy_graph(
                family_candidates,
                clean_keys,
                num_nodes,
                alpha_bar,
                torch.Generator().manual_seed(args.seed + 10000 * repeat + 100 + step),
            )
            counts = dict(
                zip(
                    range(16),
                    torch.bincount(noisy_family.long(), minlength=16).tolist(),
                )
            )
            shuffled_edges, shuffled_family = shuffled_graph(
                family_candidates,
                counts,
                torch.Generator().manual_seed(args.seed + 10000 * repeat + 200 + step),
            )
            for kind, edge_index, edge_family, metric_blocks in (
                ("forward_noisy", noisy_edges, noisy_family, blocks),
                ("family_shuffled", shuffled_edges, shuffled_family, template_blocks),
            ):
                metrics = graph_metrics(edge_index, edge_family, num_nodes, metric_blocks)
                records.append(flatten_record(kind, time_value, repeat, alpha_bar, metrics))
                raw.append({**records[-1], "family_counts": metrics["family_counts"]})
            records.append(
                flatten_record(
                    "block_marginal",
                    time_value,
                    repeat,
                    alpha_bar,
                    block_metrics,
                )
            )
            raw.append({**records[-1], "family_counts": block_metrics["family_counts"]})

    csv_path = args.output_dir / "role_alignment_metrics.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    with open(args.output_dir / "role_alignment_raw.json", "w", encoding="utf-8") as handle:
        json.dump(raw, handle, indent=2)

    summary = {}
    scalar_keys = [key for key in records[0] if key not in {"kind", "time", "repeat", "alpha_bar"}]
    for kind in sorted({record["kind"] for record in records}):
        for time_value in args.times:
            group = [
                record for record in records
                if record["kind"] == kind and math.isclose(record["time"], float(time_value))
            ]
            summary[f"{kind}@{time_value:.2f}"] = {
                key: {
                    "mean": float(np.mean([record[key] for record in group])),
                    "std": float(np.std([record[key] for record in group])),
                }
                for key in scalar_keys
            }
    with open(args.output_dir / "role_alignment_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Saved: {csv_path}")
    print(f"Saved: {args.output_dir / 'role_alignment_summary.json'}")
    for key, values in summary.items():
        print(
            f"{key:24s} edges={values['edges']['mean']:.1f} "
            f"deg_cv={values['degree_cv']['mean']:.3f} "
            f"gini={values['degree_gini']['mean']:.3f} "
            f"maxdeg={values['degree_max']['mean']:.1f} "
            f"cluster={values['clustering']['mean']:.3f} "
            f"tri={values['triangles']['mean']:.1f} "
            f"lcc={values['lcc']['mean']:.1f} "
            f"intra={values['intra_ratio']['mean']:.3f}"
        )


if __name__ == "__main__":
    main()
