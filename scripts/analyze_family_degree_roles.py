#!/usr/bin/env python3
"""Family-conditioned degree-role diagnostics for fixed-node graph samples."""

import argparse
import csv
import json
import math
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


ROLES = ("low", "mid", "high")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/PubMed_baseline_subgraphs/processed"),
    )
    parser.add_argument(
        "--generated",
        action="append",
        default=[],
        help="NAME=path/to/generated_samples.pkl; repeat as needed",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/family_degree_roles"),
    )
    parser.add_argument(
        "--intervention",
        action="append",
        default=[],
        help="NAME=path/to/posterior_intervention.jsonl",
    )
    return parser.parse_args()


def load_generated(spec):
    if "=" not in spec:
        raise ValueError(f"Expected NAME=PATH, got {spec}")
    name, path = spec.split("=", 1)
    with open(path, "rb") as handle:
        graph = pickle.load(handle)
    return name, graph


def edge_labels(graph):
    labels = graph.edge_attr
    return labels.argmax(dim=-1).long() if labels.dim() > 1 else labels.long()


def node_types_from_reference(reference):
    return reference.node_type.long().cpu()


def degree_vector(edge_index, num_nodes):
    degree = torch.zeros(num_nodes, dtype=torch.float64)
    if edge_index.numel():
        endpoints = torch.cat([edge_index[0], edge_index[1]]).long().cpu()
        degree.scatter_add_(
            0, endpoints, torch.ones(endpoints.numel(), dtype=torch.float64)
        )
    return degree.numpy()


def family_name_by_label(vocab):
    mapping = {}
    for label_name, label_id in vocab["edge_label2id"].items():
        mapping[int(label_id)] = label_name.split(":", 1)[0]
    return mapping


def family_endpoints(name, node_type2id):
    parts = name.split("_")
    return int(node_type2id[parts[1]]), int(node_type2id[parts[2]])


def orient_edges(graph, node_types, label_to_family, node_type2id):
    labels = edge_labels(graph).cpu()
    edges = graph.edge_index.long().cpu()
    result = defaultdict(list)
    for index, label in enumerate(labels.tolist()):
        if label == 0 or label not in label_to_family:
            continue
        family = label_to_family[label]
        src_type, dst_type = family_endpoints(family, node_type2id)
        u, v = int(edges[0, index]), int(edges[1, index])
        tu, tv = int(node_types[u]), int(node_types[v])
        if src_type == dst_type:
            result[family].append((u, v))
        elif tu == src_type and tv == dst_type:
            result[family].append((u, v))
        elif tv == src_type and tu == dst_type:
            result[family].append((v, u))
        else:
            raise AssertionError(
                f"{family} endpoint mismatch for edge {(u, v)}: {(tu, tv)}"
            )
    return result


def thresholds_from_real(real_degree, node_types, node_type_names):
    thresholds = {}
    for type_id, type_name in enumerate(node_type_names):
        values = real_degree[node_types.numpy() == type_id]
        thresholds[type_id] = {
            "node_type": type_name,
            "p50": float(np.quantile(values, 0.5)),
            "p80": float(np.quantile(values, 0.8)),
            "max_real": float(values.max(initial=0.0)),
        }
    return thresholds


def role(value, threshold):
    if value <= threshold["p50"]:
        return "low"
    if value <= threshold["p80"]:
        return "mid"
    return "high"


def safe_corr(x, y):
    if len(x) < 2 or np.std(x) <= 0 or np.std(y) <= 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def jsd(p, q):
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = p / max(p.sum(), 1.0)
    q = q / max(q.sum(), 1.0)
    m = 0.5 * (p + q)

    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log(a[mask] / b[mask])))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def analyze_graph(
    graph,
    name,
    node_types,
    node_type_names,
    node_type2id,
    label_to_family,
    thresholds,
):
    num_nodes = int(node_types.numel())
    degree = degree_vector(graph.edge_index, num_nodes)
    families = orient_edges(
        graph, node_types, label_to_family, node_type2id
    )
    family_degree = {}
    for family, edges in families.items():
        values = np.zeros(num_nodes, dtype=float)
        for src, dst in edges:
            values[src] += 1
            values[dst] += 1
        family_degree[family] = values

    observations = []
    family_observations = defaultdict(list)
    role_counts = defaultdict(lambda: defaultdict(int))
    for family, edges in families.items():
        src_type, dst_type = family_endpoints(family, node_type2id)
        for src, dst in edges:
            pairs = [(src, dst)]
            if src_type == dst_type:
                pairs.append((dst, src))
            for left, right in pairs:
                item = (degree[left], degree[right], family)
                observations.append(item)
                family_observations[family].append(item)
            src_role = role(degree[src], thresholds[src_type])
            dst_role = role(degree[dst], thresholds[dst_type])
            if src_type == dst_type and ROLES.index(src_role) > ROLES.index(dst_role):
                src_role, dst_role = dst_role, src_role
            role_counts[family][(src_role, dst_role)] += 1

    x = np.asarray([item[0] for item in observations], dtype=float)
    y = np.asarray([item[1] for item in observations], dtype=float)
    mu_x, mu_y = float(x.mean()), float(y.mean())
    sxx = float(np.square(x - mu_x).sum())
    syy = float(np.square(y - mu_y).sum())
    denominator = math.sqrt(max(sxx * syy, 0.0))
    global_corr = (
        float(((x - mu_x) * (y - mu_y)).sum() / denominator)
        if denominator > 0
        else float("nan")
    )

    summaries = []
    strict_numerators = {}
    for family in sorted(families):
        obs = family_observations[family]
        fx = np.asarray([item[0] for item in obs], dtype=float)
        fy = np.asarray([item[1] for item in obs], dtype=float)
        strict_n = float(((fx - mu_x) * (fy - mu_y)).sum())
        strict_numerators[family] = strict_n
        src_type, dst_type = family_endpoints(family, node_type2id)
        edges = families[family]
        src_nodes = np.asarray([src for src, _ in edges], dtype=int)
        dst_nodes = np.asarray([dst for _, dst in edges], dtype=int)
        corr_total = safe_corr(degree[src_nodes], degree[dst_nodes])
        corr_family = safe_corr(
            family_degree[family][src_nodes],
            family_degree[family][dst_nodes],
        )
        summaries.append(
            {
                "graph": name,
                "family": family,
                "edges": len(edges),
                "edge_share": len(edges) / max(graph.edge_index.shape[1], 1),
                "corr_total_degree": corr_total,
                "corr_family_degree": corr_family,
                "strict_cov_numerator": strict_n,
                "strict_corr_contribution": (
                    strict_n / denominator if denominator > 0 else float("nan")
                ),
                "global_schema_assortativity": global_corr,
            }
        )

    abs_sum = sum(abs(value) for value in strict_numerators.values())
    for row in summaries:
        value = row["strict_cov_numerator"]
        row["strict_direction_share"] = value / max(abs_sum, 1e-12)

    strict_reconstructed = sum(
        row["strict_corr_contribution"] for row in summaries
    )
    assert abs(strict_reconstructed - global_corr) < 1e-9
    assert sum(len(value) for value in families.values()) == graph.edge_index.shape[1]

    joint_rows = []
    for family in sorted(families):
        family_total = len(families[family])
        assert sum(role_counts[family].values()) == family_total
        for src_role in ROLES:
            for dst_role in ROLES:
                count = role_counts[family][(src_role, dst_role)]
                joint_rows.append(
                    {
                        "graph": name,
                        "family": family,
                        "src_role": src_role,
                        "dst_role": dst_role,
                        "count": count,
                        "probability": count / max(family_total, 1),
                    }
                )

    type_rows = []
    for type_id, threshold in thresholds.items():
        mask = node_types.numpy() == type_id
        type_rows.append(
            {
                "graph": name,
                **threshold,
                "gen_above_real_max_ratio": float(
                    np.mean(degree[mask] > threshold["max_real"])
                ),
            }
        )
    return summaries, joint_rows, type_rows


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reference = torch.load(
        args.processed_dir / "train.pt",
        map_location="cpu",
        weights_only=False,
    )[0]
    with open(args.processed_dir / "vocab.json", encoding="utf-8") as handle:
        vocab = json.load(handle)
    generated = [load_generated(spec) for spec in args.generated]
    node_types = node_types_from_reference(reference)
    real_degree = degree_vector(reference.edge_index, int(node_types.numel()))
    thresholds = thresholds_from_real(
        real_degree, node_types, vocab["node_type_names"]
    )
    label_to_family = family_name_by_label(vocab)

    analyses = [("real", reference)] + generated
    summary_by_graph = {}
    joint_by_graph = {}
    all_type_rows = []
    for name, graph in analyses:
        summary, joint, type_rows = analyze_graph(
            graph=graph,
            name=name,
            node_types=node_types,
            node_type_names=vocab["node_type_names"],
            node_type2id=vocab["node_type2id"],
            label_to_family=label_to_family,
            thresholds=thresholds,
        )
        summary_by_graph[name] = {row["family"]: row for row in summary}
        joint_by_graph[name] = joint
        all_type_rows.extend(type_rows)

    family_rows = []
    real_summary = summary_by_graph["real"]
    for name, _ in generated:
        for family, gen in summary_by_graph[name].items():
            real = real_summary[family]
            family_rows.append(
                {
                    "graph": name,
                    "family": family,
                    "edges_real": real["edges"],
                    "edges_gen": gen["edges"],
                    "corr_total_degree_real": real["corr_total_degree"],
                    "corr_total_degree_gen": gen["corr_total_degree"],
                    "corr_total_degree_gap": (
                        gen["corr_total_degree"] - real["corr_total_degree"]
                    ),
                    "corr_family_degree_real": real["corr_family_degree"],
                    "corr_family_degree_gen": gen["corr_family_degree"],
                    "corr_family_degree_gap": (
                        gen["corr_family_degree"]
                        - real["corr_family_degree"]
                    ),
                    "heuristic_Cf": gen["edge_share"]
                    * (
                        gen["corr_total_degree"]
                        - real["corr_total_degree"]
                    ),
                    "strict_cov_contribution_real": real[
                        "strict_corr_contribution"
                    ],
                    "strict_cov_contribution_gen": gen[
                        "strict_corr_contribution"
                    ],
                    "strict_cov_contribution_gap": (
                        gen["strict_corr_contribution"]
                        - real["strict_corr_contribution"]
                    ),
                    "strict_direction_share_real": real[
                        "strict_direction_share"
                    ],
                    "strict_direction_share_gen": gen[
                        "strict_direction_share"
                    ],
                }
            )

    role_rows = []
    real_joint = {
        (row["family"], row["src_role"], row["dst_role"]): row
        for row in joint_by_graph["real"]
    }
    for name, _ in generated:
        gen_joint = {
            (row["family"], row["src_role"], row["dst_role"]): row
            for row in joint_by_graph[name]
        }
        families = sorted({key[0] for key in real_joint})
        for family in families:
            keys = [
                (family, src_role, dst_role)
                for src_role in ROLES
                for dst_role in ROLES
            ]
            p = [real_joint[key]["probability"] for key in keys]
            q = [gen_joint[key]["probability"] for key in keys]
            tv = 0.5 * sum(abs(a - b) for a, b in zip(p, q))
            divergence = jsd(p, q)
            assert 0.0 <= tv <= 1.0 + 1e-12
            assert 0.0 <= divergence <= math.log(2.0) + 1e-12
            for key in keys:
                real = real_joint[key]
                gen = gen_joint[key]
                role_rows.append(
                    {
                        "graph": name,
                        "family": family,
                        "src_role": key[1],
                        "dst_role": key[2],
                        "count_real": real["count"],
                        "count_gen": gen["count"],
                        "prob_real": real["probability"],
                        "prob_gen": gen["probability"],
                        "tv_family": tv,
                        "jsd_family": divergence,
                    }
                )

    write_csv(args.output_dir / "family_summary.csv", family_rows)
    write_csv(args.output_dir / "degree_role_joint.csv", role_rows)
    write_csv(args.output_dir / "type_degree_thresholds.csv", all_type_rows)
    intervention_rows = []
    for spec in args.intervention:
        if "=" not in spec:
            raise ValueError(f"Expected NAME=PATH, got {spec}")
        name, path = spec.split("=", 1)
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                for row in record["rows"]:
                    intervention_rows.append(
                        {
                            "graph": name,
                            "t": record["t"],
                            "s": record["s"],
                            "alpha_bar_s": record["alpha_bar_s"],
                            "lambda": record["lambda"],
                            **row,
                        }
                    )
    write_csv(
        args.output_dir / "posterior_intervention.csv",
        intervention_rows,
    )
    with open(
        args.output_dir / "type_degree_thresholds.json",
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(thresholds, handle, indent=2, ensure_ascii=False)
    print(f"Diagnostics written to {args.output_dir}")


if __name__ == "__main__":
    main()
