#!/usr/bin/env python3
"""Evaluate generated typed graphs with HGEN's Table 3 metrics.

Run this script in HGEN's environment, which provides ``powerlaw`` and
``torch_two_sample``.  It deliberately reads the target only after generation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import networkx as nx
import powerlaw
import torch
from torch_two_sample import MMDStatistic


def _degrees_and_edges(payload):
    degrees = np.zeros(int(payload["num_nodes"]), dtype=np.float32)
    edges = set()
    for family in payload["families"].values():
        for u, v in family.get("edges", []):
            u, v = int(u), int(v)
            degrees[u] += 1
            degrees[v] += 1
            edges.add((min(u, v), max(u, v)))
    return degrees, edges


def _graph_metrics(payload):
    degrees, edges = _degrees_and_edges(payload)
    graph = nx.Graph()
    graph.add_nodes_from(range(int(payload["num_nodes"])))
    graph.add_edges_from(edges)
    triangles = float(sum(nx.triangles(graph).values()) / 3.0)
    lcc = float(
        max((len(component) for component in nx.connected_components(graph)), default=0)
    )
    assortativity = nx.degree_assortativity_coefficient(graph)
    if not np.isfinite(assortativity):
        assortativity = 0.0
    alpha = powerlaw.Fit(
        degrees,
        xmin=max(float(degrees.min()), 1.0),
        verbose=False,
    ).power_law.alpha
    return degrees, edges, {
        "edges": float(len(edges)),
        "lcc": lcc,
        "triangles": triangles,
        "clustering": float(nx.average_clustering(graph)),
        "transitivity": float(nx.transitivity(graph)),
        "powerlaw_alpha": float(alpha),
        "assortativity": float(assortativity),
    }


def _mean(values):
    return float(sum(values) / max(len(values), 1))


def _std(values):
    mean = _mean(values)
    return float((sum((value - mean) ** 2 for value in values) / max(len(values), 1)) ** 0.5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--generated", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    target_degrees, target_edges, target_metrics = _graph_metrics(
        json.loads(args.target.read_text(encoding="utf-8"))
    )
    rows = []
    generated_fingerprints = []
    for index, path in enumerate(args.generated):
        generated_degrees, generated_edges, metrics = _graph_metrics(
            json.loads(path.read_text(encoding="utf-8"))
        )
        statistic = MMDStatistic(len(target_degrees), len(generated_degrees))
        degree_mmd = statistic(
            torch.tensor(target_degrees)[:, None],
            torch.tensor(generated_degrees)[:, None],
            alphas=[4.0],
            ret_matrix=False,
        )
        generated_fingerprints.append(tuple(sorted(generated_edges)))
        metrics.update(
            {
                "sample": index,
                "degree_mmd": float(degree_mmd),
                "edge_overlap": len(target_edges & generated_edges)
                / max(len(target_edges), 1),
                "edge_overlap_percent": 100.0
                * len(target_edges & generated_edges)
                / max(len(target_edges), 1),
            }
        )
        rows.append(metrics)
    keys = (
        "edges",
        "lcc",
        "triangles",
        "clustering",
        "transitivity",
        "powerlaw_alpha",
        "assortativity",
        "degree_mmd",
        "edge_overlap",
        "edge_overlap_percent",
    )
    report = {
        "format": "dihug-hgen-table3-metrics-v2",
        "target": target_metrics,
        "target_powerlaw_alpha": target_metrics["powerlaw_alpha"],
        "uniqueness_percent": 100.0
        * len(set(generated_fingerprints))
        / max(len(generated_fingerprints), 1),
        "rows": rows,
        "mean": {key: _mean([row[key] for row in rows]) for key in keys},
        "std": {key: _std([row[key] for row in rows]) for key in keys},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
