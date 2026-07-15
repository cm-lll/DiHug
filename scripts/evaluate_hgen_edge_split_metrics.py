#!/usr/bin/env python3
"""Evaluate generated typed graphs with HGEN's degree metrics.

Run this script in HGEN's environment, which provides ``powerlaw`` and
``torch_two_sample``.  It deliberately reads the target only after generation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
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

    target_degrees, target_edges = _degrees_and_edges(
        json.loads(args.target.read_text(encoding="utf-8"))
    )
    rows = []
    for index, path in enumerate(args.generated):
        generated_degrees, generated_edges = _degrees_and_edges(
            json.loads(path.read_text(encoding="utf-8"))
        )
        statistic = MMDStatistic(len(target_degrees), len(generated_degrees))
        degree_mmd = statistic(
            torch.tensor(target_degrees)[:, None],
            torch.tensor(generated_degrees)[:, None],
            alphas=[4.0],
            ret_matrix=False,
        )
        alpha = powerlaw.Fit(
            generated_degrees,
            xmin=max(float(generated_degrees.min()), 1.0),
            verbose=False,
        ).power_law.alpha
        rows.append(
            {
                "sample": index,
                "degree_mmd": float(degree_mmd),
                "powerlaw_alpha": float(alpha),
                "edge_overlap": len(target_edges & generated_edges)
                / max(len(target_edges), 1),
            }
        )
    keys = ("degree_mmd", "powerlaw_alpha", "edge_overlap")
    target_alpha = powerlaw.Fit(
        target_degrees,
        xmin=max(float(target_degrees.min()), 1.0),
        verbose=False,
    ).power_law.alpha
    report = {
        "format": "dihug-hgen-degree-metrics-v1",
        "target_powerlaw_alpha": float(target_alpha),
        "rows": rows,
        "mean": {key: _mean([row[key] for row in rows]) for key in keys},
        "std": {key: _std([row[key] for row in rows]) for key in keys},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
