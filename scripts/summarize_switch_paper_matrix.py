#!/usr/bin/env python3
"""Create raw and aggregate paper tables for switch-diffusion experiments."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from sparse_diffusion.structure_energy.graph import load_typed_graph
from sparse_diffusion.structure_energy.switch_chain import graph_metrics


METRICS = (
    "triangles",
    "clustering",
    "transitivity",
    "assortativity",
    "lcc",
    "edges",
    "sum_degree_sq",
    "max_degree",
)


def relative_error(value: float, target: float) -> float:
    return abs(float(value) - float(target)) / max(abs(float(target)), 1.0e-12)


def mean(values: Iterable[float]) -> float:
    data = list(values)
    return float(statistics.fmean(data)) if data else float("nan")


def sample_std(values: Iterable[float]) -> float:
    data = list(values)
    return float(statistics.stdev(data)) if len(data) > 1 else 0.0


def read_rows(roots: list[Path], real: Mapping[str, float], dataset: str) -> list[dict]:
    rows = []
    for root in roots:
        for report_path in sorted(root.glob("**/sampling_report.json")):
            payload = json.loads(report_path.read_text(encoding="utf-8"))
            final = payload["final_metrics"]
            initial = payload["initial_metrics"]
            train = payload.get("train_report", {})
            config = payload.get("sampling_config", {})
            contract = payload.get("inference_contract", {})
            trace = payload.get("trace", [])
            applied = int(
                payload.get(
                    "applied",
                    trace[-1].get("applied_switches", 0) if trace else 0,
                )
            )
            row = {
                "dataset": dataset,
                "run_label": str(payload.get("run_label", report_path.parent.name)),
                "run": str(report_path.parent.relative_to(root)),
                "seed": int(payload.get("seed", -1)),
                "mode": str(payload.get("sampling_mode", "model")),
                "conditioning_mode": str(payload.get("conditioning_mode", "unknown")),
                "clean_target_at_inference": int(
                    bool(contract.get("clean_graphlet_target", False))
                ),
                "learned_condition": int(
                    bool(contract.get("learned_graphlet_condition", False))
                ),
                "noop_guard": int(bool(config.get("learned_condition_noop", False))),
                "requested_steps": int(config.get("requested_steps", -1)),
                "candidates": int(config.get("candidates_per_step", -1)),
                "proposed": int(payload.get("proposed", 0)),
                "applied": applied,
                "sampling_seconds": float(payload.get("sampling_seconds", float("nan"))),
                "condition_rank": int(train.get("learned_condition_rank", 0)),
                "condition_rmse": float(
                    train.get("learned_condition_rmse", float("nan"))
                ),
            }
            for metric in METRICS:
                value = float(final[metric])
                row[metric] = value
                row[f"{metric}_rel_err"] = relative_error(value, float(real[metric]))
                initial_value = float(initial[metric])
                row[f"initial_{metric}"] = initial_value
                row[f"initial_{metric}_rel_err"] = relative_error(
                    initial_value, float(real[metric])
                )
            rows.append(row)
    return rows


def write_tsv(path: Path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        key = (
            row["dataset"],
            row["run_label"],
            row["mode"],
            row["conditioning_mode"],
            row["clean_target_at_inference"],
            row["learned_condition"],
            row["noop_guard"],
        )
        grouped[key].append(row)
    result = []
    for key, group in sorted(grouped.items()):
        row = {
            "dataset": key[0],
            "run_label": key[1],
            "mode": key[2],
            "conditioning_mode": key[3],
            "clean_target_at_inference": key[4],
            "learned_condition": key[5],
            "noop_guard": key[6],
            "n": len(group),
            "applied_mean": mean(item["applied"] for item in group),
            "sampling_seconds_mean": mean(
                item["sampling_seconds"] for item in group
            ),
        }
        for metric in METRICS:
            row[f"{metric}_mean"] = mean(item[metric] for item in group)
            row[f"{metric}_std"] = sample_std(item[metric] for item in group)
            row[f"{metric}_rel_err_mean"] = mean(
                item[f"{metric}_rel_err"] for item in group
            )
            row[f"initial_{metric}_mean"] = mean(
                item[f"initial_{metric}"] for item in group
            )
            row[f"initial_{metric}_std"] = sample_std(
                item[f"initial_{metric}"] for item in group
            )
            row[f"initial_{metric}_rel_err_mean"] = mean(
                item[f"initial_{metric}_rel_err"] for item in group
            )
        result.append(row)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--root", type=Path, action="append", required=True)
    parser.add_argument("--raw-output", type=Path, required=True)
    parser.add_argument("--aggregate-output", type=Path, required=True)
    args = parser.parse_args()

    real = graph_metrics(load_typed_graph(args.dataset_dir, active_only=True))
    rows = read_rows(args.root, real, args.dataset)
    if not rows:
        raise RuntimeError("no sampling_report.json files found")
    raw_columns = list(rows[0])
    aggregate_rows = aggregate(rows)
    aggregate_columns = list(aggregate_rows[0])
    write_tsv(args.raw_output, rows, raw_columns)
    write_tsv(args.aggregate_output, aggregate_rows, aggregate_columns)
    print(
        "[SWITCH-PAPER-SUMMARY] "
        f"dataset={args.dataset} runs={len(rows)} groups={len(aggregate_rows)} "
        f"raw={args.raw_output} aggregate={args.aggregate_output}"
    )


if __name__ == "__main__":
    main()
