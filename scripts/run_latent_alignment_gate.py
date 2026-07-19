#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sparse_diffusion.structure_energy.graph import load_typed_graph
from sparse_diffusion.structure_energy.latent_edit_representation import (
    build_representation_warmup_dataset,
    train_role_conditioned_alignment,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--trajectories", type=int, default=6)
    parser.add_argument("--noise-levels", default="0.25,0.5,0.75,1.0")
    parser.add_argument("--queries-per-family", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2.0e-4)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--prototype-count", type=int, default=3)
    parser.add_argument("--min-edge-gain", type=float, default=0.02)
    parser.add_argument("--max-role-rmse", type=float, default=0.95)
    parser.add_argument("--min-latent-cosine", type=float, default=0.5)
    parser.add_argument("--min-role-sensitivity", type=float, default=0.005)
    parser.add_argument("--min-context-gain", type=float, default=0.01)
    parser.add_argument("--min-latent-context-gain", type=float, default=0.03)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = load_typed_graph(args.dataset_dir, active_only=True)
    dataset = build_representation_warmup_dataset(
        source,
        trajectories=args.trajectories,
        noise_levels=tuple(float(value) for value in args.noise_levels.split(",")),
        seed=args.seed,
    )
    checkpoint = torch.load(args.base_checkpoint, map_location="cpu", weights_only=False)
    model, report = train_role_conditioned_alignment(
        dataset,
        base_model_state=checkpoint["model_state"],
        epochs=args.epochs,
        queries_per_family=args.queries_per_family,
        learning_rate=args.learning_rate,
        prototype_count=args.prototype_count,
        device=args.device,
        seed=args.seed,
        hidden_dim=args.hidden_dim,
        heads=args.heads,
        layers=args.layers,
        dropout=args.dropout,
    )
    report = dict(report)
    informative = [
        metrics
        for level, metrics in report["by_noise"].items()
        if float(level) < 1.0
    ]
    required = min(2, len(informative))
    report["edge_positive_levels"] = sum(
        row["edge_gain_over_current"] >= args.min_edge_gain for row in informative
    )
    report["role_positive_levels"] = sum(
        row["role_rmse"] <= args.max_role_rmse for row in informative
    )
    report["latent_positive_levels"] = sum(
        row["latent_clean_cosine"] >= args.min_latent_cosine for row in informative
    )
    report["sensitivity_positive_levels"] = sum(
        row["role_shuffle_distance"] >= args.min_role_sensitivity
        for row in informative
    )
    report["context_positive_levels"] = sum(
        row["edge_context_gain"] >= args.min_context_gain
        or row["latent_context_gain"] >= args.min_latent_context_gain
        for row in informative
    )
    report["gates"] = {
        "edge": report["edge_positive_levels"] >= required,
        "role": report["role_positive_levels"] >= required,
        "latent": report["latent_positive_levels"] >= required,
        "role_sensitivity": report["sensitivity_positive_levels"] >= required,
        "graph_context": report["context_positive_levels"] >= required,
    }
    report["passed"] = bool(all(report["gates"].values()))
    report["dataset_dir"] = str(args.dataset_dir)
    report["seed"] = int(args.seed)
    torch.save(
        {
            "format": "dihug-role-conditioned-latent-alignment-v1",
            "model_state": model.state_dict(),
            "report": report,
        },
        args.output_dir / "latent_alignment.pt",
    )
    (args.output_dir / "alignment_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print("[LATENT-ALIGN-REPORT] " + json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
