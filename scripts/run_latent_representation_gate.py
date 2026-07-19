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
    train_representation_warmup,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--trajectories", type=int, default=6)
    parser.add_argument("--noise-levels", default="0.25,0.5,0.75,1.0")
    parser.add_argument("--queries-per-family", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--edge-loss-weight", type=float, default=1.0)
    parser.add_argument("--role-loss-weight", type=float, default=1.0)
    parser.add_argument("--variance-loss-weight", type=float, default=0.05)
    parser.add_argument("--min-edge-gain-over-current", type=float, default=0.02)
    parser.add_argument("--min-role-mae-reduction", type=float, default=0.0)
    parser.add_argument("--min-embedding-std", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = load_typed_graph(args.dataset_dir, active_only=True)
    levels = tuple(float(value) for value in args.noise_levels.split(","))
    dataset = build_representation_warmup_dataset(
        source,
        trajectories=args.trajectories,
        noise_levels=levels,
        seed=args.seed,
    )
    model, report = train_representation_warmup(
        dataset,
        epochs=args.epochs,
        queries_per_family=args.queries_per_family,
        learning_rate=args.learning_rate,
        edge_loss_weight=args.edge_loss_weight,
        role_loss_weight=args.role_loss_weight,
        variance_loss_weight=args.variance_loss_weight,
        device=args.device,
        seed=args.seed,
        hidden_dim=args.hidden_dim,
        heads=args.heads,
        layers=args.layers,
        dropout=args.dropout,
    )
    report = dict(report)
    report["dataset_dir"] = str(args.dataset_dir)
    report["seed"] = int(args.seed)
    report["gate_thresholds"] = {
        "min_edge_gain_over_current": float(args.min_edge_gain_over_current),
        "min_role_mae_reduction": float(args.min_role_mae_reduction),
        "min_embedding_std": float(args.min_embedding_std),
    }
    informative_levels = [
        metrics
        for level, metrics in report["by_noise"].items()
        if float(level) < 1.0
    ]
    report["edge_positive_noise_levels"] = sum(
        metrics["edge_gain_over_current"] >= args.min_edge_gain_over_current
        for metrics in informative_levels
    )
    report["role_positive_noise_levels"] = sum(
        metrics["role_mae_reduction"] >= args.min_role_mae_reduction
        for metrics in informative_levels
    )
    required_levels = min(2, len(informative_levels))
    report["gate1_edge"] = bool(
        report["edge_positive_noise_levels"] >= required_levels
    )
    report["gate2_role"] = bool(
        report["role_positive_noise_levels"] >= required_levels
    )
    report["gate3_variance"] = bool(
        report["embedding_mean_feature_std"] >= args.min_embedding_std
    )
    report["passed"] = bool(
        report["gate1_edge"] and report["gate2_role"] and report["gate3_variance"]
    )
    torch.save(
        {
            "format": "dihug-latent-representation-warmup-v1",
            "model_state": model.state_dict(),
            "model_config": {
                "node_feature_dim": len(source.family_specs) + 3,
                "type_count": len(source.type_names),
                "family_count": len(source.family_specs),
                "hidden_dim": int(args.hidden_dim),
                "heads": int(args.heads),
                "layers": int(args.layers),
                "dropout": float(args.dropout),
            },
            "report": report,
        },
        args.output_dir / "representation_warmup.pt",
    )
    (args.output_dir / "representation_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print("[LATENT-REP-REPORT] " + json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
