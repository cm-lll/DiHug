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
from sparse_diffusion.structure_energy.hgt_edit_denoiser import graph_node_features
from sparse_diffusion.structure_energy.latent_edit_representation import (
    WarmupHGTEncoder,
    build_representation_warmup_dataset,
)
from sparse_diffusion.structure_energy.latent_edit_response import (
    train_local_edit_response_gate,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--alignment-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--trajectories", type=int, default=6)
    parser.add_argument("--noise-levels", default="0.25,0.5,0.75,1.0")
    parser.add_argument("--candidates", type=int, default=16)
    parser.add_argument("--validation-candidate-sets", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=5.0e-4)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--prototype-count", type=int, default=3)
    parser.add_argument("--min-expected-utility-gain", type=float, default=0.01)
    parser.add_argument("--min-response-reduction", type=float, default=0.0)
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
    sample = dataset.snapshots[0]
    representation = WarmupHGTEncoder(
        node_feature_dim=graph_node_features(sample.graph, sample.time_fraction).shape[1],
        type_count=dataset.type_count,
        family_count=dataset.family_count,
        hidden_dim=args.hidden_dim,
        heads=args.heads,
        layers=args.layers,
        dropout=args.dropout,
        role_condition_dim=dataset.family_count,
        prototype_count=args.prototype_count,
    )
    checkpoint = torch.load(
        args.alignment_checkpoint, map_location="cpu", weights_only=False
    )
    representation.load_state_dict(checkpoint["model_state"], strict=True)
    model, report = train_local_edit_response_gate(
        dataset,
        representation,
        candidates_per_example=args.candidates,
        validation_candidate_sets=args.validation_candidate_sets,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        device=args.device,
        seed=args.seed,
    )
    report = dict(report)
    informative = [
        metrics
        for level, metrics in report["by_noise"].items()
        if float(level) < 1.0
    ]
    required = min(2, len(informative))
    report["utility_positive_levels"] = sum(
        row["expected_utility_gain"] >= args.min_expected_utility_gain
        for row in informative
    )
    report["response_positive_levels"] = sum(
        row["response_mse_reduction"] >= args.min_response_reduction
        for row in informative
    )
    report["correlation_positive_levels"] = sum(
        row["gain_utility_correlation"] > 0.0 for row in informative
    )
    report["mean_selected_utility_gain"] = float(
        sum(row["selected_utility_gain"] for row in informative)
        / max(len(informative), 1)
    )
    report["gates"] = {
        "response": report["response_positive_levels"] >= required,
        "expected_utility": report["utility_positive_levels"] >= required,
        "latent_correlation": report["correlation_positive_levels"] >= required,
        "selected_non_degrade": report["mean_selected_utility_gain"] >= 0.0,
    }
    report["passed"] = bool(all(report["gates"].values()))
    report["dataset_dir"] = str(args.dataset_dir)
    report["seed"] = int(args.seed)
    torch.save(
        {
            "format": "dihug-local-edit-response-gate-v1",
            "model_state": model.state_dict(),
            "report": report,
        },
        args.output_dir / "local_response.pt",
    )
    (args.output_dir / "response_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print("[LATENT-RESPONSE-REPORT] " + json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
