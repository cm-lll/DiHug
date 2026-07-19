#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sparse_diffusion.structure_energy.edge_energy_edit_gate import (
    train_edge_energy_fusion_gate,
)
from sparse_diffusion.structure_energy.graph import load_typed_graph
from sparse_diffusion.structure_energy.hgt_edit_denoiser import graph_node_features
from sparse_diffusion.structure_energy.latent_edit_representation import (
    WarmupHGTEncoder,
    build_representation_warmup_dataset,
)
from sparse_diffusion.structure_energy.mixed_edit_diffusion import (
    load_mixed_edit_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--legacy-checkpoint", type=Path, required=True)
    parser.add_argument("--alignment-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--trajectories", type=int, default=6)
    parser.add_argument("--noise-levels", default="0.25,0.5,0.75,1.0")
    parser.add_argument("--candidates", type=int, default=16)
    parser.add_argument("--train-candidate-sets", type=int, default=4)
    parser.add_argument("--validation-candidate-sets", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--target-temperature", type=float, default=0.5)
    parser.add_argument("--l2-weight", type=float, default=0.01)
    parser.add_argument("--min-expected-gain", type=float, default=0.01)
    parser.add_argument("--max-selected-regression", type=float, default=0.0)
    return parser.parse_args()


def alignment_model_from_checkpoint(
    path: Path,
    *,
    node_feature_dim: int,
    type_count: int,
    family_count: int,
) -> tuple[WarmupHGTEncoder, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != "dihug-role-conditioned-latent-alignment-v1":
        raise ValueError("unsupported latent-alignment checkpoint")
    state = payload["model_state"]
    hidden_dim = int(state["node_input.0.weight"].shape[0])
    role_dim = int(state["role_input.0.weight"].shape[1])
    layer_ids = {int(key.split(".")[1]) for key in state if key.startswith("layers.")}
    heads = int(state["layers.0.relation_pri"].shape[1])
    prototype_count = int(state["prototype_head.2.weight"].shape[0] // hidden_dim)
    if role_dim != int(family_count):
        raise ValueError("alignment role dimension does not match graph families")
    model = WarmupHGTEncoder(
        node_feature_dim=int(node_feature_dim),
        type_count=int(type_count),
        family_count=int(family_count),
        hidden_dim=hidden_dim,
        heads=heads,
        layers=len(layer_ids),
        dropout=0.0,
        role_condition_dim=role_dim,
        prototype_count=prototype_count,
    )
    model.load_state_dict(state, strict=True)
    return model, {
        "hidden_dim": hidden_dim,
        "heads": heads,
        "layers": len(layer_ids),
        "role_condition_dim": role_dim,
        "prototype_count": prototype_count,
    }


def main() -> None:
    args = parse_args()
    source = load_typed_graph(args.dataset_dir, active_only=True)
    dataset = build_representation_warmup_dataset(
        source,
        trajectories=args.trajectories,
        noise_levels=tuple(float(value) for value in args.noise_levels.split(",")),
        seed=args.seed,
    )
    sample = dataset.snapshots[0]
    representation, model_shape = alignment_model_from_checkpoint(
        args.alignment_checkpoint,
        node_feature_dim=graph_node_features(
            sample.graph, sample.time_fraction
        ).shape[1],
        type_count=dataset.type_count,
        family_count=dataset.family_count,
    )
    legacy, _, metadata = load_mixed_edit_checkpoint(
        args.legacy_checkpoint, device="cpu"
    )
    feature_profile = str(
        metadata.get("feature_profile", "typed_closure_compact_v2")
    )
    fusion, report = train_edge_energy_fusion_gate(
        dataset,
        representation,
        legacy,
        feature_profile=feature_profile,
        candidates_per_example=args.candidates,
        train_candidate_sets=args.train_candidate_sets,
        validation_candidate_sets=args.validation_candidate_sets,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        target_temperature=args.target_temperature,
        l2_weight=args.l2_weight,
        device=args.device,
        seed=args.seed,
    )
    report = dict(report)
    informative = [
        row for level, row in report["by_noise"].items() if float(level) < 1.0
    ]
    required = min(2, len(informative))
    positive_levels = sum(
        row["expected_utility_gain"] >= float(args.min_expected_gain)
        for row in informative
    )
    correlation_levels = sum(
        row["edge_utility_correlation"] > 0.0 for row in informative
    )
    learned_weight = max(report["action_weights"].values()) > 1.0e-6
    report.update(
        {
            "dataset_dir": str(args.dataset_dir),
            "legacy_checkpoint": str(args.legacy_checkpoint),
            "alignment_checkpoint": str(args.alignment_checkpoint),
            "alignment_model": model_shape,
            "seed": int(args.seed),
            "positive_expected_levels": int(positive_levels),
            "positive_correlation_levels": int(correlation_levels),
            "gates": {
                "learned_nonzero_weight": bool(learned_weight),
                "expected_utility": positive_levels >= required,
                "edge_signal": correlation_levels >= required,
                "selected_non_regression": report["overall"][
                    "selected_utility_gain"
                ]
                >= -float(args.max_selected_regression),
            },
        }
    )
    report["passed"] = bool(all(report["gates"].values()))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "dihug-edge-energy-edit-gate-v1",
            "model_state": fusion.state_dict(),
            "report": report,
        },
        args.output_dir / "edge_energy_fusion.pt",
    )
    (args.output_dir / "edge_energy_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print("[EDGE-ENERGY-REPORT] " + json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
