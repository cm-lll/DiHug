#!/usr/bin/env python3
"""Evaluate role-guided structural edit diffusion from a fair fixed-M prior."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sparse_diffusion.structure_energy.cardinality_conditioned_edit import (
    FamilyCountCondition,
    family_edge_counts,
)
from sparse_diffusion.structure_energy.graph import load_typed_graph
from sparse_diffusion.structure_energy.mixed_edit_diffusion import (
    load_mixed_edit_checkpoint,
)
from sparse_diffusion.structure_energy.role_guided_edit_diffusion import (
    ExchangeableRoleDegreePrior,
    sample_role_guided_edit_diffusion,
)
from sparse_diffusion.structure_energy.switch_chain import graph_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--model-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--candidates", type=int, default=16)
    parser.add_argument("--metrics-every", type=int, default=500)
    parser.add_argument("--role-weight", type=float, default=0.1)
    parser.add_argument("--gain-scale", type=float, default=10.0)
    parser.add_argument("--schedule-power", type=float, default=1.0)
    parser.add_argument("--schedule-floor", type=float, default=0.0)
    parser.add_argument("--role-error-per-node-threshold", type=float, default=0.0)
    parser.add_argument("--model-time-horizon-steps", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = load_typed_graph(args.dataset_dir, active_only=True)
    model, marginal_prior, metadata = load_mixed_edit_checkpoint(
        args.model_checkpoint, device=args.device
    )
    role_path = args.output_dir / "exchangeable_role_degree_prior.json"
    ExchangeableRoleDegreePrior.from_training_graph(source).save(role_path)
    role_prior = ExchangeableRoleDegreePrior.load(role_path)
    initial = FamilyCountCondition.from_training_graph(source).sample_graph(
        args.seed + 71, mode="fixed"
    )
    initial_counts = family_edge_counts(initial)
    result = sample_role_guided_edit_diffusion(
        marginal_prior,
        model,
        role_prior,
        steps=args.steps,
        candidates_per_step=args.candidates,
        seed=args.seed,
        device=args.device,
        metrics_every=args.metrics_every,
        initial_graph=initial,
        feature_profile=metadata["feature_profile"],
        role_weight=args.role_weight,
        gain_scale=args.gain_scale,
        schedule_power=args.schedule_power,
        schedule_floor=args.schedule_floor,
        role_error_per_node_threshold=args.role_error_per_node_threshold,
        model_time_horizon_steps=(
            None if args.model_time_horizon_steps <= 0 else args.model_time_horizon_steps
        ),
    )
    final_counts = family_edge_counts(result.graph)
    if final_counts != initial_counts:
        raise RuntimeError("role-guided sampler changed family counts")
    report = {
        "format": "dihug-role-guided-replacement-gate-v1",
        "seed": args.seed,
        "steps": args.steps,
        "candidates": args.candidates,
        "role_weight": args.role_weight,
        "gain_scale": args.gain_scale,
        "schedule_power": args.schedule_power,
        "schedule_floor": args.schedule_floor,
        "role_error_per_node_threshold": args.role_error_per_node_threshold,
        "model_time_horizon_steps": (
            args.steps
            if args.model_time_horizon_steps <= 0
            else args.model_time_horizon_steps
        ),
        "initial": graph_metrics(initial),
        "final": graph_metrics(result.graph),
        "real": graph_metrics(source),
        "trace": result.trace,
        "action_counts": result.action_counts,
        "family_counts_preserved": True,
        "sampling_uses_clean_adjacency": False,
        "role_prior_contract": role_prior.to_json_dict()["inference_contract"],
    }
    (args.output_dir / "gate_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    for row in result.trace:
        print(
            "[ROLE-GUIDED-TRACE] "
            f"step={int(row['step'])} tri={int(row['triangles'])} "
            f"clustering={row['clustering']:.6f} lcc={int(row['lcc'])} "
            f"sum_deg2={int(row['sum_degree_sq'])} "
            f"role_error={row['relation_degree_error']:.0f} "
            f"role_error_per_node={row['relation_degree_error_per_node']:.4f} "
            f"phase={row['phase']} "
            f"transition={row['role_phase_transition_step']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
