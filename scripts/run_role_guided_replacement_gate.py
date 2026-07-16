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
from sparse_diffusion.structure_energy.graph import TypedGraph, load_typed_graph
from sparse_diffusion.structure_energy.mixed_edit_diffusion import (
    load_mixed_edit_checkpoint,
)
from sparse_diffusion.structure_energy.role_guided_edit_diffusion import (
    ExchangeableRoleDegreePrior,
    sample_role_guided_edit_diffusion,
)
from sparse_diffusion.structure_energy.rooted_context_edit import (
    load_rooted_context_checkpoint,
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
    parser.add_argument("--model-time-start-step", type=int, default=0)
    parser.add_argument("--initial-graph", type=Path)
    parser.add_argument("--closure-candidate-fraction", type=float, default=0.0)
    parser.add_argument("--closure-degree-bias-power", type=float, default=0.0)
    parser.add_argument("--auto-role-balance", type=float, default=0.0)
    parser.add_argument("--role-calibration-batches", type=int, default=32)
    parser.add_argument("--targeted-role-candidate-fraction", type=float, default=0.0)
    parser.add_argument("--context-checkpoint", type=Path)
    parser.add_argument("--context-refresh-steps", type=int, default=20)
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
    context_model = None
    if args.context_checkpoint is not None:
        context_model, _ = load_rooted_context_checkpoint(
            args.context_checkpoint, model, device=args.device
        )
    initial = (
        TypedGraph.from_json_dict(
            json.loads(args.initial_graph.read_text(encoding="utf-8"))
        )
        if args.initial_graph is not None
        else FamilyCountCondition.from_training_graph(source).sample_graph(
            args.seed + 71, mode="fixed"
        )
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
        model_time_start_step=args.model_time_start_step,
        closure_candidate_fraction=args.closure_candidate_fraction,
        closure_degree_bias_power=args.closure_degree_bias_power,
        auto_role_balance=args.auto_role_balance,
        role_calibration_batches=args.role_calibration_batches,
        targeted_role_candidate_fraction=args.targeted_role_candidate_fraction,
        context_model=context_model,
        context_refresh_steps=args.context_refresh_steps,
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
        "model_time_start_step": args.model_time_start_step,
        "closure_candidate_fraction": args.closure_candidate_fraction,
        "closure_degree_bias_power": args.closure_degree_bias_power,
        "auto_role_balance": args.auto_role_balance,
        "effective_role_weight": result.trace[0]["effective_role_weight"],
        "targeted_role_candidate_fraction": args.targeted_role_candidate_fraction,
        "context_checkpoint": (
            None if args.context_checkpoint is None else str(args.context_checkpoint)
        ),
        "context_refresh_steps": args.context_refresh_steps,
        "initial_graph_source": (
            "fixed_family_count_prior"
            if args.initial_graph is None
            else str(args.initial_graph)
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
    (args.output_dir / "final_graph.json").write_text(
        json.dumps(result.graph.to_json_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
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
