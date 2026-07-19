#!/usr/bin/env python3
"""Evaluate role-guided structural edit diffusion from a family-count prior."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from sparse_diffusion.structure_energy.cardinality_conditioned_edit import (
    FamilyCountCondition,
    family_edge_counts,
)
from sparse_diffusion.structure_energy.graph import TypedGraph, load_typed_graph
from sparse_diffusion.structure_energy.mixed_edit_diffusion import (
    MarginalGraphPrior,
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
    parser.add_argument("--model-checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--sampling-backend",
        choices=("cpu", "gpu"),
        default="cpu",
        help=(
            "gpu keeps graph state, candidate construction, compact features, "
            "scoring, selection, and mutation resident on CUDA"
        ),
    )
    parser.add_argument(
        "--gpu-proposal-pool-size",
        type=int,
        default=0,
        help="streamed GPU proposal count per action pool; 0 keeps the parity gate",
    )
    parser.add_argument("--gpu-proposal-chunk-size", type=int, default=1024)
    parser.add_argument("--gpu-metric-edge-chunk-size", type=int, default=2048)
    parser.add_argument("--gpu-phase-check-interval", type=int, default=32)
    parser.add_argument("--gpu-disable-triton", action="store_true")
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
    parser.add_argument(
        "--edits-per-outer-step",
        type=int,
        default=1,
        help="ordered graph edits sharing one diffusion time value",
    )
    parser.add_argument("--initial-graph", type=Path)
    parser.add_argument(
        "--family-count-mode",
        choices=("fixed", "binomial"),
        default="fixed",
        help=(
            "fixed evaluates p(G|N,M); binomial samples each M_f from the "
            "checkpoint marginal before drawing a uniform schema-valid graph"
        ),
    )
    parser.add_argument("--closure-candidate-fraction", type=float, default=0.0)
    parser.add_argument("--closure-degree-bias-power", type=float, default=0.0)
    parser.add_argument("--auto-role-balance", type=float, default=0.0)
    parser.add_argument("--role-calibration-batches", type=int, default=32)
    parser.add_argument("--targeted-role-candidate-fraction", type=float, default=0.0)
    parser.add_argument("--context-checkpoint", type=Path)
    parser.add_argument("--context-refresh-steps", type=int, default=20)
    parser.add_argument("--context-scale", type=float, default=-1.0)
    parser.add_argument(
        "--phase-policy",
        choices=("auto", "unified", "relocation_only", "switch_only"),
        default="auto",
    )
    parser.add_argument(
        "--selection-mode", choices=("model", "random"), default="model"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = load_typed_graph(args.dataset_dir, active_only=True)
    if args.model_checkpoint is None and args.context_checkpoint is None:
        raise ValueError("provide --model-checkpoint or a self-contained --context-checkpoint")
    model = None
    marginal_prior = None
    metadata = None
    if args.model_checkpoint is not None:
        model, marginal_prior, metadata = load_mixed_edit_checkpoint(
            args.model_checkpoint, device=args.device
        )
    role_path = args.output_dir / "exchangeable_role_degree_prior.json"
    role_prior = None
    context_model = None
    if args.context_checkpoint is not None:
        context_model, context_payload = load_rooted_context_checkpoint(
            args.context_checkpoint, legacy_scorer=None, device=args.device
        )
        embedded = context_payload["legacy_checkpoint_payload"]
        embedded_prior = MarginalGraphPrior.from_json_dict(embedded["prior"])
        embedded_role = context_payload.get("exchangeable_role_prior")
        if not isinstance(embedded_role, dict):
            raise ValueError("context bundle does not contain an exchangeable role prior")
        role_prior = ExchangeableRoleDegreePrior.from_json_dict(embedded_role)
        if model is None:
            model = context_model.joint.legacy_scorer
            marginal_prior = embedded_prior
            metadata = {
                "feature_profile": str(
                    embedded.get("feature_profile", "legacy_local_v1")
                )
            }
        elif embedded_prior.to_json_dict() != marginal_prior.to_json_dict():
            raise ValueError("context bundle and model checkpoint use different priors")
        if args.context_scale >= 0.0:
            context_model.set_residual_scale(args.context_scale)
    assert model is not None and marginal_prior is not None and metadata is not None
    if role_prior is None:
        role_prior = ExchangeableRoleDegreePrior.from_training_graph(source)
    if tuple(source.type_names) != role_prior.type_names:
        raise ValueError("dataset schema and role prior type names disagree")
    role_prior.save(role_path)
    if args.initial_graph is not None:
        state_path = args.initial_graph.with_name("sampling_state.json")
        if state_path.exists():
            previous_state = json.loads(state_path.read_text(encoding="utf-8"))
            previous_seed = int(previous_state["generation_seed"])
            if previous_seed != int(args.seed):
                raise RuntimeError(
                    "continuation seed would change the anonymous role assignment: "
                    f"previous={previous_seed}, requested={args.seed}"
                )
    initial = (
        TypedGraph.from_json_dict(
            json.loads(args.initial_graph.read_text(encoding="utf-8"))
        )
        if args.initial_graph is not None
        else FamilyCountCondition(
            marginal=marginal_prior,
            observed_counts=role_prior.family_counts(
                marginal_prior.family_endpoints
            ),
        ).sample_graph(args.seed + 71, mode=args.family_count_mode)
    )
    initial_counts = family_edge_counts(initial)
    cuda_benchmark = args.sampling_backend == "gpu" and torch.cuda.is_available()
    if cuda_benchmark:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    sampling_started = time.perf_counter()
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
        edits_per_outer_step=args.edits_per_outer_step,
        closure_candidate_fraction=args.closure_candidate_fraction,
        closure_degree_bias_power=args.closure_degree_bias_power,
        auto_role_balance=args.auto_role_balance,
        role_calibration_batches=args.role_calibration_batches,
        targeted_role_candidate_fraction=args.targeted_role_candidate_fraction,
        context_model=context_model,
        context_refresh_steps=args.context_refresh_steps,
        sampling_backend=args.sampling_backend,
        gpu_proposal_pool_size=args.gpu_proposal_pool_size,
        gpu_proposal_chunk_size=args.gpu_proposal_chunk_size,
        gpu_metric_edge_chunk_size=args.gpu_metric_edge_chunk_size,
        gpu_phase_check_interval=args.gpu_phase_check_interval,
        gpu_use_triton=not args.gpu_disable_triton,
        phase_policy=args.phase_policy,
        selection_mode=args.selection_mode,
    )
    if cuda_benchmark:
        torch.cuda.synchronize()
    sampling_seconds = time.perf_counter() - sampling_started
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
        "edits_per_outer_step": args.edits_per_outer_step,
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
        "family_count_mode": args.family_count_mode,
        "context_checkpoint": (
            None if args.context_checkpoint is None else str(args.context_checkpoint)
        ),
        "context_refresh_steps": args.context_refresh_steps,
        "sampling_backend": args.sampling_backend,
        "gpu_proposal_pool_size": args.gpu_proposal_pool_size,
        "gpu_proposal_chunk_size": args.gpu_proposal_chunk_size,
        "gpu_phase_check_interval": args.gpu_phase_check_interval,
        "gpu_use_triton": (
            args.sampling_backend == "gpu" and not args.gpu_disable_triton
        ),
        "gpu_peak_memory_allocated_bytes": (
            int(torch.cuda.max_memory_allocated()) if cuda_benchmark else 0
        ),
        "gpu_peak_memory_reserved_bytes": (
            int(torch.cuda.max_memory_reserved()) if cuda_benchmark else 0
        ),
        "sampling_seconds": sampling_seconds,
        "sampling_steps_per_second": args.steps / max(sampling_seconds, 1.0e-9),
        "context_scale": (
            None
            if context_model is None
            else float(context_model.residual_scale.item())
        ),
        "phase_policy": args.phase_policy,
        "selection_mode": args.selection_mode,
        "initial_graph_source": (
            f"{args.family_count_mode}_family_count_prior"
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
    sampling_state = {
        "format": "dihug-role-guided-sampling-state-v1",
        "generation_seed": int(args.seed),
        "role_assignment_seed": int(result.trace[0]["role_assignment_seed"]),
        "role_assignment_hash": str(result.trace[0]["role_assignment_hash"]),
        "family_counts": final_counts,
        "context_checkpoint": (
            None if args.context_checkpoint is None else str(args.context_checkpoint)
        ),
        "context_scale": (
            None
            if context_model is None
            else float(context_model.residual_scale.item())
        ),
    }
    (args.output_dir / "sampling_state.json").write_text(
        json.dumps(sampling_state, indent=2, sort_keys=True), encoding="utf-8"
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
