#!/usr/bin/env python3
"""Generate multiple independent graphs with one batched context forward."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from sparse_diffusion.structure_energy.cardinality_conditioned_edit import (
    FamilyCountCondition,
    family_edge_counts,
)
from sparse_diffusion.structure_energy.gpu_role_guided_edit import (
    sample_gpu_role_guided_edit_diffusion_batched,
)
from sparse_diffusion.structure_energy.graph import load_typed_graph
from sparse_diffusion.structure_energy.mixed_edit_diffusion import MarginalGraphPrior
from sparse_diffusion.structure_energy.role_guided_edit_diffusion import (
    ExchangeableRoleDegreePrior,
)
from sparse_diffusion.structure_energy.rooted_context_edit import (
    load_rooted_context_checkpoint,
)
from sparse_diffusion.structure_energy.switch_chain import graph_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evaluation-dataset-dir",
        type=Path,
        help="optional clean graph used only after generation for reporting",
    )
    parser.add_argument("--context-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", default="0,1")
    parser.add_argument("--mode", choices=("legacy", "context"), default="context")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--family-count-mode", choices=("fixed", "binomial"), default="binomial")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--candidates", type=int, default=16)
    parser.add_argument("--metrics-every", type=int, default=1000)
    parser.add_argument("--model-time-horizon-steps", type=int, default=20000)
    parser.add_argument("--role-weight", type=float, default=0.05)
    parser.add_argument("--gain-scale", type=float, default=10.0)
    parser.add_argument("--role-error-per-node-threshold", type=float, default=5.5)
    parser.add_argument("--relative-role-error-threshold", type=float)
    parser.add_argument("--context-refresh-steps", type=int, default=20)
    parser.add_argument("--gpu-candidate-mode", choices=("legacy_pool", "canonical_ordered"), default="canonical_ordered")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() or torch.device(args.device).type != "cuda":
        raise RuntimeError("batched context generation requires CUDA")
    seeds = tuple(int(value) for value in args.seeds.split(",") if value.strip())
    if not seeds:
        raise ValueError("provide at least one generation seed")
    context, payload = load_rooted_context_checkpoint(
        args.context_checkpoint, legacy_scorer=None, device=args.device
    )
    embedded = payload["legacy_checkpoint_payload"]
    marginal = MarginalGraphPrior.from_json_dict(embedded["prior"])
    role_prior = ExchangeableRoleDegreePrior.from_json_dict(
        payload["exchangeable_role_prior"]
    )
    if tuple(marginal.type_names) != role_prior.type_names:
        raise ValueError("marginal and role-prior checkpoint schemas disagree")
    condition = FamilyCountCondition(
        marginal=marginal,
        observed_counts=role_prior.family_counts(marginal.family_endpoints),
    )
    graphs = [
        condition.sample_graph(seed + 71, mode=args.family_count_mode)
        for seed in seeds
    ]
    roles = [
        role_prior.sample_assignment_for_graph(graph, seed + 91_919)
        for graph, seed in zip(graphs, seeds)
    ]
    initial_counts = [family_edge_counts(graph) for graph in graphs]
    context_model = context if args.mode == "context" else None
    legacy = context.joint.legacy_scorer

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    results = sample_gpu_role_guided_edit_diffusion_batched(
        graphs,
        legacy,
        roles,
        steps=args.steps,
        candidates_per_step=args.candidates,
        seeds=seeds,
        device=args.device,
        metrics_every=args.metrics_every,
        role_weight=args.role_weight,
        gain_scale=args.gain_scale,
        schedule_power=1.0,
        schedule_floor=1.0,
        role_error_per_node_threshold=args.role_error_per_node_threshold,
        relative_role_error_threshold=args.relative_role_error_threshold,
        model_time_horizon_steps=args.model_time_horizon_steps,
        model_time_start_step=0,
        closure_candidate_fraction=1.0,
        targeted_role_candidate_fraction=0.5,
        auto_role_balance=0.5,
        context_model=context_model,
        context_refresh_steps=args.context_refresh_steps,
        candidate_mode=args.gpu_candidate_mode,
        use_triton=True,
    )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    peak_reserved = int(torch.cuda.max_memory_reserved())
    real_metrics = None
    if args.evaluation_dataset_dir is not None:
        source = load_typed_graph(args.evaluation_dataset_dir, active_only=True)
        if (
            tuple(source.type_names) != tuple(marginal.type_names)
            or source.num_nodes != graphs[0].num_nodes
        ):
            raise ValueError("evaluation graph and checkpoint schemas disagree")
        real_metrics = graph_metrics(source)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed, initial, role_rows, expected_counts, result in zip(
        seeds, graphs, roles, initial_counts, results
    ):
        if family_edge_counts(result.graph) != expected_counts:
            raise RuntimeError(f"seed {seed} changed family counts")
        assignment_hash = hashlib.sha256(
            "|".join(role_prior.family_names).encode("utf-8")
            + np.ascontiguousarray(role_rows).tobytes()
        ).hexdigest()
        result.trace[0]["role_assignment_seed"] = seed + 91_919
        result.trace[0]["role_assignment_hash"] = assignment_hash
        seed_dir = args.output_dir / f"seed{seed}" / args.mode
        seed_dir.mkdir(parents=True, exist_ok=True)
        report = {
            "format": "dihug-batched-context-generation-v1",
            "seed": seed,
            "mode": args.mode,
            "batch_size": len(seeds),
            "steps": args.steps,
            "candidates": args.candidates,
            "family_count_mode": args.family_count_mode,
            "sampling_backend": "gpu_resident_batched",
            "gpu_candidate_mode": args.gpu_candidate_mode,
            "context_checkpoint": str(args.context_checkpoint),
            "context_refresh_steps": args.context_refresh_steps,
            "role_error_per_node_threshold": args.role_error_per_node_threshold,
            "relative_role_error_threshold": args.relative_role_error_threshold,
            "initial": graph_metrics(initial),
            "final": graph_metrics(result.graph),
            "trace": result.trace,
            "action_counts": result.action_counts,
            "family_counts_preserved": True,
            "sampling_uses_clean_adjacency": False,
        }
        if real_metrics is not None:
            report["evaluation_real"] = real_metrics
        (seed_dir / "gate_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
        )
        (seed_dir / "final_graph.json").write_text(
            json.dumps(result.graph.to_json_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        rows.append({"seed": seed, "final": report["final"]})
    summary = {
        "format": "dihug-batched-context-generation-summary-v1",
        "mode": args.mode,
        "seeds": list(seeds),
        "batch_size": len(seeds),
        "total_seconds": elapsed,
        "chain_steps_per_second": len(seeds) * args.steps / max(elapsed, 1.0e-9),
        "peak_reserved_bytes": peak_reserved,
        "rows": rows,
    }
    (args.output_dir / f"summary_{args.mode}.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
