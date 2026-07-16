#!/usr/bin/env python3
"""Train and sample the global-HGT macro relocation/switch gate."""

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
from sparse_diffusion.structure_energy.graph import load_typed_graph
from sparse_diffusion.structure_energy.macro_edit_diffusion import (
    build_macro_rs_dataset,
    build_macro_structure_utility_dataset,
    sample_macro_rs_diffusion,
    train_macro_rs_denoiser,
)
from sparse_diffusion.structure_energy.mixed_edit_diffusion import (
    load_mixed_edit_checkpoint,
)
from sparse_diffusion.structure_energy.switch_chain import graph_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--legacy-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--trajectories", type=int, default=4)
    parser.add_argument("--forward-macro-steps", type=int, default=80)
    parser.add_argument("--record-every", type=int, default=2)
    parser.add_argument("--max-budget", type=int, default=4)
    parser.add_argument("--candidates", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--sample-macro-steps", type=int, default=100)
    parser.add_argument("--count-mode", choices=("fixed", "binomial"), default="binomial")
    parser.add_argument("--energy-guidance-scale", type=float, default=0.0)
    parser.add_argument(
        "--training-objective",
        choices=("inverse", "structure_utility"),
        default="inverse",
    )
    parser.add_argument("--utility-temperature", type=float, default=0.05)
    parser.add_argument("--max-structure-features", type=int, default=48)
    parser.add_argument(
        "--forward-budget-schedule",
        choices=("linear", "constant"),
        default="linear",
    )
    parser.add_argument(
        "--record-schedule",
        choices=("uniform", "quadratic"),
        default="uniform",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = load_typed_graph(args.dataset_dir, active_only=True)
    legacy, _, metadata = load_mixed_edit_checkpoint(
        args.legacy_checkpoint, device=args.device
    )
    feature_profile = metadata["feature_profile"]
    started = time.perf_counter()
    structure_target = None
    if args.training_objective == "structure_utility":
        dataset, structure_target = build_macro_structure_utility_dataset(
            source,
            trajectories=args.trajectories,
            macro_steps=args.forward_macro_steps,
            record_every=args.record_every,
            max_budget=args.max_budget,
            candidates=args.candidates,
            seed=args.seed,
            feature_profile=feature_profile,
            utility_temperature=args.utility_temperature,
            max_structure_features=args.max_structure_features,
        )
    else:
        dataset = build_macro_rs_dataset(
            source,
            trajectories=args.trajectories,
            macro_steps=args.forward_macro_steps,
            record_every=args.record_every,
            max_budget=args.max_budget,
            candidates=args.candidates,
            seed=args.seed,
            feature_profile=feature_profile,
            budget_schedule=args.forward_budget_schedule,
            record_schedule=args.record_schedule,
        )
    dataset_seconds = time.perf_counter() - started
    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    train_started = time.perf_counter()
    model, train_report = train_macro_rs_denoiser(
        dataset,
        legacy,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device,
        seed=args.seed,
        max_budget=args.max_budget,
    )
    train_seconds = time.perf_counter() - train_started
    peak_mb = (
        float(torch.cuda.max_memory_allocated() / (1024**2))
        if torch.cuda.is_available() and str(args.device).startswith("cuda")
        else 0.0
    )
    condition = FamilyCountCondition.from_training_graph(source)
    initial = condition.sample_graph(args.seed + 71, mode=args.count_mode)
    initial_counts = family_edge_counts(initial)
    sample_started = time.perf_counter()
    result = sample_macro_rs_diffusion(
        initial,
        model,
        macro_steps=args.sample_macro_steps,
        candidates_per_step=args.candidates,
        seed=args.seed + 991,
        device=args.device,
        feature_profile=feature_profile,
        metrics_every=max(args.sample_macro_steps // 10, 1),
        energy_guidance_scale=args.energy_guidance_scale,
    )
    sample_seconds = time.perf_counter() - sample_started
    final_counts = family_edge_counts(result.graph)
    if final_counts != initial_counts:
        raise RuntimeError("macro sampler changed its family count condition")
    payload = {
        "format": "dihug-macro-rs-gate-v1",
        "seed": int(args.seed),
        "count_mode": args.count_mode,
        "feature_profile": feature_profile,
        "dataset_examples": int(dataset.budgets.shape[0]),
        "train": dict(train_report),
        "initial": result.trace[0],
        "final": graph_metrics(result.graph),
        "real": graph_metrics(source),
        "trace": list(result.trace),
        "action_counts": dict(result.action_counts),
        "initial_counts": initial_counts,
        "final_counts": final_counts,
        "timing": {
            "dataset_seconds": dataset_seconds,
            "train_seconds": train_seconds,
            "sample_seconds": sample_seconds,
            "peak_allocated_mb": peak_mb,
        },
        "sampling_uses_clean_graph": False,
        "energy_guidance_scale": float(args.energy_guidance_scale),
        "training_objective": args.training_objective,
        "structure_target_used_at_sampling": False,
        "forward_budget_schedule": args.forward_budget_schedule,
        "record_schedule": args.record_schedule,
    }
    (args.output_dir / "gate_report.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    checkpoint_payload = {
            "format": "dihug-macro-rs-denoiser-v1",
            "state_dict": model.state_dict(),
            "feature_profile": feature_profile,
            "max_budget": int(args.max_budget),
            "train_report": dict(train_report),
            "training_objective": args.training_objective,
        }
    if structure_target is not None:
        checkpoint_payload["structure_target_audit"] = {
            "selection": structure_target.selection.to_json_dict(),
            "target_vector": structure_target.target_vector.tolist(),
            "null_distance": float(structure_target.null_distance),
            "used_at_sampling": False,
        }
    torch.save(checkpoint_payload, args.output_dir / "macro_rs_denoiser.pt")
    print(
        "[MACRO-RS-GATE] "
        f"examples={payload['dataset_examples']} "
        f"positive_top1={train_report['validation_positive_top1']:.4f} "
        f"budget_acc={train_report['validation_budget_accuracy']:.4f} "
        f"energy_mae={train_report['validation_energy_mae']:.4f} "
        f"tri={payload['initial']['triangles']:.0f}->{payload['final']['triangles']:.0f} "
        f"clustering={payload['initial']['clustering']:.6f}->"
        f"{payload['final']['clustering']:.6f} peak_mb={peak_mb:.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
