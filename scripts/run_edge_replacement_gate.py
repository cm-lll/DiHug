#!/usr/bin/env python3
"""Train and evaluate fixed-cardinality edge-replacement diffusion."""

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
from sparse_diffusion.structure_energy.edge_replacement_diffusion import (
    build_family_domains,
    build_replacement_examples,
    sample_replacement_reverse,
    train_replacement_denoiser,
)
from sparse_diffusion.structure_energy.graph import load_typed_graph
from sparse_diffusion.structure_energy.switch_chain import graph_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--trajectories", type=int, default=8)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--record-every", type=int, default=10)
    parser.add_argument("--rate", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--negatives-per-positive", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--chunk-size", type=int, default=8192)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = load_typed_graph(args.dataset_dir, active_only=True)
    domains = build_family_domains(source)
    started = time.perf_counter()
    examples = build_replacement_examples(
        source,
        trajectories=args.trajectories,
        steps=args.steps,
        record_every=args.record_every,
        rate=args.rate,
        seed=args.seed,
        domains=domains,
    )
    dataset_seconds = time.perf_counter() - started
    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    train_started = time.perf_counter()
    model, train_report = train_replacement_denoiser(
        examples,
        source,
        domains,
        epochs=args.epochs,
        negatives_per_positive=args.negatives_per_positive,
        learning_rate=args.learning_rate,
        device=args.device,
        seed=args.seed,
    )
    train_seconds = time.perf_counter() - train_started
    condition = FamilyCountCondition.from_training_graph(source)
    initial = condition.sample_graph(args.seed + 71, mode="fixed")
    initial_counts = family_edge_counts(initial)
    sample_started = time.perf_counter()
    generated, trace = sample_replacement_reverse(
        initial,
        model,
        domains,
        steps=args.steps,
        rate=args.rate,
        device=args.device,
        chunk_size=args.chunk_size,
        metrics_every=max(args.steps // 10, 1),
    )
    sample_seconds = time.perf_counter() - sample_started
    final_counts = family_edge_counts(generated)
    if final_counts != initial_counts:
        raise RuntimeError("replacement sampler changed family counts")
    report = {
        "format": "dihug-edge-replacement-gate-v1",
        "seed": int(args.seed),
        "rate": float(args.rate),
        "steps": int(args.steps),
        "examples": len(examples),
        "train": train_report,
        "initial": trace[0],
        "final": graph_metrics(generated),
        "real": graph_metrics(source),
        "trace": trace,
        "initial_counts": initial_counts,
        "final_counts": final_counts,
        "sampling_uses_clean_graph": False,
        "timing": {
            "dataset_seconds": dataset_seconds,
            "train_seconds": train_seconds,
            "sample_seconds": sample_seconds,
            "peak_allocated_mb": (
                float(torch.cuda.max_memory_allocated() / (1024**2))
                if torch.cuda.is_available() and str(args.device).startswith("cuda")
                else 0.0
            ),
        },
    }
    (args.output_dir / "gate_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    torch.save(
        {
            "format": "dihug-edge-replacement-denoiser-v1",
            "state_dict": model.state_dict(),
            "train_report": train_report,
        },
        args.output_dir / "edge_replacement_denoiser.pt",
    )
    print(
        "[REPLACEMENT-GATE] "
        f"delete_recall={train_report['delete_recall']:.4f} "
        f"add_recall={train_report['add_sampled_recall']:.4f} "
        f"tri={trace[0]['triangles']:.0f}->{report['final']['triangles']:.0f} "
        f"clustering={trace[0]['clustering']:.6f}->{report['final']['clustering']:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
