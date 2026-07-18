#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from sparse_diffusion.structure_energy.graph import load_typed_graph
from sparse_diffusion.structure_energy.mixed_edit_diffusion import (
    load_mixed_edit_checkpoint,
)
from sparse_diffusion.structure_energy.role_guided_edit_diffusion import (
    ExchangeableRoleDegreePrior,
)
from sparse_diffusion.structure_energy.sequential_macro_context import (
    build_sequential_macro_dataset,
    save_sequential_macro_context_checkpoint,
    train_sequential_macro_context,
)
from sparse_diffusion.structure_energy.subtype_decoder import load_subtype_schema


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--legacy-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trajectories", type=int, default=4)
    parser.add_argument("--macro-steps", type=int, default=100)
    parser.add_argument("--micro-steps", type=int, default=20)
    parser.add_argument("--record-every", type=int, default=2)
    parser.add_argument("--candidates", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--base-aux-weight", type=float, default=0.25)
    parser.add_argument("--trust-weight", type=float, default=0.02)
    parser.add_argument("--teacher-temperature", type=float, default=0.5)
    parser.add_argument("--freeze-legacy", action="store_true")
    parser.add_argument(
        "--supervision",
        choices=("exact_inverse", "rooted_utility"),
        default="exact_inverse",
    )
    parser.add_argument(
        "--actions",
        nargs="+",
        choices=("relocation", "switch"),
        default=("switch",),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = load_typed_graph(args.dataset_dir, active_only=True)
    legacy, _, metadata = load_mixed_edit_checkpoint(
        args.legacy_checkpoint, device="cpu"
    )
    feature_profile = str(
        metadata.get("feature_profile", "typed_closure_compact_v2")
    )
    started = time.perf_counter()
    dataset = build_sequential_macro_dataset(
        source,
        legacy,
        trajectories=args.trajectories,
        macro_steps=args.macro_steps,
        micro_steps=args.micro_steps,
        record_every=args.record_every,
        seed=args.seed,
        feature_profile=feature_profile,
        actions=args.actions,
    )
    dataset_seconds = time.perf_counter() - started
    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    train_started = time.perf_counter()
    model, report = train_sequential_macro_context(
        dataset,
        legacy,
        epochs=args.epochs,
        candidates_per_micro=args.candidates,
        learning_rate=args.learning_rate,
        base_aux_weight=args.base_aux_weight,
        trust_weight=args.trust_weight,
        teacher_temperature=args.teacher_temperature,
        finetune_legacy=not args.freeze_legacy,
        supervision=args.supervision,
        device=args.device,
        seed=args.seed,
        hidden_dim=args.hidden_dim,
        layers=args.layers,
        heads=args.heads,
        dropout=args.dropout,
    )
    train_seconds = time.perf_counter() - train_started
    peak_mb = (
        float(torch.cuda.max_memory_allocated() / (1024**2))
        if torch.cuda.is_available() and str(args.device).startswith("cuda")
        else 0.0
    )
    role_prior = ExchangeableRoleDegreePrior.from_training_graph(source)
    subtype_schema = load_subtype_schema(args.dataset_dir)
    checkpoint = args.output_dir / "sequential_macro_context.pt"
    save_sequential_macro_context_checkpoint(
        checkpoint,
        model,
        report,
        args.legacy_checkpoint,
        role_prior=role_prior.to_json_dict(),
        subtype_schema=subtype_schema.to_json_dict(),
    )
    payload = {
        "format": "dihug-sequential-macro-context-gate-v1",
        "dataset": str(args.dataset_dir),
        "legacy_checkpoint": str(args.legacy_checkpoint),
        "checkpoint": str(checkpoint),
        "feature_profile": feature_profile,
        "nodes": source.num_nodes,
        "edges": source.num_edges,
        "trajectories": args.trajectories,
        "macro_steps": args.macro_steps,
        "micro_steps": args.micro_steps,
        "record_every": args.record_every,
        "actions": list(args.actions),
        "supervision": args.supervision,
        "context_refresh_steps": args.micro_steps,
        "dataset_seconds": dataset_seconds,
        "train_seconds": train_seconds,
        "peak_allocated_mb": peak_mb,
        **dict(report),
    }
    (args.output_dir / "gate_report.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    print("[SEQUENTIAL-MACRO-GATE] " + json.dumps(payload, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
