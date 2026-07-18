#!/usr/bin/env python3
"""Evaluate a trained macro RS checkpoint without retraining it."""

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
from sparse_diffusion.structure_energy.hgt_edit_denoiser import graph_node_features
from sparse_diffusion.structure_energy.macro_edit_diffusion import (
    MacroRSEditDenoiser,
    sample_macro_rs_diffusion,
)
from sparse_diffusion.structure_energy.mixed_edit_diffusion import (
    load_mixed_edit_checkpoint,
)
from sparse_diffusion.structure_energy.switch_chain import graph_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--legacy-checkpoint", type=Path, required=True)
    parser.add_argument("--macro-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--macro-steps", type=int, default=100)
    parser.add_argument("--candidates", type=int, default=16)
    parser.add_argument("--count-mode", choices=("fixed", "binomial"), default="binomial")
    parser.add_argument("--energy-guidance-scale", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = load_typed_graph(args.dataset_dir, active_only=True)
    legacy, _, legacy_metadata = load_mixed_edit_checkpoint(
        args.legacy_checkpoint, device=args.device
    )
    payload = torch.load(args.macro_checkpoint, map_location="cpu")
    model = MacroRSEditDenoiser(
        legacy,
        node_feature_dim=graph_node_features(source, 0.0).shape[-1] - 1,
        candidate_feature_dim=legacy.input_dim,
        type_count=len(source.type_names),
        family_count=len(source.family_specs),
        max_budget=int(payload["max_budget"]),
    )
    model.load_state_dict(payload["state_dict"], strict=True)
    condition = FamilyCountCondition.from_training_graph(source)
    initial = condition.sample_graph(args.seed + 71, mode=args.count_mode)
    initial_counts = family_edge_counts(initial)
    started = time.perf_counter()
    result = sample_macro_rs_diffusion(
        initial,
        model,
        macro_steps=args.macro_steps,
        candidates_per_step=args.candidates,
        seed=args.seed + 991,
        device=args.device,
        feature_profile=payload.get(
            "feature_profile", legacy_metadata["feature_profile"]
        ),
        metrics_every=max(args.macro_steps // 10, 1),
        energy_guidance_scale=args.energy_guidance_scale,
    )
    final_counts = family_edge_counts(result.graph)
    if final_counts != initial_counts:
        raise RuntimeError("macro sampler changed its family count condition")
    report = {
        "format": "dihug-macro-rs-checkpoint-eval-v1",
        "seed": int(args.seed),
        "count_mode": args.count_mode,
        "energy_guidance_scale": float(args.energy_guidance_scale),
        "initial": result.trace[0],
        "final": graph_metrics(result.graph),
        "real": graph_metrics(source),
        "trace": list(result.trace),
        "action_counts": dict(result.action_counts),
        "initial_counts": initial_counts,
        "final_counts": final_counts,
        "sample_seconds": time.perf_counter() - started,
        "sampling_uses_clean_graph": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        "[MACRO-RS-EVAL] "
        f"guidance={args.energy_guidance_scale:g} "
        f"tri={report['initial']['triangles']:.0f}->{report['final']['triangles']:.0f} "
        f"clustering={report['initial']['clustering']:.6f}->"
        f"{report['final']['clustering']:.6f} "
        f"seconds={report['sample_seconds']:.2f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
