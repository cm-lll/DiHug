#!/usr/bin/env python3
"""Paired family-cardinality and edit-kernel gate on identical initial graphs."""

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
    sample_mixed_edit_diffusion,
)
from sparse_diffusion.structure_energy.switch_chain import graph_metrics


KERNELS = {
    "rs": ("relocation", "switch"),
    "bdrs": ("birth", "death", "relocation", "switch"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kernel", choices=tuple(KERNELS), required=True)
    parser.add_argument("--count-mode", choices=("fixed", "binomial"), required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--candidates", type=int, default=16)
    parser.add_argument("--metrics-every", type=int, default=250)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = load_typed_graph(args.dataset_dir, active_only=True)
    condition = FamilyCountCondition.from_training_graph(source)
    initial = condition.sample_graph(args.seed + 71, mode=args.count_mode)
    initial_counts = family_edge_counts(initial)
    model, prior, metadata = load_mixed_edit_checkpoint(
        args.checkpoint, device=args.device
    )
    result = sample_mixed_edit_diffusion(
        prior,
        model,
        steps=args.steps,
        candidates_per_step=args.candidates,
        seed=args.seed,
        device=args.device,
        temperature=0.0,
        metrics_every=args.metrics_every,
        actions=KERNELS[args.kernel],
        initial_graph=initial,
        feature_profile=metadata["feature_profile"],
    )
    final_counts = family_edge_counts(result.graph)
    counts_preserved = initial_counts == final_counts
    if args.kernel == "rs" and not counts_preserved:
        raise RuntimeError("relocation/switch changed a family edge count")
    payload = {
        "format": "dihug-cardinality-rs-gate-v1",
        "kernel": args.kernel,
        "count_mode": args.count_mode,
        "seed": int(args.seed),
        "conditional_on_exact_training_counts": args.count_mode == "fixed",
        "initial": result.trace[0],
        "final": graph_metrics(result.graph),
        "real": graph_metrics(source),
        "initial_counts": initial_counts,
        "final_counts": final_counts,
        "counts_preserved": counts_preserved,
        "action_counts": dict(result.action_counts),
        "trace": list(result.trace),
        "count_condition": condition.to_json_dict()["inference_contract"],
        "inference_uses_clean_adjacency": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        "[CARDINALITY-RS-GATE] "
        f"mode={args.count_mode} kernel={args.kernel} "
        f"edges={payload['initial']['edges']:.0f}->{payload['final']['edges']:.0f} "
        f"tri={payload['initial']['triangles']:.0f}->{payload['final']['triangles']:.0f} "
        f"clustering={payload['initial']['clustering']:.6f}->"
        f"{payload['final']['clustering']:.6f} counts_preserved={counts_preserved}",
        flush=True,
    )


if __name__ == "__main__":
    main()
