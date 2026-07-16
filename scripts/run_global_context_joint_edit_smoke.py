#!/usr/bin/env python3
"""GPU smoke for the zero-initialized global-context joint-edit residual."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from sparse_diffusion.structure_energy.global_context_joint_edit import (
    GlobalContextJointEditResidual,
)
from sparse_diffusion.structure_energy.graph import load_typed_graph
from sparse_diffusion.structure_energy.hgt_edit_denoiser import candidate_endpoint_matrix
from sparse_diffusion.structure_energy.mixed_edit_diffusion import (
    EDIT_ACTIONS,
    MarginalGraphPrior,
    align_mixed_edit_features_for_model,
    load_mixed_edit_checkpoint,
    mixed_edit_candidate_matrix,
    sample_edit_candidates,
)
from sparse_diffusion.structure_energy.switch_diffusion import SwitchDenoiserMLP


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data/IMDB_movie_subgraphs/subgraph_000"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--candidates", type=int, default=16)
    parser.add_argument("--node-hidden-dim", type=int, default=128)
    parser.add_argument("--edge-hidden-dim", type=int, default=64)
    parser.add_argument(
        "--legacy-checkpoint",
        type=Path,
        default=None,
        help="Optional trained mixed-edit checkpoint used for the exact fallback check.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    source = load_typed_graph(args.dataset_dir, active_only=True)
    checkpoint_metadata = {}
    if args.legacy_checkpoint is None:
        prior = MarginalGraphPrior.from_training_graph(source)
        legacy = None
    else:
        legacy, prior, checkpoint_metadata = load_mixed_edit_checkpoint(
            args.legacy_checkpoint, device=device
        )
        legacy.eval()
    graph = prior.sample(args.seed + 17)
    rng = np.random.default_rng(args.seed + 29)
    candidates, _ = sample_edit_candidates(
        graph, rng, args.candidates, actions=EDIT_ACTIONS
    )
    feature_profile = str(
        checkpoint_metadata.get("feature_profile", "typed_closure_full_v2")
    )
    features_np = mixed_edit_candidate_matrix(
        graph,
        candidates,
        1.0,
        feature_profile=feature_profile,
    )
    if legacy is not None:
        features_np = align_mixed_edit_features_for_model(
            features_np,
            legacy,
            family_count=len(graph.family_specs),
        )
    endpoints_np = candidate_endpoint_matrix(candidates)

    features = torch.from_numpy(features_np)[None].float().to(device)
    endpoints = torch.from_numpy(endpoints_np)[None].long().to(device)
    node_hidden = torch.randn(
        1, graph.num_nodes, args.node_hidden_dim, device=device
    )
    edge_hidden = torch.randn(
        1,
        args.candidates,
        4,
        args.edge_hidden_dim,
        device=device,
    )
    if legacy is None:
        legacy = SwitchDenoiserMLP(
            input_dim=features.shape[-1], hidden_dim=256, layers=3, dropout=0.0
        ).to(device).eval()
    if legacy.input_dim != features.shape[-1]:
        raise RuntimeError(
            "legacy checkpoint feature contract mismatch: "
            f"model={legacy.input_dim}, candidates={features.shape[-1]}"
        )
    fused = GlobalContextJointEditResidual(
        legacy,
        node_hidden_dim=args.node_hidden_dim,
        candidate_feature_dim=features.shape[-1],
        edge_slot_dim=args.edge_hidden_dim,
        hidden_dim=128,
    ).to(device).eval()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.no_grad():
        legacy_logits = legacy(features)
        fused_logits = fused(
            features,
            node_hidden=node_hidden,
            candidate_endpoints=endpoints,
            edge_slot_hidden=edge_hidden,
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    max_abs_diff = float((legacy_logits - fused_logits).abs().max().cpu())
    if max_abs_diff != 0.0:
        raise RuntimeError(f"zero residual changed legacy logits by {max_abs_diff}")

    payload = {
        "format": "dihug-global-context-joint-edit-smoke-v1",
        "dataset": str(args.dataset_dir),
        "nodes": int(graph.num_nodes),
        "candidates": int(args.candidates),
        "candidate_feature_dim": int(features.shape[-1]),
        "node_hidden_dim": int(args.node_hidden_dim),
        "edge_hidden_dim": int(args.edge_hidden_dim),
        "legacy_checkpoint": (
            str(args.legacy_checkpoint) if args.legacy_checkpoint is not None else None
        ),
        "legacy_feature_profile": feature_profile,
        "legacy_exact_max_abs_diff": max_abs_diff,
        "forward_seconds": float(elapsed),
        "peak_allocated_mb": (
            float(torch.cuda.max_memory_allocated(device) / (1024**2))
            if device.type == "cuda"
            else 0.0
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print("[GLOBAL-CONTEXT-JOINT-EDIT] " + json.dumps(payload, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
