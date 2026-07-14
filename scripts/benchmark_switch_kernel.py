#!/usr/bin/env python3
"""Benchmark the exact batched switch-statistic tensor kernel."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from sparse_diffusion.structure_energy.checkpoint import StructureEnergyCheckpoint
from sparse_diffusion.structure_energy.graph import load_typed_graph
from sparse_diffusion.structure_energy.switch_accelerator import (
    BatchedClosedTriangleKernel,
)
from sparse_diffusion.structure_energy.switch_diffusion import (
    proposal_delta,
    randomize_role_realization,
    sample_switch_candidates,
)
from sparse_diffusion.structure_energy.features import GraphletFeatureExtractor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--structure-checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--candidates", type=int, default=64)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--randomization-steps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    source = load_typed_graph(args.dataset_dir, active_only=True)
    structure = StructureEnergyCheckpoint.load(args.structure_checkpoint)
    graph = structure.realize(args.seed + 10_000)
    graph = randomize_role_realization(
        graph, args.randomization_steps, args.seed + 20_000
    )
    rng = np.random.default_rng(args.seed + 30_000)
    candidates, _ = sample_switch_candidates(graph, rng, args.candidates)
    selection = structure.selection
    extractor = GraphletFeatureExtractor(graph, tracked_keys=selection.keys)
    kernel = BatchedClosedTriangleKernel(graph, selection, device)

    def python_delta() -> np.ndarray:
        return np.stack(
            [
                selection.delta_vector(proposal_delta(graph, extractor, proposal))
                for proposal in candidates
            ]
        )

    expected = python_delta()
    observed = kernel.delta_matrix(candidates).cpu().numpy()
    max_error = float(np.max(np.abs(expected - observed)))
    if max_error > 1.0e-10:
        raise AssertionError(f"tensor switch kernel is not exact: {max_error}")

    start = time.perf_counter()
    for _ in range(args.repetitions):
        python_delta()
    python_seconds = time.perf_counter() - start

    synchronize(device)
    start = time.perf_counter()
    for _ in range(args.repetitions):
        kernel.delta_matrix(candidates)
    synchronize(device)
    tensor_seconds = time.perf_counter() - start
    report = {
        "candidates": int(args.candidates),
        "device": str(device),
        "features": len(selection.keys),
        "max_absolute_error": max_error,
        "python_seconds": python_seconds,
        "repetitions": int(args.repetitions),
        "speedup": python_seconds / max(tensor_seconds, 1.0e-12),
        "tensor_seconds": tensor_seconds,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
        )
    print("[SWITCH-KERNEL-BENCHMARK] " + json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
