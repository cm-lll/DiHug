#!/usr/bin/env python3
"""Write a target-free, scale-adaptive mixed-edit experiment profile."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from sparse_diffusion.structure_energy.adaptive_profile import (
    build_adaptive_mixed_edit_profile,
)
from sparse_diffusion.structure_energy.graph import load_typed_graph


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--role-coverage", type=float, default=0.95)
    parser.add_argument("--noise-steps-per-edge", type=float, default=2.0)
    parser.add_argument("--target-optimizer-updates", type=int, default=5000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    graph = load_typed_graph(args.dataset_dir, active_only=True)
    profile = build_adaptive_mixed_edit_profile(
        graph,
        role_coverage=args.role_coverage,
        noise_steps_per_edge=args.noise_steps_per_edge,
        target_optimizer_updates=args.target_optimizer_updates,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(profile.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(profile.to_dict(), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
