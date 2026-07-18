#!/usr/bin/env python3
"""Generate a heterogeneous graph from an identity-free structure checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
repo_root_str = str(REPO_ROOT)
sys.path = [entry for entry in sys.path if entry != repo_root_str]
sys.path.insert(0, repo_root_str)

from sparse_diffusion.structure_energy.checkpoint import StructureEnergyCheckpoint
from sparse_diffusion.structure_energy.features import GraphletFeatureExtractor
from sparse_diffusion.structure_energy.graph import TypedGraph
from sparse_diffusion.structure_energy.roles import realize_role_plan, validate_role_realization
from sparse_diffusion.structure_energy.switch_chain import graph_metrics, run_switch_chain


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--initial-graph", type=Path)
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--burn-in", type=int, default=20_000)
    parser.add_argument("--thin", type=int, default=1_000)
    parser.add_argument("--verify-every", type=int, default=0)
    parser.add_argument("--curveball-probability", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = StructureEnergyCheckpoint.load(args.checkpoint)
    schema, plan = checkpoint.sample_role_plan(args.seed)
    if args.initial_graph is None:
        initial = realize_role_plan(schema, plan)
    else:
        with args.initial_graph.open("r", encoding="utf-8") as handle:
            initial = TypedGraph.from_json_dict(json.load(handle))
        validate_role_realization(initial, plan)
    result = run_switch_chain(
        initial,
        GraphletFeatureExtractor(initial, tracked_keys=checkpoint.selection.keys),
        steps=args.steps,
        burn_in=args.burn_in,
        thin=args.thin,
        seed=args.seed + 17,
        energy=checkpoint.energy(),
        verify_every=args.verify_every,
        curveball_probability=args.curveball_probability,
    )
    validate_role_realization(result.graph, plan)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "generated_graph.json").open("w", encoding="utf-8") as handle:
        json.dump(result.graph.to_json_dict(), handle, indent=2, sort_keys=True)
    payload = {
        "seed": args.seed,
        "attempted": result.attempted,
        "valid": result.valid,
        "accepted": result.accepted,
        "valid_rate": result.valid_rate,
        "acceptance_rate": result.acceptance_rate,
        "final_metrics": graph_metrics(result.graph),
        "trace": [
            {
                "step": sample.step,
                "energy": sample.energy,
                "metrics": sample.metrics,
            }
            for sample in result.samples
        ],
    }
    with (args.output_dir / "sampling_report.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    print(
        "[CHECKPOINT-SAMPLE] "
        f"seed={args.seed} triangles={payload['final_metrics']['triangles']:.0f} "
        f"clustering={payload['final_metrics']['clustering']:.6f} "
        f"accept={result.acceptance_rate:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
