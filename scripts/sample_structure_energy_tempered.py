#!/usr/bin/env python3
"""Replica-exchange sampling from a frozen structure-energy checkpoint."""

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
from sparse_diffusion.structure_energy.roles import realize_role_plan, validate_role_realization
from sparse_diffusion.structure_energy.switch_chain import run_parallel_tempering


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--betas", default="0,0.2,0.4,0.6,0.8,1")
    parser.add_argument("--rounds", type=int, default=40)
    parser.add_argument("--local-steps", type=int, default=5_000)
    parser.add_argument("--initialization-steps", type=int, default=5_000)
    parser.add_argument("--verify-every", type=int, default=0)
    parser.add_argument("--curveball-probability", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    betas = [float(value) for value in args.betas.split(",") if value.strip()]
    checkpoint = StructureEnergyCheckpoint.load(args.checkpoint)
    schema, plan = checkpoint.sample_role_plan(args.seed)
    initial = realize_role_plan(schema, plan)
    result = run_parallel_tempering(
        initial,
        checkpoint.energy(),
        temperatures=betas,
        rounds=args.rounds,
        local_steps=args.local_steps,
        initialization_steps=args.initialization_steps,
        seed=args.seed + 17,
        verify_every=args.verify_every,
        curveball_probability=args.curveball_probability,
    )
    validate_role_realization(result.graph, plan)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "generated_graph.json").open("w", encoding="utf-8") as handle:
        json.dump(result.graph.to_json_dict(), handle, indent=2, sort_keys=True)
    payload = {
        "seed": args.seed,
        "betas": betas,
        "exchange_attempted": result.exchange_attempted,
        "exchange_accepted": result.exchange_accepted,
        "exchange_rate": result.exchange_rate,
        "local_acceptance_rates": result.local_acceptance_rates,
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
    final = result.samples[-1].metrics
    print(
        "[TEMPERED-SAMPLE] "
        f"seed={args.seed} triangles={final['triangles']:.0f} "
        f"clustering={final['clustering']:.6f} "
        f"exchange={result.exchange_rate:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
