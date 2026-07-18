#!/usr/bin/env python3
"""Build the anonymous structural condition required by switch diffusion."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from sparse_diffusion.structure_energy.checkpoint import StructureEnergyCheckpoint
from sparse_diffusion.structure_energy.features import (
    GraphletFeatureExtractor,
    select_features,
)
from sparse_diffusion.structure_energy.graph import load_typed_graph
from sparse_diffusion.structure_energy.roles import realize_role_plan, sample_role_plan
from sparse_diffusion.structure_energy.switch_chain import (
    LinearGraphletEnergy,
    graph_metrics,
    run_switch_chain,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--null-chains", type=int, default=4)
    parser.add_argument("--null-steps", type=int, default=10000)
    parser.add_argument("--null-thin", type=int, default=500)
    parser.add_argument("--max-features", type=int, default=64)
    parser.add_argument("--min-support", type=float, default=2.0)
    parser.add_argument(
        "--feature-prefix",
        action="append",
        default=[],
        help="Keep only automatically discovered keys with one of these prefixes.",
    )
    parser.add_argument(
        "--reserve-prefix",
        action="append",
        default=[],
        metavar="PREFIX=COUNT",
        help=(
            "Reserve slots for a feature prefix before global ranking. "
            "COUNT=-1 keeps every supported feature with that prefix."
        ),
    )
    parser.add_argument("--verify-every", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.null_chains < 2:
        raise ValueError("at least two null chains are required")
    source = load_typed_graph(args.dataset_dir, active_only=True)
    real_features = GraphletFeatureExtractor(source).extract(source)
    null_samples = []
    chain_reports = []
    for chain_id in range(args.null_chains):
        chain_seed = args.seed + 1009 * chain_id
        plan = sample_role_plan(source, seed=chain_seed)
        initial = realize_role_plan(source, plan)
        result = run_switch_chain(
            initial,
            GraphletFeatureExtractor(initial),
            steps=args.null_steps,
            burn_in=0,
            thin=args.null_thin,
            seed=chain_seed + 100_000,
            verify_every=args.verify_every,
        )
        null_samples.extend(sample.features for sample in result.samples)
        chain_reports.append(
            {
                "chain": chain_id,
                "samples": len(result.samples),
                "valid": result.valid,
                "accepted": result.accepted,
                "metrics": graph_metrics(result.graph),
            }
        )
    if len(null_samples) < 4:
        raise RuntimeError("null process produced too few feature samples")
    if args.feature_prefix:
        prefixes = tuple(str(value) for value in args.feature_prefix)
        real_features = {
            key: value for key, value in real_features.items() if key.startswith(prefixes)
        }
        null_samples = [
            {key: value for key, value in sample.items() if key.startswith(prefixes)}
            for sample in null_samples
        ]

    reserve_prefix_counts = {}
    for value in args.reserve_prefix:
        prefix, separator, count = str(value).rpartition("=")
        if not separator or not prefix:
            raise ValueError(
                f"invalid --reserve-prefix {value!r}; expected PREFIX=COUNT"
            )
        reserve_prefix_counts[prefix] = int(count)

    selection = select_features(
        real_features,
        null_samples,
        max_features=args.max_features,
        min_support=args.min_support,
        reserve_prefix_counts=reserve_prefix_counts,
    )
    # Switch diffusion consumes the selected moments and anonymous roles. The
    # failed linear-energy fit is deliberately not part of this checkpoint.
    empty_energy = LinearGraphletEnergy(
        selection,
        np.zeros(len(selection.keys), dtype=np.float64),
    )
    checkpoint = StructureEnergyCheckpoint.from_training_graph(source, empty_energy)
    checkpoint.save(args.output)
    report = {
        "dataset_dir": str(args.dataset_dir),
        "seed": args.seed,
        "source_metrics": graph_metrics(source),
        "null_chains": chain_reports,
        "null_samples": len(null_samples),
        "selected_features": list(selection.keys),
        "feature_prefixes": list(args.feature_prefix),
        "reserve_prefix_counts": reserve_prefix_counts,
    }
    report_path = args.output.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(
        "[SWITCH-STRUCTURE-CHECKPOINT] "
        f"nodes={source.num_nodes} edges={source.num_edges} "
        f"null_samples={len(null_samples)} features={len(selection.keys)} "
        f"output={args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
