#!/usr/bin/env python3
"""Run role, null, convex-energy, and learned-chain feasibility gates."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
repo_root_str = str(REPO_ROOT)
sys.path = [entry for entry in sys.path if entry != repo_root_str]
sys.path.insert(0, repo_root_str)

from sparse_diffusion.structure_energy.pipeline import GateConfig, run_gates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data/IMDB_movie_subgraphs/subgraph_000"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--null-chains", type=int, default=4)
    parser.add_argument("--null-steps", type=int, default=50_000)
    parser.add_argument("--null-burn-in", type=int, default=10_000)
    parser.add_argument("--energy-chains", type=int, default=4)
    parser.add_argument("--energy-steps", type=int, default=100_000)
    parser.add_argument("--energy-burn-in", type=int, default=20_000)
    parser.add_argument("--thin", type=int, default=500)
    parser.add_argument("--verify-every", type=int, default=0)
    parser.add_argument("--max-features", type=int, default=64)
    parser.add_argument("--min-feature-support", type=float, default=2.0)
    parser.add_argument("--nce-l2", type=float, default=0.1)
    parser.add_argument("--nce-max-weight", type=float, default=8.0)
    parser.add_argument("--moment-rounds", type=int, default=24)
    parser.add_argument("--moment-steps-per-round", type=int, default=5_000)
    parser.add_argument("--moment-samples-per-round", type=int, default=4)
    parser.add_argument("--moment-learning-rate", type=float, default=0.2)
    parser.add_argument("--moment-l2", type=float, default=0.01)
    parser.add_argument("--moment-tolerance", type=float, default=0.10)
    parser.add_argument("--structure-tolerance", type=float, default=0.10)
    parser.add_argument("--rhat-threshold", type=float, default=1.10)
    parser.add_argument("--min-ess-per-chain", type=float, default=20.0)
    parser.add_argument("--curveball-probability", type=float, default=0.0)
    parser.add_argument("--chain-drift-z-threshold", type=float, default=2.0)
    parser.add_argument("--chain-max-extensions", type=int, default=4)
    parser.add_argument("--chain-extension-steps", type=int, default=100_000)
    parser.add_argument("--diagnostic-window", type=int, default=64)
    parser.add_argument("--moment-max-batches-per-round", type=int, default=16)
    parser.add_argument("--moment-min-samples-per-chain", type=int, default=16)
    parser.add_argument("--moment-min-ess-per-chain", type=float, default=8.0)
    parser.add_argument("--moment-rhat-threshold", type=float, default=1.10)
    parser.add_argument("--moment-drift-z-threshold", type=float, default=2.0)
    parser.add_argument("--moment-warmup-steps-per-round", type=int, default=0)
    parser.add_argument("--moment-step-decay", type=float, default=0.60)
    parser.add_argument("--moment-update-norm-clip", type=float, default=0.5)
    parser.add_argument("--moment-diagnostic-quantile", type=float, default=0.90)
    parser.add_argument(
        "--moment-max-retained-samples-per-chain", type=int, default=256
    )
    parser.add_argument("--moment-min-successful-updates", type=int, default=1)
    parser.add_argument("--moment-pilot-only", action="store_true")
    parser.add_argument(
        "--moment-update-method",
        choices=("diagonal", "natural"),
        default="diagonal",
    )
    parser.add_argument(
        "--moment-natural-gradient-damping", type=float, default=0.05
    )
    parser.add_argument("--moment-covariance-shrinkage", type=float, default=0.10)
    parser.add_argument("--moment-kl-trust-radius", type=float, default=0.0)
    parser.add_argument("--moment-continuation-stages", type=int, default=0)
    parser.add_argument(
        "--moment-continuation-start-fraction", type=float, default=0.05
    )
    parser.add_argument(
        "--moment-continuation-stage-tolerance", type=float, default=0.25
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_gates(
        GateConfig(
            dataset_dir=args.dataset_dir,
            output_dir=args.output_dir,
            seed=args.seed,
            null_chains=args.null_chains,
            null_steps=args.null_steps,
            null_burn_in=args.null_burn_in,
            energy_chains=args.energy_chains,
            energy_steps=args.energy_steps,
            energy_burn_in=args.energy_burn_in,
            thin=args.thin,
            verify_every=args.verify_every,
            max_features=args.max_features,
            min_feature_support=args.min_feature_support,
            nce_l2=args.nce_l2,
            nce_max_weight=args.nce_max_weight,
            moment_rounds=args.moment_rounds,
            moment_steps_per_round=args.moment_steps_per_round,
            moment_samples_per_round=args.moment_samples_per_round,
            moment_learning_rate=args.moment_learning_rate,
            moment_l2=args.moment_l2,
            moment_tolerance=args.moment_tolerance,
            structure_tolerance=args.structure_tolerance,
            rhat_threshold=args.rhat_threshold,
            min_ess_per_chain=args.min_ess_per_chain,
            curveball_probability=args.curveball_probability,
            chain_drift_z_threshold=args.chain_drift_z_threshold,
            chain_max_extensions=args.chain_max_extensions,
            chain_extension_steps=args.chain_extension_steps,
            diagnostic_window=args.diagnostic_window,
            moment_max_batches_per_round=args.moment_max_batches_per_round,
            moment_min_samples_per_chain=args.moment_min_samples_per_chain,
            moment_min_ess_per_chain=args.moment_min_ess_per_chain,
            moment_rhat_threshold=args.moment_rhat_threshold,
            moment_drift_z_threshold=args.moment_drift_z_threshold,
            moment_warmup_steps_per_round=args.moment_warmup_steps_per_round,
            moment_step_decay=args.moment_step_decay,
            moment_update_norm_clip=args.moment_update_norm_clip,
            moment_diagnostic_quantile=args.moment_diagnostic_quantile,
            moment_max_retained_samples_per_chain=(
                args.moment_max_retained_samples_per_chain
            ),
            moment_min_successful_updates=args.moment_min_successful_updates,
            moment_pilot_only=args.moment_pilot_only,
            moment_update_method=args.moment_update_method,
            moment_natural_gradient_damping=(
                args.moment_natural_gradient_damping
            ),
            moment_covariance_shrinkage=args.moment_covariance_shrinkage,
            moment_kl_trust_radius=args.moment_kl_trust_radius,
            moment_continuation_stages=args.moment_continuation_stages,
            moment_continuation_start_fraction=(
                args.moment_continuation_start_fraction
            ),
            moment_continuation_stage_tolerance=(
                args.moment_continuation_stage_tolerance
            ),
        )
    )


if __name__ == "__main__":
    main()
