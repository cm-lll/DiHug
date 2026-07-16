#!/usr/bin/env python3
"""Run fair mixed edit-switch diffusion gates, training, and sampling."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from sparse_diffusion.structure_energy.graph import load_typed_graph
from sparse_diffusion.structure_energy.exchangeable_role_prior import (
    ExchangeableJointRolePrior,
)
from sparse_diffusion.structure_energy.factorized_edit_denoiser import (
    build_factorized_edit_dataset,
    evaluate_factorized_fixed_t,
    load_factorized_edit_checkpoint,
    sample_factorized_edit_diffusion,
    save_factorized_edit_checkpoint,
    train_factorized_edit_denoiser,
)
from sparse_diffusion.structure_energy.mixed_edit_diffusion import (
    EDIT_ACTIONS,
    MIXED_EDIT_FEATURE_PROFILES,
    MarginalGraphPrior,
    build_mixed_edit_diffusion_dataset,
    edge_edit_distance,
    evaluate_fixed_t_denoising,
    load_mixed_edit_checkpoint,
    mixed_edit_feature_width,
    run_edit_distance_oracle,
    sample_mixed_edit_diffusion,
    save_mixed_edit_checkpoint,
)
from sparse_diffusion.structure_energy.switch_chain import graph_metrics
from sparse_diffusion.structure_energy.switch_diffusion import train_switch_denoiser


KERNELS = {
    "bd": ("birth", "death"),
    "bdr": ("birth", "death", "relocation"),
    "bdrs": EDIT_ACTIONS,
    "switch": ("switch",),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data/IMDB_movie_subgraphs/subgraph_000"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("oracle", "train", "sample"), required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--denoiser", choices=("factorized", "listwise"), default="factorized"
    )
    parser.add_argument("--kernel", choices=tuple(KERNELS), default="bdrs")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--trajectories", type=int, default=16)
    parser.add_argument("--forward-steps", type=int, default=1000)
    parser.add_argument("--record-every", type=int, default=10)
    parser.add_argument("--train-candidates", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--sample-steps", type=int, default=20000)
    parser.add_argument("--sample-candidates", type=int, default=64)
    parser.add_argument("--sample-temperature", type=float, default=0.0)
    parser.add_argument(
        "--factorized-decode",
        choices=("hierarchical", "joint_leaf"),
        default="hierarchical",
    )
    parser.add_argument("--metrics-every", type=int, default=1000)
    parser.add_argument("--fixed-t-trials", type=int, default=64)
    parser.add_argument(
        "--dataset-cache",
        type=Path,
        help="Optional shared NPZ cache for listwise candidate tensors.",
    )
    parser.add_argument("--balanced-action-aux-weight", type=float, default=0.1)
    parser.add_argument(
        "--feature-profile",
        choices=MIXED_EDIT_FEATURE_PROFILES,
        default="typed_closure_full_v2",
    )
    parser.add_argument(
        "--training-objective",
        choices=("inverse_ce", "utility_kl", "hybrid"),
        default="inverse_ce",
    )
    parser.add_argument("--inverse-loss-weight", type=float, default=1.0)
    parser.add_argument("--utility-loss-weight", type=float, default=1.0)
    parser.add_argument("--utility-temperature", type=float, default=1.0)
    parser.add_argument(
        "--role-corruption-fraction",
        type=float,
        default=0.0,
        help="Fraction of training trajectories bridged to weak anonymous role priors.",
    )
    parser.add_argument(
        "--role-correlation-strengths",
        default="0,0.25,0.5,0.75,1",
        help="Comma-separated rho schedule for role-corruption trajectories.",
    )
    parser.add_argument("--role-prior-components", type=int, default=64)
    parser.add_argument("--role-prior-randomization-steps-per-edge", type=float, default=10.0)
    return parser.parse_args()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "oracle":
        source = load_typed_graph(args.dataset_dir, active_only=True)
        clean_metrics = graph_metrics(source)
        prior = MarginalGraphPrior.from_training_graph(source)
        result = run_edit_distance_oracle(
            prior,
            source,
            steps=args.sample_steps,
            candidates_per_step=args.sample_candidates,
            seed=args.seed,
            actions=KERNELS[args.kernel],
            metrics_every=args.metrics_every,
        )
        payload = {
            "format": "dihug-mixed-edit-oracle-gate-v1",
            "diagnostic_only": True,
            "oracle_objective": "clean_edge_edit_distance",
            "kernel": args.kernel,
            "seed": args.seed,
            "prior": prior.to_json_dict(),
            "clean_metrics": clean_metrics,
            "initial": result.trace[0],
            "final": {
                "edge_edit_distance": edge_edit_distance(result.graph, source),
                **graph_metrics(result.graph),
            },
            "trace": list(result.trace),
            "action_counts": dict(result.action_counts),
            "proposed": result.proposed,
            "applied": result.applied,
        }
        write_json(args.output_dir / "oracle_report.json", payload)
        print(
            "[MIXED-EDIT-ORACLE] "
            f"kernel={args.kernel} "
            f"distance={result.trace[0]['edge_edit_distance']:.0f}->"
            f"{payload['final']['edge_edit_distance']:.0f} "
            f"tri={result.trace[0]['triangles']:.0f}->"
            f"{payload['final']['triangles']:.0f}",
            flush=True,
        )
        return

    if args.mode == "train":
        source = load_typed_graph(args.dataset_dir, active_only=True)
        prior = MarginalGraphPrior.from_training_graph(source)
        if not 0.0 <= float(args.role_corruption_fraction) <= 1.0:
            raise ValueError("--role-corruption-fraction must be in [0, 1]")
        if args.role_corruption_fraction > 0.0 and args.denoiser != "listwise":
            raise ValueError("role corruption currently supports only listwise denoisers")
        if args.role_corruption_fraction > 0.0 and args.kernel != "bdrs":
            raise ValueError("role corruption requires the reversible bdrs edit kernel")
        if args.role_corruption_fraction > 0.0 and args.dataset_cache is not None:
            raise ValueError("role corruption cannot reuse an unlabelled dataset cache")

        role_corruption_targets = {}
        role_corruption_metadata = {
            "enabled": False,
            "fraction": 0.0,
            "strengths": [],
            "trajectory_ids": [],
        }
        if args.role_corruption_fraction > 0.0:
            strengths = tuple(
                float(value.strip())
                for value in args.role_correlation_strengths.split(",")
                if value.strip()
            )
            if not strengths or any(not 0.0 <= value <= 1.0 for value in strengths):
                raise ValueError("role correlation strengths must all be in [0, 1]")
            role_count = min(
                int(args.trajectories),
                max(1, int(round(args.trajectories * args.role_corruption_fraction))),
            )
            role_ids = np.rint(
                np.linspace(0, int(args.trajectories) - 1, role_count)
            ).astype(np.int64)
            if len(set(role_ids.tolist())) != role_count:
                raise AssertionError("role corruption trajectory allocation is not unique")
            base_role_prior = ExchangeableJointRolePrior.from_training_graph(
                source,
                components=args.role_prior_components,
                seed=args.seed,
                randomization_steps_per_edge=(
                    args.role_prior_randomization_steps_per_edge
                ),
                role_model="categorical",
                role_component_sampling="systematic",
            )
            assigned_strengths = []
            for index, trajectory_id in enumerate(role_ids.tolist()):
                strength = strengths[index % len(strengths)]
                assigned_strengths.append(strength)
                role_corruption_targets[int(trajectory_id)] = replace(
                    base_role_prior,
                    role_correlation_strength=float(strength),
                ).sample(args.seed + 10_000_019 + int(trajectory_id))
            role_corruption_metadata = {
                "enabled": True,
                "fraction": float(args.role_corruption_fraction),
                "strength_schedule": list(strengths),
                "strengths": assigned_strengths,
                "trajectory_ids": role_ids.tolist(),
                "components": int(args.role_prior_components),
                "randomization_steps_per_edge": float(
                    args.role_prior_randomization_steps_per_edge
                ),
                "target_contains_training_adjacency": False,
                "corruption_bridge": "exact_birth_death_relocation",
            }
        if args.dataset_cache is not None and args.denoiser != "listwise":
            raise ValueError("--dataset-cache currently supports only listwise denoisers")
        if args.dataset_cache is not None and args.dataset_cache.exists():
            cached = np.load(args.dataset_cache)
            from sparse_diffusion.structure_energy.switch_diffusion import (
                SwitchDiffusionDataset,
            )

            dataset = SwitchDiffusionDataset(
                candidate_features=cached["candidate_features"],
                candidate_improvements=cached["candidate_improvements"],
                targets=cached["targets"],
                trajectory_ids=cached["trajectory_ids"],
            )
            expected_width = mixed_edit_feature_width(
                args.feature_profile,
                family_count=len(source.family_specs),
                type_count=len(source.type_names),
            )
            if int(dataset.candidate_features.shape[-1]) != expected_width:
                raise ValueError(
                    f"cached feature width {dataset.candidate_features.shape[-1]} "
                    f"does not match {args.feature_profile} width {expected_width}"
                )
            if "feature_profile" in cached.files:
                cached_profile = str(cached["feature_profile"].item())
                if cached_profile != args.feature_profile:
                    raise ValueError(
                        f"cached feature profile {cached_profile!r} does not match "
                        f"requested profile {args.feature_profile!r}"
                    )
            if "kernel" in cached.files:
                cached_kernel = str(cached["kernel"].item())
                if cached_kernel != args.kernel:
                    raise ValueError(
                        f"cached kernel {cached_kernel!r} does not match requested "
                        f"kernel {args.kernel!r}"
                    )
        elif args.denoiser == "factorized":
            dataset = build_factorized_edit_dataset(
                source,
                trajectories=args.trajectories,
                forward_steps=args.forward_steps,
                record_every=args.record_every,
                candidates_per_example=args.train_candidates,
                seed=args.seed,
                actions=KERNELS[args.kernel],
            )
        else:
            dataset = build_mixed_edit_diffusion_dataset(
                source,
                trajectories=args.trajectories,
                forward_steps=args.forward_steps,
                record_every=args.record_every,
                candidates_per_example=args.train_candidates,
                seed=args.seed,
                actions=KERNELS[args.kernel],
                feature_profile=args.feature_profile,
                role_corruption_targets=role_corruption_targets,
            )
        dataset_path = args.output_dir / "mixed_edit_dataset.npz"
        np.savez_compressed(
            dataset_path,
            candidate_features=dataset.candidate_features,
            candidate_improvements=dataset.candidate_improvements,
            targets=dataset.targets,
            trajectory_ids=dataset.trajectory_ids,
            feature_profile=np.asarray(args.feature_profile),
            kernel=np.asarray(args.kernel),
            role_corruption=np.asarray(json.dumps(role_corruption_metadata)),
        )
        if args.dataset_cache is not None and not args.dataset_cache.exists():
            args.dataset_cache.parent.mkdir(parents=True, exist_ok=True)
            args.dataset_cache.write_bytes(dataset_path.read_bytes())
        if args.denoiser == "factorized":
            model, train_report = train_factorized_edit_denoiser(
                dataset,
                type_count=len(source.type_names),
                family_count=len(source.family_specs),
                hidden_dim=args.hidden_dim,
                layers=args.layers,
                dropout=args.dropout,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                balanced_action_aux_weight=args.balanced_action_aux_weight,
                device=args.device,
                seed=args.seed,
            )
            checkpoint = args.output_dir / "factorized_edit_diffusion.pt"
            save_factorized_edit_checkpoint(
                checkpoint,
                model,
                prior,
                hidden_dim=args.hidden_dim,
                layers=args.layers,
                dropout=args.dropout,
                actions=KERNELS[args.kernel],
                report=train_report,
            )
            fixed_t = evaluate_factorized_fixed_t(
                source,
                model,
                time_fractions=(0.1, 0.5, 0.9),
                trials_per_time=args.fixed_t_trials,
                forward_steps=args.forward_steps,
                candidates_per_example=args.train_candidates,
                seed=args.seed + 991,
                actions=KERNELS[args.kernel],
                device=args.device,
            )
            serialized_train_report = dict(train_report)
        else:
            model, train_report = train_switch_denoiser(
                dataset,
                hidden_dim=args.hidden_dim,
                layers=args.layers,
                dropout=args.dropout,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                training_objective=args.training_objective,
                inverse_loss_weight=args.inverse_loss_weight,
                utility_loss_weight=args.utility_loss_weight,
                utility_temperature=args.utility_temperature,
                device=args.device,
                seed=args.seed,
            )
            checkpoint = args.output_dir / "mixed_edit_diffusion.pt"
            save_mixed_edit_checkpoint(
                checkpoint,
                model,
                prior,
                hidden_dim=args.hidden_dim,
                layers=args.layers,
                dropout=args.dropout,
                report=train_report,
                actions=KERNELS[args.kernel],
                feature_profile=args.feature_profile,
                training_noise=role_corruption_metadata,
            )
            fixed_t = evaluate_fixed_t_denoising(
                source,
                model,
                time_fractions=(0.1, 0.5, 0.9),
                trials_per_time=args.fixed_t_trials,
                forward_steps=args.forward_steps,
                candidates_per_example=args.train_candidates,
                seed=args.seed + 991,
                device=args.device,
                actions=KERNELS[args.kernel],
            )
            serialized_train_report = train_report.to_json_dict()
        write_json(
            args.output_dir / "train_report.json",
            {
                "format": "dihug-mixed-edit-train-v1",
                "denoiser": args.denoiser,
                "training_objective": args.training_objective,
                "feature_profile": args.feature_profile,
                "train": serialized_train_report,
                "fixed_t": fixed_t,
                "prior": prior.to_json_dict(),
                "training_noise": role_corruption_metadata,
            },
        )
        print(
            "[MIXED-EDIT-TRAIN] "
            f"denoiser={args.denoiser} "
            f"examples={len(dataset.targets)} "
            f"top1={serialized_train_report.get('validation_candidate_top1', serialized_train_report.get('validation_top1', float('nan'))):.4f} "
            f"fixed_t={json.dumps(fixed_t, sort_keys=True)}",
            flush=True,
        )
        return

    if args.checkpoint is None:
        raise ValueError("--checkpoint is required in sample mode")
    if args.denoiser == "factorized":
        model, prior, checkpoint_metadata = load_factorized_edit_checkpoint(
            args.checkpoint, device=args.device
        )
    else:
        model, prior, checkpoint_metadata = load_mixed_edit_checkpoint(
            args.checkpoint, device=args.device
        )
    checkpoint_actions = tuple(checkpoint_metadata["actions"])
    if checkpoint_actions != tuple(KERNELS[args.kernel]):
        raise ValueError(
            f"checkpoint kernel {checkpoint_actions} does not match --kernel "
            f"{KERNELS[args.kernel]}"
        )
    sampler = (
        sample_factorized_edit_diffusion
        if args.denoiser == "factorized"
        else sample_mixed_edit_diffusion
    )
    sample_kwargs = {
        "steps": args.sample_steps,
        "candidates_per_step": args.sample_candidates,
        "seed": args.seed,
        "device": args.device,
        "temperature": args.sample_temperature,
        "metrics_every": args.metrics_every,
        "actions": KERNELS[args.kernel],
    }
    if args.denoiser == "factorized":
        sample_kwargs["decode_mode"] = args.factorized_decode
    result = sampler(prior, model, **sample_kwargs)
    payload = {
        "format": "dihug-mixed-edit-sampling-v1",
        "denoiser": args.denoiser,
        "seed": args.seed,
        "initial": result.trace[0],
        "final": graph_metrics(result.graph),
        "trace": list(result.trace),
        "action_counts": dict(result.action_counts),
        "proposed": result.proposed,
        "applied": result.applied,
        "prior": prior.to_json_dict(),
        "train_report": checkpoint_metadata["train_report"],
        "sampling_config": {
            "kernel": args.kernel,
            "steps": args.sample_steps,
            "candidates": args.sample_candidates,
            "temperature": args.sample_temperature,
            "factorized_decode": (
                args.factorized_decode if args.denoiser == "factorized" else None
            ),
        },
    }
    write_json(args.output_dir / "sampling_report.json", payload)
    write_json(args.output_dir / "generated_graph.json", result.graph.to_json_dict())
    print(
        "[MIXED-EDIT-SAMPLE] "
        f"tri={payload['initial']['triangles']:.0f}->{payload['final']['triangles']:.0f} "
        f"clustering={payload['initial']['clustering']:.6f}->"
        f"{payload['final']['clustering']:.6f} "
        f"edges={payload['initial']['edges']:.0f}->{payload['final']['edges']:.0f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
