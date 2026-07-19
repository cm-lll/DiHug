#!/usr/bin/env python3
"""Train and sample strictly paired explicit-MLP and HSD-context scorers."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from sparse_diffusion.structure_energy.aligned_context_edit import (
    save_aligned_context_checkpoint,
    train_aligned_context_edit_denoiser,
)
from sparse_diffusion.structure_energy.graph import load_typed_graph
from sparse_diffusion.structure_energy.hgt_edit_denoiser import build_hgt_edit_dataset
from sparse_diffusion.structure_energy.mixed_edit_diffusion import (
    EDIT_ACTIONS,
    MarginalGraphPrior,
    load_mixed_edit_checkpoint,
    save_mixed_edit_checkpoint,
)
from sparse_diffusion.structure_energy.role_guided_edit_diffusion import (
    ExchangeableRoleDegreePrior,
)
from sparse_diffusion.structure_energy.switch_diffusion import train_switch_denoiser


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("prepare", "train-context"), required=True)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data/IMDB_movie_subgraphs/subgraph_000"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-cache", type=Path)
    parser.add_argument("--base-checkpoint", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--trajectories", type=int, default=32)
    parser.add_argument("--forward-steps", type=int, default=1000)
    parser.add_argument("--record-every", type=int, default=10)
    parser.add_argument("--candidates", type=int, default=16)
    parser.add_argument("--feature-profile", default="typed_closure_compact_v2")
    parser.add_argument(
        "--supervision",
        choices=("exact_inverse", "free_x0", "rooted_utility"),
        default="exact_inverse",
    )
    parser.add_argument(
        "--forward-mode", choices=("stepwise", "closed_form"), default="stepwise"
    )
    parser.add_argument(
        "--candidate-mode", choices=("random", "unified"), default="random"
    )
    parser.add_argument("--targeted-role-candidate-fraction", type=float, default=0.5)
    parser.add_argument("--closure-candidate-fraction", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--base-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--legacy-learning-rate-scale", type=float, default=1.0)
    parser.add_argument("--legacy-init", choices=("random", "pretrained"), default="random")
    parser.add_argument("--legacy-unfreeze-epoch", type=int, default=0)
    parser.add_argument(
        "--training-objective",
        choices=("inverse_ce", "utility_kl", "utility_role_kl"),
        default="inverse_ce",
    )
    parser.add_argument("--utility-temperature", type=float, default=1.0)
    parser.add_argument("--trust-weight", type=float, default=0.0)
    parser.add_argument("--role-weight", type=float, default=0.05)
    parser.add_argument("--role-gain-scale", type=float, default=10.0)
    parser.add_argument("--role-schedule-power", type=float, default=1.0)
    parser.add_argument("--role-schedule-floor", type=float, default=1.0)
    parser.add_argument("--role-teacher-weight", type=float, default=0.5)
    parser.add_argument("--residual-min-utility-gain", type=float, default=0.05)
    parser.add_argument("--context-mode", choices=("none", "current", "counterfactual", "both"), default="both")
    parser.add_argument(
        "--explicit-feature-mode",
        choices=("full", "no_degree", "no_common", "no_closure", "no_time", "minimal"),
        default="full",
    )
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--role-conditioned-hgt", action="store_true")
    parser.add_argument("--query-edges-per-family", type=int, default=0)
    parser.add_argument("--query-context-gate-init", type=float, default=0.2)
    return parser.parse_args()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = load_typed_graph(args.dataset_dir, active_only=True)
    prior = MarginalGraphPrior.from_training_graph(source)
    role_prior = ExchangeableRoleDegreePrior.from_training_graph(source)
    cache = args.dataset_cache or args.output_dir / "aligned_hgt_edit_dataset.pt"

    if args.mode == "prepare":
        dataset = build_hgt_edit_dataset(
            source,
            trajectories=args.trajectories,
            forward_steps=args.forward_steps,
            record_every=args.record_every,
            candidates_per_example=args.candidates,
            seed=args.seed,
            actions=EDIT_ACTIONS,
            supervision=args.supervision,
            feature_profile=args.feature_profile,
            role_conditioning=args.role_conditioned_hgt,
            query_edges_per_family=args.query_edges_per_family,
            forward_mode=args.forward_mode,
            candidate_mode=args.candidate_mode,
            targeted_role_candidate_fraction=args.targeted_role_candidate_fraction,
            closure_candidate_fraction=args.closure_candidate_fraction,
        )
        cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dataset, cache)
        model, report = train_switch_denoiser(
            dataset.as_listwise_dataset(),
            hidden_dim=256,
            layers=3,
            dropout=args.dropout,
            epochs=args.base_epochs,
            batch_size=64,
            learning_rate=1.0e-3,
            training_objective=(
                "utility_kl"
                if args.supervision in {"free_x0", "rooted_utility"}
                else "inverse_ce"
            ),
            inverse_loss_weight=1.0,
            utility_loss_weight=(
                1.0
                if args.supervision in {"free_x0", "rooted_utility"}
                else 0.0
            ),
            utility_temperature=args.utility_temperature,
            device=args.device,
            seed=args.seed,
        )
        checkpoint = args.output_dir / "paired_explicit_mlp.pt"
        save_mixed_edit_checkpoint(
            checkpoint,
            model,
            prior,
            hidden_dim=256,
            layers=3,
            dropout=args.dropout,
            report=report,
            actions=EDIT_ACTIONS,
            feature_profile=args.feature_profile,
            training_noise={
                "forward_mode": args.forward_mode,
                "candidate_mode": args.candidate_mode,
                "trajectories": args.trajectories,
                "forward_steps": args.forward_steps,
                "record_every": args.record_every,
                "supervision": args.supervision,
            },
        )
        payload = {
            "mode": "prepare",
            "dataset_cache": str(cache),
            "checkpoint": str(checkpoint),
            "examples": int(dataset.targets.shape[0]),
            "feature_profile": args.feature_profile,
            "supervision": args.supervision,
            "forward_mode": args.forward_mode,
            "candidate_mode": args.candidate_mode,
            "train_report": report.to_json_dict(),
        }
        write_json(args.output_dir / "prepare_report.json", payload)
        print(
            "[ALIGNED-PREPARE] "
            f"examples={payload['examples']} "
            f"top1={payload['train_report']['validation_top1']:.4f}",
            flush=True,
        )
        return

    if args.base_checkpoint is None or not args.base_checkpoint.exists():
        raise ValueError("train-context requires --base-checkpoint")
    if not cache.exists():
        raise ValueError(f"dataset cache does not exist: {cache}")
    dataset = torch.load(cache, map_location="cpu", weights_only=False)
    legacy, checkpoint_prior, metadata = load_mixed_edit_checkpoint(
        args.base_checkpoint, device="cpu"
    )
    if checkpoint_prior.to_json_dict() != prior.to_json_dict():
        raise ValueError("base checkpoint and dataset use different priors")
    model, report = train_aligned_context_edit_denoiser(
        dataset,
        legacy,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        legacy_learning_rate_scale=args.legacy_learning_rate_scale,
        legacy_init=args.legacy_init,
        legacy_unfreeze_epoch=args.legacy_unfreeze_epoch,
        training_objective=args.training_objective,
        utility_temperature=args.utility_temperature,
        trust_weight=args.trust_weight,
        role_weight=args.role_weight,
        role_gain_scale=args.role_gain_scale,
        role_schedule_power=args.role_schedule_power,
        role_schedule_floor=args.role_schedule_floor,
        role_teacher_weight=args.role_teacher_weight,
        residual_min_utility_gain=args.residual_min_utility_gain,
        device=args.device,
        seed=args.seed,
        hidden_dim=args.hidden_dim,
        heads=args.heads,
        layers=args.layers,
        dropout=args.dropout,
        context_mode=args.context_mode,
        explicit_feature_mode=args.explicit_feature_mode,
        role_conditioning=args.role_conditioned_hgt,
        query_edges_per_family=args.query_edges_per_family,
        query_context_gate_init=args.query_context_gate_init,
    )
    checkpoint = args.output_dir / "aligned_context_edit.pt"
    save_aligned_context_checkpoint(
        checkpoint,
        model,
        prior,
        role_prior,
        report,
        feature_profile=str(metadata["feature_profile"]),
        actions=metadata["actions"],
        training_noise=dict(metadata.get("training_noise", {})),
    )
    payload = {
        "mode": "train-context",
        "dataset_cache": str(cache),
        "base_checkpoint": str(args.base_checkpoint),
        "checkpoint": str(checkpoint),
        "context_mode": args.context_mode,
        "explicit_feature_mode": args.explicit_feature_mode,
        "legacy_init": args.legacy_init,
        "legacy_unfreeze_epoch": args.legacy_unfreeze_epoch,
        "training_objective": args.training_objective,
        "role_weight": args.role_weight,
        "role_gain_scale": args.role_gain_scale,
        "role_schedule_power": args.role_schedule_power,
        "role_schedule_floor": args.role_schedule_floor,
        "role_teacher_weight": args.role_teacher_weight,
        "residual_min_utility_gain": args.residual_min_utility_gain,
        "role_conditioning": args.role_conditioned_hgt,
        "query_edges_per_family": args.query_edges_per_family,
        "train_report": dict(report),
    }
    write_json(args.output_dir / "train_report.json", payload)
    print(
        "[ALIGNED-CONTEXT-DONE] "
        f"context={args.context_mode} explicit={args.explicit_feature_mode} "
        f"top1={report['validation_top1']:.4f} "
        f"base_top1={report['validation_base_top1']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
