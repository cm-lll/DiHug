#!/usr/bin/env python3
"""Run fair mixed edit-switch diffusion gates, training, and sampling."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from sparse_diffusion.structure_energy.graph import load_typed_graph
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
    MarginalGraphPrior,
    build_mixed_edit_diffusion_dataset,
    edge_edit_distance,
    evaluate_fixed_t_denoising,
    load_mixed_edit_checkpoint,
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
    parser.add_argument("--balanced-action-aux-weight", type=float, default=0.1)
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
        if args.denoiser == "factorized":
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
            )
        np.savez_compressed(
            args.output_dir / "mixed_edit_dataset.npz",
            candidate_features=dataset.candidate_features,
            candidate_improvements=dataset.candidate_improvements,
            targets=dataset.targets,
            trajectory_ids=dataset.trajectory_ids,
        )
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
                training_objective="inverse_ce",
                inverse_loss_weight=1.0,
                utility_loss_weight=0.0,
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
                "train": serialized_train_report,
                "fixed_t": fixed_t,
                "prior": prior.to_json_dict(),
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
