#!/usr/bin/env python3
"""Paired HGT-vs-MLP listwise edit gate on identical states and candidates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from sparse_diffusion.structure_energy.graph import load_typed_graph
from sparse_diffusion.structure_energy.hgt_edit_denoiser import (
    build_hgt_edit_dataset,
    evaluate_hgt_fixed_t,
    load_hgt_edit_checkpoint,
    sample_hgt_listwise_edit_diffusion,
    save_hgt_edit_checkpoint,
    train_hgt_listwise_edit_denoiser,
    train_paired_mlp,
)
from sparse_diffusion.structure_energy.mixed_edit_diffusion import (
    EDIT_ACTIONS,
    MarginalGraphPrior,
    evaluate_fixed_t_denoising,
)
from sparse_diffusion.structure_energy.switch_chain import graph_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("paired-train", "sample"), required=True)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data/IMDB_movie_subgraphs/subgraph_000"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--trajectories", type=int, default=8)
    parser.add_argument("--forward-steps", type=int, default=1000)
    parser.add_argument("--record-every", type=int, default=20)
    parser.add_argument("--candidates", type=int, choices=(16, 64), default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--hgt-layers", type=int, default=2)
    parser.add_argument("--scorer-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--fixed-t-trials", type=int, default=64)
    parser.add_argument("--sample-steps", type=int, default=2000)
    parser.add_argument("--metrics-every", type=int, default=250)
    return parser.parse_args()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    actions = EDIT_ACTIONS
    if args.mode == "paired-train":
        source = load_typed_graph(args.dataset_dir, active_only=True)
        prior = MarginalGraphPrior.from_training_graph(source)
        dataset = build_hgt_edit_dataset(
            source,
            trajectories=args.trajectories,
            forward_steps=args.forward_steps,
            record_every=args.record_every,
            candidates_per_example=args.candidates,
            seed=args.seed,
            actions=actions,
        )
        hgt, hgt_report = train_hgt_listwise_edit_denoiser(
            dataset,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            device=args.device,
            seed=args.seed,
            hidden_dim=args.hidden_dim,
            heads=args.heads,
            hgt_layers=args.hgt_layers,
            scorer_layers=args.scorer_layers,
            dropout=args.dropout,
        )
        mlp, mlp_report = train_paired_mlp(
            dataset,
            hidden_dim=256,
            layers=3,
            dropout=args.dropout,
            epochs=args.epochs,
            batch_size=64,
            learning_rate=1.0e-3,
            training_objective="inverse_ce",
            inverse_loss_weight=1.0,
            utility_loss_weight=0.0,
            device=args.device,
            seed=args.seed,
        )
        hgt_fixed_t = evaluate_hgt_fixed_t(
            source,
            hgt,
            time_fractions=(0.1, 0.5, 0.9),
            trials_per_time=args.fixed_t_trials,
            forward_steps=args.forward_steps,
            candidates_per_example=args.candidates,
            seed=args.seed + 991,
            actions=actions,
            device=args.device,
        )
        mlp_fixed_t = evaluate_fixed_t_denoising(
            source,
            mlp,
            time_fractions=(0.1, 0.5, 0.9),
            trials_per_time=args.fixed_t_trials,
            forward_steps=args.forward_steps,
            candidates_per_example=args.candidates,
            seed=args.seed + 991,
            actions=actions,
            device=args.device,
        )
        report = {
            "format": "dihug-hgt-listwise-paired-gate-v1",
            "candidate_count": args.candidates,
            "examples": int(dataset.targets.shape[0]),
            "same_states_and_candidates": True,
            "supervision": "injected_exact_inverse_only",
            "hgt": {"train": dict(hgt_report), "fixed_t": hgt_fixed_t},
            "mlp": {"train": dict(mlp_report), "fixed_t": mlp_fixed_t},
            "inference_uses_clean_graph": False,
        }
        save_hgt_edit_checkpoint(
            args.output_dir / "hgt_listwise_edit.pt", hgt, prior, report, actions
        )
        write_json(args.output_dir / "paired_gate_report.json", report)
        print(
            "[HGT-LISTWISE-GATE] "
            f"candidates={args.candidates} "
            f"hgt_top1={hgt_report['top1']:.4f} mlp_top1={mlp_report['validation_top1']:.4f} "
            f"hgt_fixed_t={json.dumps(hgt_fixed_t, sort_keys=True)}",
            flush=True,
        )
        return

    if args.checkpoint is None:
        raise ValueError("--checkpoint is required in sample mode")
    model, prior, metadata = load_hgt_edit_checkpoint(args.checkpoint, args.device)
    result = sample_hgt_listwise_edit_diffusion(
        prior,
        model,
        steps=args.sample_steps,
        candidates_per_step=args.candidates,
        seed=args.seed,
        device=args.device,
        metrics_every=args.metrics_every,
        actions=tuple(metadata["actions"]),
    )
    payload = {
        "format": "dihug-hgt-listwise-sampling-v1",
        "seed": args.seed,
        "initial": result.trace[0],
        "final": graph_metrics(result.graph),
        "trace": list(result.trace),
        "action_counts": dict(result.action_counts),
        "sampling_config": {"steps": args.sample_steps, "candidates": args.candidates},
        "inference_uses_clean_graph": False,
    }
    write_json(args.output_dir / "sampling_report.json", payload)
    write_json(args.output_dir / "generated_graph.json", result.graph.to_json_dict())
    print(
        f"[HGT-LISTWISE-SAMPLE] tri={payload['initial']['triangles']:.0f}->"
        f"{payload['final']['triangles']:.0f} clustering="
        f"{payload['initial']['clustering']:.6f}->{payload['final']['clustering']:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
