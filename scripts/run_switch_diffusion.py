#!/usr/bin/env python3
"""Train and sample the degree-preserving switch-space diffusion gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from sparse_diffusion.structure_energy.checkpoint import StructureEnergyCheckpoint
from sparse_diffusion.structure_energy.graph import load_typed_graph
from sparse_diffusion.structure_energy.switch_chain import graph_metrics
from sparse_diffusion.structure_energy.switch_diffusion import (
    build_switch_diffusion_dataset,
    load_switch_checkpoint,
    randomize_role_realization,
    sample_switch_diffusion,
    save_sampling_result,
    save_switch_checkpoint,
    train_switch_denoiser,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data/IMDB_movie_subgraphs/subgraph_000"),
    )
    parser.add_argument("--structure-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--trajectories", type=int, default=8)
    parser.add_argument("--forward-steps", type=int, default=2000)
    parser.add_argument("--record-every", type=int, default=20)
    parser.add_argument("--train-candidates", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--sample-steps", type=int, default=2000)
    parser.add_argument("--sample-candidates", type=int, default=64)
    parser.add_argument("--sample-temperature", type=float, default=0.1)
    parser.add_argument(
        "--sampling-mode",
        choices=("model", "random", "oracle"),
        default="model",
    )
    parser.add_argument("--metrics-every", type=int, default=100)
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--test-only", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.test_only is None:
        source = load_typed_graph(args.dataset_dir, active_only=True)
        structure = StructureEnergyCheckpoint.load(args.structure_checkpoint)
        dataset, target_vector = build_switch_diffusion_dataset(
            source,
            structure.selection,
            trajectories=args.trajectories,
            forward_steps=args.forward_steps,
            record_every=args.record_every,
            candidates_per_example=args.train_candidates,
            seed=args.seed,
        )
        np.savez_compressed(
            args.output_dir / "switch_dataset.npz",
            candidate_features=dataset.candidate_features,
            candidate_improvements=dataset.candidate_improvements,
            targets=dataset.targets,
            trajectory_ids=dataset.trajectory_ids,
        )
        print(
            "[SWITCH-DIFFUSION-DATA] "
            f"examples={len(dataset.targets)} candidates={dataset.candidate_features.shape[1]} "
            f"features={dataset.candidate_features.shape[2]}",
            flush=True,
        )
        model, report = train_switch_denoiser(
            dataset,
            hidden_dim=args.hidden_dim,
            layers=args.layers,
            dropout=args.dropout,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            device=args.device,
            seed=args.seed,
        )
        checkpoint_path = args.output_dir / "switch_diffusion.pt"
        save_switch_checkpoint(
            checkpoint_path,
            model,
            structure,
            target_vector,
            hidden_dim=args.hidden_dim,
            layers=args.layers,
            dropout=args.dropout,
            report=report,
        )
        (args.output_dir / "train_report.json").write_text(
            json.dumps(report.to_json_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(
            "[SWITCH-DIFFUSION-TRAIN] "
            f"loss={report.validation_loss:.4f} top1={report.validation_top1:.4f} "
            f"top5={report.validation_top5:.4f} margin={report.validation_margin:.4f}",
            f"inverse_oracle_agreement={report.validation_oracle_top1:.4f}",
            flush=True,
        )
        if args.train_only:
            return
    else:
        checkpoint_path = args.test_only

    model, structure, target_vector, train_report = load_switch_checkpoint(
        checkpoint_path, args.device
    )
    initial = structure.realize(args.seed + 100_000)
    initial = randomize_role_realization(
        initial, steps=args.sample_steps, seed=args.seed + 200_000
    )
    initial_metrics = graph_metrics(initial)
    result = sample_switch_diffusion(
        initial,
        model,
        structure.selection,
        target_vector,
        steps=args.sample_steps,
        candidates_per_step=args.sample_candidates,
        seed=args.seed + 300_000,
        device=args.device,
        temperature=args.sample_temperature,
        metrics_every=args.metrics_every,
        selection_mode=args.sampling_mode,
        verbose=True,
    )
    save_sampling_result(args.output_dir / "generated_switch_graph.json", result)
    final_metrics = graph_metrics(result.graph)
    summary = {
        "initial_metrics": initial_metrics,
        "final_metrics": final_metrics,
        "train_report": train_report,
        "trace": list(result.trace),
        "sampling_mode": result.selection_mode,
        "oracle_agreement": result.oracle_agreement,
    }
    (args.output_dir / "sampling_report.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        "[SWITCH-DIFFUSION-SAMPLE] "
        f"tri={initial_metrics['triangles']:.0f}->{final_metrics['triangles']:.0f} "
        f"clustering={initial_metrics['clustering']:.4f}->{final_metrics['clustering']:.4f} "
        f"lcc={initial_metrics['lcc']:.0f}->{final_metrics['lcc']:.0f}",
        f"mode={result.selection_mode} oracle_agreement={result.oracle_agreement:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
