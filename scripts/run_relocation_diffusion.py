#!/usr/bin/env python3
"""Train and sample edge-count-conditioned relocation diffusion."""

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
from sparse_diffusion.structure_energy.relocation_diffusion import (
    EdgeCountCondition,
    build_relocation_diffusion_dataset,
    load_relocation_checkpoint,
    sample_relocation_diffusion,
    save_relocation_checkpoint,
)
from sparse_diffusion.structure_energy.switch_chain import graph_metrics
from sparse_diffusion.structure_energy.switch_diffusion import train_switch_denoiser


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data/IMDB_movie_subgraphs/subgraph_000"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--trajectories", type=int, default=8)
    parser.add_argument("--forward-steps", type=int, default=2000)
    parser.add_argument("--record-every", type=int, default=20)
    parser.add_argument("--train-candidates", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--sample-steps", type=int, default=2000)
    parser.add_argument("--sample-candidates", type=int, default=64)
    parser.add_argument("--sample-temperature", type=float, default=0.0)
    parser.add_argument("--metrics-every", type=int, default=100)
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--test-only", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.test_only is None:
        source = load_typed_graph(args.dataset_dir, active_only=True)
        condition = EdgeCountCondition.from_training_graph(source)
        dataset = build_relocation_diffusion_dataset(
            source,
            trajectories=args.trajectories,
            forward_steps=args.forward_steps,
            record_every=args.record_every,
            candidates_per_example=args.train_candidates,
            seed=args.seed,
        )
        np.savez_compressed(
            args.output_dir / "relocation_dataset.npz",
            candidate_features=dataset.candidate_features,
            candidate_improvements=dataset.candidate_improvements,
            targets=dataset.targets,
            trajectory_ids=dataset.trajectory_ids,
        )
        print(
            "[RELOCATION-DIFFUSION-DATA] "
            f"examples={len(dataset.targets)} "
            f"candidates={dataset.candidate_features.shape[1]} "
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
            training_objective="inverse_ce",
            inverse_loss_weight=1.0,
            utility_loss_weight=0.0,
            device=args.device,
            seed=args.seed,
        )
        checkpoint_path = args.output_dir / "relocation_diffusion.pt"
        save_relocation_checkpoint(
            checkpoint_path,
            model,
            condition,
            hidden_dim=args.hidden_dim,
            layers=args.layers,
            dropout=args.dropout,
            report=report,
        )
        condition.save(args.output_dir / "edge_count_condition.json")
        (args.output_dir / "train_report.json").write_text(
            json.dumps(report.to_json_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(
            "[RELOCATION-DIFFUSION-TRAIN] "
            f"loss={report.validation_loss:.4f} "
            f"top1={report.validation_top1:.4f} "
            f"top5={report.validation_top5:.4f} "
            f"margin={report.validation_margin:.4f}",
            flush=True,
        )
        if args.train_only:
            return
        train_report = report.to_json_dict()
    else:
        model, condition, train_report = load_relocation_checkpoint(
            args.test_only, device=args.device
        )

    result = sample_relocation_diffusion(
        condition,
        model,
        steps=args.sample_steps,
        candidates_per_step=args.sample_candidates,
        seed=args.seed,
        device=args.device,
        temperature=args.sample_temperature,
        metrics_every=args.metrics_every,
        verbose=True,
    )
    initial = dict(result.trace[0])
    final = graph_metrics(result.graph)
    inference_contract = {
        "node_type_counts": True,
        "family_schema": True,
        "family_edge_counts": True,
        "relation_degree_rows": False,
        "graphlet_target": False,
        "training_adjacency": False,
    }
    payload = {
        "format": "dihug-relocation-sampling-v1",
        "seed": int(args.seed),
        "initial_metrics": initial,
        "final_metrics": final,
        "trace": list(result.trace),
        "proposed": int(result.proposed),
        "applied": int(result.applied),
        "sampling_config": {
            "steps": int(args.sample_steps),
            "candidates_per_step": int(args.sample_candidates),
            "temperature": float(args.sample_temperature),
        },
        "inference_contract": inference_contract,
        "train_report": train_report,
    }
    (args.output_dir / "sampling_report.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    (args.output_dir / "generated_relocation_graph.json").write_text(
        json.dumps(result.graph.to_json_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(
        "[RELOCATION-DIFFUSION-SAMPLE] "
        f"tri={initial['triangles']:.0f}->{final['triangles']:.0f} "
        f"clustering={initial['clustering']:.6f}->{final['clustering']:.6f} "
        f"degree_sum_sq={initial['sum_degree_sq']:.0f}->{final['sum_degree_sq']:.0f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
