#!/usr/bin/env python3
"""Train and sample the degree-preserving switch-space diffusion gate."""

from __future__ import annotations

import argparse
import json
import sys
import time
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
    parser.add_argument(
        "--run-label",
        default="",
        help="Stable method label written to sampling_report.json for aggregation.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--trajectories", type=int, default=8)
    parser.add_argument("--forward-steps", type=int, default=2000)
    parser.add_argument(
        "--diffusion-steps",
        type=int,
        default=0,
        help="Number of discrete macro diffusion layers; 0 keeps legacy micro time.",
    )
    parser.add_argument(
        "--switches-per-diffusion-step",
        type=int,
        default=0,
        help="Legal micro switches composed inside each macro diffusion layer.",
    )
    parser.add_argument("--record-every", type=int, default=20)
    parser.add_argument("--train-candidates", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--training-objective",
        choices=("inverse_ce", "utility_kl", "hybrid", "scheduled_hybrid"),
        default="inverse_ce",
    )
    parser.add_argument("--inverse-loss-weight", type=float, default=0.2)
    parser.add_argument("--utility-loss-weight", type=float, default=1.0)
    parser.add_argument("--utility-temperature", type=float, default=1.0)
    parser.add_argument("--utility-schedule-power", type=float, default=1.0)
    parser.add_argument(
        "--learned-condition",
        action="store_true",
        help=(
            "Fit and store the anonymous graphlet condition as model parameters; "
            "requires state_only + utility_kl"
        ),
    )
    parser.add_argument("--learned-condition-ridge", type=float, default=1.0e-6)
    parser.add_argument(
        "--dataset-cache",
        type=Path,
        help="Reuse a previously generated switch_dataset.npz",
    )
    parser.add_argument(
        "--conditioning-mode",
        choices=("target_residual", "state_only"),
        default="target_residual",
        help=(
            "state_only trains a target-free reverse kernel and omits clean "
            "graph statistics from the saved denoiser checkpoint"
        ),
    )
    parser.add_argument("--sample-steps", type=int, default=2000)
    parser.add_argument(
        "--randomization-steps",
        type=int,
        default=None,
        help="Forward random-switch burn-in; defaults to --sample-steps",
    )
    parser.add_argument("--sample-candidates", type=int, default=64)
    parser.add_argument("--sample-temperature", type=float, default=0.1)
    parser.add_argument(
        "--sampling-mode",
        choices=("model", "random", "oracle"),
        default="model",
    )
    parser.add_argument("--metrics-every", type=int, default=100)
    parser.add_argument(
        "--switch-kernel",
        choices=("auto", "python", "tensor"),
        default="auto",
        help="Exact candidate-delta backend; auto uses tensor batches on CUDA",
    )
    parser.add_argument(
        "--learned-condition-noop",
        action="store_true",
        help="Reject switches with non-positive learned-condition utility",
    )
    parser.add_argument("--learned-condition-noop-patience", type=int, default=0)
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--build-dataset-only", action="store_true")
    parser.add_argument("--test-only", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (args.diffusion_steps > 0) != (args.switches_per_diffusion_step > 0):
        raise ValueError(
            "--diffusion-steps and --switches-per-diffusion-step must be set together"
        )
    macro_schedule = args.diffusion_steps > 0
    scheduled_micro_steps = (
        int(args.diffusion_steps) * int(args.switches_per_diffusion_step)
        if macro_schedule
        else 0
    )
    training_forward_steps = (
        scheduled_micro_steps if macro_schedule else int(args.forward_steps)
    )
    sampling_steps = scheduled_micro_steps if macro_schedule else int(args.sample_steps)
    if macro_schedule:
        print(
            "[SWITCH-DIFFUSION-SCHEDULE] "
            f"macro_steps={args.diffusion_steps} "
            f"switches_per_step={args.switches_per_diffusion_step} "
            f"micro_steps={scheduled_micro_steps}",
            flush=True,
        )
    if args.learned_condition and (
        args.conditioning_mode != "state_only"
        or args.training_objective != "utility_kl"
    ):
        raise ValueError(
            "--learned-condition requires --conditioning-mode state_only "
            "and --training-objective utility_kl"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.test_only is None:
        source = load_typed_graph(args.dataset_dir, active_only=True)
        structure = StructureEnergyCheckpoint.load(args.structure_checkpoint)
        if args.dataset_cache is None:
            dataset, target_vector = build_switch_diffusion_dataset(
                source,
                structure.selection,
                trajectories=args.trajectories,
                forward_steps=training_forward_steps,
                record_every=args.record_every,
                candidates_per_example=args.train_candidates,
                seed=args.seed,
                conditioning_mode=args.conditioning_mode,
                diffusion_steps=args.diffusion_steps,
                switches_per_diffusion_step=args.switches_per_diffusion_step,
            )
        else:
            from sparse_diffusion.structure_energy.switch_diffusion import (
                SwitchDiffusionDataset,
            )

            cached = np.load(args.dataset_cache)
            dataset = SwitchDiffusionDataset(
                candidate_features=cached["candidate_features"],
                candidate_improvements=cached["candidate_improvements"],
                targets=cached["targets"],
                trajectory_ids=cached["trajectory_ids"],
            )
            target_vector = (
                np.asarray(cached["target_vector"], dtype=np.float64)
                if "target_vector" in cached.files
                else None
            )
            if args.conditioning_mode == "target_residual" and target_vector is None:
                raise ValueError(
                    "target-conditioned training needs target_vector in dataset cache"
                )
        cache_payload = dict(
            candidate_features=dataset.candidate_features,
            candidate_improvements=dataset.candidate_improvements,
            targets=dataset.targets,
            trajectory_ids=dataset.trajectory_ids,
        )
        if (
            target_vector is not None
            and args.conditioning_mode == "target_residual"
        ):
            cache_payload["target_vector"] = target_vector
        np.savez_compressed(
            args.output_dir / "switch_dataset.npz", **cache_payload
        )
        print(
            "[SWITCH-DIFFUSION-DATA] "
            f"examples={len(dataset.targets)} candidates={dataset.candidate_features.shape[1]} "
            f"features={dataset.candidate_features.shape[2]}",
            flush=True,
        )
        if args.build_dataset_only:
            return
        model, report = train_switch_denoiser(
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
            utility_schedule_power=args.utility_schedule_power,
            learned_condition_feature_count=(
                len(structure.selection.keys) if args.learned_condition else 0
            ),
            learned_condition_ridge=args.learned_condition_ridge,
            device=args.device,
            seed=args.seed,
        )
        checkpoint_path = args.output_dir / "switch_diffusion.pt"
        save_switch_checkpoint(
            checkpoint_path,
            model,
            structure,
            (
                target_vector
                if args.conditioning_mode == "target_residual"
                else None
            ),
            hidden_dim=args.hidden_dim,
            layers=args.layers,
            dropout=args.dropout,
            report=report,
            conditioning_mode=args.conditioning_mode,
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
            f"model_oracle_agreement={report.validation_model_oracle_top1:.4f}",
            f"selected_utility={report.validation_selected_utility:.4f}",
            f"condition_rank={report.learned_condition_rank}",
            f"condition_rmse={report.learned_condition_rmse:.3e}",
            flush=True,
        )
        if args.train_only:
            return
    else:
        checkpoint_path = args.test_only

    (
        model,
        structure,
        target_vector,
        train_report,
        conditioning_mode,
    ) = load_switch_checkpoint(
        checkpoint_path, args.device
    )
    initial = structure.realize(args.seed + 100_000)
    randomization_start = time.perf_counter()
    initial = randomize_role_realization(
        initial,
        steps=(
            sampling_steps
            if args.randomization_steps is None
            else args.randomization_steps
        ),
        seed=args.seed + 200_000,
    )
    randomization_seconds = time.perf_counter() - randomization_start
    initial_metrics = graph_metrics(initial)
    sampling_start = time.perf_counter()
    result = sample_switch_diffusion(
        initial,
        model,
        structure.selection,
        target_vector,
        steps=sampling_steps,
        candidates_per_step=args.sample_candidates,
        seed=args.seed + 300_000,
        device=args.device,
        temperature=args.sample_temperature,
        metrics_every=args.metrics_every,
        selection_mode=args.sampling_mode,
        conditioning_mode=conditioning_mode,
        switch_kernel=args.switch_kernel,
        learned_condition_noop=args.learned_condition_noop,
        learned_condition_noop_patience=args.learned_condition_noop_patience,
        diffusion_steps=args.diffusion_steps,
        switches_per_diffusion_step=args.switches_per_diffusion_step,
        verbose=True,
    )
    sampling_seconds = time.perf_counter() - sampling_start
    save_sampling_result(args.output_dir / "generated_switch_graph.json", result)
    final_metrics = graph_metrics(result.graph)
    summary = {
        "initial_metrics": initial_metrics,
        "final_metrics": final_metrics,
        "train_report": train_report,
        "trace": list(result.trace),
        "sampling_mode": result.selection_mode,
        "oracle_agreement": result.oracle_agreement,
        "run_label": args.run_label or args.output_dir.name,
        "seed": int(args.seed),
        "conditioning_mode": conditioning_mode,
        "inference_contract": {
            "clean_graphlet_target": target_vector is not None,
            "training_adjacency": False,
            "anonymous_relation_degree_rows": True,
            "learned_graphlet_condition": model.learned_target is not None,
        },
        "sampling_config": {
            "requested_steps": int(sampling_steps),
            "diffusion_steps": int(args.diffusion_steps),
            "switches_per_diffusion_step": int(
                args.switches_per_diffusion_step
            ),
            "randomization_steps": int(
                sampling_steps
                if args.randomization_steps is None
                else args.randomization_steps
            ),
            "candidates_per_step": int(args.sample_candidates),
            "temperature": float(args.sample_temperature),
            "switch_kernel": str(args.switch_kernel),
            "learned_condition_noop": bool(args.learned_condition_noop),
            "learned_condition_noop_patience": int(
                args.learned_condition_noop_patience
            ),
        },
        "proposed": int(result.proposed),
        "applied": int(result.applied),
        "randomization_seconds": float(randomization_seconds),
        "sampling_seconds": float(sampling_seconds),
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
