#!/usr/bin/env python3
"""Train and sample the typed quotient-energy edit branch."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from sparse_diffusion.structure_energy.exchangeable_role_prior import (
    ExchangeableJointRolePrior,
)
from sparse_diffusion.structure_energy.graph import load_typed_graph
from sparse_diffusion.structure_energy.mixed_edit_diffusion import (
    permute_typed_graph_within_types,
)
from sparse_diffusion.structure_energy.quotient_energy import (
    add_on_policy_replay_negatives,
    build_quotient_energy_dataset,
    calibrate_minimum_energy_drop,
    graph_energies,
    load_quotient_energy_checkpoint,
    sample_quotient_energy_edits,
    save_quotient_energy_checkpoint,
    train_quotient_energy,
    write_quotient_report,
)
from sparse_diffusion.structure_energy.switch_chain import graph_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("train", "sample"), required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--trajectories", type=int, default=12)
    parser.add_argument("--noise-fractions", default="0.05,0.2,1.0")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--sample-steps", type=int, default=1000)
    parser.add_argument("--sample-candidates", type=int, default=16)
    parser.add_argument("--metrics-every", type=int, default=100)
    parser.add_argument(
        "--minimum-energy-drop",
        type=float,
        default=-1.0,
        help="Negative uses the held-out clean-orbit calibration in the checkpoint.",
    )
    parser.add_argument("--replay-rounds", type=int, default=0)
    parser.add_argument("--replay-steps", type=int, default=100)
    parser.add_argument("--replay-candidates", type=int, default=16)
    parser.add_argument("--noop-patience", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = load_typed_graph(args.dataset_dir, active_only=True)
    prior = ExchangeableJointRolePrior.load(args.prior)
    if tuple(prior.marginal.type_names) != tuple(source.type_names):
        raise ValueError("prior and dataset type schemas differ")

    if args.mode == "train":
        fractions = tuple(
            float(value.strip())
            for value in args.noise_fractions.split(",")
            if value.strip()
        )
        dataset = build_quotient_energy_dataset(
            source,
            prior.sample,
            trajectories=args.trajectories,
            noise_fractions=fractions,
            seed=args.seed,
        )
        model, report = train_quotient_energy(
            dataset,
            epochs=args.epochs,
            hidden_dim=args.hidden_dim,
            heads=args.heads,
            layers=args.layers,
            dropout=args.dropout,
            learning_rate=args.learning_rate,
            device=args.device,
            seed=args.seed,
        )
        replay_reports = []
        for replay_round in range(int(args.replay_rounds)):
            dataset = add_on_policy_replay_negatives(
                dataset,
                model,
                prior.sample,
                steps=args.replay_steps,
                candidates_per_step=args.replay_candidates,
                seed=args.seed + 60_000_133 * (replay_round + 1),
                device=args.device,
            )
            model, replay_report = train_quotient_energy(
                dataset,
                epochs=args.epochs,
                hidden_dim=args.hidden_dim,
                heads=args.heads,
                layers=args.layers,
                dropout=args.dropout,
                learning_rate=args.learning_rate,
                device=args.device,
                seed=args.seed + replay_round + 1,
                initial_model=model,
            )
            replay_reports.append(dict(replay_report))
            report = replay_report
        model.eval()
        clean = source
        permuted = permute_typed_graph_within_types(source, args.seed + 90_000_011)
        with torch.no_grad():
            orbit_energies = graph_energies(
                model, (clean, permuted), args.device
            ).detach().cpu()
        report = {
            **dict(report),
            "orbit_energy_absolute_difference": float(
                abs(orbit_energies[0] - orbit_energies[1])
            ),
            "noise_fractions": list(fractions),
            "examples": len(dataset.examples),
            "replay_rounds": int(args.replay_rounds),
            "replay_reports": replay_reports,
            "sampling_uses_clean_graph": False,
        }
        report.update(
            calibrate_minimum_energy_drop(
                model,
                source,
                trials_per_action=32,
                seed=args.seed + 91_000_021,
                device=args.device,
            )
        )
        checkpoint = args.output_dir / "quotient_energy.pt"
        save_quotient_energy_checkpoint(checkpoint, model, report)
        write_quotient_report(args.output_dir / "train_report.json", report)
        print(
            "[QUOTIENT-ENERGY-TRAIN] "
            f"ranking={report['overall_ranking_accuracy']:.4f} "
            f"prior={report['prior_ranking_accuracy']:.4f} "
            f"relocation={report['relocation_ranking_accuracy']:.4f} "
            f"switch={report['switch_ranking_accuracy']:.4f} "
            f"orbit_diff={report['orbit_energy_absolute_difference']:.3e}",
            flush=True,
        )
        return

    if args.checkpoint is None:
        raise ValueError("sample mode requires --checkpoint")
    model, metadata = load_quotient_energy_checkpoint(args.checkpoint, args.device)
    minimum_energy_drop = float(args.minimum_energy_drop)
    if minimum_energy_drop < 0.0:
        minimum_energy_drop = float(
            metadata.get("train_report", {}).get(
                "recommended_minimum_energy_drop", 0.0
            )
        )
    initial = prior.sample(args.seed + 71)
    result = sample_quotient_energy_edits(
        initial,
        model,
        steps=args.sample_steps,
        candidates_per_step=args.sample_candidates,
        seed=args.seed,
        device=args.device,
        metrics_every=args.metrics_every,
        minimum_energy_drop=minimum_energy_drop,
        noop_patience=args.noop_patience,
    )
    (args.output_dir / "generated_graph.json").write_text(
        json.dumps(result.graph.to_json_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    payload = {
        "format": "dihug-typed-quotient-energy-sample-v1",
        "seed": args.seed,
        "initial": result.trace[0],
        "final": graph_metrics(result.graph),
        "trace": list(result.trace),
        "action_counts": dict(result.action_counts),
        "proposed": result.proposed,
        "applied": result.applied,
        "sampling_uses_clean_graph": False,
        "target_metrics_used_by_sampler": False,
        "checkpoint_contract": metadata["inference_contract"],
        "minimum_energy_drop": minimum_energy_drop,
    }
    write_quotient_report(args.output_dir / "sampling_report.json", payload)
    print(
        "[QUOTIENT-ENERGY-SAMPLE] "
        f"tri={payload['initial']['triangles']:.0f}->{payload['final']['triangles']:.0f} "
        f"clustering={payload['initial']['clustering']:.6f}->"
        f"{payload['final']['clustering']:.6f} applied={result.applied}",
        flush=True,
    )


if __name__ == "__main__":
    main()
