"""Training entry point for the anonymous DiHug release."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Sequence

from dihug.cli import PROFILES, ROOT, _relative_or_absolute, _resolve_device
from dihug.core.graph import load_typed_graph
from dihug.core.role_guided_edit_diffusion import (
    ExchangeableRoleDegreePrior,
)


@dataclass(frozen=True)
class WarmupProfile:
    trajectories: int
    forward_steps: int
    record_every: int
    epochs: int
    learning_rate: float
    role_corruption: bool = False


WARMUP = {
    "imdb": WarmupProfile(32, 1_000, 10, 100, 1.0e-3),
    "pubmed": WarmupProfile(16, 3_000, 30, 20, 1.0e-3, True),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train DiHug from the prepared graph. No packaged checkpoint or "
            "candidate cache is used."
        )
    )
    parser.add_argument("--dataset", required=True, choices=tuple(PROFILES))
    parser.add_argument(
        "--device",
        default="auto",
        help="'auto' (default), 'cpu', 'cuda', or a CUDA device such as 'cuda:0'.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Override the prepared dataset directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Default: outputs/training/<dataset>.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--stage",
        choices=("all", "warmup", "calibrate"),
        default="all",
        help="Run the complete pipeline or one resumable training stage.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Tiny one-epoch configuration for an execution check, not full results.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Retrain stages whose output checkpoint already exists.",
    )
    parser.add_argument(
        "--disable-time-conditioning",
        action="store_true",
        help="Train the reported no-time ablation during fixed-count calibration.",
    )
    parser.add_argument(
        "--trajectories",
        type=int,
        help="Override the number of fixed-count calibration trajectories.",
    )
    parser.add_argument(
        "--forward-steps",
        type=int,
        help="Override fixed-count forward edits per calibration trajectory.",
    )
    parser.add_argument(
        "--record-every",
        type=int,
        help="Override the calibration supervision interval.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Override fixed-count calibration epochs.",
    )
    parser.add_argument(
        "--train-candidates",
        type=int,
        default=16,
        help="Candidate edits per listwise training example.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved stages without running them.",
    )
    return parser


def _command_base(
    data_dir: Path,
    output_dir: Path,
    device: str,
    seed: int,
) -> list[str]:
    return [
        sys.executable,
        "-u",
        "-m",
        "dihug.train_model",
        "--mode",
        "train",
        "--denoiser",
        "listwise",
        "--dataset-dir",
        _relative_or_absolute(data_dir),
        "--output-dir",
        _relative_or_absolute(output_dir),
        "--device",
        device,
        "--seed",
        str(seed),
        "--feature-profile",
        "typed_closure_compact_v2",
        "--training-objective",
        "inverse_ce",
        "--train-candidates",
        "16",
        "--batch-size",
        "64",
        "--hidden-dim",
        "256",
        "--layers",
        "3",
        "--dropout",
        "0.1",
        "--weight-decay",
        "1e-4",
    ]


def _warmup_command(
    dataset: str,
    data_dir: Path,
    output_dir: Path,
    device: str,
    seed: int,
    quick: bool,
    train_candidates: int,
) -> list[str] | None:
    if dataset not in WARMUP:
        return None
    profile = WARMUP[dataset]
    trajectories = 2 if quick else profile.trajectories
    forward_steps = 4 if quick else profile.forward_steps
    record_every = 1 if quick else profile.record_every
    epochs = 1 if quick else profile.epochs
    command = _command_base(data_dir, output_dir, device, seed)
    command[command.index("--train-candidates") + 1] = str(train_candidates)
    command.extend(
        (
            "--kernel",
            "bdrs",
            "--forward-mode",
            "micro",
            "--trajectories",
            str(trajectories),
            "--forward-steps",
            str(forward_steps),
            "--record-every",
            str(record_every),
            "--epochs",
            str(epochs),
            "--learning-rate",
            str(profile.learning_rate),
            "--fixed-t-trials",
            "0" if quick else "16",
        )
    )
    if profile.role_corruption and not quick:
        command.extend(
            (
                "--role-corruption-fraction",
                "1",
                "--role-correlation-strengths",
                "0,0.25,0.5,0.75,1",
                "--role-prior-components",
                "512",
                "--role-prior-randomization-steps-per-edge",
                "2",
            )
        )
    return command


def _calibration_defaults(
    dataset: str,
    edge_count: int,
    quick: bool,
) -> tuple[int, int, int, int, float]:
    if quick:
        return 2, 4, 1, 1, 1.0e-3
    if dataset == "dblp":
        return 32, 20_000, 200, 20, 1.0e-3
    forward_steps = 20 * edge_count
    return 4, forward_steps, max(1, forward_steps // 200), 5, 1.0e-5


def build_commands(
    args: argparse.Namespace,
    device: str,
) -> tuple[list[tuple[str, list[str], Path]], Path, Path]:
    profile = PROFILES[args.dataset]
    data_dir = args.data_dir or Path(profile.data_dir)
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir
    output_dir = args.output_dir or Path("outputs", "training", args.dataset)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir

    graph = load_typed_graph(data_dir, active_only=True)
    warmup_dir = output_dir / "warmup"
    model_dir = output_dir / (
        "model_no_time" if args.disable_time_conditioning else "model"
    )
    commands: list[tuple[str, list[str], Path]] = []

    if args.stage in {"all", "warmup"}:
        warmup = _warmup_command(
            args.dataset,
            data_dir,
            warmup_dir,
            device,
            args.seed,
            args.quick,
            args.train_candidates,
        )
        if warmup is not None:
            commands.append(
                ("warmup", warmup, warmup_dir / "mixed_edit_diffusion.pt")
            )

    if args.stage in {"all", "calibrate"}:
        defaults = _calibration_defaults(
            args.dataset, graph.num_edges, args.quick
        )
        trajectories = defaults[0] if args.trajectories is None else args.trajectories
        forward_steps = defaults[1] if args.forward_steps is None else args.forward_steps
        record_every = defaults[2] if args.record_every is None else args.record_every
        epochs = defaults[3] if args.epochs is None else args.epochs
        learning_rate = defaults[4]
        for name, value in (
            ("trajectories", trajectories),
            ("forward-steps", forward_steps),
            ("record-every", record_every),
            ("epochs", epochs),
            ("train-candidates", args.train_candidates),
        ):
            if value <= 0:
                raise ValueError(f"--{name} must be positive")

        command = _command_base(
            data_dir, model_dir, device, args.seed + 1701
        )
        candidate_index = command.index("--train-candidates") + 1
        command[candidate_index] = str(args.train_candidates)
        command.extend(
            (
                "--kernel",
                "rs",
                "--forward-mode",
                "fixed_count",
                "--trajectories",
                str(trajectories),
                "--forward-steps",
                str(forward_steps),
                "--record-every",
                str(record_every),
                "--epochs",
                str(epochs),
                "--learning-rate",
                str(learning_rate),
                "--fixed-t-trials",
                "0" if args.quick else "16",
            )
        )
        if args.dataset in WARMUP:
            warmup_checkpoint = warmup_dir / "mixed_edit_diffusion.pt"
            command.extend(
                ("--init-checkpoint", _relative_or_absolute(warmup_checkpoint))
            )
        if args.disable_time_conditioning:
            command.append("--disable-time-conditioning")
        commands.append(
            ("calibrate", command, model_dir / "mixed_edit_diffusion.pt")
        )
    return commands, data_dir, output_dir


def _write_training_metadata(
    output_dir: Path,
    dataset: str,
    data_dir: Path,
    device: str,
    seed: int,
    quick: bool,
) -> None:
    graph = load_typed_graph(data_dir, active_only=True)
    role_path = output_dir / "role_prior.json"
    ExchangeableRoleDegreePrior.from_training_graph(graph).save(role_path)
    payload = {
        "format": "dihug-training-run-v1",
        "dataset": dataset,
        "seed": seed,
        "device": device,
        "quick_check_only": quick,
        "nodes": graph.num_nodes,
        "edges": graph.num_edges,
        "relations": len(graph.family_specs),
        "role_prior": str(role_path.relative_to(output_dir)),
        "uses_packaged_checkpoint": False,
        "uses_packaged_candidate_cache": False,
    }
    (output_dir / "training_run.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        device = _resolve_device(args.device)
        commands, data_dir, output_dir = build_commands(args, device)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(
        f"[DiHug train] dataset={args.dataset} device={device} "
        f"stage={args.stage} quick={args.quick}"
    )
    for stage, command, _ in commands:
        print(f"[DiHug train:{stage}] {shlex.join(command)}")
    if args.dry_run:
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_training_metadata(
        output_dir,
        args.dataset,
        data_dir,
        device,
        args.seed,
        args.quick,
    )
    env = os.environ.copy()
    current_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(ROOT)
        if not current_pythonpath
        else f"{ROOT}{os.pathsep}{current_pythonpath}"
    )
    for stage, command, checkpoint in commands:
        if checkpoint.exists() and not args.overwrite:
            print(f"[DiHug train:{stage}] reuse={checkpoint}")
            continue
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        try:
            subprocess.run(command, cwd=ROOT, env=env, check=True)
        except subprocess.CalledProcessError as exc:
            print(
                f"error: {stage} failed with exit code {exc.returncode}: "
                f"{shlex.join(command)}",
                file=sys.stderr,
            )
            return int(exc.returncode)
    model_dir_name = (
        "model_no_time" if args.disable_time_conditioning else "model"
    )
    if args.stage != "warmup":
        print(
            "[DiHug train] model="
            f"{output_dir / model_dir_name / 'mixed_edit_diffusion.pt'}"
        )
    print(f"[DiHug train] role_prior={output_dir / 'role_prior.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
