"""Small command-line interface for DiHug generation and ablations."""

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


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class DatasetProfile:
    data_dir: str
    steps: int
    candidates: int
    fixed_role_steps: int | None


PROFILES = {
    "imdb": DatasetProfile("data/imdb", 10_000, 64, 2_240),
    "pubmed": DatasetProfile("data/pubmed", 10_000, 16, 7_488),
    "dblp": DatasetProfile("data/dblp", 50_000, 16, None),
}

MODES = (
    "full",
    "prior",
    "random",
    "relocation",
    "switch",
    "no-score",
    "no-target",
    "no-time",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a heterogeneous graph with a trained DiHug model. "
            "Dataset-specific reported settings are used unless overridden."
        )
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=tuple(PROFILES),
        help="Prepared dataset and matching training output.",
    )
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
        "--mode",
        default="full",
        choices=MODES,
        help=(
            "Paper setting: full model; weak prior; random, relocation-only, "
            "or switch-only edits; or remove learned score, degree target, "
            "or diffusion-time input."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--steps",
        type=int,
        help="Override the dataset's validated reverse-edit budget.",
    )
    parser.add_argument(
        "--candidates",
        type=int,
        help="Override the number of candidate edits scored at each step.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Default: outputs/<dataset>/<mode>/seed<seed>.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help=(
            "Trained model checkpoint. Default: "
            "outputs/training/<dataset>/model/mixed_edit_diffusion.pt."
        ),
    )
    parser.add_argument(
        "--role-prior",
        type=Path,
        help=(
            "Anonymous degree target learned during training. Default: "
            "outputs/training/<dataset>/role_prior.json."
        ),
    )
    parser.add_argument(
        "--role-weight",
        type=float,
        default=0.05,
        help="Base weight of improvement toward the sampled degree target.",
    )
    parser.add_argument(
        "--role-threshold",
        type=float,
        default=5.5,
        help="Average squared degree-target error used to enter Stage 2.",
    )
    parser.add_argument(
        "--targeted-fraction",
        type=float,
        default=0.5,
        help="Fraction of Stage-1 candidates targeted from surplus to deficit nodes.",
    )
    parser.add_argument(
        "--closure-fraction",
        type=float,
        default=1.0,
        help="Fraction of candidates proposed from open typed wedges.",
    )
    parser.add_argument(
        "--auto-role-balance",
        type=float,
        default=0.5,
        help="Scale role improvement relative to learned-score variation.",
    )
    parser.add_argument(
        "--fixed-role-steps",
        type=int,
        help="Stage-1 length for --mode no-target; defaults are dataset specific.",
    )
    parser.add_argument(
        "--time-horizon",
        type=int,
        help="Time-conditioning horizon; defaults to the selected edit budget.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved commands without executing them.",
    )
    return parser


def _resolve_device(requested: str) -> str:
    requested = requested.lower()
    if requested == "auto":
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cpu":
        return requested
    if requested == "cuda" or requested.startswith("cuda:"):
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but torch.cuda.is_available() is false. "
                "Use --device cpu or install a CUDA-enabled PyTorch build."
            )
        return requested
    raise ValueError("--device must be auto, cpu, cuda, or cuda:<index>")


def _resolve_input_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _require_inputs(
    dataset: str,
    data_dir: Path,
    checkpoint: Path,
    role_prior: Path,
) -> None:
    required = (
        _resolve_input_path(data_dir),
        _resolve_input_path(checkpoint),
        _resolve_input_path(role_prior),
    )
    missing = [path for path in required if not path.exists()]
    if missing:
        formatted = "\n".join(
            f"  - {_relative_or_absolute(path)}" for path in missing
        )
        raise FileNotFoundError(
            "Required inputs are missing:\n"
            f"{formatted}\n"
            "Prepare the benchmark data, then run "
            f"`python train.py --dataset {dataset}` before generation."
        )


def _relative_or_absolute(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def _print_generation_summary(report_path: Path) -> None:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    generated = report["final"]
    reference = report.get("real", {})
    rows = (
        ("edges", "Edges", ".0f"),
        ("lcc", "LCC", ".0f"),
        ("triangles", "Triangles", ".0f"),
        ("clustering", "Clustering", ".6f"),
        ("assortativity", "Assortativity", ".6f"),
        ("max_degree", "Max degree", ".0f"),
    )
    print("[DiHug] final graph compared with the reference graph")
    print(f"{'metric':<16}{'generated':>14}{'reference':>14}")
    for key, label, spec in rows:
        generated_value = generated.get(key)
        reference_value = reference.get(key)
        generated_text = (
            "--" if generated_value is None else format(generated_value, spec)
        )
        reference_text = (
            "--" if reference_value is None else format(reference_value, spec)
        )
        print(f"{label:<16}{generated_text:>14}{reference_text:>14}")


def build_commands(
    args: argparse.Namespace,
    device: str,
) -> tuple[list[list[str]], Path, str]:
    profile = PROFILES[args.dataset]
    data_dir = args.data_dir or Path(profile.data_dir)
    if args.mode in {"no-score", "no-target"} and args.dataset == "dblp":
        raise ValueError(
            f"--mode {args.mode} is defined only for IMDB and PubMed"
        )
    if args.mode == "no-time" and args.dataset == "dblp":
        raise ValueError(
            "--mode no-time is defined only for IMDB and PubMed"
        )

    default_steps = profile.steps
    steps = default_steps if args.steps is None else args.steps
    if steps < 0:
        raise ValueError("--steps must be nonnegative")
    if args.mode == "prior":
        steps = 0
    candidates = profile.candidates if args.candidates is None else args.candidates
    if candidates <= 0:
        raise ValueError("--candidates must be positive")
    horizon = (
        (steps if steps > 0 else default_steps)
        if args.time_horizon is None
        else args.time_horizon
    )
    if horizon <= 0:
        raise ValueError("--time-horizon must be positive")

    output_dir = args.output_dir or Path(
        "outputs", args.dataset, args.mode, f"seed{args.seed}"
    )
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir

    model_path = args.checkpoint or Path(
        "outputs",
        "training",
        args.dataset,
        "model",
        "mixed_edit_diffusion.pt",
    )
    role_prior_path = args.role_prior or Path(
        "outputs", "training", args.dataset, "role_prior.json"
    )
    commands: list[list[str]] = []
    if args.mode == "no-time":
        model_path = Path(_relative_or_absolute(output_dir / "artifacts" / "model_no_time.pt"))
        commands.append(
            [
                sys.executable,
                "-m",
                "dihug.time_ablation",
                "--input",
                _relative_or_absolute(
                    _resolve_input_path(
                        args.checkpoint
                        or Path(
                            "outputs",
                            "training",
                            args.dataset,
                            "model",
                            "mixed_edit_diffusion.pt",
                        )
                    )
                ),
                "--output",
                str(model_path),
            ]
        )

    phase_policy = "auto"
    selection_mode = "model"
    role_weight = args.role_weight
    auto_role_balance = args.auto_role_balance
    targeted_fraction = args.targeted_fraction
    closure_fraction = args.closure_fraction
    fixed_role_steps: int | None = None

    reference_candidate_modes = {"prior", "random", "relocation", "switch"}
    sampling_backend = (
        "gpu"
        if device.startswith("cuda") and args.mode not in reference_candidate_modes
        else "cpu"
    )

    if args.mode == "random":
        selection_mode = "random"
        targeted_fraction = 0.0
        closure_fraction = 0.0
    elif args.mode == "relocation":
        phase_policy = "relocation_only"
        closure_fraction = 0.0
    elif args.mode == "switch":
        phase_policy = "switch_only"
        role_weight = 0.0
        targeted_fraction = 0.0
    elif args.mode == "no-score":
        selection_mode = "role_only"
    elif args.mode == "no-target":
        phase_policy = "fixed"
        role_weight = 0.0
        auto_role_balance = 0.0
        targeted_fraction = 0.0
        if args.fixed_role_steps is not None:
            fixed_role_steps = args.fixed_role_steps
        elif profile.fixed_role_steps is not None:
            fixed_role_steps = round(
                profile.fixed_role_steps * steps / profile.steps
            )
        else:
            raise ValueError("--fixed-role-steps is required for this dataset")
        if not 0 <= fixed_role_steps <= steps:
            raise ValueError("--fixed-role-steps must be between 0 and --steps")

    command = [
        sys.executable,
        "-u",
        "-m",
        "dihug.generate",
        "--dataset-dir",
        _relative_or_absolute(_resolve_input_path(data_dir)),
        "--model-checkpoint",
        _relative_or_absolute(_resolve_input_path(model_path)),
        "--role-prior",
        _relative_or_absolute(_resolve_input_path(role_prior_path)),
        "--strict-target-free",
        "--family-count-mode",
        "binomial",
        "--seed",
        str(args.seed),
        "--device",
        device,
        "--sampling-backend",
        sampling_backend,
        "--steps",
        str(steps),
        "--model-time-horizon-steps",
        str(horizon),
        "--metrics-every",
        str(max(1, steps)),
        "--candidates",
        str(candidates),
        "--role-weight",
        str(role_weight),
        "--gain-scale",
        "10",
        "--schedule-power",
        "1",
        "--schedule-floor",
        "1",
        "--role-error-per-node-threshold",
        str(args.role_threshold),
        "--closure-candidate-fraction",
        str(closure_fraction),
        "--auto-role-balance",
        str(auto_role_balance),
        "--targeted-role-candidate-fraction",
        str(targeted_fraction),
        "--phase-policy",
        phase_policy,
        "--selection-mode",
        selection_mode,
        "--decoder",
        "map",
        "--output-dir",
        _relative_or_absolute(output_dir),
    ]
    if sampling_backend == "gpu":
        command.extend(("--gpu-candidate-mode", "canonical_ordered"))
    if fixed_role_steps is not None:
        command.extend(("--fixed-role-steps", str(fixed_role_steps)))
    commands.append(command)
    return commands, output_dir, sampling_backend


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        device = _resolve_device(args.device)
        checkpoint = args.checkpoint or Path(
            "outputs",
            "training",
            args.dataset,
            "model",
            "mixed_edit_diffusion.pt",
        )
        role_prior = args.role_prior or Path(
            "outputs", "training", args.dataset, "role_prior.json"
        )
        data_dir = args.data_dir or Path(PROFILES[args.dataset].data_dir)
        if not args.dry_run:
            _require_inputs(args.dataset, data_dir, checkpoint, role_prior)
        commands, output_dir, backend = build_commands(args, device)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(
        f"[DiHug] dataset={args.dataset} mode={args.mode} "
        f"device={device} candidate_backend={backend}"
    )
    for command in commands:
        print(f"[DiHug] {shlex.join(command)}")
    if args.dry_run:
        return 0

    env = os.environ.copy()
    current_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(ROOT)
        if not current_pythonpath
        else f"{ROOT}{os.pathsep}{current_pythonpath}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    for command in commands:
        try:
            subprocess.run(command, cwd=ROOT, env=env, check=True)
        except subprocess.CalledProcessError as exc:
            print(
                f"error: command failed with exit code {exc.returncode}: "
                f"{shlex.join(command)}",
                file=sys.stderr,
            )
            return int(exc.returncode)
    report_path = output_dir / "gate_report.json"
    _print_generation_summary(report_path)
    print(f"[DiHug] result={report_path}")
    return 0
