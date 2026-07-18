#!/usr/bin/env python3
"""Fit and gate a type-exchangeable joint relation-degree graph prior."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from sparse_diffusion.structure_energy.exchangeable_role_prior import (
    ExchangeableJointRolePrior,
)
from sparse_diffusion.structure_energy.graph import (
    TypedGraph,
    hgen_edge_cover_view,
    load_typed_graph,
    uniform_edge_view,
)
from sparse_diffusion.structure_energy.mixed_edit_diffusion import (
    MarginalGraphPrior,
    load_mixed_edit_checkpoint,
    sample_mixed_edit_diffusion,
)
from sparse_diffusion.structure_energy.switch_chain import graph_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("fit-gate", "sample"), required=True)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data/IMDB_movie_subgraphs/subgraph_000"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prior", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--components",
        type=int,
        default=64,
        help="Maximum role prototypes per type; 0 keeps the full empirical codebook.",
    )
    parser.add_argument(
        "--role-model", choices=("categorical", "log_gaussian"), default="categorical"
    )
    parser.add_argument(
        "--role-component-sampling",
        choices=("iid", "systematic"),
        default="systematic",
    )
    parser.add_argument("--prior-samples", type=int, default=16)
    parser.add_argument("--randomization-steps-per-edge", type=float, default=2.0)
    parser.add_argument(
        "--role-correlation-strength",
        type=float,
        default=1.0,
        help=(
            "Retain this fraction of learned cross-family relation-degree coupling. "
            "One keeps the joint role prior; zero independently permutes every "
            "family-degree marginal within node type."
        ),
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--sample-steps", type=int, default=2000)
    parser.add_argument("--sample-candidates", type=int, default=16)
    parser.add_argument(
        "--sample-actions",
        choices=("checkpoint", "switch", "relocation_switch"),
        default="checkpoint",
    )
    parser.add_argument(
        "--allow-kernel-override",
        action="store_true",
        help="Legacy ablation only: sample with a kernel not used by the checkpoint.",
    )
    parser.add_argument(
        "--motif-refinement-stage",
        action="store_true",
        help=(
            "Use a count-preserving terminal subset of a checkpoint trained with "
            "the corresponding transitions. 'switch' preserves relation degrees; "
            "'relocation_switch' preserves total and per-family edge counts while "
            "allowing relation-degree correction."
        ),
    )
    parser.add_argument("--metrics-every", type=int, default=250)
    parser.add_argument("--initial-graph", type=Path)
    parser.add_argument(
        "--edge-view-count",
        type=int,
        help=(
            "Sample a full graph from the learned prior, then uniformly retain "
            "exactly this many edges before denoising. This does not read target "
            "adjacency, family counts, or degrees."
        ),
    )
    parser.add_argument(
        "--edge-view-source",
        choices=("uniform", "hgen_split"),
        default="uniform",
        help="Target-free transform applied to a learned full-graph prior sample.",
    )
    parser.add_argument("--fixed-time-fraction", type=float)
    parser.add_argument(
        "--time-schedule-steps",
        type=int,
        help=(
            "Number of initial sampling steps used to anneal t from one to zero. "
            "Any remaining --sample-steps continue at t=0. Defaults to the full "
            "sampling horizon."
        ),
    )
    return parser.parse_args()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def histogram_tv(left: np.ndarray, right: np.ndarray) -> float:
    size = int(max(left.max(initial=0), right.max(initial=0))) + 1
    left_hist = np.bincount(left.astype(np.int64), minlength=size).astype(np.float64)
    right_hist = np.bincount(right.astype(np.int64), minlength=size).astype(np.float64)
    left_hist /= max(left_hist.sum(), 1.0)
    right_hist /= max(right_hist.sum(), 1.0)
    return float(0.5 * np.abs(left_hist - right_hist).sum())


def relation_degree_correlation_error(clean: TypedGraph, graph: TypedGraph) -> float:
    """Mean pairwise correlation error across incident families within node type."""
    errors = []
    for type_id in range(len(clean.type_names)):
        nodes = np.flatnonzero(clean.node_types == type_id)
        incident = [
            family
            for family, spec in clean.family_specs.items()
            if type_id in (spec.src_type, spec.dst_type)
        ]
        for left_index, left_family in enumerate(incident):
            for right_family in incident[left_index + 1 :]:
                clean_left = np.log1p(clean.family_degrees(left_family)[nodes])
                clean_right = np.log1p(clean.family_degrees(right_family)[nodes])
                graph_left = np.log1p(graph.family_degrees(left_family)[nodes])
                graph_right = np.log1p(graph.family_degrees(right_family)[nodes])
                if min(
                    np.std(clean_left),
                    np.std(clean_right),
                    np.std(graph_left),
                    np.std(graph_right),
                ) <= 1e-12:
                    continue
                clean_corr = float(np.corrcoef(clean_left, clean_right)[0, 1])
                graph_corr = float(np.corrcoef(graph_left, graph_right)[0, 1])
                errors.append(abs(clean_corr - graph_corr))
    return float(np.mean(errors)) if errors else 0.0


def degree_distances(clean: TypedGraph, graph: TypedGraph) -> dict:
    report = {"total_degree_tv": histogram_tv(clean.degrees(), graph.degrees())}
    family = {}
    for name, spec in clean.family_specs.items():
        clean_degree = clean.family_degrees(name)
        generated_degree = graph.family_degrees(name)
        nodes = sorted(set(spec.src_nodes) | set(spec.dst_nodes))
        family[name] = histogram_tv(clean_degree[nodes], generated_degree[nodes])
    report["family_degree_tv_mean"] = float(np.mean(list(family.values())))
    report["family_degree_tv"] = family
    report["relation_degree_correlation_error"] = relation_degree_correlation_error(
        clean, graph
    )
    return report


def summarize_samples(clean: TypedGraph, samples: list[TypedGraph]) -> dict:
    metrics = [graph_metrics(graph) for graph in samples]
    distances = [degree_distances(clean, graph) for graph in samples]
    keys = metrics[0]
    return {
        "samples": len(samples),
        "metrics_mean": {key: float(np.mean([row[key] for row in metrics])) for key in keys},
        "metrics_std": {key: float(np.std([row[key] for row in metrics])) for key in keys},
        "total_degree_tv_mean": float(np.mean([row["total_degree_tv"] for row in distances])),
        "family_degree_tv_mean": float(np.mean([row["family_degree_tv_mean"] for row in distances])),
        "relation_degree_correlation_error_mean": float(
            np.mean([row["relation_degree_correlation_error"] for row in distances])
        ),
        "edge_count_unique": len({int(row["edges"]) for row in metrics}),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "fit-gate":
        clean = load_typed_graph(args.dataset_dir, active_only=True)
        role_prior = ExchangeableJointRolePrior.from_training_graph(
            clean,
            components=args.components,
            seed=args.seed,
            randomization_steps_per_edge=args.randomization_steps_per_edge,
            role_model=args.role_model,
            role_component_sampling=args.role_component_sampling,
            role_correlation_strength=args.role_correlation_strength,
        )
        marginal = MarginalGraphPrior.from_training_graph(clean)
        role_samples = [role_prior.sample(args.seed + index) for index in range(args.prior_samples)]
        marginal_samples = [marginal.sample(args.seed + index) for index in range(args.prior_samples)]
        role_report = summarize_samples(clean, role_samples)
        marginal_report = summarize_samples(clean, marginal_samples)
        role_prior.save(args.output_dir / "exchangeable_role_prior.json")
        payload = {
            "format": "dihug-exchangeable-role-prior-gate-v1",
            "clean_metrics": graph_metrics(clean),
            "role_prior": role_report,
            "bernoulli_prior": marginal_report,
            "gate": {
                "all_samples_realized": len(role_samples) == args.prior_samples,
                "edge_counts_random": role_report["edge_count_unique"] > 1,
                "degree_tv_improved": role_report["total_degree_tv_mean"] < marginal_report["total_degree_tv_mean"],
                "family_degree_tv_improved": role_report["family_degree_tv_mean"] < marginal_report["family_degree_tv_mean"],
            },
            "inference_contract": role_prior.to_json_dict()["inference_contract"],
        }
        write_json(args.output_dir / "prior_gate_report.json", payload)
        print(
            "[ROLE-PRIOR-GATE] "
            f"degree_tv={marginal_report['total_degree_tv_mean']:.4f}->"
            f"{role_report['total_degree_tv_mean']:.4f} "
            f"family_tv={marginal_report['family_degree_tv_mean']:.4f}->"
            f"{role_report['family_degree_tv_mean']:.4f} "
            f"role_corr_err={role_report['relation_degree_correlation_error_mean']:.4f} "
            f"tri={marginal_report['metrics_mean']['triangles']:.1f}->"
            f"{role_report['metrics_mean']['triangles']:.1f}",
            flush=True,
        )
        return

    if args.prior is None or args.checkpoint is None:
        raise ValueError("sample mode requires --prior and --checkpoint")
    role_prior = ExchangeableJointRolePrior.load(args.prior)
    model, checkpoint_prior, metadata = load_mixed_edit_checkpoint(
        args.checkpoint, device=args.device
    )
    if role_prior.marginal.type_names != checkpoint_prior.type_names:
        raise ValueError("role prior and denoiser checkpoint schemas disagree")
    if args.sample_actions == "switch":
        sample_actions = ("switch",)
    elif args.sample_actions == "relocation_switch":
        sample_actions = ("relocation", "switch")
    else:
        sample_actions = tuple(metadata["actions"])
    checkpoint_actions = tuple(metadata["actions"])
    terminal_count_preserving_subset = (
        args.motif_refinement_stage
        and sample_actions in (("switch",), ("relocation", "switch"))
        and set(sample_actions).issubset(checkpoint_actions)
    )
    if args.motif_refinement_stage and not terminal_count_preserving_subset:
        raise ValueError(
            "--motif-refinement-stage requires a count-preserving action subset "
            "and a checkpoint trained with every requested transition"
        )
    if (
        sample_actions != checkpoint_actions
        and not terminal_count_preserving_subset
        and not args.allow_kernel_override
    ):
        raise ValueError(
            f"checkpoint kernel {checkpoint_actions} does not match sampling kernel "
            f"{sample_actions}; use --motif-refinement-stage for a trained switch "
            "subset, or --allow-kernel-override only for a named legacy ablation"
        )
    if args.initial_graph is not None and args.edge_view_count is not None:
        raise ValueError("--initial-graph and --edge-view-count are mutually exclusive")
    initial_graph = None
    if args.initial_graph is not None:
        initial_graph = TypedGraph.from_json_dict(
            json.loads(args.initial_graph.read_text(encoding="utf-8"))
        )
    elif args.edge_view_count is not None:
        full_prior_sample = role_prior.sample(args.seed + 71)
        if args.edge_view_source == "hgen_split":
            initial_graph = hgen_edge_cover_view(
                full_prior_sample,
                edge_count=args.edge_view_count,
                seed=args.seed + 193,
            )
        else:
            initial_graph = uniform_edge_view(
                full_prior_sample,
                edge_count=args.edge_view_count,
                seed=args.seed + 193,
            )

    result = sample_mixed_edit_diffusion(
        role_prior,
        model,
        steps=args.sample_steps,
        candidates_per_step=args.sample_candidates,
        seed=args.seed,
        device=args.device,
        temperature=0.0,
        metrics_every=args.metrics_every,
        actions=sample_actions,
        initial_graph=initial_graph,
        fixed_time_fraction=args.fixed_time_fraction,
        time_schedule_steps=args.time_schedule_steps,
    )
    if (
        terminal_count_preserving_subset
        and result.graph.num_edges != int(result.trace[0]["edges"])
    ):
        raise AssertionError("terminal count-preserving refinement changed edge count")
    payload = {
        "format": "dihug-exchangeable-role-mixed-edit-sample-v1",
        "initial": result.trace[0],
        "final": graph_metrics(result.graph),
        "trace": list(result.trace),
        "action_counts": dict(result.action_counts),
        "inference_uses_clean_graph": False,
        "sampling_config": {
            "steps": args.sample_steps,
            "candidates": args.sample_candidates,
            "seed": args.seed,
            "actions": list(sample_actions),
            "checkpoint_actions": list(checkpoint_actions),
            "kernel_contract": (
                "terminal_count_preserving_subset"
                if terminal_count_preserving_subset
                else (
                    "legacy_override"
                    if sample_actions != checkpoint_actions
                    else "checkpoint_exact"
                )
            ),
            "kernel_override": (
                sample_actions != checkpoint_actions
                and not terminal_count_preserving_subset
            ),
            "initial_graph": args.initial_graph is not None,
            "edge_view_count": args.edge_view_count,
            "edge_view_source": (
                f"{args.edge_view_source}_view_of_learned_prior_sample"
                if args.edge_view_count is not None
                else None
            ),
            "fixed_time_fraction": args.fixed_time_fraction,
            "time_schedule_steps": args.time_schedule_steps,
        },
    }
    write_json(args.output_dir / "sampling_report.json", payload)
    write_json(args.output_dir / "generated_graph.json", result.graph.to_json_dict())
    print(
        "[ROLE-PRIOR-SAMPLE] "
        f"tri={payload['initial']['triangles']:.0f}->{payload['final']['triangles']:.0f} "
        f"clustering={payload['initial']['clustering']:.6f}->"
        f"{payload['final']['clustering']:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
