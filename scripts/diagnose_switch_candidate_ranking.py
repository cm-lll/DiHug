#!/usr/bin/env python3
"""Separate switch proposal coverage from learned scorer ranking quality."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from sparse_diffusion.structure_energy.graph import TypedGraph
from sparse_diffusion.structure_energy.mixed_edit_diffusion import (
    align_mixed_edit_features_for_model,
    degree_normalized_closure_delta,
    load_mixed_edit_checkpoint,
    mixed_edit_candidate_matrix,
    typed_closure_delta,
)
from sparse_diffusion.structure_energy.role_guided_edit_diffusion import (
    sample_switch_refinement_candidates,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--model-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-counts", type=int, nargs="+", default=(16, 64))
    parser.add_argument("--trials", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--time-fraction", type=float, default=0.0)
    parser.add_argument("--closure-candidate-fraction", type=float, default=0.0)
    parser.add_argument("--closure-degree-bias-power", type=float, default=0.0)
    return parser.parse_args()


def _safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    if left.size < 2 or float(left.std()) == 0.0 or float(right.std()) == 0.0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def diagnose(
    graph: TypedGraph,
    model: torch.nn.Module,
    *,
    feature_profile: str,
    candidate_count: int,
    trials: int,
    seed: int,
    device: torch.device,
    time_fraction: float,
    closure_candidate_fraction: float,
    closure_degree_bias_power: float,
) -> dict:
    rng = np.random.default_rng(int(seed))
    selected_triangle = []
    oracle_triangle = []
    random_triangle = []
    selected_clustering = []
    oracle_clustering = []
    random_clustering = []
    positive_triangle = []
    positive_clustering = []
    triangle_agreement = []
    clustering_agreement = []
    all_logits = []
    all_triangle = []
    all_clustering = []
    model = model.to(device).eval()
    with torch.no_grad():
        for _ in range(int(trials)):
            candidates = sample_switch_refinement_candidates(
                graph,
                rng,
                int(candidate_count),
                closure_fraction=closure_candidate_fraction,
                degree_bias_power=closure_degree_bias_power,
            )
            matrix = mixed_edit_candidate_matrix(
                graph,
                candidates,
                float(time_fraction),
                feature_profile=feature_profile,
            )
            matrix = align_mixed_edit_features_for_model(
                matrix, model, family_count=len(graph.family_specs)
            )
            logits = (
                model(torch.from_numpy(matrix)[None].float().to(device))
                .squeeze(0)
                .cpu()
                .numpy()
            )
            triangle = np.asarray(
                [typed_closure_delta(graph, item).sum() for item in candidates],
                dtype=np.float64,
            )
            clustering = np.asarray(
                [degree_normalized_closure_delta(graph, item) for item in candidates],
                dtype=np.float64,
            )
            selected = int(logits.argmax())
            triangle_oracle = int(triangle.argmax())
            clustering_oracle = int(clustering.argmax())
            random_index = int(rng.integers(len(candidates)))
            selected_triangle.append(float(triangle[selected]))
            oracle_triangle.append(float(triangle[triangle_oracle]))
            random_triangle.append(float(triangle[random_index]))
            selected_clustering.append(float(clustering[selected]))
            oracle_clustering.append(float(clustering[clustering_oracle]))
            random_clustering.append(float(clustering[random_index]))
            positive_triangle.append(float(triangle.max() > 0.0))
            positive_clustering.append(float(clustering.max() > 0.0))
            triangle_agreement.append(float(selected == triangle_oracle))
            clustering_agreement.append(float(selected == clustering_oracle))
            all_logits.extend(logits.tolist())
            all_triangle.extend(triangle.tolist())
            all_clustering.extend(clustering.tolist())

    def summarize(selected: list[float], oracle: list[float], random: list[float]) -> dict:
        selected_mean = float(np.mean(selected))
        oracle_mean = float(np.mean(oracle))
        random_mean = float(np.mean(random))
        scale = max(oracle_mean - random_mean, 1.0e-8)
        return {
            "selected_mean": selected_mean,
            "oracle_mean": oracle_mean,
            "random_mean": random_mean,
            "normalized_regret": float(
                np.clip((oracle_mean - selected_mean) / scale, 0.0, 2.0)
            ),
        }

    return {
        "candidate_count": int(candidate_count),
        "trials": int(trials),
        "triangle": summarize(selected_triangle, oracle_triangle, random_triangle),
        "clustering_sum": summarize(
            selected_clustering, oracle_clustering, random_clustering
        ),
        "positive_candidate_coverage_triangle": float(np.mean(positive_triangle)),
        "positive_candidate_coverage_clustering": float(
            np.mean(positive_clustering)
        ),
        "model_triangle_oracle_top1": float(np.mean(triangle_agreement)),
        "model_clustering_oracle_top1": float(np.mean(clustering_agreement)),
        "score_triangle_correlation": _safe_correlation(
            np.asarray(all_logits), np.asarray(all_triangle)
        ),
        "score_clustering_correlation": _safe_correlation(
            np.asarray(all_logits), np.asarray(all_clustering)
        ),
    }


def main() -> None:
    args = parse_args()
    if args.trials <= 0 or any(value < 2 for value in args.candidate_counts):
        raise ValueError("trials must be positive and candidate counts at least two")
    graph = TypedGraph.from_json_dict(json.loads(args.graph.read_text(encoding="utf-8")))
    model, _, metadata = load_mixed_edit_checkpoint(
        args.model_checkpoint, device=args.device
    )
    device = torch.device(args.device)
    rows = [
        diagnose(
            graph,
            model,
            feature_profile=metadata["feature_profile"],
            candidate_count=count,
            trials=args.trials,
            seed=args.seed + 10_007 * index,
            device=device,
            time_fraction=args.time_fraction,
            closure_candidate_fraction=args.closure_candidate_fraction,
            closure_degree_bias_power=args.closure_degree_bias_power,
        )
        for index, count in enumerate(args.candidate_counts)
    ]
    payload = {
        "format": "dihug-switch-candidate-ranking-diagnostic-v1",
        "graph": str(args.graph),
        "checkpoint": str(args.model_checkpoint),
        "feature_profile": metadata["feature_profile"],
        "time_fraction": args.time_fraction,
        "closure_candidate_fraction": args.closure_candidate_fraction,
        "closure_degree_bias_power": args.closure_degree_bias_power,
        "sampling_uses_clean_adjacency": False,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    for row in rows:
        print(
            "[SWITCH-RANK-DIAG] "
            f"c={row['candidate_count']} "
            f"coverage={row['positive_candidate_coverage_triangle']:.3f} "
            f"selected_tri={row['triangle']['selected_mean']:.3f} "
            f"oracle_tri={row['triangle']['oracle_mean']:.3f} "
            f"tri_regret={row['triangle']['normalized_regret']:.3f} "
            f"selected_cluster={row['clustering_sum']['selected_mean']:.5f} "
            f"oracle_cluster={row['clustering_sum']['oracle_mean']:.5f} "
            f"corr={row['score_clustering_correlation']:.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
