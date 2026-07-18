from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence

import numpy as np

from .graph import TypedGraph
from .mixed_edit_diffusion import (
    EDIT_ACTIONS,
    MarginalGraphPrior,
    MixedKernelSchedule,
    apply_edit,
    edit_distance_improvement,
    inverse_edit,
    sample_edit_candidates,
    sample_mixed_forward_transition,
    sample_valid_edit,
    mixed_edit_state_vector,
)
from .switch_chain import graph_metrics


def _edge_disagreement(left: TypedGraph, right: TypedGraph) -> float:
    differing = 0
    union = 0
    for family in left.family_specs:
        left_edges = left.family_edge_sets[family]
        right_edges = right.family_edge_sets[family]
        differing += len(left_edges ^ right_edges)
        union += len(left_edges | right_edges)
    return float(differing) / max(int(union), 1)


def _relative_count_distance(left: TypedGraph, right: TypedGraph) -> float:
    difference = sum(
        abs(len(left.family_edges[family]) - len(right.family_edges[family]))
        for family in left.family_specs
    )
    scale = 0.5 * (left.num_edges + right.num_edges)
    return float(difference) / max(float(scale), 1.0)


def _relative_degree_distance(left: TypedGraph, right: TypedGraph) -> float:
    left_degree = left.degrees().astype(np.float64)
    right_degree = right.degrees().astype(np.float64)
    difference = float(np.abs(left_degree - right_degree).sum())
    scale = 0.5 * float(left_degree.sum() + right_degree.sum())
    return difference / max(scale, 1.0)


def _snapshot(left: TypedGraph, right: TypedGraph) -> Dict[str, float]:
    left_metrics = graph_metrics(left)
    right_metrics = graph_metrics(right)
    return {
        "edge_disagreement": _edge_disagreement(left, right),
        "family_count_distance": _relative_count_distance(left, right),
        "degree_distance": _relative_degree_distance(left, right),
        "left_edges": float(left.num_edges),
        "right_edges": float(right.num_edges),
        "left_triangles": float(left_metrics["triangles"]),
        "right_triangles": float(right_metrics["triangles"]),
        "left_clustering": float(left_metrics["clustering"]),
        "right_clustering": float(right_metrics["clustering"]),
    }


def run_forward_coupling_diagnostic(
    source: TypedGraph,
    steps: int,
    checkpoints: Sequence[int],
    replicas: int,
    seed: int,
    actions: Sequence[str] = EDIT_ACTIONS,
    schedule: Optional[MixedKernelSchedule] = None,
) -> Mapping:
    """Couple clean-start and prior-start forward chains with shared randomness.

    Re-seeding both marginal transitions at every step is a valid common-random-
    number coupling: each chain still receives the exact forward kernel, while
    equal random tapes expose whether the kernel forgets its initial state.
    """

    if int(steps) <= 0:
        raise ValueError("steps must be positive")
    if int(replicas) <= 0:
        raise ValueError("replicas must be positive")
    selected = tuple(sorted({0, int(steps), *(int(v) for v in checkpoints)}))
    if selected[0] < 0 or selected[-1] > int(steps):
        raise ValueError("checkpoints must be inside [0, steps]")
    schedule = schedule or MixedKernelSchedule()
    prior = MarginalGraphPrior.from_training_graph(source)
    samples: Dict[int, list] = {step: [] for step in selected}

    for replica in range(int(replicas)):
        left = source.clone()
        right = prior.sample(int(seed) + 1_000_003 * replica + 17)
        samples[0].append(_snapshot(left, right))
        for step in range(1, int(steps) + 1):
            shared_seed = int(seed) + 10_000_019 * replica + 97_409 * step
            time_fraction = float(step) / float(steps)
            for graph in (left, right):
                rng = np.random.default_rng(shared_seed)
                proposal = sample_mixed_forward_transition(
                    graph,
                    prior,
                    rng,
                    time_fraction,
                    schedule,
                    actions=actions,
                )
                apply_edit(graph, proposal)
            if step in samples:
                samples[step].append(_snapshot(left, right))

    trajectory = []
    for step in selected:
        row = {"step": int(step), "time_fraction": float(step) / float(steps)}
        keys = samples[step][0]
        for key in keys:
            values = np.asarray([item[key] for item in samples[step]], dtype=float)
            row[f"{key}_mean"] = float(values.mean())
            row[f"{key}_std"] = float(values.std(ddof=0))
        trajectory.append(row)
    initial = trajectory[0]["edge_disagreement_mean"]
    final = trajectory[-1]["edge_disagreement_mean"]
    return {
        "format": "dihug-forward-coupling-diagnostic-v1",
        "interpretation": "diagnostic_not_a_mixing_time_proof",
        "steps": int(steps),
        "replicas": int(replicas),
        "actions": list(actions),
        "trajectory": trajectory,
        "edge_disagreement_contraction": float(final) / max(float(initial), 1e-12),
    }


def audit_state_summary_aliasing(
    source: TypedGraph,
    trials: int,
    seed: int,
    time_fraction: float = 0.5,
) -> Mapping:
    """Measure structural changes hidden by the target-free global summary.

    A legal 2-switch preserves every family count and every endpoint degree.
    Therefore the current global state summary is identical before and after the
    switch even when adjacency-level motif statistics change.
    """

    if int(trials) <= 0:
        raise ValueError("trials must be positive")
    rng = np.random.default_rng(int(seed))
    accepted = 0
    summary_collisions = 0
    triangle_changes = []
    clustering_changes = []
    attempts = 0
    max_attempts = max(256, int(trials) * 32)
    while accepted < int(trials) and attempts < max_attempts:
        attempts += 1
        proposal = sample_valid_edit(source, rng, actions=("switch",))
        if proposal is None:
            continue
        changed = source.clone()
        before_state = mixed_edit_state_vector(source, time_fraction)
        before_metrics = graph_metrics(source)
        apply_edit(changed, proposal)
        after_state = mixed_edit_state_vector(changed, time_fraction)
        after_metrics = graph_metrics(changed)
        accepted += 1
        summary_collisions += int(np.array_equal(before_state, after_state))
        triangle_changes.append(
            float(after_metrics["triangles"] - before_metrics["triangles"])
        )
        clustering_changes.append(
            float(after_metrics["clustering"] - before_metrics["clustering"])
        )
    if accepted == 0:
        raise RuntimeError("no legal switch was available for the aliasing audit")
    triangle = np.asarray(triangle_changes, dtype=float)
    clustering = np.asarray(clustering_changes, dtype=float)
    return {
        "format": "dihug-state-summary-aliasing-v1",
        "interpretation": "global_summary_collision_not_full_feature_collision",
        "requested_trials": int(trials),
        "accepted_trials": int(accepted),
        "attempts": int(attempts),
        "summary_collision_rate": float(summary_collisions) / float(accepted),
        "motif_change_rate": float(
            np.count_nonzero((triangle != 0.0) | (clustering != 0.0))
        )
        / float(accepted),
        "triangle_delta_mean": float(triangle.mean()),
        "triangle_delta_abs_mean": float(np.abs(triangle).mean()),
        "triangle_delta_max_abs": float(np.abs(triangle).max(initial=0.0)),
        "clustering_delta_abs_mean": float(np.abs(clustering).mean()),
    }


def _proposal_key(proposal) -> tuple:
    return (
        proposal.action,
        "" if proposal.action == "noop" else proposal.family,
        tuple(sorted(proposal.old_edges)),
        tuple(sorted(proposal.new_edges)),
    )


def run_free_proposal_coverage_diagnostic(
    source: TypedGraph,
    time_fractions: Sequence[float],
    trials_per_time: int,
    forward_steps: int,
    candidate_counts: Sequence[int],
    seed: int,
    actions: Sequence[str] = EDIT_ACTIONS,
    schedule: Optional[MixedKernelSchedule] = None,
) -> Mapping:
    """Audit inference-time candidates without injecting the clean inverse."""

    if int(trials_per_time) <= 0 or int(forward_steps) <= 0:
        raise ValueError("trials_per_time and forward_steps must be positive")
    counts = tuple(sorted({int(value) for value in candidate_counts}))
    if not counts or counts[0] < 2:
        raise ValueError("candidate counts must contain values >= 2")
    fractions = tuple(float(value) for value in time_fractions)
    if not fractions or any(value <= 0.0 or value > 1.0 for value in fractions):
        raise ValueError("time fractions must be inside (0, 1]")
    prior = MarginalGraphPrior.from_training_graph(source)
    schedule = schedule or MixedKernelSchedule()
    output = {}

    for time_index, fraction in enumerate(fractions):
        per_count = {
            count: {
                "improving": 0,
                "best_improvement": [],
                "inverse_hits": 0,
                "inverse_active_support": 0,
                "best_actions": {action: 0 for action in ("noop",) + tuple(actions)},
                "represented_actions": {action: 0 for action in ("noop",) + tuple(actions)},
            }
            for count in counts
        }
        for trial in range(int(trials_per_time)):
            rng = np.random.default_rng(
                int(seed) + 10_000_019 * time_index + 1_000_003 * trial
            )
            graph = source.clone()
            last = None
            steps = max(1, int(round(fraction * int(forward_steps))))
            for step in range(1, steps + 1):
                last = sample_mixed_forward_transition(
                    graph,
                    prior,
                    rng,
                    float(step) / float(forward_steps),
                    schedule,
                    actions=actions,
                )
                apply_edit(graph, last)
            candidates, _ = sample_edit_candidates(
                graph,
                rng,
                counts[-1],
                actions=actions,
                positive=None,
            )
            inverse = inverse_edit(last)
            inverse_key = _proposal_key(inverse)
            for count in counts:
                subset = candidates[:count]
                improvements = np.asarray(
                    [edit_distance_improvement(graph, source, item) for item in subset],
                    dtype=np.int64,
                )
                best_index = int(improvements.argmax())
                best = int(improvements[best_index])
                stats = per_count[count]
                stats["improving"] += int(best > 0)
                stats["best_improvement"].append(best)
                stats["best_actions"][subset[best_index].action] += 1
                represented = {item.action for item in subset}
                for action in represented:
                    stats["represented_actions"][action] += 1
                if inverse.action != "noop":
                    stats["inverse_active_support"] += 1
                    stats["inverse_hits"] += int(
                        any(_proposal_key(item) == inverse_key for item in subset)
                    )

        serialized = {}
        for count in counts:
            stats = per_count[count]
            best = np.asarray(stats.pop("best_improvement"), dtype=np.float64)
            support = int(stats.pop("inverse_active_support"))
            hits = int(stats.pop("inverse_hits"))
            serialized[str(count)] = {
                "trials": int(trials_per_time),
                "improving_candidate_rate": float(stats.pop("improving"))
                / float(trials_per_time),
                "best_edit_improvement_mean": float(best.mean()),
                "best_edit_improvement_std": float(best.std(ddof=0)),
                "active_inverse_support": support,
                "active_inverse_coverage": float(hits) / max(support, 1),
                "best_action_counts": stats["best_actions"],
                "action_representation_rate": {
                    action: float(value) / float(trials_per_time)
                    for action, value in stats["represented_actions"].items()
                },
            }
        output[f"{fraction:.3f}"] = serialized

    return {
        "format": "dihug-free-proposal-coverage-v1",
        "diagnostic_only": True,
        "clean_graph_usage": "edge_edit_distance_evaluation_only",
        "positive_injection": False,
        "forward_steps": int(forward_steps),
        "trials_per_time": int(trials_per_time),
        "candidate_counts": list(counts),
        "time_fractions": list(fractions),
        "results": output,
    }
