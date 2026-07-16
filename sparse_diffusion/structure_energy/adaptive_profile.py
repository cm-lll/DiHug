from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from itertools import combinations_with_replacement
from typing import Dict

import numpy as np

from .graph import TypedGraph


@dataclass(frozen=True)
class AdaptiveMixedEditProfile:
    """Scale-normalized settings derived only from the training graph."""

    num_nodes: int
    num_edges: int
    active_families: int
    triangle_capable_schema: bool
    role_components: int
    role_coverage: float
    randomization_steps_per_edge: float
    trajectories: int
    forward_steps: int
    record_every: int
    training_examples: int
    epochs: int
    train_candidates: int
    sample_steps: int
    sample_candidates: int
    kernel: str = "rs"
    sample_actions: str = "checkpoint"
    feature_profile: str = "typed_closure_compact_v2"

    def to_dict(self) -> dict:
        return asdict(self)


def schema_supports_triangle(graph: TypedGraph) -> bool:
    """Return whether any typed three-cycle is legal under the schema."""

    supported_pairs = {
        tuple(sorted((int(spec.src_type), int(spec.dst_type))))
        for spec in graph.family_specs.values()
    }
    for left, middle, right in combinations_with_replacement(
        range(len(graph.type_names)), 3
    ):
        pairs = (
            tuple(sorted((left, middle))),
            tuple(sorted((middle, right))),
            tuple(sorted((left, right))),
        )
        if all(pair in supported_pairs for pair in pairs):
            return True
    return False


def _role_prototypes_for_coverage(
    graph: TypedGraph,
    coverage: float,
) -> Dict[int, int]:
    if not 0.0 < float(coverage) <= 1.0:
        raise ValueError("role coverage must be in (0, 1]")
    family_degrees = graph.all_family_degrees()
    required = {}
    for type_id in range(len(graph.type_names)):
        nodes = np.flatnonzero(graph.node_types == type_id)
        incident = tuple(
            family
            for family, spec in graph.family_specs.items()
            if type_id in (int(spec.src_type), int(spec.dst_type))
        )
        if len(nodes) == 0 or not incident:
            required[type_id] = 1
            continue
        rows = np.stack(
            [family_degrees[family][nodes] for family in incident], axis=1
        )
        _, counts = np.unique(rows, axis=0, return_counts=True)
        counts = np.sort(counts)[::-1]
        cumulative = np.cumsum(counts) / max(int(counts.sum()), 1)
        exact = int(np.searchsorted(cumulative, float(coverage), side="left") + 1)
        # The categorical codebook reserves one additional prototype for the
        # merged rare tail when common roles do not cover every observed row.
        required[type_id] = min(len(counts), exact + int(exact < len(counts)))
    return required


def select_role_component_budget(
    graph: TypedGraph,
    coverage: float = 0.95,
    minimum: int = 4,
    maximum: int = 512,
) -> int:
    """Choose a power-of-two codebook covering common type-local roles."""

    if int(minimum) <= 0 or int(maximum) < int(minimum):
        raise ValueError("invalid role component bounds")
    required = max(_role_prototypes_for_coverage(graph, coverage).values())
    rounded = 1 << max(int(math.ceil(math.log2(max(required, 1)))), 0)
    return int(np.clip(rounded, int(minimum), int(maximum)))


def build_adaptive_mixed_edit_profile(
    graph: TypedGraph,
    *,
    role_coverage: float = 0.95,
    noise_steps_per_edge: float = 2.0,
    target_optimizer_updates: int = 5000,
    batch_size: int = 64,
) -> AdaptiveMixedEditProfile:
    """Derive comparable training exposure across heterogeneous graph scales."""

    if graph.num_edges <= 0:
        raise ValueError("adaptive mixed-edit profile requires at least one edge")
    if float(noise_steps_per_edge) <= 0.0:
        raise ValueError("noise steps per edge must be positive")
    if int(target_optimizer_updates) <= 0 or int(batch_size) <= 0:
        raise ValueError("optimizer update budget and batch size must be positive")

    edge_count = int(graph.num_edges)
    forward_steps = int(
        np.clip(
            math.ceil(float(noise_steps_per_edge) * edge_count / 100.0) * 100,
            1000,
            50_000,
        )
    )
    snapshots_per_trajectory = int(np.clip(math.ceil(edge_count / 16.0), 100, 1000))
    record_every = max(1, forward_steps // snapshots_per_trajectory)
    snapshots = int(forward_steps // record_every)
    trajectories = 32
    training_examples = trajectories * snapshots
    epochs = int(
        np.clip(
            math.ceil(
                int(target_optimizer_updates)
                * int(batch_size)
                / max(training_examples, 1)
            ),
            20,
            100,
        )
    )
    return AdaptiveMixedEditProfile(
        num_nodes=int(graph.num_nodes),
        num_edges=edge_count,
        active_families=len(graph.family_specs),
        triangle_capable_schema=schema_supports_triangle(graph),
        role_components=select_role_component_budget(
            graph, coverage=float(role_coverage)
        ),
        role_coverage=float(role_coverage),
        randomization_steps_per_edge=2.0,
        trajectories=trajectories,
        forward_steps=forward_steps,
        record_every=record_every,
        training_examples=training_examples,
        epochs=epochs,
        train_candidates=16,
        sample_steps=forward_steps,
        sample_candidates=64,
    )
