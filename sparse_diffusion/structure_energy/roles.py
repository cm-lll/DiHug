from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping

import networkx as nx
import numpy as np

from .graph import Edge, TypedGraph


@dataclass
class RolePlan:
    """Anonymous per-node relation-degree plan.

    ``permutation[old_node]`` is the anonymous node receiving the complete
    relation-degree vector of ``old_node``. Keeping the vector intact preserves
    cross-family role correlations without retaining an adjacency matrix.
    """

    permutation: np.ndarray
    family_degrees: Dict[str, np.ndarray]

    def validate(self, source: TypedGraph) -> None:
        if self.permutation.shape != (source.num_nodes,):
            raise ValueError("role permutation has the wrong size")
        if set(self.permutation.tolist()) != set(range(source.num_nodes)):
            raise ValueError("role permutation is not bijective")
        for old_node, new_node in enumerate(self.permutation.tolist()):
            if source.node_types[old_node] != source.node_types[new_node]:
                raise ValueError("role permutation crossed node types")
        for family in source.family_specs:
            values = self.family_degrees.get(family)
            if values is None or values.shape != (source.num_nodes,):
                raise ValueError(f"missing or malformed degree plan for {family}")


def sample_role_plan(source: TypedGraph, seed: int) -> RolePlan:
    rng = np.random.default_rng(int(seed))
    permutation = np.arange(source.num_nodes, dtype=np.int64)
    for type_id in range(len(source.type_names)):
        nodes = np.flatnonzero(source.node_types == type_id)
        permutation[nodes] = rng.permutation(nodes)

    source_degrees = source.all_family_degrees()
    planned = {
        family: np.zeros(source.num_nodes, dtype=np.int64)
        for family in source.family_specs
    }
    for family, degree in source_degrees.items():
        planned[family][permutation] = degree
    plan = RolePlan(permutation=permutation, family_degrees=planned)
    plan.validate(source)
    return plan


def permuted_witness(source: TypedGraph, plan: RolePlan) -> TypedGraph:
    """Return the proof witness; never use this as a generation algorithm."""

    plan.validate(source)
    edges = {
        family: [
            (int(plan.permutation[u]), int(plan.permutation[v]))
            for u, v in source.family_edges[family]
        ]
        for family in source.family_specs
    }
    witness = TypedGraph(
        source.node_types.copy(),
        source.type_names,
        source.family_specs,
        edges,
    )
    validate_role_realization(witness, plan)
    return witness


def realize_role_plan(source: TypedGraph, plan: RolePlan) -> TypedGraph:
    """Construct a legal graph from degrees only, without source adjacency."""

    plan.validate(source)
    realized_edges: Dict[str, list[Edge]] = {}
    for family, spec in source.family_specs.items():
        degree = plan.family_degrees[family]
        if spec.bipartite:
            left = list(spec.src_nodes)
            right = list(spec.dst_nodes)
            left_degree = [int(degree[node]) for node in left]
            right_degree = [int(degree[node]) for node in right]
            if sum(left_degree) != sum(right_degree):
                raise ValueError(
                    f"bipartite degree sums disagree for {family}: "
                    f"{sum(left_degree)} != {sum(right_degree)}"
                )
            local = nx.algorithms.bipartite.havel_hakimi_graph(
                left_degree,
                right_degree,
                create_using=nx.Graph(),
            )
            split = len(left)
            realized_edges[family] = [
                (left[u], right[v - split]) if u < split else (left[v], right[u - split])
                for u, v in local.edges()
            ]
        else:
            nodes = list(spec.src_nodes)
            sequence = [int(degree[node]) for node in nodes]
            if not nx.is_graphical(sequence, method="eg"):
                raise ValueError(f"nongraphical degree sequence for {family}")
            # NetworkX 2.8 compresses zero-degree positions internally in its
            # Havel-Hakimi implementation. Pass only active nodes and map them
            # explicitly so degree roles remain attached to the intended IDs.
            active = [
                (node, value) for node, value in zip(nodes, sequence) if value > 0
            ]
            local = nx.havel_hakimi_graph(
                [value for _, value in active], create_using=nx.Graph()
            )
            realized_edges[family] = [
                (active[u][0], active[v][0]) for u, v in local.edges()
            ]

    realized = TypedGraph(
        source.node_types.copy(),
        source.type_names,
        source.family_specs,
        realized_edges,
    )
    validate_role_realization(realized, plan)
    return realized


def validate_role_realization(graph: TypedGraph, plan: RolePlan) -> None:
    graph.validate()
    for family, target in plan.family_degrees.items():
        actual = graph.family_degrees(family)
        if not np.array_equal(actual, target):
            mismatch = np.flatnonzero(actual != target)[:10].tolist()
            raise AssertionError(
                f"role realization mismatch for {family}; nodes={mismatch}"
            )
