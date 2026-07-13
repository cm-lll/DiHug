from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .features import FeatureCounter, FeatureSelection, GraphletFeatureExtractor
from .graph import Edge, TypedGraph


@dataclass(frozen=True)
class SwitchProposal:
    family: str
    indices: Tuple[int, ...]
    old_edges: Tuple[Edge, ...]
    new_edges: Tuple[Edge, ...]


@dataclass
class ChainSample:
    step: int
    energy: float
    features: Dict[str, int]
    metrics: Dict[str, float]


@dataclass
class ChainResult:
    graph: TypedGraph
    samples: List[ChainSample]
    attempted: int
    valid: int
    accepted: int

    @property
    def valid_rate(self) -> float:
        return float(self.valid) / max(self.attempted, 1)

    @property
    def acceptance_rate(self) -> float:
        return float(self.accepted) / max(self.valid, 1)


@dataclass
class ParallelTemperingResult:
    graph: TypedGraph
    samples: List[ChainSample]
    exchange_attempted: int
    exchange_accepted: int
    local_acceptance_rates: List[float]

    @property
    def exchange_rate(self) -> float:
        return float(self.exchange_accepted) / max(self.exchange_attempted, 1)


class LinearGraphletEnergy:
    def __init__(self, selection: FeatureSelection, weights: np.ndarray) -> None:
        weights = np.asarray(weights, dtype=np.float64)
        if weights.shape != (len(selection.keys),):
            raise ValueError("energy weights do not match selected features")
        self.selection = selection
        self.weights = weights

    def value(self, features: Mapping[str, int | float]) -> float:
        return float(np.dot(self.weights, self.selection.vector(features)))

    def delta(self, values: Mapping[str, int | float]) -> float:
        return float(np.dot(self.weights, self.selection.delta_vector(values)))

    def to_json_dict(self) -> dict:
        return {
            "selection": self.selection.to_json_dict(),
            "weights": self.weights.tolist(),
        }


def _legal_new_edges(
    graph: TypedGraph,
    family: str,
    old_edges: Sequence[Edge],
    new_edges: Sequence[Edge],
) -> bool:
    oriented = tuple(graph.orient_edge(family, edge) for edge in new_edges)
    if oriented[0] == oriented[1]:
        return False
    old_canonical = {graph.canonical_edge(*edge) for edge in old_edges}
    seen = set()
    for edge in oriented:
        canonical = graph.canonical_edge(*edge)
        if canonical in seen:
            return False
        seen.add(canonical)
        existing = graph.edge_family.get(canonical)
        if existing is not None and canonical not in old_canonical:
            return False
    return True


def propose_switch(
    graph: TypedGraph, family: str, rng: np.random.Generator
) -> Optional[SwitchProposal]:
    edges = graph.family_edges[family]
    if len(edges) < 2:
        return None
    first, second = sorted(rng.choice(len(edges), size=2, replace=False).tolist())
    edge_a = edges[first]
    edge_b = edges[second]
    spec = graph.family_specs[family]
    if spec.bipartite:
        a, b = edge_a
        c, d = edge_b
        if a == c or b == d:
            return None
        proposed = ((a, d), (c, b))
    else:
        a, b = edge_a
        c, d = edge_b
        if len({a, b, c, d}) < 4:
            return None
        if bool(rng.integers(0, 2)):
            proposed = ((a, c), (b, d))
        else:
            proposed = ((a, d), (b, c))
    try:
        proposed = tuple(graph.orient_edge(family, edge) for edge in proposed)
    except ValueError:
        return None
    old = (edge_a, edge_b)
    if not _legal_new_edges(graph, family, old, proposed):
        return None
    return SwitchProposal(
        family=family,
        indices=(first, second),
        old_edges=old,
        new_edges=proposed,
    )


def propose_curveball(
    graph: TypedGraph, family: str, rng: np.random.Generator
) -> Optional[SwitchProposal]:
    """Symmetric multi-edge trade for a bipartite relation family."""

    spec = graph.family_specs[family]
    if not spec.bipartite or len(spec.src_nodes) < 2:
        return None
    left, right = sorted(
        rng.choice(np.asarray(spec.src_nodes), size=2, replace=False).tolist()
    )
    right_type = int(spec.dst_type)
    left_neighbors = {
        node
        for node in graph.adjacency[left]
        if int(graph.node_types[node]) == right_type
    }
    right_neighbors = {
        node
        for node in graph.adjacency[right]
        if int(graph.node_types[node]) == right_type
    }
    left_only = left_neighbors - right_neighbors
    right_only = right_neighbors - left_neighbors
    trade_pool = sorted(left_only | right_only)
    if not left_only or not right_only:
        return None

    left_count = len(left_only)
    chosen = set(
        rng.choice(np.asarray(trade_pool), size=left_count, replace=False).tolist()
    )
    if chosen == left_only:
        return None
    new_left = chosen
    new_right = set(trade_pool) - chosen
    old_edges = tuple(
        [graph.orient_edge(family, (left, node)) for node in sorted(left_only)]
        + [graph.orient_edge(family, (right, node)) for node in sorted(right_only)]
    )
    new_edges = tuple(
        [graph.orient_edge(family, (left, node)) for node in sorted(new_left)]
        + [graph.orient_edge(family, (right, node)) for node in sorted(new_right)]
    )
    edge_positions = {
        edge: index for index, edge in enumerate(graph.family_edges[family])
    }
    indices = tuple(edge_positions[edge] for edge in old_edges)
    if not _legal_new_edges(graph, family, old_edges, new_edges):
        return None
    return SwitchProposal(
        family=family,
        indices=indices,
        old_edges=old_edges,
        new_edges=new_edges,
    )


def graph_metrics(graph: TypedGraph) -> Dict[str, float]:
    degree = graph.degrees().astype(np.float64)
    triangles_per_node = np.zeros(graph.num_nodes, dtype=np.int64)
    triangles = 0
    for u in range(graph.num_nodes):
        for v in graph.adjacency[u]:
            if v <= u:
                continue
            for w in graph.adjacency[u] & graph.adjacency[v]:
                if w <= v:
                    continue
                triangles += 1
                triangles_per_node[u] += 1
                triangles_per_node[v] += 1
                triangles_per_node[w] += 1

    local_denominator = degree * (degree - 1.0) / 2.0
    local_clustering = np.divide(
        triangles_per_node,
        local_denominator,
        out=np.zeros_like(degree),
        where=local_denominator > 0,
    )
    wedges = float(local_denominator.sum())
    transitivity = 0.0 if wedges <= 0 else float(3.0 * triangles / wedges)

    visited = np.zeros(graph.num_nodes, dtype=bool)
    lcc = 0
    for start in range(graph.num_nodes):
        if visited[start]:
            continue
        stack = [start]
        visited[start] = True
        size = 0
        while stack:
            node = stack.pop()
            size += 1
            for other in graph.adjacency[node]:
                if not visited[other]:
                    visited[other] = True
                    stack.append(other)
        lcc = max(lcc, size)

    left_degree = []
    right_degree = []
    for u, v in graph.edge_family:
        left_degree.extend((degree[u], degree[v]))
        right_degree.extend((degree[v], degree[u]))
    if len(left_degree) < 2 or np.std(left_degree) == 0 or np.std(right_degree) == 0:
        assortativity = 0.0
    else:
        assortativity = float(np.corrcoef(left_degree, right_degree)[0, 1])
    return {
        "edges": float(graph.num_edges),
        "triangles": float(triangles),
        "clustering": float(local_clustering.mean()),
        "transitivity": transitivity,
        "lcc": float(lcc),
        "assortativity": assortativity,
        "sum_degree_sq": float(np.square(degree).sum()),
        "max_degree": float(degree.max(initial=0.0)),
    }


def run_switch_chain(
    initial_graph: TypedGraph,
    extractor: GraphletFeatureExtractor,
    steps: int,
    burn_in: int,
    thin: int,
    seed: int,
    energy: Optional[LinearGraphletEnergy] = None,
    verify_every: int = 0,
    curveball_probability: float = 0.0,
) -> ChainResult:
    graph = initial_graph.clone()
    rng = np.random.default_rng(int(seed))
    current_features = extractor.extract(graph)
    current_energy = 0.0 if energy is None else energy.value(current_features)
    families = [
        name for name, edges in graph.family_edges.items() if len(edges) >= 2
    ]
    if not families:
        raise ValueError("no family contains enough edges for a switch")
    family_weights = np.asarray(
        [len(graph.family_edges[name]) for name in families], dtype=np.float64
    )
    family_weights /= family_weights.sum()
    if not 0.0 <= float(curveball_probability) <= 1.0:
        raise ValueError("curveball_probability must be in [0, 1]")

    attempted = valid = accepted = 0
    samples: List[ChainSample] = []
    for step in range(1, int(steps) + 1):
        attempted += 1
        family = str(rng.choice(families, p=family_weights))
        use_curveball = (
            float(curveball_probability) > 0.0
            and graph.family_specs[family].bipartite
            and rng.random() < float(curveball_probability)
        )
        proposal = (
            propose_curveball(graph, family, rng)
            if use_curveball
            else propose_switch(graph, family, rng)
        )
        if proposal is not None:
            valid += 1
            delta = extractor.switch_delta(
                graph,
                proposal.family,
                proposal.indices,
                proposal.old_edges,
                proposal.new_edges,
            )
            delta_energy = 0.0 if energy is None else energy.delta(delta)
            accept = delta_energy >= 0.0 or math.log(
                max(rng.random(), 1.0e-300)
            ) < delta_energy
            if accept:
                accepted += 1
                extractor.apply_delta(current_features, delta)
                current_energy += delta_energy
            else:
                graph.replace_edges(
                    proposal.family,
                    proposal.indices,
                    proposal.new_edges,
                    proposal.old_edges,
                )

        if verify_every > 0 and step % int(verify_every) == 0:
            exact = extractor.extract(graph)
            if exact != current_features:
                keys = set(exact) | set(current_features)
                mismatch = {
                    key: (current_features.get(key, 0), exact.get(key, 0))
                    for key in keys
                    if current_features.get(key, 0) != exact.get(key, 0)
                }
                raise AssertionError(
                    f"incremental graphlet state drifted: {list(mismatch.items())[:10]}"
                )
            if energy is not None:
                exact_energy = energy.value(exact)
                if not math.isclose(exact_energy, current_energy, abs_tol=1.0e-8):
                    raise AssertionError(
                        f"incremental energy drifted: {current_energy} != {exact_energy}"
                    )
            graph.validate()

        if step >= int(burn_in) and (step - int(burn_in)) % max(int(thin), 1) == 0:
            samples.append(
                ChainSample(
                    step=step,
                    energy=float(current_energy),
                    features=dict(current_features),
                    metrics=graph_metrics(graph),
                )
            )

    graph.validate()
    return ChainResult(
        graph=graph,
        samples=samples,
        attempted=attempted,
        valid=valid,
        accepted=accepted,
    )


def run_parallel_tempering(
    initial_graph: TypedGraph,
    energy: LinearGraphletEnergy,
    temperatures: Sequence[float],
    rounds: int,
    local_steps: int,
    seed: int,
    initialization_steps: int = 2_000,
    verify_every: int = 0,
    curveball_probability: float = 0.0,
) -> ParallelTemperingResult:
    """Sample the frozen energy with exact replica-exchange MCMC.

    ``temperatures`` are inverse temperatures (beta), must start at zero and
    end at one. Local kernels and replica exchanges each satisfy detailed
    balance for the joint tempered distribution; the beta=1 marginal is the
    requested graphlet-energy model.
    """

    betas = np.asarray(temperatures, dtype=np.float64)
    if betas.ndim != 1 or betas.size < 2:
        raise ValueError("parallel tempering needs at least two temperatures")
    if not np.all(np.diff(betas) > 0.0):
        raise ValueError("temperatures must be strictly increasing")
    if not math.isclose(float(betas[0]), 0.0, abs_tol=1.0e-12):
        raise ValueError("the first inverse temperature must be zero")
    if not math.isclose(float(betas[-1]), 1.0, abs_tol=1.0e-12):
        raise ValueError("the final inverse temperature must be one")
    if int(rounds) <= 0 or int(local_steps) <= 0:
        raise ValueError("tempering rounds and local steps must be positive")

    rng = np.random.default_rng(int(seed))
    replicas = []
    for replica_id in range(len(betas)):
        if int(initialization_steps) > 0:
            randomized = run_switch_chain(
                initial_graph,
                GraphletFeatureExtractor(initial_graph),
                steps=int(initialization_steps),
                burn_in=int(initialization_steps),
                thin=max(int(initialization_steps), 1),
                seed=int(seed) + 7919 * replica_id,
                energy=None,
                verify_every=int(verify_every),
                curveball_probability=float(curveball_probability),
            ).graph
        else:
            randomized = initial_graph.clone()
        replicas.append(randomized)

    exchange_attempted = exchange_accepted = 0
    local_valid = np.zeros(len(betas), dtype=np.float64)
    local_accepted = np.zeros(len(betas), dtype=np.float64)
    samples: List[ChainSample] = []
    for round_id in range(int(rounds)):
        for replica_id, beta in enumerate(betas.tolist()):
            scaled = LinearGraphletEnergy(
                energy.selection, energy.weights * float(beta)
            )
            result = run_switch_chain(
                replicas[replica_id],
                GraphletFeatureExtractor(
                    replicas[replica_id], tracked_keys=energy.selection.keys
                ),
                steps=int(local_steps),
                burn_in=int(local_steps),
                thin=max(int(local_steps), 1),
                seed=int(seed) + 1_000_003 * round_id + 1009 * replica_id,
                energy=scaled,
                verify_every=int(verify_every),
                curveball_probability=float(curveball_probability),
            )
            replicas[replica_id] = result.graph
            local_valid[replica_id] += result.valid
            local_accepted[replica_id] += result.accepted

        features = [
            GraphletFeatureExtractor(
                graph, tracked_keys=energy.selection.keys
            ).extract(graph)
            for graph in replicas
        ]
        base_energy = [energy.value(values) for values in features]
        parity = round_id % 2
        for left in range(parity, len(betas) - 1, 2):
            right = left + 1
            exchange_attempted += 1
            log_ratio = float(betas[left] - betas[right]) * (
                base_energy[right] - base_energy[left]
            )
            if log_ratio >= 0.0 or math.log(max(rng.random(), 1.0e-300)) < log_ratio:
                exchange_accepted += 1
                replicas[left], replicas[right] = replicas[right], replicas[left]
                features[left], features[right] = features[right], features[left]
                base_energy[left], base_energy[right] = (
                    base_energy[right],
                    base_energy[left],
                )

        samples.append(
            ChainSample(
                step=(round_id + 1) * int(local_steps),
                energy=float(base_energy[-1]),
                features=dict(features[-1]),
                metrics=graph_metrics(replicas[-1]),
            )
        )

    return ParallelTemperingResult(
        graph=replicas[-1],
        samples=samples,
        exchange_attempted=exchange_attempted,
        exchange_accepted=exchange_accepted,
        local_acceptance_rates=[
            float(accepted) / max(float(valid), 1.0)
            for accepted, valid in zip(local_accepted, local_valid)
        ],
    )
