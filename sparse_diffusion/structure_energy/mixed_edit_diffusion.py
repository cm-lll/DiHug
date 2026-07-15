from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from .graph import Edge, FamilySpec, TypedGraph
from .switch_chain import SwitchProposal, graph_metrics, propose_switch
from .switch_diffusion import (
    SwitchDenoiserMLP,
    SwitchDiffusionDataset,
    SwitchDiffusionTrainReport,
)


EDIT_ACTIONS = ("birth", "death", "relocation", "switch")
SCORER_ACTIONS = ("noop",) + EDIT_ACTIONS


@dataclass(frozen=True)
class MarginalGraphPrior:
    """Learned marginal prior without exact edge counts or node roles."""

    type_names: Tuple[str, ...]
    type_counts: Tuple[int, ...]
    family_names: Tuple[str, ...]
    family_endpoints: Mapping[str, Tuple[int, int]]
    family_probabilities: Mapping[str, float]

    @classmethod
    def from_training_graph(cls, graph: TypedGraph) -> "MarginalGraphPrior":
        graph.validate_disjoint_family_domains()
        family_names = tuple(sorted(graph.family_specs))
        probabilities = {}
        endpoints = {}
        for family in family_names:
            spec = graph.family_specs[family]
            capacity = family_domain_size(spec)
            probabilities[family] = float(len(graph.family_edges[family])) / max(
                capacity, 1
            )
            endpoints[family] = (int(spec.src_type), int(spec.dst_type))
        return cls(
            type_names=tuple(graph.type_names),
            type_counts=tuple(
                int(np.count_nonzero(graph.node_types == type_id))
                for type_id in range(len(graph.type_names))
            ),
            family_names=family_names,
            family_endpoints=endpoints,
            family_probabilities=probabilities,
        )

    def schema_graph(self) -> TypedGraph:
        offsets = np.cumsum((0,) + self.type_counts)
        node_types = np.concatenate(
            [
                np.full(count, type_id, dtype=np.int64)
                for type_id, count in enumerate(self.type_counts)
            ]
        )
        nodes_by_type = {
            type_id: tuple(range(int(offsets[type_id]), int(offsets[type_id + 1])))
            for type_id in range(len(self.type_names))
        }
        specs = {
            family: FamilySpec(
                family,
                int(self.family_endpoints[family][0]),
                int(self.family_endpoints[family][1]),
                nodes_by_type[int(self.family_endpoints[family][0])],
                nodes_by_type[int(self.family_endpoints[family][1])],
            )
            for family in self.family_names
        }
        return TypedGraph(
            node_types,
            self.type_names,
            specs,
            {family: [] for family in self.family_names},
        )

    def sample(self, seed: int) -> TypedGraph:
        """Sample independent family edges, equivalently M_f ~ Binomial(C_f,p_f)."""

        graph = self.schema_graph()
        rng = np.random.default_rng(int(seed))
        for family in self.family_names:
            spec = graph.family_specs[family]
            capacity = family_domain_size(spec)
            probability = float(self.family_probabilities[family])
            count = int(rng.binomial(capacity, probability))
            if count == 0:
                continue
            ranks = rng.choice(capacity, size=count, replace=False)
            for rank in np.sort(ranks).tolist():
                graph.add_edge(family, domain_edge_from_rank(spec, int(rank)))
        graph.validate()
        return graph

    def to_json_dict(self) -> dict:
        return {
            "format": "dihug-marginal-graph-prior-v1",
            "type_names": list(self.type_names),
            "type_counts": list(self.type_counts),
            "family_names": list(self.family_names),
            "family_endpoints": {
                family: list(self.family_endpoints[family])
                for family in self.family_names
            },
            "family_probabilities": {
                family: float(self.family_probabilities[family])
                for family in self.family_names
            },
            "inference_contract": {
                "node_type_counts": True,
                "family_schema": True,
                "family_marginal_probabilities": True,
                "exact_family_edge_counts": False,
                "relation_degree_rows": False,
                "graphlet_target": False,
                "training_adjacency": False,
            },
        }

    @classmethod
    def from_json_dict(cls, payload: Mapping) -> "MarginalGraphPrior":
        if payload.get("format") != "dihug-marginal-graph-prior-v1":
            raise ValueError("unsupported marginal graph prior format")
        forbidden = {
            "family_edge_counts",
            "role_rows_by_type",
            "selection",
            "target_vector",
            "weights",
        }
        leaked = forbidden.intersection(payload)
        if leaked:
            raise ValueError(f"marginal prior contains forbidden fields: {leaked}")
        contract = payload.get("inference_contract", {})
        required_false = (
            "exact_family_edge_counts",
            "relation_degree_rows",
            "graphlet_target",
            "training_adjacency",
        )
        if any(contract.get(key) is not False for key in required_false):
            raise ValueError(f"unsafe marginal prior contract: {contract}")
        return cls(
            type_names=tuple(str(value) for value in payload["type_names"]),
            type_counts=tuple(int(value) for value in payload["type_counts"]),
            family_names=tuple(str(value) for value in payload["family_names"]),
            family_endpoints={
                str(family): (int(value[0]), int(value[1]))
                for family, value in payload["family_endpoints"].items()
            },
            family_probabilities={
                str(family): float(value)
                for family, value in payload["family_probabilities"].items()
            },
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_json_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "MarginalGraphPrior":
        return cls.from_json_dict(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass(frozen=True)
class EditProposal:
    action: str
    family: str
    old_edges: Tuple[Edge, ...]
    new_edges: Tuple[Edge, ...]

    def __post_init__(self) -> None:
        expected = {
            "noop": (0, 0),
            "birth": (0, 1),
            "death": (1, 0),
            "relocation": (1, 1),
            "switch": (2, 2),
        }
        if self.action not in expected:
            raise ValueError(f"unknown edit action {self.action!r}")
        if (len(self.old_edges), len(self.new_edges)) != expected[self.action]:
            raise ValueError(f"malformed {self.action} proposal")


@dataclass(frozen=True)
class MixedKernelSchedule:
    """Time-inhomogeneous mixture; every component preserves the terminal prior."""

    resample_floor: float = 0.05
    relocation_floor: float = 0.05
    switch_floor: float = 0.05

    def weights(self, time_fraction: float) -> Mapping[str, float]:
        t = float(np.clip(time_fraction, 0.0, 1.0))
        raw = {
            "resample": self.resample_floor + t * t,
            "relocation": self.relocation_floor + 2.0 * t * (1.0 - t),
            "switch": self.switch_floor + (1.0 - t) * (1.0 - t),
        }
        total = sum(raw.values())
        return {key: value / total for key, value in raw.items()}


@dataclass(frozen=True)
class MixedEditSamplingResult:
    graph: TypedGraph
    trace: Sequence[Mapping[str, float]]
    action_counts: Mapping[str, int]
    proposed: int
    applied: int


def family_domain_size(spec: FamilySpec) -> int:
    if spec.bipartite:
        return len(spec.src_nodes) * len(spec.dst_nodes)
    return len(spec.src_nodes) * (len(spec.src_nodes) - 1) // 2


def domain_edge_from_rank(spec: FamilySpec, rank: int) -> Edge:
    capacity = family_domain_size(spec)
    if rank < 0 or rank >= capacity:
        raise IndexError(f"domain rank {rank} outside [0,{capacity})")
    if spec.bipartite:
        width = len(spec.dst_nodes)
        return int(spec.src_nodes[rank // width]), int(spec.dst_nodes[rank % width])
    n = len(spec.src_nodes)
    starts = np.arange(n, dtype=np.int64)
    starts = starts * (2 * n - starts - 1) // 2
    left_index = int(np.searchsorted(starts, rank, side="right") - 1)
    offset = int(rank - starts[left_index])
    right_index = left_index + 1 + offset
    left = int(spec.src_nodes[left_index])
    right = int(spec.src_nodes[right_index])
    return (left, right) if left < right else (right, left)


def sample_domain_edge(spec: FamilySpec, rng: np.random.Generator) -> Edge:
    return domain_edge_from_rank(spec, int(rng.integers(family_domain_size(spec))))


def inverse_edit(proposal: EditProposal) -> EditProposal:
    inverse_action = {
        "noop": "noop",
        "birth": "death",
        "death": "birth",
        "relocation": "relocation",
        "switch": "switch",
    }[proposal.action]
    return EditProposal(
        inverse_action,
        proposal.family,
        proposal.new_edges,
        proposal.old_edges,
    )


def apply_edit(graph: TypedGraph, proposal: EditProposal) -> None:
    old_canonical = {graph.canonical_edge(*edge) for edge in proposal.old_edges}
    for edge in proposal.old_edges:
        if graph.family_of(*edge) != proposal.family:
            raise ValueError(f"missing old edge {proposal.family}:{edge}")
    seen_new = set()
    for edge in proposal.new_edges:
        oriented = graph.orient_edge(proposal.family, edge)
        canonical = graph.canonical_edge(*oriented)
        if canonical in seen_new:
            raise ValueError("proposal creates duplicate edges")
        seen_new.add(canonical)
        existing = graph.family_of(*oriented)
        if existing is not None and canonical not in old_canonical:
            raise ValueError(f"new edge already exists: {oriented}")
    removed = []
    try:
        for edge in proposal.old_edges:
            removed.append(graph.remove_edge(proposal.family, edge))
        for edge in proposal.new_edges:
            graph.add_edge(proposal.family, edge)
    except Exception:
        for edge in tuple(proposal.new_edges):
            if graph.family_of(*edge) == proposal.family:
                graph.remove_edge(proposal.family, edge)
        for edge in removed:
            if graph.family_of(*edge) is None:
                graph.add_edge(proposal.family, edge)
        raise


def marginal_log_probability(graph: TypedGraph, prior: MarginalGraphPrior) -> float:
    """Log probability under the independent typed-edge terminal prior."""

    value = 0.0
    for family in prior.family_names:
        probability = float(prior.family_probabilities[family])
        count = len(graph.family_edges[family])
        capacity = family_domain_size(graph.family_specs[family])
        if probability <= 0.0:
            if count:
                return -math.inf
            continue
        if probability >= 1.0:
            if count != capacity:
                return -math.inf
            continue
        value += count * math.log(probability)
        value += (capacity - count) * math.log1p(-probability)
    return float(value)


def mixed_transition_probability(
    graph: TypedGraph,
    prior: MarginalGraphPrior,
    proposal: EditProposal,
    time_fraction: float,
    schedule: MixedKernelSchedule,
) -> float:
    """Probability of one specific non-null proposal in the forward kernel."""

    weights = schedule.weights(time_fraction)
    if proposal.action == "noop":
        raise ValueError("noop probability is an aggregate over rejected micro moves")
    family_factor = 1.0 / max(len(prior.family_names), 1)
    family = proposal.family
    capacity = family_domain_size(graph.family_specs[family])
    if proposal.action == "birth":
        return (
            float(weights["resample"])
            * family_factor
            / capacity
            * float(prior.family_probabilities[family])
        )
    if proposal.action == "death":
        return (
            float(weights["resample"])
            * family_factor
            / capacity
            * (1.0 - float(prior.family_probabilities[family]))
        )
    if proposal.action == "relocation":
        edge_count = len(graph.family_edges[family])
        if edge_count <= 0:
            return 0.0
        return (
            float(weights["relocation"])
            * family_factor
            / edge_count
            / capacity
        )
    edge_count = len(graph.family_edges[family])
    if edge_count < 2:
        return 0.0
    pairing_factor = 1.0 if graph.family_specs[family].bipartite else 0.5
    return (
        float(weights["switch"])
        * family_factor
        / math.comb(edge_count, 2)
        * pairing_factor
    )


def _family_names(graph: TypedGraph) -> Tuple[str, ...]:
    return tuple(sorted(graph.family_specs))


def sample_mixed_forward_transition(
    graph: TypedGraph,
    prior: MarginalGraphPrior,
    rng: np.random.Generator,
    time_fraction: float,
    schedule: MixedKernelSchedule,
    actions: Sequence[str] = EDIT_ACTIONS,
) -> EditProposal:
    """One exact micro transition of the known forward kernel.

    Edge resampling is a Bernoulli heat-bath update. Relocation and switch are
    symmetric within a fixed family edge count. Therefore every component, and
    any time-dependent mixture of them, leaves the marginal prior invariant.
    """

    actions = tuple(str(action) for action in actions)
    if ("birth" in actions) != ("death" in actions):
        raise ValueError("birth and death must be enabled together")
    components = []
    if "birth" in actions:
        components.append("resample")
    if "relocation" in actions:
        components.append("relocation")
    if "switch" in actions:
        components.append("switch")
    if not components:
        raise ValueError("forward kernel has no enabled component")
    all_weights = schedule.weights(time_fraction)
    total = sum(all_weights[name] for name in components)
    weights = {name: all_weights[name] / total for name in components}
    component = str(
        rng.choice(tuple(weights), p=np.asarray(tuple(weights.values())))
    )
    family = str(rng.choice(_family_names(graph)))
    spec = graph.family_specs[family]
    if component == "resample":
        edge = sample_domain_edge(spec, rng)
        exists = graph.family_of(*edge) == family
        desired = bool(rng.random() < float(prior.family_probabilities[family]))
        if exists == desired:
            return EditProposal("noop", family, (), ())
        return EditProposal(
            "birth" if desired else "death",
            family,
            () if desired else (edge,),
            (edge,) if desired else (),
        )
    if component == "relocation":
        edges = graph.family_edges[family]
        if not edges:
            return EditProposal("noop", family, (), ())
        old_edge = edges[int(rng.integers(len(edges)))]
        new_edge = sample_domain_edge(spec, rng)
        if graph.has_edge(*new_edge):
            return EditProposal("noop", family, (), ())
        return EditProposal("relocation", family, (old_edge,), (new_edge,))
    proposal = propose_switch(graph, family, rng)
    if proposal is None:
        return EditProposal("noop", family, (), ())
    return EditProposal("switch", family, proposal.old_edges, proposal.new_edges)


def edge_edit_distance(graph: TypedGraph, clean: TypedGraph) -> int:
    if set(graph.family_specs) != set(clean.family_specs):
        raise ValueError("graphs use different family schemas")
    return sum(
        len(graph.family_edge_sets[family] ^ clean.family_edge_sets[family])
        for family in graph.family_specs
    )


def edit_distance_improvement(
    graph: TypedGraph, clean: TypedGraph, proposal: EditProposal
) -> int:
    clean_edges = clean.family_edge_sets[proposal.family]
    improvement = 0
    for edge in proposal.old_edges:
        oriented = graph.orient_edge(proposal.family, edge)
        improvement += -1 if oriented in clean_edges else 1
    for edge in proposal.new_edges:
        oriented = graph.orient_edge(proposal.family, edge)
        improvement += 1 if oriented in clean_edges else -1
    return int(improvement)


def _sample_absent_edge(
    graph: TypedGraph,
    family: str,
    rng: np.random.Generator,
    max_attempts: int = 256,
) -> Optional[Edge]:
    spec = graph.family_specs[family]
    if len(graph.family_edges[family]) >= family_domain_size(spec):
        return None
    for _ in range(int(max_attempts)):
        edge = sample_domain_edge(spec, rng)
        if not graph.has_edge(*edge):
            return graph.orient_edge(family, edge)
    start = int(rng.integers(family_domain_size(spec)))
    for offset in range(family_domain_size(spec)):
        edge = domain_edge_from_rank(spec, (start + offset) % family_domain_size(spec))
        if not graph.has_edge(*edge):
            return graph.orient_edge(family, edge)
    return None


def sample_valid_edit(
    graph: TypedGraph,
    rng: np.random.Generator,
    actions: Sequence[str] = EDIT_ACTIONS,
    family: Optional[str] = None,
) -> Optional[EditProposal]:
    actions = tuple(str(action) for action in actions)
    unknown = set(actions) - set(EDIT_ACTIONS)
    if unknown:
        raise ValueError(f"unknown edit actions: {sorted(unknown)}")
    if family is not None and str(family) not in graph.family_specs:
        raise ValueError(f"unknown family {family!r}")
    for _ in range(max(64, 16 * len(actions))):
        action = str(rng.choice(actions))
        selected_family = (
            str(family)
            if family is not None
            else str(rng.choice(_family_names(graph)))
        )
        edges = graph.family_edges[selected_family]
        if action == "birth":
            new_edge = _sample_absent_edge(graph, selected_family, rng)
            if new_edge is not None:
                return EditProposal("birth", selected_family, (), (new_edge,))
        elif action == "death":
            if edges:
                old_edge = edges[int(rng.integers(len(edges)))]
                return EditProposal("death", selected_family, (old_edge,), ())
        elif action == "relocation":
            if edges:
                old_edge = edges[int(rng.integers(len(edges)))]
                new_edge = _sample_absent_edge(graph, selected_family, rng)
                if new_edge is not None:
                    return EditProposal(
                        "relocation", selected_family, (old_edge,), (new_edge,)
                    )
        else:
            proposal = propose_switch(graph, selected_family, rng)
            if proposal is not None:
                return EditProposal(
                    "switch",
                    selected_family,
                    proposal.old_edges,
                    proposal.new_edges,
                )
    return None


def sample_edit_candidates(
    graph: TypedGraph,
    rng: np.random.Generator,
    count: int,
    actions: Sequence[str] = EDIT_ACTIONS,
    positive: Optional[EditProposal] = None,
) -> Tuple[List[EditProposal], int]:
    if int(count) < 2:
        raise ValueError("candidate count must be at least two")
    candidates: List[EditProposal] = []
    keys = set()

    def add(proposal: EditProposal) -> None:
        key = (
            proposal.action,
            "" if proposal.action == "noop" else proposal.family,
            tuple(sorted(proposal.old_edges)),
            tuple(sorted(proposal.new_edges)),
        )
        if key not in keys:
            keys.add(key)
            candidates.append(proposal)

    if positive is not None:
        add(positive)
    add(EditProposal("noop", _family_names(graph)[0], (), ()))
    attempts = 0
    while len(candidates) < int(count) and attempts < int(count) * 256:
        attempts += 1
        proposal = sample_valid_edit(graph, rng, actions=actions)
        if proposal is not None:
            add(proposal)
    if len(candidates) != int(count):
        raise RuntimeError(f"sampled only {len(candidates)}/{count} legal edits")
    target = 0
    if positive is not None:
        order = rng.permutation(len(candidates))
        candidates = [candidates[index] for index in order.tolist()]
        target = int(np.flatnonzero(order == 0)[0])
    return candidates, target


def _edge_slots(proposal: EditProposal) -> Tuple[Optional[Edge], ...]:
    old = tuple(proposal.old_edges) + (None,) * (2 - len(proposal.old_edges))
    new = tuple(proposal.new_edges) + (None,) * (2 - len(proposal.new_edges))
    return old + new


def mixed_edit_state_vector(graph: TypedGraph, time_fraction: float) -> np.ndarray:
    """Target-free whole-graph summary shared by all edit candidates."""

    degrees = graph.degrees().astype(np.float64)
    log_scale = max(float(np.log1p(graph.num_nodes)), 1.0)
    total_capacity = sum(family_domain_size(spec) for spec in graph.family_specs.values())
    values = [
        float(time_fraction),
        float(graph.num_edges) / max(total_capacity, 1),
        float(np.log1p(graph.num_edges)) / log_scale,
    ]
    for type_id in range(len(graph.type_names)):
        selected = degrees[graph.node_types == type_id]
        logged = np.log1p(selected) / log_scale
        values.extend(
            (
                float(logged.mean()) if logged.size else 0.0,
                float(logged.std()) if logged.size else 0.0,
                float(logged.max()) if logged.size else 0.0,
                float(np.mean(selected == 0)) if selected.size else 0.0,
            )
        )
    for family in _family_names(graph):
        spec = graph.family_specs[family]
        family_degree = graph.family_degrees(family).astype(np.float64)
        source = family_degree[np.asarray(spec.src_nodes, dtype=np.int64)]
        target = family_degree[np.asarray(spec.dst_nodes, dtype=np.int64)]
        capacity = family_domain_size(spec)
        values.extend(
            (
                len(graph.family_edges[family]) / max(capacity, 1),
                float(np.log1p(source).mean() / log_scale),
                float(np.log1p(target).mean() / log_scale),
                float(np.log1p(source.max(initial=0.0)) / log_scale),
                float(np.log1p(target.max(initial=0.0)) / log_scale),
            )
        )
    return np.asarray(values, dtype=np.float32)


def mixed_edit_candidate_matrix(
    graph: TypedGraph,
    candidates: Sequence[EditProposal],
    time_fraction: float,
) -> np.ndarray:
    family_names = _family_names(graph)
    family_to_id = {name: index for index, name in enumerate(family_names)}
    action_to_id = {name: index for index, name in enumerate(SCORER_ACTIONS)}
    degrees = graph.degrees().astype(np.float64)
    family_degrees = {
        family: graph.family_degrees(family).astype(np.float64)
        for family in family_names
    }
    degree_scale = max(float(np.log1p(graph.num_nodes)), 1.0)
    state = mixed_edit_state_vector(graph, time_fraction)
    rows = []
    for proposal in candidates:
        action = np.zeros(len(SCORER_ACTIONS), dtype=np.float64)
        action[action_to_id[proposal.action]] = 1.0
        family = np.zeros(len(family_names), dtype=np.float64)
        if proposal.action != "noop":
            family[family_to_id[proposal.family]] = 1.0
        endpoint = []
        common = []
        for edge in _edge_slots(proposal):
            if edge is None:
                endpoint.extend((0.0, 0.0, 0.0, 0.0))
                common.append(0.0)
                continue
            u, v = int(edge[0]), int(edge[1])
            endpoint.extend(
                (
                    np.log1p(degrees[u]) / degree_scale,
                    np.log1p(degrees[v]) / degree_scale,
                    np.log1p(family_degrees[proposal.family][u]) / degree_scale,
                    np.log1p(family_degrees[proposal.family][v]) / degree_scale,
                )
            )
            common.append(
                len(graph.adjacency[u] & graph.adjacency[v])
                / max(float(np.sqrt(graph.num_nodes)), 1.0)
            )
        scalar = np.asarray(
            [
                float(time_fraction),
                0.0
                if proposal.action == "noop"
                else len(graph.family_edges[proposal.family])
                / max(family_domain_size(graph.family_specs[proposal.family]), 1),
                float(len(proposal.old_edges)),
                float(len(proposal.new_edges)),
            ],
            dtype=np.float64,
        )
        rows.append(
            np.concatenate(
                (action, family, endpoint, np.asarray(common), scalar, state)
            )
        )
    return np.asarray(rows, dtype=np.float32)


def build_mixed_edit_diffusion_dataset(
    source: TypedGraph,
    trajectories: int,
    forward_steps: int,
    record_every: int,
    candidates_per_example: int,
    seed: int,
    schedule: Optional[MixedKernelSchedule] = None,
    actions: Sequence[str] = EDIT_ACTIONS,
) -> SwitchDiffusionDataset:
    if int(trajectories) < 2:
        raise ValueError("mixed edit training needs at least two trajectories")
    if int(forward_steps) <= 0 or int(record_every) <= 0:
        raise ValueError("forward_steps and record_every must be positive")
    prior = MarginalGraphPrior.from_training_graph(source)
    schedule = schedule or MixedKernelSchedule()
    examples = []
    targets = []
    trajectory_ids = []
    for trajectory_id in range(int(trajectories)):
        rng = np.random.default_rng(int(seed) + 1_000_003 * trajectory_id)
        graph = source.clone()
        recorded = 0
        for step in range(1, int(forward_steps) + 1):
            proposal = sample_mixed_forward_transition(
                graph,
                prior,
                rng,
                float(step) / float(forward_steps),
                schedule,
                actions=actions,
            )
            apply_edit(graph, proposal)
            if step % int(record_every) != 0:
                continue
            positive = inverse_edit(proposal)
            candidates, target = sample_edit_candidates(
                graph,
                rng,
                candidates_per_example,
                actions=actions,
                positive=positive,
            )
            examples.append(
                mixed_edit_candidate_matrix(
                    graph, candidates, float(step) / float(forward_steps)
                )
            )
            targets.append(target)
            trajectory_ids.append(trajectory_id)
            recorded += 1
        if recorded == 0:
            raise RuntimeError(f"trajectory {trajectory_id} produced no examples")
    shape = (len(examples), int(candidates_per_example))
    return SwitchDiffusionDataset(
        candidate_features=np.stack(examples),
        candidate_improvements=np.zeros(shape, dtype=np.float32),
        targets=np.asarray(targets, dtype=np.int64),
        trajectory_ids=np.asarray(trajectory_ids, dtype=np.int64),
    )


def sample_mixed_edit_diffusion(
    prior: MarginalGraphPrior,
    model: SwitchDenoiserMLP,
    steps: int,
    candidates_per_step: int,
    seed: int,
    device: str | torch.device = "cpu",
    temperature: float = 0.0,
    metrics_every: int = 100,
    actions: Sequence[str] = EDIT_ACTIONS,
) -> MixedEditSamplingResult:
    graph = prior.sample(seed=int(seed) + 71)
    rng = np.random.default_rng(int(seed))
    device = torch.device(device)
    generator = torch.Generator(device=device).manual_seed(int(seed))
    model = model.to(device).eval()
    trace = [{"step": 0.0, **graph_metrics(graph)}]
    action_counts = {action: 0 for action in EDIT_ACTIONS}
    action_counts["noop"] = 0
    proposed = 0
    applied = 0
    for reverse_step in range(int(steps), 0, -1):
        time_fraction = float(reverse_step) / float(steps)
        candidates, _ = sample_edit_candidates(
            graph, rng, candidates_per_step, actions=actions
        )
        proposed += len(candidates)
        matrix = mixed_edit_candidate_matrix(graph, candidates, time_fraction)
        with torch.no_grad():
            logits = model(torch.from_numpy(matrix)[None].to(device)).squeeze(0)
        if float(temperature) <= 0.0:
            chosen = int(logits.argmax().item())
        else:
            probabilities = torch.softmax(logits / float(temperature), dim=0)
            chosen = int(
                torch.multinomial(probabilities, 1, generator=generator).item()
            )
        proposal = candidates[chosen]
        apply_edit(graph, proposal)
        action_counts[proposal.action] += 1
        applied += int(proposal.action != "noop")
        completed = int(steps) - reverse_step + 1
        if completed == int(steps) or (
            int(metrics_every) > 0 and completed % int(metrics_every) == 0
        ):
            trace.append({"step": float(completed), **graph_metrics(graph)})
    graph.validate()
    return MixedEditSamplingResult(
        graph=graph,
        trace=trace,
        action_counts=action_counts,
        proposed=proposed,
        applied=applied,
    )


def evaluate_fixed_t_denoising(
    source: TypedGraph,
    model: SwitchDenoiserMLP,
    time_fractions: Sequence[float],
    trials_per_time: int,
    forward_steps: int,
    candidates_per_example: int,
    seed: int,
    device: str | torch.device = "cpu",
    schedule: Optional[MixedKernelSchedule] = None,
    actions: Sequence[str] = EDIT_ACTIONS,
) -> Mapping[str, Mapping[str, float]]:
    """Diagnostic only: identify a known inverse at fixed forward noise levels."""

    prior = MarginalGraphPrior.from_training_graph(source)
    schedule = schedule or MixedKernelSchedule()
    device = torch.device(device)
    model = model.to(device).eval()
    report = {}
    for time_index, fraction in enumerate(time_fractions):
        fraction = float(fraction)
        if not 0.0 < fraction <= 1.0:
            raise ValueError("fixed-t fractions must be in (0,1]")
        correct = 0
        top5 = 0
        margins = []
        action_correct = 0
        completed = 0
        for trial in range(int(trials_per_time)):
            rng = np.random.default_rng(
                int(seed) + 10_000_019 * time_index + 1_000_003 * trial
            )
            graph = source.clone()
            last = None
            steps = max(1, int(round(fraction * int(forward_steps))))
            for step in range(1, steps + 1):
                proposal = sample_mixed_forward_transition(
                    graph,
                    prior,
                    rng,
                    float(step) / float(forward_steps),
                    schedule,
                    actions=actions,
                )
                apply_edit(graph, proposal)
                last = proposal
            if last is None:
                continue
            positive = inverse_edit(last)
            candidates, target = sample_edit_candidates(
                graph,
                rng,
                candidates_per_example,
                actions=actions,
                positive=positive,
            )
            matrix = mixed_edit_candidate_matrix(graph, candidates, fraction)
            with torch.no_grad():
                logits = model(torch.from_numpy(matrix)[None].to(device)).squeeze(0)
            order = torch.argsort(logits, descending=True)
            selected = int(order[0].item())
            correct += int(selected == target)
            top5 += int(target in order[: min(5, len(candidates))].tolist())
            action_correct += int(candidates[selected].action == positive.action)
            negative = torch.cat((logits[:target], logits[target + 1 :]))
            margins.append(float((logits[target] - negative.max()).item()))
            completed += 1
        if completed == 0:
            raise RuntimeError(f"fixed-t={fraction} produced no valid trials")
        report[f"{fraction:.3f}"] = {
            "trials": float(completed),
            "top1": float(correct) / completed,
            "top5": float(top5) / completed,
            "action_top1": float(action_correct) / completed,
            "margin": float(np.mean(margins)),
        }
    return report


def run_edit_distance_oracle(
    prior: MarginalGraphPrior,
    clean: TypedGraph,
    steps: int,
    candidates_per_step: int,
    seed: int,
    actions: Sequence[str] = EDIT_ACTIONS,
    metrics_every: int = 100,
) -> MixedEditSamplingResult:
    """Candidate-coverage diagnostic. Clean edges never enter main sampling."""

    graph = prior.sample(seed=int(seed) + 71)
    rng = np.random.default_rng(int(seed))
    trace = [
        {
            "step": 0.0,
            "edge_edit_distance": float(edge_edit_distance(graph, clean)),
            **graph_metrics(graph),
        }
    ]
    action_counts = {action: 0 for action in EDIT_ACTIONS}
    action_counts["noop"] = 0
    applied = 0
    proposed = 0
    for step in range(1, int(steps) + 1):
        candidates, _ = sample_edit_candidates(
            graph, rng, candidates_per_step, actions=actions
        )
        proposed += len(candidates)
        improvements = np.asarray(
            [edit_distance_improvement(graph, clean, item) for item in candidates]
        )
        best = int(np.argmax(improvements))
        if int(improvements[best]) > 0:
            proposal = candidates[best]
            apply_edit(graph, proposal)
            action_counts[proposal.action] += 1
            applied += 1
        if step == int(steps) or (
            int(metrics_every) > 0 and step % int(metrics_every) == 0
        ):
            trace.append(
                {
                    "step": float(step),
                    "edge_edit_distance": float(edge_edit_distance(graph, clean)),
                    **graph_metrics(graph),
                }
            )
        if edge_edit_distance(graph, clean) == 0:
            break
    graph.validate()
    return MixedEditSamplingResult(
        graph=graph,
        trace=trace,
        action_counts=action_counts,
        proposed=proposed,
        applied=applied,
    )


def save_mixed_edit_checkpoint(
    path: str | Path,
    model: SwitchDenoiserMLP,
    prior: MarginalGraphPrior,
    hidden_dim: int,
    layers: int,
    dropout: float,
    report: SwitchDiffusionTrainReport,
    actions: Sequence[str] = EDIT_ACTIONS,
) -> None:
    payload = {
        "format": "dihug-mixed-edit-diffusion-v1",
        "prior": prior.to_json_dict(),
        "input_dim": int(model.input_dim),
        "hidden_dim": int(hidden_dim),
        "layers": int(layers),
        "dropout": float(dropout),
        "state_dict": model.state_dict(),
        "train_report": report.to_json_dict(),
        "actions": list(actions),
        "inference_contract": prior.to_json_dict()["inference_contract"],
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_mixed_edit_checkpoint(
    path: str | Path, device: str | torch.device = "cpu"
) -> Tuple[SwitchDenoiserMLP, MarginalGraphPrior, Mapping]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != "dihug-mixed-edit-diffusion-v1":
        raise ValueError("unsupported mixed edit diffusion checkpoint")
    prior = MarginalGraphPrior.from_json_dict(payload["prior"])
    model = SwitchDenoiserMLP(
        input_dim=int(payload["input_dim"]),
        hidden_dim=int(payload["hidden_dim"]),
        layers=int(payload["layers"]),
        dropout=float(payload["dropout"]),
    )
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    metadata = {
        "train_report": payload.get("train_report", {}),
        "actions": tuple(payload.get("actions", EDIT_ACTIONS)),
    }
    return model, prior, metadata
