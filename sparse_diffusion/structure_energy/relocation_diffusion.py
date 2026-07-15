from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from .graph import Edge, FamilySpec, TypedGraph
from .switch_chain import SwitchProposal, graph_metrics
from .switch_diffusion import (
    SwitchDenoiserMLP,
    SwitchDiffusionDataset,
    SwitchDiffusionTrainReport,
    SwitchSamplingResult,
    apply_switch,
    inverse_switch,
    train_switch_denoiser,
)


@dataclass(frozen=True)
class EdgeCountCondition:
    """Identity-free generation condition with no degree or motif targets."""

    type_names: Tuple[str, ...]
    type_counts: Tuple[int, ...]
    family_names: Tuple[str, ...]
    family_endpoints: Mapping[str, Tuple[int, int]]
    family_edge_counts: Mapping[str, int]

    @classmethod
    def from_training_graph(cls, graph: TypedGraph) -> "EdgeCountCondition":
        graph.validate_disjoint_family_domains()
        return cls(
            type_names=tuple(graph.type_names),
            type_counts=tuple(
                int(np.count_nonzero(graph.node_types == type_id))
                for type_id in range(len(graph.type_names))
            ),
            family_names=tuple(sorted(graph.family_specs)),
            family_endpoints={
                family: (
                    int(graph.family_specs[family].src_type),
                    int(graph.family_specs[family].dst_type),
                )
                for family in sorted(graph.family_specs)
            },
            family_edge_counts={
                family: len(graph.family_edges[family])
                for family in sorted(graph.family_specs)
            },
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

    def sample_uniform(self, seed: int) -> TypedGraph:
        graph = self.schema_graph()
        rng = np.random.default_rng(int(seed))
        for family in self.family_names:
            spec = graph.family_specs[family]
            target = int(self.family_edge_counts[family])
            domain_size = (
                len(spec.src_nodes) * len(spec.dst_nodes)
                if spec.bipartite
                else len(spec.src_nodes) * (len(spec.src_nodes) - 1) // 2
            )
            if target < 0 or target > domain_size:
                raise ValueError(
                    f"illegal edge count for {family}: {target}/{domain_size}"
                )
            selected = set()
            attempts = 0
            max_attempts = max(10_000, 100 * max(target, 1))
            while len(selected) < target and attempts < max_attempts:
                attempts += 1
                edge = _sample_domain_edge(graph, family, rng)
                if edge in selected:
                    continue
                selected.add(edge)
            if len(selected) != target:
                raise RuntimeError(
                    f"sampled only {len(selected)}/{target} edges for {family}"
                )
            for edge in sorted(selected):
                graph.add_edge(family, edge)
        graph.validate()
        return graph

    def to_json_dict(self) -> dict:
        return {
            "format": "dihug-edge-count-condition-v1",
            "type_names": list(self.type_names),
            "type_counts": list(self.type_counts),
            "family_names": list(self.family_names),
            "family_endpoints": {
                family: list(self.family_endpoints[family])
                for family in self.family_names
            },
            "family_edge_counts": {
                family: int(self.family_edge_counts[family])
                for family in self.family_names
            },
        }

    @classmethod
    def from_json_dict(cls, payload: Mapping) -> "EdgeCountCondition":
        if payload.get("format") != "dihug-edge-count-condition-v1":
            raise ValueError("unsupported edge-count condition format")
        forbidden = {"role_rows_by_type", "selection", "target_vector", "weights"}
        leaked = forbidden.intersection(payload)
        if leaked:
            raise ValueError(f"edge-count condition contains forbidden fields: {leaked}")
        return cls(
            type_names=tuple(str(value) for value in payload["type_names"]),
            type_counts=tuple(int(value) for value in payload["type_counts"]),
            family_names=tuple(str(value) for value in payload["family_names"]),
            family_endpoints={
                str(family): (int(value[0]), int(value[1]))
                for family, value in payload["family_endpoints"].items()
            },
            family_edge_counts={
                str(family): int(value)
                for family, value in payload["family_edge_counts"].items()
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
    def load(cls, path: str | Path) -> "EdgeCountCondition":
        return cls.from_json_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _sample_domain_edge(
    graph: TypedGraph,
    family: str,
    rng: np.random.Generator,
) -> Edge:
    spec = graph.family_specs[family]
    if spec.bipartite:
        edge = (
            int(rng.choice(spec.src_nodes)),
            int(rng.choice(spec.dst_nodes)),
        )
    else:
        left, right = rng.choice(spec.src_nodes, size=2, replace=False).tolist()
        edge = (int(left), int(right))
    return graph.orient_edge(family, edge)


def sample_valid_relocation(
    graph: TypedGraph,
    rng: np.random.Generator,
    family: Optional[str] = None,
    max_attempts: int = 256,
) -> Optional[SwitchProposal]:
    families = [
        name
        for name, edges in graph.family_edges.items()
        if edges and len(edges) < _family_domain_size(graph.family_specs[name])
    ]
    if family is not None:
        families = [str(family)] if str(family) in families else []
    if not families:
        return None
    weights = np.asarray(
        [len(graph.family_edges[name]) for name in families], dtype=np.float64
    )
    weights /= weights.sum()
    for _ in range(int(max_attempts)):
        selected = (
            str(family)
            if family is not None
            else str(rng.choice(families, p=weights))
        )
        edges = graph.family_edges[selected]
        index = int(rng.integers(len(edges)))
        old_edge = edges[index]
        new_edge = _sample_domain_edge(graph, selected, rng)
        if graph.has_edge(*new_edge):
            continue
        return SwitchProposal(
            family=selected,
            indices=(index,),
            old_edges=(old_edge,),
            new_edges=(new_edge,),
        )
    return None


def _family_domain_size(spec: FamilySpec) -> int:
    if spec.bipartite:
        return len(spec.src_nodes) * len(spec.dst_nodes)
    return len(spec.src_nodes) * (len(spec.src_nodes) - 1) // 2


def _proposal_key(proposal: SwitchProposal) -> tuple:
    return (
        proposal.family,
        proposal.old_edges,
        proposal.new_edges,
    )


def sample_relocation_candidates(
    graph: TypedGraph,
    rng: np.random.Generator,
    count: int,
    positive: Optional[SwitchProposal] = None,
    max_attempt_factor: int = 64,
) -> tuple[List[SwitchProposal], int]:
    if int(count) < 2:
        raise ValueError("candidate set must contain at least two relocations")
    candidates: List[SwitchProposal] = []
    keys = set()
    if positive is not None:
        candidates.append(positive)
        keys.add(_proposal_key(positive))
    attempts = 0
    limit = int(count) * int(max_attempt_factor)
    while len(candidates) < int(count) and attempts < limit:
        attempts += 1
        proposal = sample_valid_relocation(graph, rng)
        if proposal is None:
            continue
        key = _proposal_key(proposal)
        if key in keys:
            continue
        keys.add(key)
        candidates.append(proposal)
    if len(candidates) != int(count):
        raise RuntimeError(
            f"sampled only {len(candidates)}/{count} unique legal relocations"
        )
    target = 0
    if positive is not None:
        order = rng.permutation(len(candidates))
        candidates = [candidates[index] for index in order.tolist()]
        target = int(np.flatnonzero(order == 0)[0])
    return candidates, target


def _typed_common_neighbors(graph: TypedGraph, edge: Edge) -> np.ndarray:
    common = graph.adjacency[int(edge[0])] & graph.adjacency[int(edge[1])]
    values = np.zeros(len(graph.type_names), dtype=np.float64)
    for node in common:
        values[int(graph.node_types[node])] += 1.0
    return values


def relocation_feature_vector(
    graph: TypedGraph,
    proposal: SwitchProposal,
    family_to_id: Mapping[str, int],
    time_fraction: float,
) -> np.ndarray:
    family_names = tuple(sorted(graph.family_specs))
    total_degree = graph.degrees().astype(np.float64)
    family_degree = np.stack(
        [graph.family_degrees(family) for family in family_names], axis=1
    ).astype(np.float64)
    return _relocation_feature_vector_from_state(
        graph,
        proposal,
        family_to_id,
        time_fraction,
        total_degree,
        family_degree,
    )


def _relocation_feature_vector_from_state(
    graph: TypedGraph,
    proposal: SwitchProposal,
    family_to_id: Mapping[str, int],
    time_fraction: float,
    total_degree: np.ndarray,
    family_degree: np.ndarray,
) -> np.ndarray:
    if len(proposal.old_edges) != 1 or len(proposal.new_edges) != 1:
        raise ValueError("relocation features require one removed and one added edge")
    old_edge = proposal.old_edges[0]
    new_edge = proposal.new_edges[0]
    nodes = tuple(int(value) for value in old_edge + new_edge)
    degree_scale = max(float(np.log1p(graph.num_nodes)), 1.0)
    node_total = np.log1p(total_degree[list(nodes)]) / degree_scale
    node_family = np.log1p(family_degree[list(nodes)]).reshape(-1) / degree_scale
    old_common = _typed_common_neighbors(graph, old_edge)
    new_common = _typed_common_neighbors(graph, new_edge)
    common_scale = max(float(np.sqrt(graph.num_nodes)), 1.0)
    common = np.concatenate((old_common, new_common)) / common_scale
    family = np.zeros(len(family_to_id), dtype=np.float64)
    family[int(family_to_id[proposal.family])] = 1.0
    overlap = np.asarray(
        [
            float(old_edge[0] == new_edge[0]),
            float(old_edge[0] == new_edge[1]),
            float(old_edge[1] == new_edge[0]),
            float(old_edge[1] == new_edge[1]),
        ],
        dtype=np.float64,
    )
    scalar = np.asarray(
        [
            float(time_fraction),
            float(old_common.sum() / common_scale),
            float(new_common.sum() / common_scale),
            float((new_common.sum() - old_common.sum()) / common_scale),
        ],
        dtype=np.float64,
    )
    return np.concatenate(
        (node_total, node_family, common, family, overlap, scalar)
    ).astype(np.float32)


def relocation_candidate_matrix(
    graph: TypedGraph,
    candidates: Sequence[SwitchProposal],
    family_to_id: Mapping[str, int],
    time_fraction: float,
) -> np.ndarray:
    family_names = tuple(sorted(graph.family_specs))
    total_degree = graph.degrees().astype(np.float64)
    family_degree = np.stack(
        [graph.family_degrees(family) for family in family_names], axis=1
    ).astype(np.float64)
    return np.stack(
        [
            _relocation_feature_vector_from_state(
                graph,
                proposal,
                family_to_id,
                time_fraction,
                total_degree,
                family_degree,
            )
            for proposal in candidates
        ]
    )


def build_relocation_diffusion_dataset(
    source: TypedGraph,
    trajectories: int,
    forward_steps: int,
    record_every: int,
    candidates_per_example: int,
    seed: int,
) -> SwitchDiffusionDataset:
    if int(trajectories) < 2:
        raise ValueError("relocation training needs at least two trajectories")
    if int(forward_steps) <= 0 or int(record_every) <= 0:
        raise ValueError("forward_steps and record_every must be positive")
    family_to_id = {
        name: index for index, name in enumerate(sorted(source.family_specs))
    }
    examples = []
    targets = []
    trajectory_ids = []
    for trajectory_id in range(int(trajectories)):
        rng = np.random.default_rng(int(seed) + 1_000_003 * trajectory_id)
        graph = source.clone()
        for step in range(1, int(forward_steps) + 1):
            forward = sample_valid_relocation(graph, rng)
            if forward is None:
                continue
            apply_switch(graph, forward)
            if step % int(record_every) != 0:
                continue
            positive = inverse_switch(forward)
            candidates, target = sample_relocation_candidates(
                graph,
                rng,
                candidates_per_example,
                positive=positive,
            )
            examples.append(
                relocation_candidate_matrix(
                    graph,
                    candidates,
                    family_to_id,
                    float(step) / float(forward_steps),
                )
            )
            targets.append(target)
            trajectory_ids.append(trajectory_id)
    if not examples:
        raise RuntimeError("forward relocation process produced no examples")
    shape = (len(examples), int(candidates_per_example))
    return SwitchDiffusionDataset(
        candidate_features=np.stack(examples),
        candidate_improvements=np.zeros(shape, dtype=np.float32),
        targets=np.asarray(targets, dtype=np.int64),
        trajectory_ids=np.asarray(trajectory_ids, dtype=np.int64),
    )


def sample_relocation_diffusion(
    condition: EdgeCountCondition,
    model: SwitchDenoiserMLP,
    steps: int,
    candidates_per_step: int,
    seed: int,
    device: str | torch.device = "cpu",
    temperature: float = 0.0,
    metrics_every: int = 100,
    verbose: bool = False,
) -> SwitchSamplingResult:
    graph = condition.sample_uniform(seed=int(seed) + 71)
    initial_counts = {
        family: len(graph.family_edges[family]) for family in graph.family_specs
    }
    family_to_id = {
        name: index for index, name in enumerate(sorted(graph.family_specs))
    }
    rng = np.random.default_rng(int(seed))
    device = torch.device(device)
    generator = torch.Generator(device=device).manual_seed(int(seed))
    model = model.to(device).eval()
    trace = [{"step": 0.0, **graph_metrics(graph)}]
    proposed = 0
    for reverse_step in range(int(steps), 0, -1):
        candidates, _ = sample_relocation_candidates(
            graph, rng, candidates_per_step
        )
        proposed += len(candidates)
        matrix = relocation_candidate_matrix(
            graph,
            candidates,
            family_to_id,
            float(reverse_step) / float(steps),
        )
        with torch.no_grad():
            logits = model(torch.from_numpy(matrix)[None].to(device)).squeeze(0)
        if float(temperature) <= 0.0:
            chosen = int(logits.argmax().item())
        else:
            probability = torch.softmax(logits / float(temperature), dim=0)
            chosen = int(
                torch.multinomial(
                    probability, 1, generator=generator
                ).item()
            )
        apply_switch(graph, candidates[chosen])
        completed = int(steps) - reverse_step + 1
        if completed == int(steps) or (
            int(metrics_every) > 0 and completed % int(metrics_every) == 0
        ):
            metrics = {"step": float(completed), **graph_metrics(graph)}
            trace.append(metrics)
            if verbose:
                print(
                    "[RELOCATION-DIFFUSION-TRACE] "
                    f"step={completed}/{steps} "
                    f"tri={metrics['triangles']:.0f} "
                    f"clustering={metrics['clustering']:.6f}",
                    flush=True,
                )
    graph.validate()
    final_counts = {
        family: len(graph.family_edges[family]) for family in graph.family_specs
    }
    if final_counts != initial_counts:
        raise AssertionError("relocation diffusion changed family edge counts")
    return SwitchSamplingResult(
        graph=graph,
        trace=trace,
        proposed=proposed,
        applied=int(steps),
        selection_mode="model",
        oracle_agreement=float("nan"),
    )


def save_relocation_checkpoint(
    path: str | Path,
    model: SwitchDenoiserMLP,
    condition: EdgeCountCondition,
    hidden_dim: int,
    layers: int,
    dropout: float,
    report: SwitchDiffusionTrainReport,
) -> None:
    payload = {
        "format": "dihug-relocation-diffusion-v1",
        "condition": condition.to_json_dict(),
        "input_dim": int(model.input_dim),
        "hidden_dim": int(hidden_dim),
        "layers": int(layers),
        "dropout": float(dropout),
        "state_dict": model.state_dict(),
        "train_report": report.to_json_dict(),
        "inference_contract": {
            "node_type_counts": True,
            "family_schema": True,
            "family_edge_counts": True,
            "relation_degree_rows": False,
            "graphlet_target": False,
            "training_adjacency": False,
        },
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_relocation_checkpoint(
    path: str | Path,
    device: str | torch.device = "cpu",
) -> tuple[SwitchDenoiserMLP, EdgeCountCondition, Mapping]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != "dihug-relocation-diffusion-v1":
        raise ValueError("unsupported relocation diffusion checkpoint")
    contract = payload.get("inference_contract", {})
    expected_false = ("relation_degree_rows", "graphlet_target", "training_adjacency")
    if any(contract.get(key) is not False for key in expected_false):
        raise ValueError(f"unsafe relocation checkpoint contract: {contract}")
    condition = EdgeCountCondition.from_json_dict(payload["condition"])
    model = SwitchDenoiserMLP(
        input_dim=int(payload["input_dim"]),
        hidden_dim=int(payload["hidden_dim"]),
        layers=int(payload["layers"]),
        dropout=float(payload["dropout"]),
    )
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    return model, condition, payload.get("train_report", {})
