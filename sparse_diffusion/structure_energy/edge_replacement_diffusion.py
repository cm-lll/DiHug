from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from torch import nn

from sparse_diffusion.models.pyhgt_denoiser import HGTConv

from .graph import Edge, TypedGraph
from .hgt_edit_denoiser import graph_edge_tensors, graph_node_features
from .switch_chain import graph_metrics


@dataclass(frozen=True)
class FamilyReplacement:
    family: str
    removed: Tuple[Edge, ...]
    added: Tuple[Edge, ...]

    def inverse(self) -> "FamilyReplacement":
        return FamilyReplacement(self.family, self.added, self.removed)


@dataclass(frozen=True)
class ReplacementTransition:
    families: Tuple[FamilyReplacement, ...]

    def inverse(self) -> "ReplacementTransition":
        return ReplacementTransition(tuple(item.inverse() for item in self.families))

    @property
    def changed_edges(self) -> int:
        return sum(len(item.removed) for item in self.families)


def enumerate_family_domain(graph: TypedGraph, family: str) -> Tuple[Edge, ...]:
    spec = graph.family_specs[family]
    if spec.bipartite:
        return tuple(
            graph.orient_edge(family, (left, right))
            for left in spec.src_nodes
            for right in spec.dst_nodes
        )
    nodes = tuple(spec.src_nodes)
    return tuple(
        graph.orient_edge(family, (nodes[left], nodes[right]))
        for left in range(len(nodes))
        for right in range(left + 1, len(nodes))
    )


def build_family_domains(graph: TypedGraph) -> Dict[str, Tuple[Edge, ...]]:
    return {
        family: enumerate_family_domain(graph, family)
        for family in sorted(graph.family_specs)
    }


def replacement_budget(edge_count: int, rate: float) -> int:
    if not 0.0 <= float(rate) <= 1.0:
        raise ValueError("replacement rate must lie in [0,1]")
    if edge_count <= 0 or rate <= 0.0:
        return 0
    return min(max(int(round(float(rate) * edge_count)), 1), edge_count)


def sample_replacement_transition(
    graph: TypedGraph,
    rng: np.random.Generator,
    *,
    rate: float,
    domains: Mapping[str, Sequence[Edge]],
) -> ReplacementTransition:
    changes = []
    for family in sorted(graph.family_specs):
        current = tuple(graph.family_edges[family])
        absent = tuple(edge for edge in domains[family] if edge not in graph.family_edge_sets[family])
        budget = min(replacement_budget(len(current), rate), len(absent))
        if budget == 0:
            changes.append(FamilyReplacement(family, (), ()))
            continue
        remove_ids = rng.choice(len(current), size=budget, replace=False)
        add_ids = rng.choice(len(absent), size=budget, replace=False)
        changes.append(
            FamilyReplacement(
                family,
                tuple(current[int(index)] for index in remove_ids.tolist()),
                tuple(absent[int(index)] for index in add_ids.tolist()),
            )
        )
    return ReplacementTransition(tuple(changes))


def apply_replacement_transition(
    graph: TypedGraph, transition: ReplacementTransition
) -> None:
    for item in transition.families:
        for edge in item.removed:
            graph.remove_edge(item.family, edge)
    for item in transition.families:
        for edge in item.added:
            graph.add_edge(item.family, edge)


def replacement_log_probability(
    graph: TypedGraph,
    transition: ReplacementTransition,
    domains: Mapping[str, Sequence[Edge]],
) -> float:
    """Exact log q(G'|G) for a legal fixed-cardinality replacement."""

    value = 0.0
    items = {item.family: item for item in transition.families}
    for family in sorted(graph.family_specs):
        item = items[family]
        budget = len(item.removed)
        if budget != len(item.added):
            return -math.inf
        if not set(item.removed).issubset(graph.family_edge_sets[family]):
            return -math.inf
        if any(edge in graph.family_edge_sets[family] for edge in item.added):
            return -math.inf
        edge_count = len(graph.family_edges[family])
        absent_count = len(domains[family]) - edge_count
        if budget > min(edge_count, absent_count):
            return -math.inf
        value -= (
            math.lgamma(edge_count + 1)
            - math.lgamma(budget + 1)
            - math.lgamma(edge_count - budget + 1)
        )
        value -= (
            math.lgamma(absent_count + 1)
            - math.lgamma(budget + 1)
            - math.lgamma(absent_count - budget + 1)
        )
    return float(value)


@dataclass(frozen=True)
class ReplacementExample:
    graph: TypedGraph
    reverse: ReplacementTransition
    time_fraction: float
    trajectory: int


def build_replacement_examples(
    source: TypedGraph,
    *,
    trajectories: int,
    steps: int,
    record_every: int,
    rate: float,
    seed: int,
    domains: Mapping[str, Sequence[Edge]],
) -> Tuple[ReplacementExample, ...]:
    examples: List[ReplacementExample] = []
    for trajectory in range(int(trajectories)):
        graph = source.clone()
        rng = np.random.default_rng(int(seed) + 1_000_003 * trajectory)
        for step in range(1, int(steps) + 1):
            forward = sample_replacement_transition(
                graph, rng, rate=rate, domains=domains
            )
            apply_replacement_transition(graph, forward)
            if step % int(record_every) == 0 or step == int(steps):
                examples.append(
                    ReplacementExample(
                        graph.clone(), forward.inverse(), step / float(steps), trajectory
                    )
                )
    return tuple(examples)


class EdgeReplacementDenoiser(nn.Module):
    """Global sparse HGT with action-conditional edge-set scores."""

    def __init__(
        self,
        *,
        node_feature_dim: int,
        type_count: int,
        family_count: int,
        hidden_dim: int = 128,
        heads: int = 4,
        layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.node_input = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU()
        )
        self.layers = nn.ModuleList(
            HGTConv(
                hidden_dim,
                hidden_dim,
                num_types=type_count,
                num_relations=family_count,
                n_heads=heads,
                dropout=dropout,
                use_norm=True,
            )
            for _ in range(int(layers))
        )
        self.family = nn.Embedding(max(family_count, 1), hidden_dim // 4)
        self.action = nn.Embedding(2, hidden_dim // 4)
        self.structural_feature_dim = 5 + int(type_count)
        width = 4 * hidden_dim + hidden_dim // 2 + 1 + self.structural_feature_dim
        self.edge_head = nn.Sequential(
            nn.Linear(width, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def encode(
        self,
        node_features: torch.Tensor,
        node_types: torch.Tensor,
        edge_index: torch.Tensor,
        edge_types: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.node_input(node_features)
        for layer in self.layers:
            hidden = layer(hidden, node_types, edge_index, edge_types)
        return hidden

    def score_edges(
        self,
        hidden: torch.Tensor,
        edges: torch.Tensor,
        *,
        family_id: int,
        action_id: int,
        time_fraction: float,
        structural_features: torch.Tensor,
    ) -> torch.Tensor:
        if edges.numel() == 0:
            return hidden.new_empty((0,))
        left = hidden[edges[:, 0]]
        right = hidden[edges[:, 1]]
        family = self.family.weight[int(family_id)][None].expand(edges.shape[0], -1)
        action = self.action.weight[int(action_id)][None].expand(edges.shape[0], -1)
        time = hidden.new_full((edges.shape[0], 1), float(time_fraction))
        if structural_features.shape != (edges.shape[0], self.structural_feature_dim):
            raise ValueError("edge structural features have incompatible shape")
        features = torch.cat(
            (
                left,
                right,
                torch.abs(left - right),
                left * right,
                family,
                action,
                time,
                structural_features,
            ),
            dim=-1,
        )
        return self.edge_head(features).squeeze(-1)


def edge_structural_features(
    graph: TypedGraph, edges: Sequence[Edge], family: str
) -> np.ndarray:
    scale = max(float(np.log1p(graph.num_nodes)), 1.0)
    common_scale = max(float(np.sqrt(graph.num_nodes)), 1.0)
    degree = graph.degrees()
    family_degree = graph.family_degrees(family)
    rows = []
    for left, right in edges:
        common = graph.adjacency[int(left)] & graph.adjacency[int(right)]
        typed = np.zeros(len(graph.type_names), dtype=np.float32)
        for node in common:
            typed[int(graph.node_types[node])] += 1.0
        rows.append(
            [
                np.log1p(degree[int(left)]) / scale,
                np.log1p(degree[int(right)]) / scale,
                np.log1p(family_degree[int(left)]) / scale,
                np.log1p(family_degree[int(right)]) / scale,
                len(common) / common_scale,
                *(typed / common_scale).tolist(),
            ]
        )
    return np.asarray(rows, dtype=np.float32)


def _dense_typed_common_neighbors(
    graph: TypedGraph, device: torch.device
) -> Tuple[torch.Tensor, ...]:
    adjacency = torch.zeros(
        (graph.num_nodes, graph.num_nodes), dtype=torch.float32, device=device
    )
    if graph.edge_family:
        pairs = torch.tensor(tuple(graph.edge_family), dtype=torch.long, device=device)
        adjacency[pairs[:, 0], pairs[:, 1]] = 1.0
        adjacency[pairs[:, 1], pairs[:, 0]] = 1.0
    values = []
    node_types = torch.from_numpy(graph.node_types).long().to(device)
    for type_id in range(len(graph.type_names)):
        mask = node_types == int(type_id)
        values.append(adjacency[:, mask] @ adjacency[mask, :])
    return tuple(values)


def edge_structural_features_tensor(
    graph: TypedGraph,
    edges: torch.Tensor,
    family: str,
    typed_common: Sequence[torch.Tensor],
) -> torch.Tensor:
    scale = max(float(np.log1p(graph.num_nodes)), 1.0)
    common_scale = max(float(np.sqrt(graph.num_nodes)), 1.0)
    degree = torch.from_numpy(graph.degrees()).float().to(edges.device)
    family_degree = torch.from_numpy(graph.family_degrees(family)).float().to(edges.device)
    left, right = edges[:, 0], edges[:, 1]
    typed = torch.stack(
        [matrix[left, right] / common_scale for matrix in typed_common], dim=1
    )
    return torch.cat(
        (
            torch.log1p(degree[left])[:, None] / scale,
            torch.log1p(degree[right])[:, None] / scale,
            torch.log1p(family_degree[left])[:, None] / scale,
            torch.log1p(family_degree[right])[:, None] / scale,
            typed.sum(dim=1, keepdim=True),
            typed,
        ),
        dim=1,
    )


def _encode_graph(
    graph: TypedGraph,
    model: EdgeReplacementDenoiser,
    time_fraction: float,
    device: torch.device,
) -> torch.Tensor:
    edge_index, edge_types = graph_edge_tensors(graph)
    return model.encode(
        torch.from_numpy(graph_node_features(graph, time_fraction)).float().to(device),
        torch.from_numpy(graph.node_types).long().to(device),
        torch.from_numpy(edge_index).long().to(device),
        torch.from_numpy(edge_types).long().to(device),
    )


def _set_ranking_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    count = labels.sum().clamp_min(1.0)
    teacher = labels / count
    return -(teacher * torch.log_softmax(logits, dim=0)).sum()


def train_replacement_denoiser(
    examples: Sequence[ReplacementExample],
    source: TypedGraph,
    domains: Mapping[str, Sequence[Edge]],
    *,
    epochs: int,
    negatives_per_positive: int = 4,
    learning_rate: float = 3.0e-4,
    device: str | torch.device = "cuda",
    seed: int = 0,
) -> Tuple[EdgeReplacementDenoiser, dict]:
    device = torch.device(device)
    torch.manual_seed(int(seed))
    rng = np.random.default_rng(int(seed) + 55_555)
    family_names = tuple(sorted(source.family_specs))
    family_to_id = {name: index for index, name in enumerate(family_names)}
    validation_trajectory = max(item.trajectory for item in examples)
    train_rows = [item for item in examples if item.trajectory != validation_trajectory]
    validation_rows = [item for item in examples if item.trajectory == validation_trajectory]
    model = EdgeReplacementDenoiser(
        node_feature_dim=graph_node_features(source, 0.0).shape[-1],
        type_count=len(source.type_names),
        family_count=len(family_names),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    def example_loss(item: ReplacementExample) -> torch.Tensor:
        hidden = _encode_graph(item.graph, model, item.time_fraction, device)
        reverse = {change.family: change for change in item.reverse.families}
        losses = []
        for family in family_names:
            family_id = family_to_id[family]
            change = reverse[family]
            present = tuple(item.graph.family_edges[family])
            delete_labels = torch.tensor(
                [edge in set(change.removed) for edge in present],
                dtype=torch.float32,
                device=device,
            )
            present_tensor = torch.tensor(present, dtype=torch.long, device=device)
            delete_logits = model.score_edges(
                hidden,
                present_tensor,
                family_id=family_id,
                action_id=0,
                time_fraction=item.time_fraction,
                structural_features=torch.from_numpy(
                    edge_structural_features(item.graph, present, family)
                ).to(device),
            )
            losses.append(_set_ranking_loss(delete_logits, delete_labels))

            positive = tuple(change.added)
            positive_set = set(positive)
            absent = [
                edge
                for edge in domains[family]
                if edge not in item.graph.family_edge_sets[family] and edge not in positive_set
            ]
            negative_count = min(
                len(absent), max(len(positive) * int(negatives_per_positive), 1)
            )
            negative_ids = rng.choice(len(absent), size=negative_count, replace=False)
            addition = positive + tuple(absent[int(index)] for index in negative_ids.tolist())
            add_tensor = torch.tensor(addition, dtype=torch.long, device=device)
            add_labels = torch.zeros(len(addition), dtype=torch.float32, device=device)
            add_labels[: len(positive)] = 1.0
            add_logits = model.score_edges(
                hidden,
                add_tensor,
                family_id=family_id,
                action_id=1,
                time_fraction=item.time_fraction,
                structural_features=torch.from_numpy(
                    edge_structural_features(item.graph, addition, family)
                ).to(device),
            )
            losses.append(_set_ranking_loss(add_logits, add_labels))
        return torch.stack(losses).mean()

    final_loss = 0.0
    for epoch in range(int(epochs)):
        model.train()
        values = []
        for index in rng.permutation(len(train_rows)).tolist():
            optimizer.zero_grad(set_to_none=True)
            loss = example_loss(train_rows[index])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            values.append(float(loss.detach()))
        final_loss = float(np.mean(values))
        print(f"[REPLACEMENT-EPOCH] epoch={epoch} loss={final_loss:.6f}", flush=True)

    model.eval()
    delete_hits = delete_total = add_hits = add_total = 0
    with torch.no_grad():
        for item in validation_rows:
            hidden = _encode_graph(item.graph, model, item.time_fraction, device)
            reverse = {change.family: change for change in item.reverse.families}
            for family in family_names:
                family_id = family_to_id[family]
                change = reverse[family]
                present = tuple(item.graph.family_edges[family])
                logits = model.score_edges(
                    hidden,
                    torch.tensor(present, dtype=torch.long, device=device),
                    family_id=family_id,
                    action_id=0,
                    time_fraction=item.time_fraction,
                    structural_features=torch.from_numpy(
                        edge_structural_features(item.graph, present, family)
                    ).to(device),
                )
                budget = len(change.removed)
                selected = {present[index] for index in logits.topk(budget).indices.tolist()}
                delete_hits += len(selected & set(change.removed))
                delete_total += budget

                positive = tuple(change.added)
                positive_set = set(positive)
                absent = [
                    edge
                    for edge in domains[family]
                    if edge not in item.graph.family_edge_sets[family] and edge not in positive_set
                ]
                negative_count = min(len(absent), max(len(positive) * 8, 1))
                ids = rng.choice(len(absent), size=negative_count, replace=False)
                candidates = positive + tuple(absent[int(index)] for index in ids.tolist())
                logits = model.score_edges(
                    hidden,
                    torch.tensor(candidates, dtype=torch.long, device=device),
                    family_id=family_id,
                    action_id=1,
                    time_fraction=item.time_fraction,
                    structural_features=torch.from_numpy(
                        edge_structural_features(item.graph, candidates, family)
                    ).to(device),
                )
                selected = {candidates[index] for index in logits.topk(len(positive)).indices.tolist()}
                add_hits += len(selected & positive_set)
                add_total += len(positive)
    return model, {
        "loss": final_loss,
        "train_examples": len(train_rows),
        "validation_examples": len(validation_rows),
        "delete_recall": delete_hits / max(delete_total, 1),
        "add_sampled_recall": add_hits / max(add_total, 1),
    }


def sample_replacement_reverse(
    initial: TypedGraph,
    model: EdgeReplacementDenoiser,
    domains: Mapping[str, Sequence[Edge]],
    *,
    steps: int,
    rate: float,
    device: str | torch.device = "cuda",
    chunk_size: int = 8192,
    metrics_every: int = 10,
) -> Tuple[TypedGraph, Tuple[dict, ...]]:
    graph = initial.clone()
    device = torch.device(device)
    model = model.to(device).eval()
    family_names = tuple(sorted(graph.family_specs))
    family_to_id = {name: index for index, name in enumerate(family_names)}
    trace = [{"step": 0.0, **graph_metrics(graph)}]
    with torch.no_grad():
        for reverse_index in range(int(steps)):
            fraction = 1.0 - reverse_index / float(steps)
            hidden = _encode_graph(graph, model, fraction, device)
            typed_common = _dense_typed_common_neighbors(graph, device)
            changes = []
            for family in family_names:
                family_id = family_to_id[family]
                present = tuple(graph.family_edges[family])
                absent = tuple(
                    edge for edge in domains[family] if edge not in graph.family_edge_sets[family]
                )
                budget = min(replacement_budget(len(present), rate), len(absent))
                delete_logits = model.score_edges(
                    hidden,
                    (present_tensor := torch.tensor(present, dtype=torch.long, device=device)),
                    family_id=family_id,
                    action_id=0,
                    time_fraction=fraction,
                    structural_features=edge_structural_features_tensor(
                        graph, present_tensor, family, typed_common
                    ),
                )
                removed = tuple(
                    present[index] for index in delete_logits.topk(budget).indices.tolist()
                )
                best_values = []
                best_edges = []
                for start in range(0, len(absent), int(chunk_size)):
                    chunk = absent[start : start + int(chunk_size)]
                    values = model.score_edges(
                        hidden,
                        (chunk_tensor := torch.tensor(chunk, dtype=torch.long, device=device)),
                        family_id=family_id,
                        action_id=1,
                        time_fraction=fraction,
                        structural_features=edge_structural_features_tensor(
                            graph, chunk_tensor, family, typed_common
                        ),
                    )
                    take = min(budget, len(chunk))
                    indices = values.topk(take).indices.tolist()
                    best_values.extend(float(values[index]) for index in indices)
                    best_edges.extend(chunk[index] for index in indices)
                order = np.argsort(np.asarray(best_values))[::-1][:budget]
                added = tuple(best_edges[int(index)] for index in order.tolist())
                changes.append(FamilyReplacement(family, removed, added))
            apply_replacement_transition(graph, ReplacementTransition(tuple(changes)))
            completed = reverse_index + 1
            if completed % int(metrics_every) == 0 or completed == int(steps):
                trace.append({"step": float(completed), **graph_metrics(graph)})
    return graph, tuple(trace)
