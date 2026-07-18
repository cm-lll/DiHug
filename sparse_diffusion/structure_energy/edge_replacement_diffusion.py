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
        role_residual_features: bool = False,
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
        self.role_residual_features = bool(role_residual_features)
        self.register_buffer("role_prior_rows", torch.empty((0, 0)), persistent=False)
        self.register_buffer(
            "role_prior_types", torch.empty((0,), dtype=torch.long), persistent=False
        )
        self.structural_feature_dim = 5 + int(type_count) + (
            4 if self.role_residual_features else 0
        )
        width = 4 * hidden_dim + hidden_dim // 2 + 1 + self.structural_feature_dim
        self.edge_head = nn.Sequential(
            nn.Linear(width, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def set_role_prior(self, rows: np.ndarray, node_types: np.ndarray) -> None:
        if rows.ndim != 2 or rows.shape[0] != node_types.shape[0]:
            raise ValueError("role prior rows and node types have incompatible shape")
        self.role_prior_rows = torch.from_numpy(rows).float().to(self.family.weight.device)
        self.role_prior_types = torch.from_numpy(node_types).long().to(
            self.family.weight.device
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
    graph: TypedGraph,
    edges: Sequence[Edge],
    family: str,
    role_features: np.ndarray | None = None,
) -> np.ndarray:
    scale = max(float(np.log1p(graph.num_nodes)), 1.0)
    common_scale = max(float(np.sqrt(graph.num_nodes)), 1.0)
    degree = graph.degrees()
    family_degree = graph.family_degrees(family)
    family_names = tuple(sorted(graph.family_specs))
    family_column = 1 + family_names.index(family)
    rows = []
    for left, right in edges:
        common = graph.adjacency[int(left)] & graph.adjacency[int(right)]
        typed = np.zeros(len(graph.type_names), dtype=np.float32)
        for node in common:
            typed[int(graph.node_types[node])] += 1.0
        row = [
            np.log1p(degree[int(left)]) / scale,
            np.log1p(degree[int(right)]) / scale,
            np.log1p(family_degree[int(left)]) / scale,
            np.log1p(family_degree[int(right)]) / scale,
            len(common) / common_scale,
            *(typed / common_scale).tolist(),
        ]
        if role_features is not None:
            row.extend(
                (
                    float(role_features[int(left), 0] - row[0]),
                    float(role_features[int(right), 0] - row[1]),
                    float(role_features[int(left), family_column] - row[2]),
                    float(role_features[int(right), family_column] - row[3]),
                )
            )
        rows.append(row)
    return np.asarray(rows, dtype=np.float32)


def relation_degree_role_features(graph: TypedGraph) -> np.ndarray:
    """Anonymous joint relation-degree rows learned from the training graph."""

    scale = max(float(np.log1p(graph.num_nodes)), 1.0)
    columns = [np.log1p(graph.degrees()).astype(np.float32) / scale]
    for family in sorted(graph.family_specs):
        columns.append(
            np.log1p(graph.family_degrees(family)).astype(np.float32) / scale
        )
    return np.stack(columns, axis=1).astype(np.float32)


def sample_exchangeable_role_features(
    source: TypedGraph, seed: int
) -> np.ndarray:
    """Assign learned role rows by an independent within-type permutation."""

    rng = np.random.default_rng(int(seed))
    source_rows = relation_degree_role_features(source)
    sampled = np.empty_like(source_rows)
    for type_id in range(len(source.type_names)):
        nodes = np.flatnonzero(source.node_types == int(type_id))
        sampled[nodes] = source_rows[rng.permutation(nodes)]
    return sampled


def sample_checkpoint_role_features(
    model: EdgeReplacementDenoiser,
    node_types: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Sample an exchangeable role assignment using checkpoint-only statistics."""

    rows = model.role_prior_rows.detach().cpu().numpy()
    prior_types = model.role_prior_types.detach().cpu().numpy()
    if rows.size == 0 or prior_types.size == 0:
        raise RuntimeError("checkpoint does not contain an anonymous role prior")
    if rows.shape[0] != node_types.shape[0]:
        raise ValueError("checkpoint role prior has a different node count")
    rng = np.random.default_rng(int(seed))
    sampled = np.empty_like(rows)
    for type_id in np.unique(node_types).tolist():
        target_nodes = np.flatnonzero(node_types == int(type_id))
        source_nodes = np.flatnonzero(prior_types == int(type_id))
        if len(target_nodes) != len(source_nodes):
            raise ValueError("checkpoint role prior has incompatible type counts")
        sampled[target_nodes] = rows[rng.permutation(source_nodes)]
    return sampled


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
    role_features: np.ndarray | None = None,
) -> torch.Tensor:
    scale = max(float(np.log1p(graph.num_nodes)), 1.0)
    common_scale = max(float(np.sqrt(graph.num_nodes)), 1.0)
    degree = torch.from_numpy(graph.degrees()).float().to(edges.device)
    family_degree = torch.from_numpy(graph.family_degrees(family)).float().to(edges.device)
    left, right = edges[:, 0], edges[:, 1]
    typed = torch.stack(
        [matrix[left, right] / common_scale for matrix in typed_common], dim=1
    )
    columns = [
        torch.log1p(degree[left])[:, None] / scale,
        torch.log1p(degree[right])[:, None] / scale,
        torch.log1p(family_degree[left])[:, None] / scale,
        torch.log1p(family_degree[right])[:, None] / scale,
        typed.sum(dim=1, keepdim=True),
        typed,
    ]
    if role_features is not None:
        role = torch.from_numpy(role_features).float().to(edges.device)
        family_column = 1 + tuple(sorted(graph.family_specs)).index(family)
        columns.extend(
            (
                role[left, 0, None] - columns[0],
                role[right, 0, None] - columns[1],
                role[left, family_column, None] - columns[2],
                role[right, family_column, None] - columns[3],
            )
        )
    return torch.cat(tuple(columns), dim=1)


def _encode_graph(
    graph: TypedGraph,
    model: EdgeReplacementDenoiser,
    time_fraction: float,
    device: torch.device,
    role_features: np.ndarray | None = None,
) -> torch.Tensor:
    edge_index, edge_types = graph_edge_tensors(graph)
    node_features = graph_node_features(graph, time_fraction)
    if role_features is not None:
        if role_features.shape[0] != graph.num_nodes:
            raise ValueError("role features must have one row per node")
        node_features = np.concatenate((node_features, role_features), axis=1)
    return model.encode(
        torch.from_numpy(node_features).float().to(device),
        torch.from_numpy(graph.node_types).long().to(device),
        torch.from_numpy(edge_index).long().to(device),
        torch.from_numpy(edge_types).long().to(device),
    )


def _set_ranking_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    count = labels.sum().clamp_min(1.0)
    teacher = labels / count
    return -(teacher * torch.log_softmax(logits, dim=0)).sum()


@dataclass(frozen=True)
class X0RankingBatch:
    family: str
    edges: Tuple[Edge, ...]
    labels: np.ndarray
    present: np.ndarray
    structural_features: np.ndarray


def _build_x0_ranking_batches(
    examples: Sequence[ReplacementExample],
    source: TypedGraph,
    domains: Mapping[str, Sequence[Edge]],
    *,
    negatives_per_positive: int,
    seed: int,
    role_features: np.ndarray | None = None,
) -> Dict[int, Tuple[X0RankingBatch, ...]]:
    """Freeze sampled x0-ranking domains so epochs see identical supervision."""

    rng = np.random.default_rng(int(seed) + 88_881)
    result: Dict[int, Tuple[X0RankingBatch, ...]] = {}
    family_names = tuple(sorted(source.family_specs))
    clean_sets = {
        family: set(source.family_edges[family]) for family in family_names
    }
    for item_index, item in enumerate(examples):
        rows = []
        for family in family_names:
            clean = clean_sets[family]
            current = item.graph.family_edge_sets[family]
            required_negative = set(current) - clean
            remaining_negative = [
                edge
                for edge in domains[family]
                if edge not in clean and edge not in required_negative
            ]
            negative_budget = max(
                len(required_negative),
                len(clean) * max(int(negatives_per_positive), 1),
            )
            extra_count = min(
                max(negative_budget - len(required_negative), 0),
                len(remaining_negative),
            )
            if extra_count:
                extra_ids = rng.choice(
                    len(remaining_negative), size=extra_count, replace=False
                )
                extra = {
                    remaining_negative[int(index)] for index in extra_ids.tolist()
                }
            else:
                extra = set()
            candidates = tuple(sorted(clean | required_negative | extra))
            labels = np.asarray(
                [edge in clean for edge in candidates], dtype=np.float32
            )
            present = np.asarray(
                [edge in current for edge in candidates], dtype=np.bool_
            )
            rows.append(
                X0RankingBatch(
                    family=family,
                    edges=candidates,
                    labels=labels,
                    present=present,
                    structural_features=edge_structural_features(
                        item.graph, candidates, family, role_features=role_features
                    ),
                )
            )
        result[item_index] = tuple(rows)
    return result


def _score_x0_batch(
    model: EdgeReplacementDenoiser,
    hidden: torch.Tensor,
    batch: X0RankingBatch,
    *,
    family_id: int,
    time_fraction: float,
    device: torch.device,
) -> torch.Tensor:
    edges = torch.tensor(batch.edges, dtype=torch.long, device=device)
    structural = torch.from_numpy(batch.structural_features).to(device)
    present = torch.from_numpy(batch.present).to(device)
    logits = hidden.new_empty((len(batch.edges),))
    for is_present, action_id in ((True, 0), (False, 1)):
        mask = present if is_present else ~present
        if bool(mask.any()):
            logits[mask] = model.score_edges(
                hidden,
                edges[mask],
                family_id=family_id,
                action_id=action_id,
                time_fraction=time_fraction,
                structural_features=structural[mask],
            )
    return logits


def train_replacement_x0_denoiser(
    examples: Sequence[ReplacementExample],
    source: TypedGraph,
    domains: Mapping[str, Sequence[Edge]],
    *,
    epochs: int,
    negatives_per_positive: int = 1,
    learning_rate: float = 3.0e-4,
    device: str | torch.device = "cuda",
    seed: int = 0,
    role_condition: bool = True,
) -> Tuple[EdgeReplacementDenoiser, dict]:
    """Learn clean-edge membership instead of an unidentifiable random inverse event."""

    if len(examples) < 2:
        raise ValueError("x0 training requires at least two replacement examples")
    device = torch.device(device)
    torch.manual_seed(int(seed))
    rng = np.random.default_rng(int(seed) + 55_555)
    family_names = tuple(sorted(source.family_specs))
    family_to_id = {name: index for index, name in enumerate(family_names)}
    validation_trajectory = max(item.trajectory for item in examples)
    train_indices = [
        index
        for index, item in enumerate(examples)
        if item.trajectory != validation_trajectory
    ]
    validation_indices = [
        index
        for index, item in enumerate(examples)
        if item.trajectory == validation_trajectory
    ]
    role_features = relation_degree_role_features(source) if role_condition else None
    batches = _build_x0_ranking_batches(
        examples,
        source,
        domains,
        negatives_per_positive=negatives_per_positive,
        seed=seed,
        role_features=role_features,
    )
    node_feature_dim = graph_node_features(source, 0.0).shape[-1]
    if role_features is not None:
        node_feature_dim += role_features.shape[-1]
    model = EdgeReplacementDenoiser(
        node_feature_dim=node_feature_dim,
        type_count=len(source.type_names),
        family_count=len(family_names),
        role_residual_features=bool(role_condition),
    ).to(device)
    if role_features is not None:
        model.set_role_prior(role_features, source.node_types)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    final_loss = 0.0
    for epoch in range(int(epochs)):
        model.train()
        values = []
        for item_index in rng.permutation(train_indices).tolist():
            item = examples[item_index]
            optimizer.zero_grad(set_to_none=True)
            hidden = _encode_graph(
                item.graph,
                model,
                item.time_fraction,
                device,
                role_features=role_features,
            )
            losses = []
            for batch in batches[item_index]:
                logits = _score_x0_batch(
                    model,
                    hidden,
                    batch,
                    family_id=family_to_id[batch.family],
                    time_fraction=item.time_fraction,
                    device=device,
                )
                labels = torch.from_numpy(batch.labels).to(device)
                losses.append(_set_ranking_loss(logits, labels))
            loss = torch.stack(losses).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            values.append(float(loss.detach()))
        final_loss = float(np.mean(values))
        print(f"[REPLACEMENT-X0-EPOCH] epoch={epoch} loss={final_loss:.6f}", flush=True)

    model.eval()
    topm_hits = topm_total = add_hits = add_total = delete_hits = delete_total = 0
    clean_sets = {
        family: set(source.family_edges[family]) for family in family_names
    }
    with torch.no_grad():
        for item_index in validation_indices:
            item = examples[item_index]
            hidden = _encode_graph(
                item.graph,
                model,
                item.time_fraction,
                device,
                role_features=role_features,
            )
            for batch in batches[item_index]:
                family = batch.family
                logits = _score_x0_batch(
                    model,
                    hidden,
                    batch,
                    family_id=family_to_id[family],
                    time_fraction=item.time_fraction,
                    device=device,
                )
                clean = clean_sets[family]
                clean_count = len(clean)
                selected = {
                    batch.edges[index]
                    for index in logits.topk(min(clean_count, len(batch.edges))).indices.tolist()
                }
                topm_hits += len(selected & clean)
                topm_total += clean_count

                present_indices = np.flatnonzero(batch.present).tolist()
                false_current = set(item.graph.family_edge_sets[family]) - clean
                if false_current:
                    present_logits = logits[present_indices]
                    chosen = {
                        batch.edges[present_indices[index]]
                        for index in present_logits.topk(
                            min(len(false_current), len(present_indices)), largest=False
                        ).indices.tolist()
                    }
                    delete_hits += len(chosen & false_current)
                    delete_total += len(false_current)

                absent_indices = np.flatnonzero(~batch.present).tolist()
                missing_clean = clean - item.graph.family_edge_sets[family]
                if missing_clean:
                    absent_logits = logits[absent_indices]
                    chosen = {
                        batch.edges[absent_indices[index]]
                        for index in absent_logits.topk(
                            min(len(missing_clean), len(absent_indices))
                        ).indices.tolist()
                    }
                    add_hits += len(chosen & missing_clean)
                    add_total += len(missing_clean)
    return model, {
        "objective": "x0_set_ranking",
        "role_condition": bool(role_condition),
        "role_feature_dim": 0 if role_features is None else int(role_features.shape[-1]),
        "loss": final_loss,
        "train_examples": len(train_indices),
        "validation_examples": len(validation_indices),
        "x0_topm_recall": topm_hits / max(topm_total, 1),
        "missing_clean_add_recall": add_hits / max(add_total, 1),
        "false_current_delete_recall": delete_hits / max(delete_total, 1),
    }


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


def select_x0_replacement(
    present: Sequence[Edge],
    present_scores: Sequence[float],
    absent: Sequence[Edge],
    absent_scores: Sequence[float],
    *,
    budget: int,
) -> FamilyReplacement:
    """Select score-improving fixed-cardinality exchanges for one family."""

    if len(present) != len(present_scores) or len(absent) != len(absent_scores):
        raise ValueError("edge and score lengths must match")
    take = min(int(budget), len(present), len(absent))
    if take <= 0:
        return FamilyReplacement("", (), ())
    remove_order = np.argsort(np.asarray(present_scores, dtype=np.float64))[:take]
    add_order = np.argsort(np.asarray(absent_scores, dtype=np.float64))[::-1][:take]
    removed = []
    added = []
    for remove_index, add_index in zip(remove_order.tolist(), add_order.tolist()):
        if float(absent_scores[add_index]) <= float(present_scores[remove_index]):
            break
        removed.append(present[remove_index])
        added.append(absent[add_index])
    return FamilyReplacement("", tuple(removed), tuple(added))


def sample_replacement_x0_reverse(
    initial: TypedGraph,
    model: EdgeReplacementDenoiser,
    domains: Mapping[str, Sequence[Edge]],
    *,
    steps: int,
    rate: float,
    device: str | torch.device = "cuda",
    chunk_size: int = 8192,
    metrics_every: int = 10,
    role_features: np.ndarray | None = None,
    role_seed: int | None = None,
) -> Tuple[TypedGraph, Tuple[dict, ...]]:
    """Denoise by monotonically improving learned x0 set score at fixed counts."""

    graph = initial.clone()
    device = torch.device(device)
    model = model.to(device).eval()
    if role_features is None and role_seed is not None:
        role_features = sample_checkpoint_role_features(
            model, graph.node_types, role_seed
        )
    family_names = tuple(sorted(graph.family_specs))
    family_to_id = {name: index for index, name in enumerate(family_names)}
    trace = [{"step": 0.0, "accepted": 0.0, **graph_metrics(graph)}]
    with torch.no_grad():
        for reverse_index in range(int(steps)):
            fraction = 1.0 - reverse_index / float(steps)
            hidden = _encode_graph(
                graph,
                model,
                fraction,
                device,
                role_features=role_features,
            )
            typed_common = _dense_typed_common_neighbors(graph, device)
            changes = []
            accepted = 0
            for family in family_names:
                family_id = family_to_id[family]
                present = tuple(graph.family_edges[family])
                budget = replacement_budget(len(present), rate)
                if budget <= 0:
                    changes.append(FamilyReplacement(family, (), ()))
                    continue
                present_tensor = torch.tensor(present, dtype=torch.long, device=device)
                present_values = model.score_edges(
                    hidden,
                    present_tensor,
                    family_id=family_id,
                    action_id=0,
                    time_fraction=fraction,
                    structural_features=edge_structural_features_tensor(
                        graph,
                        present_tensor,
                        family,
                        typed_common,
                        role_features=role_features,
                    ),
                )
                absent = tuple(
                    edge
                    for edge in domains[family]
                    if edge not in graph.family_edge_sets[family]
                )
                best_values: List[float] = []
                best_edges: List[Edge] = []
                for start in range(0, len(absent), int(chunk_size)):
                    chunk = absent[start : start + int(chunk_size)]
                    if not chunk:
                        continue
                    chunk_tensor = torch.tensor(chunk, dtype=torch.long, device=device)
                    values = model.score_edges(
                        hidden,
                        chunk_tensor,
                        family_id=family_id,
                        action_id=1,
                        time_fraction=fraction,
                        structural_features=edge_structural_features_tensor(
                            graph,
                            chunk_tensor,
                            family,
                            typed_common,
                            role_features=role_features,
                        ),
                    )
                    take = min(budget, len(chunk))
                    indices = values.topk(take).indices.tolist()
                    best_values.extend(float(values[index]) for index in indices)
                    best_edges.extend(chunk[index] for index in indices)
                selected = select_x0_replacement(
                    present,
                    present_values.tolist(),
                    best_edges,
                    best_values,
                    budget=budget,
                )
                changes.append(
                    FamilyReplacement(family, selected.removed, selected.added)
                )
                accepted += len(selected.removed)
            apply_replacement_transition(graph, ReplacementTransition(tuple(changes)))
            completed = reverse_index + 1
            if completed % int(metrics_every) == 0 or completed == int(steps):
                trace.append(
                    {
                        "step": float(completed),
                        "accepted": float(accepted),
                        **graph_metrics(graph),
                    }
                )
    return graph, tuple(trace)
