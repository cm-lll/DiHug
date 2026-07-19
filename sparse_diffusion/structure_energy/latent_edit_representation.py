from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import List, Mapping, Sequence, Tuple

import numpy as np
import torch
from torch import nn

from sparse_diffusion.models.pyhgt_denoiser import HGTConv

from .graph import TypedGraph
from .hgt_edit_denoiser import graph_edge_tensors, graph_node_features
from .mixed_edit_diffusion import (
    MarginalGraphPrior,
    _family_names,
    sample_closed_form_marginal_noise,
)
from .roles import permuted_witness, sample_role_plan


@dataclass(frozen=True)
class RepresentationSnapshot:
    graph: TypedGraph
    clean: TypedGraph
    role_rows: np.ndarray
    time_fraction: float
    trajectory_id: int


@dataclass(frozen=True)
class RepresentationWarmupDataset:
    snapshots: Tuple[RepresentationSnapshot, ...]
    family_names: Tuple[str, ...]
    type_names: Tuple[str, ...]
    node_types: np.ndarray

    @property
    def family_count(self) -> int:
        return len(self.family_names)

    @property
    def type_count(self) -> int:
        return len(self.type_names)


def build_representation_warmup_dataset(
    source: TypedGraph,
    *,
    trajectories: int,
    noise_levels: Sequence[float],
    seed: int,
) -> RepresentationWarmupDataset:
    if int(trajectories) < 2:
        raise ValueError("representation warm-up needs at least two trajectories")
    levels = tuple(float(value) for value in noise_levels)
    if not levels or any(not 0.0 < value <= 1.0 for value in levels):
        raise ValueError("noise levels must lie in (0,1]")
    family_names = tuple(_family_names(source))
    prior = MarginalGraphPrior.from_training_graph(source)
    snapshots: List[RepresentationSnapshot] = []
    for trajectory_id in range(int(trajectories)):
        trajectory_seed = int(seed) + 1_000_003 * trajectory_id
        plan = sample_role_plan(source, trajectory_seed + 17)
        clean = permuted_witness(source, plan)
        role_rows = np.stack(
            [plan.family_degrees[family] for family in family_names], axis=1
        ).astype(np.float32)
        for level_id, fraction in enumerate(levels):
            alpha_bar = (1.0 - float(fraction)) ** 2
            graph = sample_closed_form_marginal_noise(
                clean,
                prior,
                alpha_bar=alpha_bar,
                seed=trajectory_seed + 97_409 * (level_id + 1),
            )
            snapshots.append(
                RepresentationSnapshot(
                    graph=graph,
                    clean=clean,
                    role_rows=role_rows,
                    time_fraction=float(fraction),
                    trajectory_id=trajectory_id,
                )
            )
    return RepresentationWarmupDataset(
        snapshots=tuple(snapshots),
        family_names=family_names,
        type_names=tuple(source.type_names),
        node_types=source.node_types.copy(),
    )


class WarmupHGTEncoder(nn.Module):
    def __init__(
        self,
        *,
        node_feature_dim: int,
        type_count: int,
        family_count: int,
        hidden_dim: int = 64,
        heads: int = 4,
        layers: int = 2,
        dropout: float = 0.1,
        role_condition_dim: int = 0,
        prototype_count: int = 0,
    ) -> None:
        super().__init__()
        if int(hidden_dim) % int(heads):
            raise ValueError("HGT hidden dimension must be divisible by heads")
        self.hidden_dim = int(hidden_dim)
        self.family_count = int(family_count)
        self.role_condition_dim = int(role_condition_dim)
        self.prototype_count = int(prototype_count)
        self.node_input = nn.Sequential(
            nn.Linear(int(node_feature_dim), int(hidden_dim)),
            nn.LayerNorm(int(hidden_dim)),
            nn.SiLU(),
        )
        self.layers = nn.ModuleList(
            HGTConv(
                int(hidden_dim),
                int(hidden_dim),
                num_types=int(type_count),
                num_relations=int(family_count),
                n_heads=int(heads),
                dropout=float(dropout),
                use_norm=True,
            )
            for _ in range(int(layers))
        )
        self.edge_family = nn.Embedding(int(family_count), int(hidden_dim))
        self.edge_head = nn.Sequential(
            nn.Linear(4 * int(hidden_dim) + 2, int(hidden_dim)),
            nn.LayerNorm(int(hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(hidden_dim), 1),
        )
        self.role_head = nn.Sequential(
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(hidden_dim), int(family_count)),
        )
        self.role_input = None
        self.prototype_type = None
        self.prototype_head = None
        if self.role_condition_dim > 0:
            self.role_input = nn.Sequential(
                nn.Linear(self.role_condition_dim, int(hidden_dim)),
                nn.LayerNorm(int(hidden_dim)),
                nn.SiLU(),
            )
        if self.prototype_count > 0:
            if self.role_condition_dim <= 0:
                raise ValueError("role prototypes require role conditions")
            self.prototype_type = nn.Embedding(int(type_count), int(hidden_dim))
            self.prototype_head = nn.Sequential(
                nn.Linear(self.role_condition_dim + int(hidden_dim), int(hidden_dim)),
                nn.SiLU(),
                nn.Linear(
                    int(hidden_dim),
                    self.prototype_count * int(hidden_dim),
                ),
            )

    def encode(
        self,
        node_features: torch.Tensor,
        node_types: torch.Tensor,
        edge_index: torch.Tensor,
        edge_types: torch.Tensor,
        role_conditions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden = self.node_input(node_features)
        if self.role_input is not None:
            if role_conditions is None:
                raise ValueError("role-conditioned encoder needs role conditions")
            hidden = hidden + self.role_input(role_conditions)
        for layer in self.layers:
            hidden = layer(hidden, node_types, edge_index, edge_types)
        return hidden

    def edge_logits(
        self,
        hidden: torch.Tensor,
        query_edges: torch.Tensor,
        query_families: torch.Tensor,
        current_exists: torch.Tensor,
        time_fraction: float,
    ) -> torch.Tensor:
        left = hidden[query_edges[0]]
        right = hidden[query_edges[1]]
        relation = self.edge_family(query_families)
        pair = torch.cat(
            (
                left,
                right,
                torch.abs(left - right),
                left * right + relation,
                current_exists[:, None],
                torch.full_like(current_exists[:, None], float(time_fraction)),
            ),
            dim=1,
        )
        return self.edge_head(pair).squeeze(1)

    def role_prediction(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.role_head(hidden)

    def role_prototypes(
        self, role_conditions: torch.Tensor, node_types: torch.Tensor
    ) -> torch.Tensor:
        if self.prototype_head is None or self.prototype_type is None:
            raise RuntimeError("prototype head is disabled")
        inputs = torch.cat(
            (role_conditions, self.prototype_type(node_types)), dim=1
        )
        return self.prototype_head(inputs).reshape(
            role_conditions.shape[0], self.prototype_count, self.hidden_dim
        )


def _sample_domain_pair(
    graph: TypedGraph, family: str, rng: np.random.Generator
) -> tuple[int, int]:
    spec = graph.family_specs[family]
    for _ in range(512):
        left = int(rng.choice(np.asarray(spec.src_nodes, dtype=np.int64)))
        right = int(rng.choice(np.asarray(spec.dst_nodes, dtype=np.int64)))
        if left == right:
            continue
        try:
            return graph.orient_edge(family, (left, right))
        except ValueError:
            continue
    raise RuntimeError(f"could not sample a domain pair for {family}")


def sample_clean_edge_queries(
    snapshot: RepresentationSnapshot,
    family_names: Sequence[str],
    *,
    queries_per_family: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    edges: List[tuple[int, int]] = []
    families: List[int] = []
    labels: List[float] = []
    current: List[float] = []
    per_class = max(1, int(queries_per_family) // 2)
    for family_id, family in enumerate(family_names):
        positives = snapshot.clean.family_edges[family]
        if positives:
            indices = rng.integers(0, len(positives), size=per_class)
            for index in indices.tolist():
                edge = snapshot.clean.orient_edge(family, positives[int(index)])
                edges.append(edge)
                families.append(family_id)
                labels.append(1.0)
                current.append(float(snapshot.graph.has_edge(*edge)))
        negative_count = per_class if positives else 2 * per_class
        attempts = 0
        added = 0
        clean_edges = snapshot.clean.family_edge_sets[family]
        while added < negative_count and attempts < max(1024, 64 * negative_count):
            attempts += 1
            edge = _sample_domain_pair(snapshot.clean, family, rng)
            if edge in clean_edges:
                continue
            edges.append(edge)
            families.append(family_id)
            labels.append(0.0)
            current.append(float(snapshot.graph.has_edge(*edge)))
            added += 1
        if added != negative_count:
            raise RuntimeError(f"insufficient negative queries for {family}")
    return (
        np.asarray(edges, dtype=np.int64).T,
        np.asarray(families, dtype=np.int64),
        np.asarray(labels, dtype=np.float32),
        np.asarray(current, dtype=np.float32),
    )


def _average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    positives = int(labels.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    ordered_labels = labels[order]
    ordered_scores = scores[order]
    cumulative_true = np.cumsum(ordered_labels)
    threshold_ends = np.flatnonzero(
        np.r_[ordered_scores[1:] != ordered_scores[:-1], True]
    )
    true_at_threshold = cumulative_true[threshold_ends]
    precision = true_at_threshold / (threshold_ends + 1)
    recall = true_at_threshold / positives
    recall_delta = np.diff(np.r_[0.0, recall])
    return float(np.sum(recall_delta * precision))


def _role_normalizer(
    role_rows: np.ndarray, node_types: np.ndarray, type_count: int
) -> tuple[np.ndarray, np.ndarray]:
    transformed = np.log1p(np.asarray(role_rows, dtype=np.float32))
    means = np.zeros((int(type_count), transformed.shape[1]), dtype=np.float32)
    scales = np.ones_like(means)
    for type_id in range(int(type_count)):
        values = transformed[np.asarray(node_types) == type_id]
        if not len(values):
            continue
        means[type_id] = values.mean(axis=0)
        scales[type_id] = np.maximum(values.std(axis=0), 0.1)
    return means, scales


def _normalize_role_rows(
    role_rows: np.ndarray,
    node_types: np.ndarray,
    means: np.ndarray,
    scales: np.ndarray,
) -> np.ndarray:
    transformed = np.log1p(np.asarray(role_rows, dtype=np.float32))
    return (
        transformed
        - means[np.asarray(node_types, dtype=np.int64)]
    ) / scales[np.asarray(node_types, dtype=np.int64)]


def train_representation_warmup(
    dataset: RepresentationWarmupDataset,
    *,
    epochs: int = 10,
    queries_per_family: int = 128,
    learning_rate: float = 3.0e-4,
    edge_loss_weight: float = 1.0,
    role_loss_weight: float = 1.0,
    variance_loss_weight: float = 0.05,
    min_feature_std: float = 0.2,
    device: str | torch.device = "cuda",
    seed: int = 0,
    **model_kwargs,
) -> tuple[WarmupHGTEncoder, Mapping[str, object]]:
    torch.manual_seed(int(seed))
    rng = np.random.default_rng(int(seed))
    device = torch.device(device)
    sample = dataset.snapshots[0]
    feature_dim = graph_node_features(sample.graph, sample.time_fraction).shape[1]
    model = WarmupHGTEncoder(
        node_feature_dim=feature_dim,
        type_count=dataset.type_count,
        family_count=dataset.family_count,
        **model_kwargs,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate))
    validation_trajectory = max(item.trajectory_id for item in dataset.snapshots)
    train_snapshots = [
        item for item in dataset.snapshots if item.trajectory_id != validation_trajectory
    ]
    validation_snapshots = [
        item for item in dataset.snapshots if item.trajectory_id == validation_trajectory
    ]
    node_types = torch.from_numpy(dataset.node_types).long().to(device)
    role_means, role_scales = _role_normalizer(
        dataset.snapshots[0].role_rows,
        dataset.node_types,
        dataset.type_count,
    )
    normalized_reference_roles = _normalize_role_rows(
        dataset.snapshots[0].role_rows,
        dataset.node_types,
        role_means,
        role_scales,
    )
    type_median_roles = np.zeros(
        (dataset.type_count, dataset.family_count), dtype=np.float32
    )
    for type_id in range(dataset.type_count):
        mask = dataset.node_types == type_id
        if np.any(mask):
            type_median_roles[type_id] = np.median(
                normalized_reference_roles[mask], axis=0
            )
    role_baseline_template = type_median_roles[dataset.node_types]
    final_loss = float("nan")
    for epoch in range(int(epochs)):
        model.train()
        losses = []
        for snapshot_index in rng.permutation(len(train_snapshots)).tolist():
            snapshot = train_snapshots[int(snapshot_index)]
            edge_index, edge_types = graph_edge_tensors(snapshot.graph)
            query_edges, query_families, labels, current = sample_clean_edge_queries(
                snapshot,
                dataset.family_names,
                queries_per_family=int(queries_per_family),
                seed=int(seed) + 10_000_019 * epoch + 97_409 * snapshot_index,
            )
            features = torch.from_numpy(
                graph_node_features(snapshot.graph, snapshot.time_fraction)
            ).float().to(device)
            target_roles = torch.from_numpy(
                _normalize_role_rows(
                    snapshot.role_rows,
                    dataset.node_types,
                    role_means,
                    role_scales,
                )
            ).float().to(device)
            optimizer.zero_grad(set_to_none=True)
            hidden = model.encode(
                features,
                node_types,
                torch.from_numpy(edge_index).long().to(device),
                torch.from_numpy(edge_types).long().to(device),
            )
            logits = model.edge_logits(
                hidden,
                torch.from_numpy(query_edges).long().to(device),
                torch.from_numpy(query_families).long().to(device),
                torch.from_numpy(current).float().to(device),
                snapshot.time_fraction,
            )
            edge_loss = nn.functional.binary_cross_entropy_with_logits(
                logits, torch.from_numpy(labels).float().to(device)
            )
            role_loss = nn.functional.smooth_l1_loss(
                model.role_prediction(hidden), target_roles
            )
            feature_std = torch.sqrt(hidden.var(dim=0, unbiased=False) + 1.0e-4)
            variance_loss = torch.relu(float(min_feature_std) - feature_std).mean()
            loss = (
                float(edge_loss_weight) * edge_loss
                + float(role_loss_weight) * role_loss
                + float(variance_loss_weight) * variance_loss
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        final_loss = float(np.mean(losses))
        print(
            f"[LATENT-REP-EPOCH] epoch={epoch} loss={final_loss:.6f}", flush=True
        )

    model.eval()
    edge_rows = []
    role_rows = []
    hidden_stds = []
    with torch.no_grad():
        for snapshot_index, snapshot in enumerate(validation_snapshots):
            edge_index, edge_types = graph_edge_tensors(snapshot.graph)
            query_edges, query_families, labels, current = sample_clean_edge_queries(
                snapshot,
                dataset.family_names,
                queries_per_family=int(queries_per_family),
                seed=int(seed) + 1_000_000_007 + 97_409 * snapshot_index,
            )
            hidden = model.encode(
                torch.from_numpy(
                    graph_node_features(snapshot.graph, snapshot.time_fraction)
                ).float().to(device),
                node_types,
                torch.from_numpy(edge_index).long().to(device),
                torch.from_numpy(edge_types).long().to(device),
            )
            probabilities = torch.sigmoid(
                model.edge_logits(
                    hidden,
                    torch.from_numpy(query_edges).long().to(device),
                    torch.from_numpy(query_families).long().to(device),
                    torch.from_numpy(current).float().to(device),
                    snapshot.time_fraction,
                )
            ).cpu().numpy()
            edge_rows.append(
                (
                    labels,
                    probabilities,
                    current,
                    query_families,
                    snapshot.time_fraction,
                )
            )
            prediction = model.role_prediction(hidden).cpu().numpy()
            target = _normalize_role_rows(
                snapshot.role_rows,
                dataset.node_types,
                role_means,
                role_scales,
            )
            role_rows.append((target, prediction, snapshot.time_fraction))
            hidden_stds.append(float(hidden.std(dim=0, unbiased=False).mean().cpu()))

    labels = np.concatenate([row[0] for row in edge_rows])
    scores = np.concatenate([row[1] for row in edge_rows])
    current_scores = np.concatenate([row[2] for row in edge_rows])
    families = np.concatenate([row[3] for row in edge_rows])
    family_ap = {}
    family_baseline = {}
    family_current_ap = {}
    for family_id, family in enumerate(dataset.family_names):
        mask = families == family_id
        family_ap[family] = _average_precision(labels[mask], scores[mask])
        family_baseline[family] = float(labels[mask].mean())
        family_current_ap[family] = _average_precision(
            labels[mask], current_scores[mask]
        )
    role_target = np.concatenate([row[0] for row in role_rows])
    role_prediction = np.concatenate([row[1] for row in role_rows])
    baseline_prediction = np.concatenate(
        [role_baseline_template for _ in role_rows], axis=0
    )
    role_mae = float(np.abs(role_prediction - role_target).mean())
    role_baseline_mae = float(np.abs(baseline_prediction - role_target).mean())
    macro_ap = float(np.nanmean(list(family_ap.values())))
    macro_baseline = float(np.nanmean(list(family_baseline.values())))
    macro_current_ap = float(np.nanmean(list(family_current_ap.values())))
    by_noise = {}
    for level in sorted({float(row[4]) for row in edge_rows}):
        level_edge_rows = [row for row in edge_rows if float(row[4]) == level]
        level_labels = np.concatenate([row[0] for row in level_edge_rows])
        level_scores = np.concatenate([row[1] for row in level_edge_rows])
        level_current = np.concatenate([row[2] for row in level_edge_rows])
        level_families = np.concatenate([row[3] for row in level_edge_rows])
        level_model_ap = []
        level_current_ap = []
        for family_id in range(dataset.family_count):
            mask = level_families == family_id
            level_model_ap.append(_average_precision(level_labels[mask], level_scores[mask]))
            level_current_ap.append(
                _average_precision(level_labels[mask], level_current[mask])
            )
        level_role_rows = [row for row in role_rows if float(row[2]) == level]
        level_role_target = np.concatenate([row[0] for row in level_role_rows])
        level_role_prediction = np.concatenate([row[1] for row in level_role_rows])
        level_role_mae = float(
            np.abs(level_role_prediction - level_role_target).mean()
        )
        level_role_baseline = float(
            np.abs(role_baseline_template - level_role_target).mean()
        )
        by_noise[str(level)] = {
            "edge_macro_ap": float(np.nanmean(level_model_ap)),
            "edge_current_state_ap": float(np.nanmean(level_current_ap)),
            "edge_gain_over_current": float(
                np.nanmean(level_model_ap) - np.nanmean(level_current_ap)
            ),
            "role_mae": level_role_mae,
            "role_type_baseline_mae": level_role_baseline,
            "role_mae_reduction": level_role_baseline - level_role_mae,
        }
    report = {
        "format": "dihug-latent-representation-warmup-v1",
        "train_loss": final_loss,
        "train_snapshots": len(train_snapshots),
        "validation_snapshots": len(validation_snapshots),
        "edge_macro_ap": macro_ap,
        "edge_prevalence_baseline": macro_baseline,
        "edge_ap_gain": macro_ap - macro_baseline,
        "edge_current_state_ap": macro_current_ap,
        "edge_gain_over_current": macro_ap - macro_current_ap,
        "edge_ap_by_family": family_ap,
        "edge_baseline_by_family": family_baseline,
        "edge_current_ap_by_family": family_current_ap,
        "role_mae": role_mae,
        "role_type_baseline_mae": role_baseline_mae,
        "role_mae_reduction": role_baseline_mae - role_mae,
        "embedding_mean_feature_std": float(np.mean(hidden_stds)),
        "noise_levels": sorted(
            {float(snapshot.time_fraction) for snapshot in validation_snapshots}
        ),
        "by_noise": by_noise,
    }
    return model, report


def _load_compatible_state(
    model: nn.Module, state: Mapping[str, torch.Tensor]
) -> tuple[str, ...]:
    current = model.state_dict()
    compatible = {
        key: value
        for key, value in state.items()
        if key in current and tuple(value.shape) == tuple(current[key].shape)
    }
    model.load_state_dict(compatible, strict=False)
    return tuple(sorted(compatible))


def _cosine_distance(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return 1.0 - nn.functional.cosine_similarity(left, right, dim=-1)


def train_role_conditioned_alignment(
    dataset: RepresentationWarmupDataset,
    *,
    base_model_state: Mapping[str, torch.Tensor],
    epochs: int = 8,
    queries_per_family: int = 128,
    learning_rate: float = 2.0e-4,
    edge_loss_weight: float = 1.0,
    role_loss_weight: float = 0.5,
    latent_loss_weight: float = 0.25,
    prototype_loss_weight: float = 0.25,
    variance_loss_weight: float = 0.05,
    min_feature_std: float = 0.2,
    ema_decay: float = 0.99,
    prototype_count: int = 3,
    device: str | torch.device = "cuda",
    seed: int = 0,
    **model_kwargs,
) -> tuple[WarmupHGTEncoder, Mapping[str, object]]:
    """Align noisy representations to fixed clean teachers and role prototypes."""

    torch.manual_seed(int(seed))
    rng = np.random.default_rng(int(seed))
    device = torch.device(device)
    sample = dataset.snapshots[0]
    feature_dim = graph_node_features(sample.graph, sample.time_fraction).shape[1]
    model = WarmupHGTEncoder(
        node_feature_dim=feature_dim,
        type_count=dataset.type_count,
        family_count=dataset.family_count,
        role_condition_dim=dataset.family_count,
        prototype_count=int(prototype_count),
        **model_kwargs,
    ).to(device)
    loaded_keys = _load_compatible_state(model, base_model_state)
    teacher = copy.deepcopy(model).to(device).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate))
    validation_trajectory = max(item.trajectory_id for item in dataset.snapshots)
    train_snapshots = [
        item for item in dataset.snapshots if item.trajectory_id != validation_trajectory
    ]
    validation_snapshots = [
        item for item in dataset.snapshots if item.trajectory_id == validation_trajectory
    ]
    node_types = torch.from_numpy(dataset.node_types).long().to(device)
    role_means, role_scales = _role_normalizer(
        dataset.snapshots[0].role_rows,
        dataset.node_types,
        dataset.type_count,
    )
    final_losses = {}
    for epoch in range(int(epochs)):
        model.train()
        epoch_rows = []
        for snapshot_index in rng.permutation(len(train_snapshots)).tolist():
            snapshot = train_snapshots[int(snapshot_index)]
            edge_index, edge_types = graph_edge_tensors(snapshot.graph)
            clean_edge_index, clean_edge_types = graph_edge_tensors(snapshot.clean)
            query_edges, query_families, labels, current = sample_clean_edge_queries(
                snapshot,
                dataset.family_names,
                queries_per_family=int(queries_per_family),
                seed=int(seed) + 10_000_019 * epoch + 97_409 * snapshot_index,
            )
            role_condition = torch.from_numpy(
                _normalize_role_rows(
                    snapshot.role_rows,
                    dataset.node_types,
                    role_means,
                    role_scales,
                )
            ).float().to(device)
            optimizer.zero_grad(set_to_none=True)
            hidden = model.encode(
                torch.from_numpy(
                    graph_node_features(snapshot.graph, snapshot.time_fraction)
                ).float().to(device),
                node_types,
                torch.from_numpy(edge_index).long().to(device),
                torch.from_numpy(edge_types).long().to(device),
                role_condition,
            )
            with torch.no_grad():
                clean_hidden = teacher.encode(
                    torch.from_numpy(graph_node_features(snapshot.clean, 0.0))
                    .float()
                    .to(device),
                    node_types,
                    torch.from_numpy(clean_edge_index).long().to(device),
                    torch.from_numpy(clean_edge_types).long().to(device),
                    role_condition,
                )
            logits = model.edge_logits(
                hidden,
                torch.from_numpy(query_edges).long().to(device),
                torch.from_numpy(query_families).long().to(device),
                torch.from_numpy(current).float().to(device),
                snapshot.time_fraction,
            )
            edge_loss = nn.functional.binary_cross_entropy_with_logits(
                logits, torch.from_numpy(labels).float().to(device)
            )
            role_loss = nn.functional.mse_loss(
                model.role_prediction(hidden), role_condition
            )
            prototypes = model.role_prototypes(role_condition, node_types)
            teacher_distance = _cosine_distance(
                prototypes, clean_hidden[:, None, :]
            )
            assignment = teacher_distance.argmin(dim=1)
            chosen = prototypes[
                torch.arange(prototypes.shape[0], device=device), assignment
            ]
            prototype_loss = _cosine_distance(chosen, clean_hidden).mean()
            latent_loss = _cosine_distance(hidden, chosen.detach()).mean()
            feature_std = torch.sqrt(hidden.var(dim=0, unbiased=False) + 1.0e-4)
            variance_loss = torch.relu(float(min_feature_std) - feature_std).mean()
            loss = (
                float(edge_loss_weight) * edge_loss
                + float(role_loss_weight) * role_loss
                + float(latent_loss_weight) * latent_loss
                + float(prototype_loss_weight) * prototype_loss
                + float(variance_loss_weight) * variance_loss
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            with torch.no_grad():
                for teacher_parameter, parameter in zip(
                    teacher.parameters(), model.parameters()
                ):
                    teacher_parameter.mul_(float(ema_decay)).add_(
                        parameter, alpha=1.0 - float(ema_decay)
                    )
            epoch_rows.append(
                (
                    float(loss.detach().cpu()),
                    float(edge_loss.detach().cpu()),
                    float(role_loss.detach().cpu()),
                    float(latent_loss.detach().cpu()),
                    float(prototype_loss.detach().cpu()),
                )
            )
        means = np.asarray(epoch_rows).mean(axis=0)
        final_losses = {
            "total": float(means[0]),
            "edge": float(means[1]),
            "role": float(means[2]),
            "latent": float(means[3]),
            "prototype": float(means[4]),
        }
        print(
            "[LATENT-ALIGN-EPOCH] "
            f"epoch={epoch} total={means[0]:.6f} edge={means[1]:.6f} "
            f"role={means[2]:.6f} latent={means[3]:.6f} "
            f"prototype={means[4]:.6f}",
            flush=True,
        )

    model.eval()
    teacher.eval()
    rows = []
    with torch.no_grad():
        for snapshot_index, snapshot in enumerate(validation_snapshots):
            edge_index, edge_types = graph_edge_tensors(snapshot.graph)
            clean_edge_index, clean_edge_types = graph_edge_tensors(snapshot.clean)
            query_edges, query_families, labels, current = sample_clean_edge_queries(
                snapshot,
                dataset.family_names,
                queries_per_family=int(queries_per_family),
                seed=int(seed) + 1_000_000_007 + 97_409 * snapshot_index,
            )
            role_array = _normalize_role_rows(
                snapshot.role_rows,
                dataset.node_types,
                role_means,
                role_scales,
            )
            role_condition = torch.from_numpy(role_array).float().to(device)
            noisy_features = torch.from_numpy(
                graph_node_features(snapshot.graph, snapshot.time_fraction)
            ).float().to(device)
            hidden = model.encode(
                noisy_features,
                node_types,
                torch.from_numpy(edge_index).long().to(device),
                torch.from_numpy(edge_types).long().to(device),
                role_condition,
            )
            no_context_hidden = model.encode(
                noisy_features,
                node_types,
                torch.empty((2, 0), dtype=torch.long, device=device),
                torch.empty((0,), dtype=torch.long, device=device),
                role_condition,
            )
            clean_hidden = teacher.encode(
                torch.from_numpy(graph_node_features(snapshot.clean, 0.0))
                .float()
                .to(device),
                node_types,
                torch.from_numpy(clean_edge_index).long().to(device),
                torch.from_numpy(clean_edge_types).long().to(device),
                role_condition,
            )
            probabilities = torch.sigmoid(
                model.edge_logits(
                    hidden,
                    torch.from_numpy(query_edges).long().to(device),
                    torch.from_numpy(query_families).long().to(device),
                    torch.from_numpy(current).float().to(device),
                    snapshot.time_fraction,
                )
            ).cpu().numpy()
            no_context_probabilities = torch.sigmoid(
                model.edge_logits(
                    no_context_hidden,
                    torch.from_numpy(query_edges).long().to(device),
                    torch.from_numpy(query_families).long().to(device),
                    torch.from_numpy(current).float().to(device),
                    snapshot.time_fraction,
                )
            ).cpu().numpy()
            role_prediction = model.role_prediction(hidden)
            prototypes = model.role_prototypes(role_condition, node_types)
            assignment = _cosine_distance(
                prototypes, clean_hidden[:, None, :]
            ).argmin(dim=1)
            chosen = prototypes[
                torch.arange(prototypes.shape[0], device=device), assignment
            ]
            shuffled = role_array.copy()
            shuffle_rng = np.random.default_rng(
                int(seed) + 90_000_119 + snapshot_index
            )
            for type_id in range(dataset.type_count):
                nodes = np.flatnonzero(dataset.node_types == type_id)
                shuffled[nodes] = shuffled[shuffle_rng.permutation(nodes)]
            shuffled_hidden = model.encode(
                noisy_features,
                node_types,
                torch.from_numpy(edge_index).long().to(device),
                torch.from_numpy(edge_types).long().to(device),
                torch.from_numpy(shuffled).float().to(device),
            )
            rows.append(
                {
                    "level": float(snapshot.time_fraction),
                    "labels": labels,
                    "scores": probabilities,
                    "no_context_scores": no_context_probabilities,
                    "current": current,
                    "families": query_families,
                    "role_squared_error": float(
                        torch.square(role_prediction - role_condition).mean().cpu()
                    ),
                    "latent_cosine": float(
                        (1.0 - _cosine_distance(hidden, clean_hidden)).mean().cpu()
                    ),
                    "no_context_latent_cosine": float(
                        (
                            1.0
                            - _cosine_distance(no_context_hidden, clean_hidden)
                        )
                        .mean()
                        .cpu()
                    ),
                    "prototype_cosine": float(
                        (1.0 - _cosine_distance(chosen, clean_hidden)).mean().cpu()
                    ),
                    "role_shuffle_distance": float(
                        _cosine_distance(hidden, shuffled_hidden).mean().cpu()
                    ),
                    "embedding_std": float(
                        hidden.std(dim=0, unbiased=False).mean().cpu()
                    ),
                }
            )

    by_noise = {}
    for level in sorted({row["level"] for row in rows}):
        selected = [row for row in rows if row["level"] == level]
        labels = np.concatenate([row["labels"] for row in selected])
        scores = np.concatenate([row["scores"] for row in selected])
        no_context_scores = np.concatenate(
            [row["no_context_scores"] for row in selected]
        )
        current = np.concatenate([row["current"] for row in selected])
        families = np.concatenate([row["families"] for row in selected])
        model_ap = []
        current_ap = []
        no_context_ap = []
        for family_id in range(dataset.family_count):
            mask = families == family_id
            model_ap.append(_average_precision(labels[mask], scores[mask]))
            current_ap.append(_average_precision(labels[mask], current[mask]))
            no_context_ap.append(
                _average_precision(labels[mask], no_context_scores[mask])
            )
        by_noise[str(level)] = {
            "edge_macro_ap": float(np.nanmean(model_ap)),
            "edge_current_state_ap": float(np.nanmean(current_ap)),
            "edge_no_context_ap": float(np.nanmean(no_context_ap)),
            "edge_context_gain": float(
                np.nanmean(model_ap) - np.nanmean(no_context_ap)
            ),
            "edge_gain_over_current": float(
                np.nanmean(model_ap) - np.nanmean(current_ap)
            ),
            "role_rmse": float(
                np.sqrt(np.mean([row["role_squared_error"] for row in selected]))
            ),
            "latent_clean_cosine": float(
                np.mean([row["latent_cosine"] for row in selected])
            ),
            "no_context_latent_clean_cosine": float(
                np.mean([row["no_context_latent_cosine"] for row in selected])
            ),
            "latent_context_gain": float(
                np.mean(
                    [
                        row["latent_cosine"] - row["no_context_latent_cosine"]
                        for row in selected
                    ]
                )
            ),
            "prototype_clean_cosine": float(
                np.mean([row["prototype_cosine"] for row in selected])
            ),
            "role_shuffle_distance": float(
                np.mean([row["role_shuffle_distance"] for row in selected])
            ),
        }
    report = {
        "format": "dihug-role-conditioned-latent-alignment-v1",
        "loaded_base_keys": len(loaded_keys),
        "prototype_count": int(prototype_count),
        "train_losses": final_losses,
        "embedding_mean_feature_std": float(
            np.mean([row["embedding_std"] for row in rows])
        ),
        "by_noise": by_noise,
    }
    return model, report
