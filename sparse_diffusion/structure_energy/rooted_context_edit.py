from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations_with_replacement
from pathlib import Path
from typing import Mapping, Sequence, Tuple

import numpy as np
import torch
from torch import nn

from sparse_diffusion.models.pyhgt_denoiser import HGTConv

from .global_context_joint_edit import GlobalContextJointEditResidual
from .graph import TypedGraph
from .hgt_edit_denoiser import (
    candidate_endpoint_matrix,
    graph_edge_tensors,
    graph_node_features,
)
from .mixed_edit_diffusion import (
    EditProposal,
    align_mixed_edit_features_for_model,
    apply_edit,
    mixed_edit_candidate_matrix,
)
from .role_guided_edit_diffusion import sample_switch_refinement_candidates
from .roles import permuted_witness, sample_role_plan
from .switch_diffusion import SwitchDenoiserMLP


DEFAULT_DEGREE_BUCKETS = (0, 1, 2, 4, 8, 16, 32)


def schema_supports_triangles(graph: TypedGraph) -> bool:
    domains = {
        tuple(sorted((int(spec.src_type), int(spec.dst_type))))
        for spec in graph.family_specs.values()
    }
    for left, center, right in combinations_with_replacement(
        range(len(graph.type_names)), 3
    ):
        required = (
            tuple(sorted((left, center))),
            tuple(sorted((left, right))),
            tuple(sorted((center, right))),
        )
        if all(domain in domains for domain in required):
            return True
    return False


def _degree_bucket_ids(
    degrees: np.ndarray, boundaries: Sequence[int] = DEFAULT_DEGREE_BUCKETS
) -> np.ndarray:
    boundaries = np.asarray(tuple(int(value) for value in boundaries), dtype=np.int64)
    if boundaries.ndim != 1 or boundaries.size == 0 or boundaries[0] != 0:
        raise ValueError("degree bucket boundaries must start at zero")
    if np.any(np.diff(boundaries) <= 0):
        raise ValueError("degree bucket boundaries must be strictly increasing")
    return np.searchsorted(boundaries, np.asarray(degrees), side="right") - 1


def rooted_triangle_counts(graph: TypedGraph) -> np.ndarray:
    """Count triangles incident to every node exactly once."""

    counts = np.zeros(graph.num_nodes, dtype=np.float64)
    for left in range(graph.num_nodes):
        for right in graph.adjacency[left]:
            if int(right) <= left:
                continue
            for third in graph.adjacency[left] & graph.adjacency[int(right)]:
                if int(third) <= int(right):
                    continue
                counts[left] += 1.0
                counts[int(right)] += 1.0
                counts[int(third)] += 1.0
    return counts


@dataclass(frozen=True)
class RootedClosureSignature:
    values: np.ndarray
    weights: np.ndarray
    type_count: int
    bucket_count: int
    boundaries: Tuple[int, ...]

    def squared_error(self, other: "RootedClosureSignature") -> float:
        if self.values.shape != other.values.shape:
            raise ValueError("rooted closure signatures have different shapes")
        return float(np.sum(self.weights * np.square(other.values - self.values)))


def rooted_closure_signature(
    graph: TypedGraph,
    boundaries: Sequence[int] = DEFAULT_DEGREE_BUCKETS,
) -> RootedClosureSignature:
    """Permutation-invariant typed, degree-conditioned closure moments.

    For every node-type/degree bucket, the signature stores both wedge-weighted
    closure and mean local clustering. The first preserves total closure mass;
    the second prevents the same triangles from being concentrated only on hubs.
    """

    boundaries = tuple(int(value) for value in boundaries)
    degrees = graph.degrees().astype(np.int64)
    buckets = _degree_bucket_ids(degrees, boundaries)
    bucket_count = len(boundaries)
    rooted = rooted_triangle_counts(graph)
    wedges = degrees * np.maximum(degrees - 1, 0) / 2.0
    local = np.divide(rooted, wedges, out=np.zeros_like(rooted), where=wedges > 0)
    values = np.zeros((len(graph.type_names), bucket_count, 2), dtype=np.float64)
    weights = np.zeros_like(values)
    total_nodes = max(graph.num_nodes, 1)
    total_wedges = max(float(wedges.sum()), 1.0)
    for type_id in range(len(graph.type_names)):
        for bucket_id in range(bucket_count):
            mask = (graph.node_types == type_id) & (buckets == bucket_id)
            node_count = int(mask.sum())
            wedge_count = float(wedges[mask].sum())
            if wedge_count > 0.0:
                values[type_id, bucket_id, 0] = float(rooted[mask].sum()) / wedge_count
            if node_count > 0:
                values[type_id, bucket_id, 1] = float(local[mask].mean())
            weights[type_id, bucket_id, 0] = wedge_count / total_wedges
            weights[type_id, bucket_id, 1] = node_count / total_nodes
    return RootedClosureSignature(
        values=values.reshape(-1),
        weights=weights.reshape(-1),
        type_count=len(graph.type_names),
        bucket_count=bucket_count,
        boundaries=boundaries,
    )


def rooted_closure_utility(
    graph: TypedGraph,
    proposal: EditProposal,
    target: RootedClosureSignature,
    current: RootedClosureSignature | None = None,
) -> float:
    """Exact decrease in rooted-signature error from one legal edit."""

    before = (
        rooted_closure_signature(graph, target.boundaries)
        if current is None
        else current
    )
    degree_delta = np.zeros(graph.num_nodes, dtype=np.int64)
    for edge in proposal.old_edges:
        degree_delta[int(edge[0])] -= 1
        degree_delta[int(edge[1])] -= 1
    for edge in proposal.new_edges:
        degree_delta[int(edge[0])] += 1
        degree_delta[int(edge[1])] += 1
    if np.any(degree_delta):
        candidate = graph.clone()
        apply_edit(candidate, proposal)
        after = rooted_closure_signature(candidate, target.boundaries)
        return target.squared_error(before) - target.squared_error(after)

    def neighbors_after(node: int) -> set[int]:
        neighbors = set(graph.adjacency[int(node)])
        for left, right in proposal.old_edges:
            if int(left) == int(node):
                neighbors.discard(int(right))
            elif int(right) == int(node):
                neighbors.discard(int(left))
        for left, right in proposal.new_edges:
            if int(left) == int(node):
                neighbors.add(int(right))
            elif int(right) == int(node):
                neighbors.add(int(left))
        return neighbors

    rooted_delta: dict[int, float] = {}

    def accumulate(edge: tuple[int, int], common: set[int], sign: float) -> None:
        left, right = int(edge[0]), int(edge[1])
        rooted_delta[left] = rooted_delta.get(left, 0.0) + sign * len(common)
        rooted_delta[right] = rooted_delta.get(right, 0.0) + sign * len(common)
        for node in common:
            rooted_delta[int(node)] = rooted_delta.get(int(node), 0.0) + sign

    for edge in proposal.old_edges:
        common = graph.adjacency[int(edge[0])] & graph.adjacency[int(edge[1])]
        accumulate(edge, common, -1.0)
    for edge in proposal.new_edges:
        common = neighbors_after(int(edge[0])) & neighbors_after(int(edge[1]))
        accumulate(edge, common, 1.0)

    degrees = graph.degrees().astype(np.int64)
    buckets = _degree_bucket_ids(degrees, target.boundaries)
    wedges = degrees * np.maximum(degrees - 1, 0) / 2.0
    group_nodes = np.zeros((target.type_count, target.bucket_count), dtype=np.float64)
    group_wedges = np.zeros_like(group_nodes)
    for type_id in range(target.type_count):
        for bucket_id in range(target.bucket_count):
            mask = (graph.node_types == type_id) & (buckets == bucket_id)
            group_nodes[type_id, bucket_id] = float(mask.sum())
            group_wedges[type_id, bucket_id] = float(wedges[mask].sum())
    value_delta = np.zeros((target.type_count, target.bucket_count, 2), dtype=np.float64)
    for node, delta in rooted_delta.items():
        type_id = int(graph.node_types[node])
        bucket_id = int(buckets[node])
        if group_wedges[type_id, bucket_id] > 0.0:
            value_delta[type_id, bucket_id, 0] += delta / group_wedges[type_id, bucket_id]
        if wedges[node] > 0.0 and group_nodes[type_id, bucket_id] > 0.0:
            value_delta[type_id, bucket_id, 1] += (
                delta / wedges[node] / group_nodes[type_id, bucket_id]
            )
    after_values = before.values + value_delta.reshape(-1)
    before_error = float(np.sum(target.weights * np.square(before.values - target.values)))
    after_error = float(np.sum(target.weights * np.square(after_values - target.values)))
    return before_error - after_error


@dataclass(frozen=True)
class RootedContextDataset:
    candidate_features: np.ndarray
    candidate_endpoints: np.ndarray
    candidate_utilities: np.ndarray
    node_features: np.ndarray
    edge_indices: Tuple[np.ndarray, ...]
    edge_types: Tuple[np.ndarray, ...]
    trajectory_ids: np.ndarray
    node_types: np.ndarray
    type_count: int
    family_count: int

    def __post_init__(self) -> None:
        examples, candidates, _ = self.candidate_features.shape
        if self.candidate_endpoints.shape != (examples, candidates, 8):
            raise ValueError("candidate endpoint shape mismatch")
        if self.candidate_utilities.shape != (examples, candidates):
            raise ValueError("candidate utility shape mismatch")
        if self.node_features.shape[:2] != (examples, self.node_types.shape[0]):
            raise ValueError("node feature shape mismatch")
        if len(self.edge_indices) != examples or len(self.edge_types) != examples:
            raise ValueError("edge snapshot count mismatch")


def build_rooted_context_dataset(
    source: TypedGraph,
    legacy_model: SwitchDenoiserMLP,
    *,
    trajectories: int = 4,
    examples_per_trajectory: int = 24,
    switches_per_example: int = 128,
    candidates_per_example: int = 16,
    closure_fraction: float = 1.0,
    seed: int = 0,
    feature_profile: str = "typed_closure_compact_v2",
) -> RootedContextDataset:
    if int(trajectories) < 2:
        raise ValueError("rooted context training needs at least two trajectories")
    target = rooted_closure_signature(source)
    features = []
    endpoints = []
    utilities = []
    node_features = []
    edge_indices = []
    edge_types = []
    trajectory_ids = []
    for trajectory_id in range(int(trajectories)):
        trajectory_seed = int(seed) + 1_000_003 * trajectory_id
        rng = np.random.default_rng(trajectory_seed)
        graph = permuted_witness(source, sample_role_plan(source, trajectory_seed + 17))
        for example_id in range(int(examples_per_trajectory)):
            for _ in range(int(switches_per_example)):
                proposal = sample_switch_refinement_candidates(
                    graph, rng, 2, closure_fraction=0.0
                )[1]
                apply_edit(graph, proposal)
            fraction = 1.0 - (example_id + 1) / float(examples_per_trajectory + 1)
            candidates = sample_switch_refinement_candidates(
                graph,
                rng,
                int(candidates_per_example),
                closure_fraction=float(closure_fraction),
            )
            matrix = mixed_edit_candidate_matrix(
                graph,
                candidates,
                fraction,
                feature_profile=feature_profile,
            )
            matrix = align_mixed_edit_features_for_model(
                matrix, legacy_model, family_count=len(graph.family_specs)
            )
            edge_index, relations = graph_edge_tensors(graph)
            current_signature = rooted_closure_signature(graph, target.boundaries)
            features.append(matrix)
            endpoints.append(candidate_endpoint_matrix(candidates))
            utilities.append(
                np.asarray(
                    [
                        rooted_closure_utility(
                            graph, item, target, current=current_signature
                        )
                        for item in candidates
                    ],
                    dtype=np.float32,
                )
            )
            node_features.append(graph_node_features(graph, fraction))
            edge_indices.append(edge_index)
            edge_types.append(relations)
            trajectory_ids.append(trajectory_id)
    return RootedContextDataset(
        candidate_features=np.stack(features).astype(np.float32),
        candidate_endpoints=np.stack(endpoints).astype(np.int64),
        candidate_utilities=np.stack(utilities).astype(np.float32),
        node_features=np.stack(node_features).astype(np.float32),
        edge_indices=tuple(edge_indices),
        edge_types=tuple(edge_types),
        trajectory_ids=np.asarray(trajectory_ids, dtype=np.int64),
        node_types=source.node_types.copy(),
        type_count=len(source.type_names),
        family_count=len(source.family_specs),
    )


class RootedContextEditResidual(nn.Module):
    """Frozen legacy listwise scorer plus zero-initialized sparse-HGT residual."""

    def __init__(
        self,
        legacy_scorer: SwitchDenoiserMLP,
        *,
        node_feature_dim: int,
        candidate_feature_dim: int,
        type_count: int,
        family_count: int,
        hidden_dim: int = 64,
        heads: int = 4,
        layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_dim % heads:
            raise ValueError("HGT hidden dimension must be divisible by head count")
        self.config = {
            "node_feature_dim": int(node_feature_dim),
            "candidate_feature_dim": int(candidate_feature_dim),
            "type_count": int(type_count),
            "family_count": int(family_count),
            "hidden_dim": int(hidden_dim),
            "heads": int(heads),
            "layers": int(layers),
            "dropout": float(dropout),
        }
        self.node_input = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
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
        self.joint = GlobalContextJointEditResidual(
            legacy_scorer,
            node_hidden_dim=hidden_dim,
            candidate_feature_dim=candidate_feature_dim,
            hidden_dim=hidden_dim,
        )
        for parameter in self.joint.legacy_scorer.parameters():
            parameter.requires_grad_(False)

    def encode(
        self,
        node_features: torch.Tensor,
        node_types: torch.Tensor,
        edge_index: torch.Tensor,
        edge_types: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, node_count, _ = node_features.shape
        hidden = self.node_input(node_features.reshape(batch_size * node_count, -1))
        for layer in self.layers:
            hidden = layer(hidden, node_types.reshape(-1), edge_index, edge_types)
        return hidden.reshape(batch_size, node_count, -1)

    def score_encoded(
        self,
        hidden: torch.Tensor,
        candidate_features: torch.Tensor,
        candidate_endpoints: torch.Tensor,
    ) -> torch.Tensor:
        return self.joint(
            candidate_features,
            node_hidden=hidden,
            candidate_endpoints=candidate_endpoints,
        )

    def forward(
        self,
        node_features: torch.Tensor,
        node_types: torch.Tensor,
        edge_index: torch.Tensor,
        edge_types: torch.Tensor,
        candidate_features: torch.Tensor,
        candidate_endpoints: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.encode(node_features, node_types, edge_index, edge_types)
        return self.score_encoded(hidden, candidate_features, candidate_endpoints)

    def encode_graph(
        self,
        graph: TypedGraph,
        time_fraction: float,
        device: torch.device,
    ) -> torch.Tensor:
        edge_index, edge_types = graph_edge_tensors(graph)
        return self.encode(
            torch.from_numpy(graph_node_features(graph, time_fraction))[None]
            .float()
            .to(device),
            torch.from_numpy(graph.node_types)[None].long().to(device),
            torch.from_numpy(edge_index).long().to(device),
            torch.from_numpy(edge_types).long().to(device),
        )

    def score_graph_candidates(
        self,
        graph: TypedGraph,
        candidates: Sequence[EditProposal],
        time_fraction: float,
        feature_profile: str,
        device: torch.device,
        encoded: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = (
            self.encode_graph(graph, time_fraction, device)
            if encoded is None
            else encoded
        )
        matrix = mixed_edit_candidate_matrix(
            graph, candidates, time_fraction, feature_profile=feature_profile
        )
        matrix = align_mixed_edit_features_for_model(
            matrix,
            self.joint.legacy_scorer,
            family_count=len(graph.family_specs),
        )
        endpoints = candidate_endpoint_matrix(candidates)
        logits = self.score_encoded(
            hidden,
            torch.from_numpy(matrix)[None].float().to(device),
            torch.from_numpy(endpoints)[None].long().to(device),
        ).squeeze(0)
        return logits, hidden


def _batch(
    dataset: RootedContextDataset, indices: np.ndarray, device: torch.device
) -> tuple[torch.Tensor, ...]:
    batch_size = len(indices)
    node_count = dataset.node_types.shape[0]
    edge_parts = []
    relation_parts = []
    for batch_id, example_id in enumerate(indices.tolist()):
        edge_parts.append(
            torch.from_numpy(dataset.edge_indices[example_id]).long()
            + batch_id * node_count
        )
        relation_parts.append(torch.from_numpy(dataset.edge_types[example_id]).long())
    return (
        torch.from_numpy(dataset.node_features[indices]).float().to(device),
        torch.from_numpy(np.tile(dataset.node_types, (batch_size, 1))).long().to(device),
        torch.cat(edge_parts, dim=1).to(device),
        torch.cat(relation_parts).to(device),
        torch.from_numpy(dataset.candidate_features[indices]).float().to(device),
        torch.from_numpy(dataset.candidate_endpoints[indices]).long().to(device),
        torch.from_numpy(dataset.candidate_utilities[indices]).float().to(device),
    )


def _teacher_distribution(utilities: torch.Tensor, temperature: float) -> torch.Tensor:
    centered = utilities - utilities.mean(dim=1, keepdim=True)
    scale = utilities.std(dim=1, keepdim=True).clamp(min=1.0e-6)
    return torch.softmax(centered / scale / float(temperature), dim=1)


def train_rooted_context_residual(
    dataset: RootedContextDataset,
    legacy_scorer: SwitchDenoiserMLP,
    *,
    epochs: int = 20,
    batch_size: int = 2,
    learning_rate: float = 3.0e-4,
    weight_decay: float = 1.0e-4,
    teacher_temperature: float = 0.5,
    trust_weight: float = 0.05,
    device: str | torch.device = "cuda",
    seed: int = 0,
    **model_kwargs,
) -> tuple[RootedContextEditResidual, Mapping[str, float]]:
    torch.manual_seed(int(seed))
    rng = np.random.default_rng(int(seed))
    device = torch.device(device)
    validation_trajectory = int(dataset.trajectory_ids.max())
    train_ids = np.flatnonzero(dataset.trajectory_ids != validation_trajectory)
    validation_ids = np.flatnonzero(dataset.trajectory_ids == validation_trajectory)
    model = RootedContextEditResidual(
        legacy_scorer,
        node_feature_dim=dataset.node_features.shape[-1],
        candidate_feature_dim=dataset.candidate_features.shape[-1],
        type_count=dataset.type_count,
        family_count=dataset.family_count,
        **model_kwargs,
    ).to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=weight_decay)
    final_loss = float("nan")
    for epoch in range(int(epochs)):
        model.train()
        losses = []
        shuffled = rng.permutation(train_ids)
        for start in range(0, len(train_ids), int(batch_size)):
            indices = shuffled[start : start + int(batch_size)]
            batch = _batch(dataset, indices, device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(*batch[:-1])
            with torch.no_grad():
                base = model.joint.legacy_scorer(batch[-3])
                teacher = _teacher_distribution(batch[-1], teacher_temperature)
                base_distribution = torch.softmax(base, dim=1)
            task = -(teacher * torch.log_softmax(logits, dim=1)).sum(dim=1).mean()
            trust = nn.functional.kl_div(
                torch.log_softmax(logits, dim=1),
                base_distribution,
                reduction="batchmean",
            )
            loss = task + float(trust_weight) * trust
            loss.backward()
            nn.utils.clip_grad_norm_(trainable, 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        final_loss = float(np.mean(losses))
        print(f"[ROOTED-CONTEXT-EPOCH] epoch={epoch} loss={final_loss:.6f}", flush=True)

    model.eval()
    base_values = []
    residual_values = []
    oracle_values = []
    with torch.no_grad():
        for start in range(0, len(validation_ids), int(batch_size)):
            indices = validation_ids[start : start + int(batch_size)]
            batch = _batch(dataset, indices, device)
            logits = model(*batch[:-1])
            base = model.joint.legacy_scorer(batch[-3])
            utility = batch[-1]
            rows = torch.arange(len(indices), device=device)
            base_values.extend(utility[rows, base.argmax(dim=1)].cpu().tolist())
            residual_values.extend(utility[rows, logits.argmax(dim=1)].cpu().tolist())
            oracle_values.extend(utility.max(dim=1).values.cpu().tolist())
    report = {
        "train_loss": final_loss,
        "train_examples": float(len(train_ids)),
        "validation_examples": float(len(validation_ids)),
        "base_selected_utility": float(np.mean(base_values)),
        "context_selected_utility": float(np.mean(residual_values)),
        "oracle_selected_utility": float(np.mean(oracle_values)),
        "utility_gain": float(np.mean(residual_values) - np.mean(base_values)),
        "relative_utility_gain": float(
            (np.mean(residual_values) - np.mean(base_values))
            / max(abs(float(np.mean(base_values))), 1.0e-12)
        ),
        "oracle_capture": float(
            np.mean(residual_values)
            / max(abs(float(np.mean(oracle_values))), 1.0e-12)
        ),
        "non_regression": float(np.mean(residual_values) >= np.mean(base_values) - 1.0e-9),
    }
    return model, report


def save_rooted_context_checkpoint(
    path: str | Path,
    model: RootedContextEditResidual,
    report: Mapping[str, float],
    legacy_checkpoint: str | Path,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "dihug-rooted-context-edit-residual-v1",
            "model_config": model.config,
            "state_dict": model.state_dict(),
            "report": dict(report),
            "legacy_checkpoint": str(legacy_checkpoint),
            "inference_contract": {
                "training_adjacency": False,
                "graph_metric_target": False,
                "rooted_closure_target_at_inference": False,
            },
        },
        path,
    )


def load_rooted_context_checkpoint(
    path: str | Path,
    legacy_scorer: SwitchDenoiserMLP,
    device: str | torch.device = "cpu",
) -> tuple[RootedContextEditResidual, Mapping]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != "dihug-rooted-context-edit-residual-v1":
        raise ValueError("unsupported rooted context edit checkpoint")
    contract = payload.get("inference_contract", {})
    forbidden = (
        "training_adjacency",
        "graph_metric_target",
        "rooted_closure_target_at_inference",
    )
    if any(contract.get(key) is not False for key in forbidden):
        raise ValueError(f"unsafe rooted context inference contract: {contract}")
    model = RootedContextEditResidual(legacy_scorer, **payload["model_config"])
    model.load_state_dict(payload["state_dict"])
    return model.to(device).eval(), payload
