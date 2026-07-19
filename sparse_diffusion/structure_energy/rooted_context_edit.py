from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations_with_replacement
from pathlib import Path
from typing import Mapping, Sequence, Tuple

import numpy as np
import torch
from torch import nn

from sparse_diffusion.models.pyhgt_denoiser import HGTConv

from .global_context_joint_edit import NodeContextJointEditResidual
from .graph import TypedGraph
from .hgt_edit_denoiser import (
    candidate_endpoint_matrix,
    graph_edge_tensors,
    graph_node_features,
)
from .mixed_edit_diffusion import (
    EditProposal,
    MarginalGraphPrior,
    _family_names,
    align_mixed_edit_features_for_model,
    apply_edit,
    mixed_edit_candidate_matrix,
    sample_closed_form_marginal_noise,
    sample_edit_candidates,
)
from .role_guided_edit_diffusion import (
    sample_role_alignment_candidates,
    sample_switch_refinement_candidates,
)
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
    role_conditions: np.ndarray
    trajectory_ids: np.ndarray
    node_types: np.ndarray
    role_means: np.ndarray
    role_scales: np.ndarray
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
        if self.role_conditions.shape != (
            examples,
            self.node_types.shape[0],
            self.family_count,
        ):
            raise ValueError("role-condition shape mismatch")
        if self.role_means.shape != (self.type_count, self.family_count):
            raise ValueError("role-mean shape mismatch")
        if self.role_scales.shape != (self.type_count, self.family_count):
            raise ValueError("role-scale shape mismatch")


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
        transformed - means[np.asarray(node_types, dtype=np.int64)]
    ) / scales[np.asarray(node_types, dtype=np.int64)]


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
    state_protocol: str = "legacy_switch_chain",
    noise_levels: Sequence[float] = (0.25, 0.5, 0.75, 1.0),
    targeted_role_candidate_fraction: float = 0.0,
    terminal_noop_examples_per_trajectory: int = 0,
) -> RootedContextDataset:
    if int(trajectories) < 2:
        raise ValueError("rooted context training needs at least two trajectories")
    state_protocol = str(state_protocol).strip().lower()
    if state_protocol not in {"legacy_switch_chain", "closed_form_marginal"}:
        raise ValueError(
            "state protocol must be legacy_switch_chain or closed_form_marginal"
        )
    levels = tuple(float(level) for level in noise_levels)
    if not levels or any(not 0.0 < level <= 1.0 for level in levels):
        raise ValueError("noise levels must lie in (0,1]")
    if not 0.0 <= float(targeted_role_candidate_fraction) <= 1.0:
        raise ValueError("targeted role candidate fraction must lie in [0,1]")
    if int(terminal_noop_examples_per_trajectory) < 0:
        raise ValueError("terminal noop examples must be non-negative")
    target = rooted_closure_signature(source)
    family_names = tuple(_family_names(source))
    reference_plan = sample_role_plan(source, int(seed) + 17)
    reference_rows = np.stack(
        [reference_plan.family_degrees[family] for family in family_names], axis=1
    ).astype(np.float32)
    role_means, role_scales = _role_normalizer(
        reference_rows, source.node_types, len(source.type_names)
    )
    prior = MarginalGraphPrior.from_training_graph(source)
    features = []
    endpoints = []
    utilities = []
    node_features = []
    edge_indices = []
    edge_types = []
    role_conditions = []
    trajectory_ids = []

    def append_example(
        graph: TypedGraph,
        candidates: Sequence[EditProposal],
        fraction: float,
        normalized_roles: np.ndarray,
        trajectory_id: int,
    ) -> None:
        if not candidates or candidates[0].action != "noop":
            raise RuntimeError("rooted context candidates must place noop first")
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
        role_conditions.append(normalized_roles)
        trajectory_ids.append(trajectory_id)

    for trajectory_id in range(int(trajectories)):
        trajectory_seed = int(seed) + 1_000_003 * trajectory_id
        rng = np.random.default_rng(trajectory_seed)
        plan = sample_role_plan(source, trajectory_seed + 17)
        clean = permuted_witness(source, plan)
        role_rows = np.stack(
            [plan.family_degrees[family] for family in family_names], axis=1
        ).astype(np.float32)
        normalized_roles = _normalize_role_rows(
            role_rows, source.node_types, role_means, role_scales
        )
        for _ in range(int(terminal_noop_examples_per_trajectory)):
            terminal_candidates = sample_switch_refinement_candidates(
                clean,
                rng,
                int(candidates_per_example),
                closure_fraction=float(closure_fraction),
            )
            append_example(
                clean,
                terminal_candidates,
                0.0,
                normalized_roles,
                trajectory_id,
            )
        graph = clean.clone()
        for example_id in range(int(examples_per_trajectory)):
            if state_protocol == "legacy_switch_chain":
                for _ in range(int(switches_per_example)):
                    proposal = sample_switch_refinement_candidates(
                        graph, rng, 2, closure_fraction=0.0
                    )[1]
                    apply_edit(graph, proposal)
                fraction = 1.0 - (example_id + 1) / float(
                    examples_per_trajectory + 1
                )
                candidates = sample_switch_refinement_candidates(
                    graph,
                    rng,
                    int(candidates_per_example),
                    closure_fraction=float(closure_fraction),
                )
            else:
                fraction = levels[example_id % len(levels)]
                alpha_bar = (1.0 - float(fraction)) ** 2
                graph = sample_closed_form_marginal_noise(
                    clean,
                    prior,
                    alpha_bar=alpha_bar,
                    seed=trajectory_seed + 97_409 * (example_id + 1),
                )
                candidates = sample_role_alignment_candidates(
                    graph,
                    role_rows,
                    family_names,
                    rng,
                    int(candidates_per_example),
                    targeted_fraction=float(targeted_role_candidate_fraction),
                    actions=("relocation", "switch"),
                )
            append_example(
                graph, candidates, fraction, normalized_roles, trajectory_id
            )
    return RootedContextDataset(
        candidate_features=np.stack(features).astype(np.float32),
        candidate_endpoints=np.stack(endpoints).astype(np.int64),
        candidate_utilities=np.stack(utilities).astype(np.float32),
        node_features=np.stack(node_features).astype(np.float32),
        edge_indices=tuple(edge_indices),
        edge_types=tuple(edge_types),
        role_conditions=np.stack(role_conditions).astype(np.float32),
        trajectory_ids=np.asarray(trajectory_ids, dtype=np.int64),
        node_types=source.node_types.copy(),
        role_means=role_means,
        role_scales=role_scales,
        type_count=len(source.type_names),
        family_count=len(source.family_specs),
    )


class RootedContextEditResidual(nn.Module):
    """Frozen listwise scorer plus a node-only sparse-HGT context residual.

    Typed edges participate in message passing through ``edge_index`` and
    ``edge_types``. No existing or candidate edge owns a persistent hidden
    vector; complete edit candidates are decoded from endpoint node states.
    """

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
        role_condition_dim: int = 0,
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
            "role_condition_dim": int(role_condition_dim),
        }
        self.type_count = int(type_count)
        self.role_condition_dim = int(role_condition_dim)
        self.node_input = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.role_input = None
        if self.role_condition_dim > 0:
            self.role_input = nn.Sequential(
                nn.Linear(self.role_condition_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
            )
            self.register_buffer(
                "role_means",
                torch.zeros(self.type_count, self.role_condition_dim),
            )
            self.register_buffer(
                "role_scales",
                torch.ones(self.type_count, self.role_condition_dim),
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
        self.joint = NodeContextJointEditResidual(
            legacy_scorer,
            node_hidden_dim=hidden_dim,
            candidate_feature_dim=candidate_feature_dim,
            hidden_dim=hidden_dim,
        )
        for parameter in self.joint.legacy_scorer.parameters():
            parameter.requires_grad_(False)
        self.register_buffer("residual_scale", torch.tensor(1.0))

    def set_residual_scale(self, value: float) -> None:
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError("residual scale must lie in [0,1]")
        self.residual_scale.fill_(float(value))

    def set_role_normalizer(
        self, means: np.ndarray | torch.Tensor, scales: np.ndarray | torch.Tensor
    ) -> None:
        if self.role_input is None:
            return
        means_tensor = torch.as_tensor(
            means, dtype=self.role_means.dtype, device=self.role_means.device
        )
        scales_tensor = torch.as_tensor(
            scales, dtype=self.role_scales.dtype, device=self.role_scales.device
        )
        if means_tensor.shape != self.role_means.shape:
            raise ValueError("role normalizer mean shape mismatch")
        if scales_tensor.shape != self.role_scales.shape or torch.any(scales_tensor <= 0):
            raise ValueError("role normalizer scale mismatch")
        self.role_means.copy_(means_tensor)
        self.role_scales.copy_(scales_tensor)

    def normalize_role_rows(
        self, role_rows: torch.Tensor, node_types: torch.Tensor
    ) -> torch.Tensor:
        if self.role_input is None:
            raise RuntimeError("role conditioning is disabled")
        transformed = torch.log1p(role_rows.float())
        return (
            transformed - self.role_means[node_types]
        ) / self.role_scales[node_types]

    def encode(
        self,
        node_features: torch.Tensor,
        node_types: torch.Tensor,
        edge_index: torch.Tensor,
        edge_types: torch.Tensor,
        role_conditions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, node_count, _ = node_features.shape
        hidden = self.node_input(node_features.reshape(batch_size * node_count, -1))
        if self.role_input is not None:
            if role_conditions is None:
                raise ValueError("role-conditioned context needs role conditions")
            hidden = hidden + self.role_input(
                role_conditions.reshape(batch_size * node_count, -1)
            )
        for layer in self.layers:
            hidden = layer(hidden, node_types.reshape(-1), edge_index, edge_types)
        return hidden.reshape(batch_size, node_count, -1)

    def score_encoded(
        self,
        hidden: torch.Tensor,
        candidate_features: torch.Tensor,
        candidate_endpoints: torch.Tensor,
    ) -> torch.Tensor:
        base = self.joint.legacy_scorer(candidate_features)
        fused = self.joint(
            candidate_features,
            node_hidden=hidden,
            candidate_endpoints=candidate_endpoints,
        )
        return base + self.residual_scale * (fused - base)

    def forward(
        self,
        node_features: torch.Tensor,
        node_types: torch.Tensor,
        edge_index: torch.Tensor,
        edge_types: torch.Tensor,
        candidate_features: torch.Tensor,
        candidate_endpoints: torch.Tensor,
        role_conditions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden = self.encode(
            node_features,
            node_types,
            edge_index,
            edge_types,
            role_conditions=role_conditions,
        )
        return self.score_encoded(hidden, candidate_features, candidate_endpoints)

    def encode_graph(
        self,
        graph: TypedGraph,
        time_fraction: float,
        device: torch.device,
        role_rows: np.ndarray | None = None,
    ) -> torch.Tensor:
        edge_index, edge_types = graph_edge_tensors(graph)
        role_conditions = None
        if self.role_input is not None:
            if role_rows is None:
                raise ValueError("role-conditioned context needs sampled role rows")
            raw_roles = torch.from_numpy(np.asarray(role_rows))[None].float().to(device)
            node_types = torch.from_numpy(graph.node_types)[None].long().to(device)
            role_conditions = self.normalize_role_rows(raw_roles, node_types)
        return self.encode(
            torch.from_numpy(graph_node_features(graph, time_fraction))[None]
            .float()
            .to(device),
            torch.from_numpy(graph.node_types)[None].long().to(device),
            torch.from_numpy(edge_index).long().to(device),
            torch.from_numpy(edge_types).long().to(device),
            role_conditions=role_conditions,
        )

    def score_graph_candidates(
        self,
        graph: TypedGraph,
        candidates: Sequence[EditProposal],
        time_fraction: float,
        feature_profile: str,
        device: torch.device,
        encoded: torch.Tensor | None = None,
        role_rows: np.ndarray | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = (
            self.encode_graph(graph, time_fraction, device, role_rows=role_rows)
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
        torch.from_numpy(dataset.role_conditions[indices]).float().to(device),
        torch.from_numpy(dataset.candidate_utilities[indices]).float().to(device),
    )


def _teacher_distribution(utilities: torch.Tensor, temperature: float) -> torch.Tensor:
    centered = utilities - utilities.mean(dim=1, keepdim=True)
    scale = utilities.std(dim=1, keepdim=True)
    # Utility is normalized by graph-level rooted statistics, so its absolute
    # magnitude shrinks with graph size. Preserve ranking information down to
    # floating-point resolution instead of imposing a dataset-scale floor.
    resolved = scale > torch.finfo(utilities.dtype).tiny
    safe_scale = torch.where(resolved, scale, torch.ones_like(scale))
    logits = centered / safe_scale / float(temperature)
    logits = torch.where(resolved, logits, torch.zeros_like(logits))
    return torch.softmax(logits, dim=1)


def initialize_context_encoder_from_alignment(
    model: RootedContextEditResidual,
    checkpoint: str | Path,
) -> Mapping[str, object]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("format") != "dihug-role-conditioned-latent-alignment-v1":
        raise ValueError("unsupported latent-alignment checkpoint")
    source = payload.get("model_state")
    if not isinstance(source, Mapping):
        raise ValueError("latent-alignment checkpoint has no model state")
    target = model.state_dict()
    prefixes = ("node_input.", "layers.", "role_input.")
    expected = {key for key in target if key.startswith(prefixes)}
    copied = {}
    incompatible = []
    for key in sorted(expected):
        value = source.get(key)
        if value is None or tuple(value.shape) != tuple(target[key].shape):
            incompatible.append(key)
            continue
        copied[key] = value
    if incompatible:
        raise ValueError(
            "latent-alignment encoder is incompatible with context model: "
            + ", ".join(incompatible[:8])
        )
    target.update(copied)
    model.load_state_dict(target)
    return {
        "alignment_checkpoint": str(checkpoint),
        "loaded_encoder_keys": len(copied),
        "alignment_passed": bool(payload.get("report", {}).get("passed", False)),
    }


def _set_context_encoder_trainable(
    model: RootedContextEditResidual, trainable: bool
) -> None:
    prefixes = ("node_input.", "layers.", "role_input.")
    for name, parameter in model.named_parameters():
        if name.startswith(prefixes):
            parameter.requires_grad_(bool(trainable))


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
    stop_loss_weight: float = 0.25,
    device: str | torch.device = "cuda",
    seed: int = 0,
    alignment_checkpoint: str | Path | None = None,
    freeze_context_encoder: bool = False,
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
    model.set_role_normalizer(dataset.role_means, dataset.role_scales)
    initialization_report: Mapping[str, object] = {}
    if alignment_checkpoint is not None:
        initialization_report = initialize_context_encoder_from_alignment(
            model, alignment_checkpoint
        )
    if freeze_context_encoder:
        _set_context_encoder_trainable(model, False)
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
            logits = model(
                batch[0],
                batch[1],
                batch[2],
                batch[3],
                batch[4],
                batch[5],
                role_conditions=(
                    batch[6] if model.role_condition_dim > 0 else None
                ),
            )
            with torch.no_grad():
                base = model.joint.legacy_scorer(batch[4])
                teacher = _teacher_distribution(batch[-1], teacher_temperature)
                base_distribution = torch.softmax(base, dim=1)
            task = -(teacher * torch.log_softmax(logits, dim=1)).sum(dim=1).mean()
            trust = nn.functional.kl_div(
                torch.log_softmax(logits, dim=1),
                base_distribution,
                reduction="batchmean",
            )
            continue_target = (
                batch[-1][:, 1:].amax(dim=1) > batch[-1][:, 0] + 1.0e-12
            ).float()
            continue_logit = torch.logsumexp(logits[:, 1:], dim=1) - logits[:, 0]
            stop = nn.functional.binary_cross_entropy_with_logits(
                continue_logit, continue_target
            )
            loss = (
                task
                + float(trust_weight) * trust
                + float(stop_loss_weight) * stop
            )
            loss.backward()
            nn.utils.clip_grad_norm_(trainable, 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        final_loss = float(np.mean(losses))
        print(f"[ROOTED-CONTEXT-EPOCH] epoch={epoch} loss={final_loss:.6f}", flush=True)

    model.eval()
    validation_rows = []
    with torch.no_grad():
        for start in range(0, len(validation_ids), int(batch_size)):
            indices = validation_ids[start : start + int(batch_size)]
            batch = _batch(dataset, indices, device)
            logits = model(
                batch[0],
                batch[1],
                batch[2],
                batch[3],
                batch[4],
                batch[5],
                role_conditions=(
                    batch[6] if model.role_condition_dim > 0 else None
                ),
            )
            base = model.joint.legacy_scorer(batch[4])
            utility = batch[-1]
            validation_rows.append((base.cpu(), logits.cpu(), utility.cpu()))
    base_logits = torch.cat([row[0] for row in validation_rows])
    fused_logits = torch.cat([row[1] for row in validation_rows])
    validation_utility = torch.cat([row[2] for row in validation_rows])
    rows = torch.arange(validation_utility.shape[0])
    scale_utilities = {}
    for scale in (0.0, 0.25, 0.5, 0.75, 1.0):
        blended = base_logits + float(scale) * (fused_logits - base_logits)
        selected = validation_utility[rows, blended.argmax(dim=1)]
        scale_utilities[float(scale)] = float(selected.mean())
    selected_scale = max(scale_utilities, key=scale_utilities.get)
    model.set_residual_scale(selected_scale)
    base_values = validation_utility[rows, base_logits.argmax(dim=1)].tolist()
    residual_values = validation_utility[
        rows,
        (base_logits + selected_scale * (fused_logits - base_logits)).argmax(dim=1),
    ].tolist()
    oracle_values = validation_utility.max(dim=1).values.tolist()
    continue_target = (
        validation_utility[:, 1:].amax(dim=1)
        > validation_utility[:, 0] + 1.0e-12
    )
    base_continue = base_logits[:, 1:].amax(dim=1) > base_logits[:, 0]
    context_continue = (
        base_logits + selected_scale * (fused_logits - base_logits)
    )[:, 1:].amax(dim=1) > (
        base_logits + selected_scale * (fused_logits - base_logits)
    )[:, 0]
    terminal = ~continue_target
    report = {
        "train_loss": final_loss,
        "train_examples": float(len(train_ids)),
        "validation_examples": float(len(validation_ids)),
        "candidate_nonzero_utility_fraction": float(
            np.mean(dataset.candidate_utilities != 0.0)
        ),
        "candidate_positive_utility_fraction": float(
            np.mean(dataset.candidate_utilities > 0.0)
        ),
        "examples_with_positive_utility_fraction": float(
            np.mean(np.max(dataset.candidate_utilities, axis=1) > 0.0)
        ),
        "mean_candidate_utility_std": float(
            np.mean(np.std(dataset.candidate_utilities, axis=1))
        ),
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
        "oracle_normalized_gain": float(
            (np.mean(residual_values) - np.mean(base_values))
            / max(abs(float(np.mean(oracle_values))), 1.0e-12)
        ),
        "residual_scale": float(selected_scale),
        "scale_utilities": {
            str(scale): float(value) for scale, value in scale_utilities.items()
        },
        "non_regression": float(np.mean(residual_values) >= np.mean(base_values) - 1.0e-9),
        "context_encoder_frozen": bool(freeze_context_encoder),
        "stop_loss_weight": float(stop_loss_weight),
        "validation_continue_fraction": float(continue_target.float().mean()),
        "base_stop_accuracy": float((base_continue == continue_target).float().mean()),
        "context_stop_accuracy": float(
            (context_continue == continue_target).float().mean()
        ),
        "base_terminal_noop_rate": float(
            (~base_continue[terminal]).float().mean() if terminal.any() else 0.0
        ),
        "context_terminal_noop_rate": float(
            (~context_continue[terminal]).float().mean() if terminal.any() else 0.0
        ),
        **dict(initialization_report),
    }
    return model, report


def save_rooted_context_checkpoint(
    path: str | Path,
    model: RootedContextEditResidual,
    report: Mapping[str, float],
    legacy_checkpoint: str | Path,
    role_prior: Mapping | None = None,
    subtype_schema: Mapping | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    legacy_payload = torch.load(
        legacy_checkpoint, map_location="cpu", weights_only=False
    )
    if legacy_payload.get("format") != "dihug-mixed-edit-diffusion-v1":
        raise ValueError("rooted context bundle needs a mixed-edit legacy checkpoint")
    torch.save(
        {
            "format": "dihug-rooted-context-edit-residual-v1",
            "model_config": model.config,
            "state_dict": model.state_dict(),
            "report": dict(report),
            "legacy_checkpoint": str(legacy_checkpoint),
            "legacy_checkpoint_payload": legacy_payload,
            "exchangeable_role_prior": None if role_prior is None else dict(role_prior),
            "subtype_schema": None if subtype_schema is None else dict(subtype_schema),
            "inference_contract": {
                "training_adjacency": False,
                "graph_metric_target": False,
                "rooted_closure_target_at_inference": False,
                "persistent_edge_hidden_state": False,
                "candidate_context": "typed_endpoint_node_states",
            },
        },
        path,
    )


def load_rooted_context_checkpoint(
    path: str | Path,
    legacy_scorer: SwitchDenoiserMLP | None = None,
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
    if contract.get("persistent_edge_hidden_state", False) is not False:
        raise ValueError("rooted context checkpoint requires persistent edge state")
    if legacy_scorer is None:
        legacy = payload.get("legacy_checkpoint_payload")
        if not isinstance(legacy, Mapping):
            raise ValueError("rooted context checkpoint is not self-contained")
        legacy_scorer = SwitchDenoiserMLP(
            input_dim=int(legacy["input_dim"]),
            hidden_dim=int(legacy["hidden_dim"]),
            layers=int(legacy["layers"]),
            dropout=float(legacy["dropout"]),
        )
        legacy_scorer.load_state_dict(legacy["state_dict"])
    model = RootedContextEditResidual(legacy_scorer, **payload["model_config"])
    missing, unexpected = model.load_state_dict(payload["state_dict"], strict=False)
    if set(missing) - {"residual_scale"} or unexpected:
        raise ValueError(
            f"rooted context state mismatch: missing={missing}, unexpected={unexpected}"
        )
    return model.to(device).eval(), payload
