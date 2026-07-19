from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn

from sparse_diffusion.models.pyhgt_denoiser import HGTConv

from .graph import TypedGraph
from .mixed_edit_diffusion import (
    EDIT_ACTIONS,
    EditProposal,
    MarginalGraphPrior,
    MixedEditSamplingResult,
    MixedKernelSchedule,
    _edge_slots,
    _family_names,
    apply_edit,
    edit_distance_improvement,
    inverse_edit,
    mixed_edit_candidate_matrix,
    sample_edit_candidates,
    sample_closed_form_marginal_noise,
    sample_mixed_forward_transition,
)
from .roles import permuted_witness, sample_role_plan
from .switch_chain import graph_metrics
from .switch_diffusion import SwitchDiffusionDataset, SwitchDenoiserMLP, train_switch_denoiser


@dataclass(frozen=True)
class HGTEditDataset:
    candidate_features: np.ndarray
    candidate_endpoints: np.ndarray
    candidate_improvements: np.ndarray
    node_features: np.ndarray
    edge_indices: Tuple[np.ndarray, ...]
    edge_types: Tuple[np.ndarray, ...]
    targets: np.ndarray
    trajectory_ids: np.ndarray
    node_types: np.ndarray
    type_count: int
    family_count: int
    edge_subtypes: Optional[Tuple[np.ndarray, ...]] = None
    role_features: Optional[np.ndarray] = None
    candidate_role_gains: Optional[np.ndarray] = None
    time_fractions: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        count, candidates, _ = self.candidate_features.shape
        if self.candidate_endpoints.shape != (count, candidates, 8):
            raise ValueError("candidate endpoints must have shape [N,C,8]")
        if self.candidate_improvements.shape != (count, candidates):
            raise ValueError("candidate improvements must have shape [N,C]")
        if self.node_features.shape[:2] != (count, self.node_types.shape[0]):
            raise ValueError("node feature snapshots have incompatible shape")
        if len(self.edge_indices) != count or len(self.edge_types) != count:
            raise ValueError("edge snapshot count does not match examples")
        if self.edge_subtypes is not None and len(self.edge_subtypes) != count:
            raise ValueError("edge subtype snapshot count does not match examples")
        if self.role_features is not None and self.role_features.shape[:2] != (
            count,
            self.node_types.shape[0],
        ):
            raise ValueError("role feature snapshots have incompatible shape")
        if self.candidate_role_gains is not None and self.candidate_role_gains.shape != (
            count,
            candidates,
        ):
            raise ValueError("candidate role gains must have shape [N,C]")
        if self.time_fractions is not None and self.time_fractions.shape != (count,):
            raise ValueError("time fractions must have shape [N]")
        if self.targets.shape != (count,) or self.trajectory_ids.shape != (count,):
            raise ValueError("target or trajectory shape does not match examples")

    def as_listwise_dataset(self) -> SwitchDiffusionDataset:
        shape = self.candidate_features.shape[:2]
        return SwitchDiffusionDataset(
            candidate_features=self.candidate_features,
            candidate_improvements=self.candidate_improvements,
            targets=self.targets,
            trajectory_ids=self.trajectory_ids,
        )


def graph_node_features(graph: TypedGraph, time_fraction: float) -> np.ndarray:
    """Anonymous node state: typed relation degrees, total degree, isolation and t."""

    scale = max(float(np.log1p(graph.num_nodes)), 1.0)
    columns = [np.log1p(graph.degrees()).astype(np.float32) / scale]
    for family in _family_names(graph):
        columns.append(
            np.log1p(graph.family_degrees(family)).astype(np.float32) / scale
        )
    columns.extend(
        (
            (graph.degrees() == 0).astype(np.float32),
            np.full(graph.num_nodes, float(time_fraction), dtype=np.float32),
        )
    )
    return np.stack(columns, axis=1)


def graph_edge_tensors(graph: TypedGraph) -> Tuple[np.ndarray, np.ndarray]:
    family_to_id = {name: index for index, name in enumerate(_family_names(graph))}
    sources: List[int] = []
    targets: List[int] = []
    relations: List[int] = []
    for family in _family_names(graph):
        relation = family_to_id[family]
        for u, v in graph.family_edges[family]:
            sources.extend((int(u), int(v)))
            targets.extend((int(v), int(u)))
            relations.extend((relation, relation))
    return (
        np.asarray((sources, targets), dtype=np.int64),
        np.asarray(relations, dtype=np.int64),
    )


def graph_role_features(graph: TypedGraph, target_rows: np.ndarray) -> np.ndarray:
    """Normalized anonymous target degrees and signed current-to-target gaps."""

    family_names = tuple(_family_names(graph))
    target = np.asarray(target_rows, dtype=np.float32)
    expected = (graph.num_nodes, len(family_names))
    if target.shape != expected:
        raise ValueError(f"target role rows have shape {target.shape}, expected {expected}")
    current = np.stack(
        [graph.family_degrees(family) for family in family_names], axis=1
    ).astype(np.float32)
    scale = max(float(np.log1p(graph.num_nodes)), 1.0)
    return np.concatenate(
        (np.log1p(target) / scale, np.arcsinh(target - current) / scale), axis=1
    ).astype(np.float32)


def sample_typed_query_edges(
    graph: TypedGraph,
    *,
    seed: int,
    edges_per_family: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample sparse schema-valid NULL links without consulting a clean graph."""

    requested = int(edges_per_family)
    if requested < 0:
        raise ValueError("query edges per family must be non-negative")
    if requested == 0:
        return np.empty((2, 0), dtype=np.int64), np.empty(0, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    sources: List[int] = []
    targets: List[int] = []
    relations: List[int] = []
    for relation, family in enumerate(_family_names(graph)):
        spec = graph.family_specs[family]
        existing = {
            (min(int(u), int(v)), max(int(u), int(v)))
            for u, v in graph.family_edges[family]
        }
        sampled: set[tuple[int, int]] = set()
        maximum = (
            len(spec.src_nodes) * len(spec.dst_nodes)
            if spec.bipartite
            else len(spec.src_nodes) * max(len(spec.src_nodes) - 1, 0) // 2
        ) - len(existing)
        target_count = min(requested, max(maximum, 0))
        attempts = 0
        while len(sampled) < target_count and attempts < max(256, requested * 64):
            attempts += 1
            u = int(rng.choice(spec.src_nodes))
            v = int(rng.choice(spec.dst_nodes))
            if u == v:
                continue
            edge = (min(u, v), max(u, v))
            if edge in existing or edge in sampled:
                continue
            sampled.add(edge)
        for u, v in sorted(sampled):
            sources.extend((u, v))
            targets.extend((v, u))
            relations.extend((relation, relation))
    return (
        np.asarray((sources, targets), dtype=np.int64),
        np.asarray(relations, dtype=np.int64),
    )


def candidate_endpoint_matrix(candidates: Sequence[EditProposal]) -> np.ndarray:
    rows = []
    for proposal in candidates:
        endpoints: List[int] = []
        for edge in _edge_slots(proposal):
            endpoints.extend((-1, -1) if edge is None else (int(edge[0]), int(edge[1])))
        rows.append(endpoints)
    return np.asarray(rows, dtype=np.int64)


def build_hgt_edit_dataset(
    source: TypedGraph,
    trajectories: int,
    forward_steps: int,
    record_every: int,
    candidates_per_example: int,
    seed: int,
    schedule: Optional[MixedKernelSchedule] = None,
    actions: Sequence[str] = EDIT_ACTIONS,
    supervision: str = "exact_inverse",
    feature_profile: str = "typed_closure_full_v2",
    role_conditioning: bool = False,
    query_edges_per_family: int = 0,
    forward_mode: str = "stepwise",
    candidate_mode: str = "random",
    targeted_role_candidate_fraction: float = 0.5,
    closure_candidate_fraction: float = 0.5,
) -> HGTEditDataset:
    if int(trajectories) < 2:
        raise ValueError("HGT edit training needs at least two trajectories")
    prior = MarginalGraphPrior.from_training_graph(source)
    schedule = schedule or MixedKernelSchedule()
    supervision = str(supervision).strip().lower()
    if supervision not in {"exact_inverse", "free_x0", "rooted_utility"}:
        raise ValueError(
            "HGT edit supervision must be exact_inverse, free_x0, or rooted_utility"
        )
    forward_mode = str(forward_mode).strip().lower()
    if forward_mode not in {"stepwise", "closed_form"}:
        raise ValueError("HGT edit forward mode must be stepwise or closed_form")
    candidate_mode = str(candidate_mode).strip().lower()
    if candidate_mode not in {"random", "unified"}:
        raise ValueError("HGT edit candidate mode must be random or unified")
    if candidate_mode == "unified" and not role_conditioning:
        raise ValueError("unified candidates require role conditioning")
    candidate_features = []
    candidate_endpoints = []
    candidate_improvements = []
    node_features = []
    edge_indices = []
    edge_types = []
    edge_subtypes = []
    role_features = []
    candidate_role_gains = []
    time_fractions = []
    targets = []
    trajectory_ids = []
    for trajectory_id in range(int(trajectories)):
        rng = np.random.default_rng(int(seed) + 1_000_003 * trajectory_id)
        if role_conditioning:
            role_plan = sample_role_plan(
                source, int(seed) + 1_000_003 * trajectory_id + 17
            )
            graph = permuted_witness(source, role_plan)
            target_rows = np.stack(
                [
                    role_plan.family_degrees[family]
                    for family in sorted(source.family_specs)
                ],
                axis=1,
            ).astype(np.int64)
            clean_target = graph.clone()
        else:
            graph = source.clone()
            target_rows = None
            clean_target = source
        rooted_target_signature = None
        if supervision == "rooted_utility":
            from .rooted_context_edit import (
                rooted_closure_signature,
                schema_supports_triangles,
            )

            if schema_supports_triangles(clean_target):
                rooted_target_signature = rooted_closure_signature(clean_target)

        def record_example(
            graph: TypedGraph,
            fraction: float,
            step: int,
            proposal: EditProposal | None,
        ) -> None:
            if supervision == "exact_inverse":
                if proposal is None:
                    raise ValueError("exact inverse supervision requires a forward edit")
                candidates, target = sample_edit_candidates(
                    graph,
                    rng,
                    candidates_per_example,
                    actions=actions,
                    positive=inverse_edit(proposal),
                )
                improvements = np.zeros(len(candidates), dtype=np.float32)
            else:
                if candidate_mode == "unified":
                    from .role_guided_edit_diffusion import (
                        sample_unified_edit_candidates,
                    )

                    candidates, _ = sample_unified_edit_candidates(
                        graph,
                        target_rows,
                        tuple(sorted(graph.family_specs)),
                        rng,
                        candidates_per_example,
                        targeted_fraction=float(targeted_role_candidate_fraction),
                        closure_fraction=float(closure_candidate_fraction),
                    )
                else:
                    candidates, _ = sample_edit_candidates(
                        graph, rng, candidates_per_example, actions=actions
                    )
                if supervision == "free_x0":
                    raw_improvements = np.asarray(
                        [
                            edit_distance_improvement(graph, clean_target, item)
                            for item in candidates
                        ],
                        dtype=np.float32,
                    )
                else:
                    from .rooted_context_edit import rooted_closure_signature, rooted_closure_utility

                    target_signature = rooted_target_signature
                    if target_signature is None:
                        raw_improvements = np.zeros(
                            len(candidates), dtype=np.float32
                        )
                    else:
                        current_signature = rooted_closure_signature(
                            graph, target_signature.boundaries
                        )
                        raw_improvements = np.asarray(
                            [
                                (
                                    rooted_closure_utility(
                                        graph,
                                        item,
                                        target_signature,
                                        current=current_signature,
                                    )
                                    if item.action in {"switch", "noop"}
                                    else 0.0
                                )
                                for item in candidates
                            ],
                            dtype=np.float32,
                        )
                centered = raw_improvements - float(raw_improvements.mean())
                scale = max(float(raw_improvements.std()), 1.0e-8)
                improvements = (centered / scale).astype(np.float32)
                target = int(improvements.argmax())
            edges, relations = graph_edge_tensors(graph)
            subtypes = np.ones(relations.shape[0], dtype=np.int64)
            if int(query_edges_per_family) > 0:
                query_edges, query_relations = sample_typed_query_edges(
                    graph,
                    seed=(
                        int(seed)
                        + 1_000_003 * trajectory_id
                        + 97_409 * step
                    ),
                    edges_per_family=int(query_edges_per_family),
                )
                if query_relations.size:
                    edges = np.concatenate((edges, query_edges), axis=1)
                    relations = np.concatenate((relations, query_relations))
                    subtypes = np.concatenate(
                        (subtypes, np.zeros(query_relations.shape[0], dtype=np.int64))
                    )
            candidate_features.append(
                mixed_edit_candidate_matrix(
                    graph,
                    candidates,
                    fraction,
                    feature_profile=feature_profile,
                )
            )
            candidate_endpoints.append(candidate_endpoint_matrix(candidates))
            candidate_improvements.append(improvements)
            node_features.append(graph_node_features(graph, fraction))
            edge_indices.append(edges)
            edge_types.append(relations)
            edge_subtypes.append(subtypes)
            if target_rows is not None:
                role_features.append(graph_role_features(graph, target_rows))
                from .role_guided_edit_diffusion import proposal_relation_degree_gain

                family_to_id = {
                    family: index
                    for index, family in enumerate(sorted(graph.family_specs))
                }
                candidate_role_gains.append(
                    np.asarray(
                        [
                            proposal_relation_degree_gain(
                                graph, item, target_rows, family_to_id
                            )
                            for item in candidates
                        ],
                        dtype=np.float32,
                    )
                )
            else:
                candidate_role_gains.append(
                    np.zeros(len(candidates), dtype=np.float32)
                )
            time_fractions.append(float(fraction))
            targets.append(target)
            trajectory_ids.append(trajectory_id)

        if forward_mode == "closed_form":
            example_count = max(1, int(forward_steps) // int(record_every))
            for example_id in range(example_count):
                fraction = float(example_id + 1) / float(example_count)
                graph = sample_closed_form_marginal_noise(
                    clean_target,
                    prior,
                    alpha_bar=(1.0 - fraction) ** 2,
                    seed=(
                        int(seed)
                        + 1_000_003 * trajectory_id
                        + 97_409 * (example_id + 1)
                    ),
                )
                record_example(graph, fraction, example_id + 1, None)
            continue

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
            record_example(
                graph,
                float(step) / float(forward_steps),
                step,
                proposal,
            )
    return HGTEditDataset(
        candidate_features=np.stack(candidate_features).astype(np.float32),
        candidate_endpoints=np.stack(candidate_endpoints),
        candidate_improvements=np.stack(candidate_improvements),
        node_features=np.stack(node_features).astype(np.float32),
        edge_indices=tuple(edge_indices),
        edge_types=tuple(edge_types),
        targets=np.asarray(targets, dtype=np.int64),
        trajectory_ids=np.asarray(trajectory_ids, dtype=np.int64),
        node_types=source.node_types.copy(),
        type_count=len(source.type_names),
        family_count=len(source.family_specs),
        edge_subtypes=tuple(edge_subtypes),
        role_features=(
            np.stack(role_features).astype(np.float32)
            if role_conditioning
            else None
        ),
        candidate_role_gains=np.stack(candidate_role_gains).astype(np.float32),
        time_fractions=np.asarray(time_fractions, dtype=np.float32),
    )


class HGTListwiseEditDenoiser(nn.Module):
    """Permutation-equivariant graph encoder with the original listwise edit head."""

    def __init__(
        self,
        node_feature_dim: int,
        candidate_feature_dim: int,
        type_count: int,
        family_count: int,
        hidden_dim: int = 128,
        heads: int = 4,
        hgt_layers: int = 2,
        scorer_layers: int = 2,
        dropout: float = 0.1,
        endpoint_dim: int = 64,
    ) -> None:
        super().__init__()
        if hidden_dim % heads:
            raise ValueError("HGT hidden dimension must be divisible by heads")
        self.config = {
            "node_feature_dim": int(node_feature_dim),
            "candidate_feature_dim": int(candidate_feature_dim),
            "type_count": int(type_count),
            "family_count": int(family_count),
            "hidden_dim": int(hidden_dim),
            "heads": int(heads),
            "hgt_layers": int(hgt_layers),
            "scorer_layers": int(scorer_layers),
            "dropout": float(dropout),
            "endpoint_dim": int(endpoint_dim),
        }
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
            for _ in range(hgt_layers)
        )
        self.endpoint_projection = nn.Sequential(
            nn.Linear(4 * hidden_dim, endpoint_dim), nn.LayerNorm(endpoint_dim), nn.SiLU()
        )
        scorer_input = candidate_feature_dim + 4 * endpoint_dim + hidden_dim
        modules: List[nn.Module] = []
        width = scorer_input
        for _ in range(scorer_layers):
            modules.extend(
                (nn.Linear(width, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(), nn.Dropout(dropout))
            )
            width = hidden_dim
        modules.append(nn.Linear(width, 1))
        self.scorer = nn.Sequential(*modules)

    def forward(
        self,
        node_features: torch.Tensor,
        node_types: torch.Tensor,
        edge_index: torch.Tensor,
        edge_types: torch.Tensor,
        candidate_features: torch.Tensor,
        candidate_endpoints: torch.Tensor,
        edge_subtypes: Optional[torch.Tensor] = None,
        role_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if node_features.ndim != 3 or candidate_features.ndim != 3:
            raise ValueError("HGT listwise inputs must be batched")
        batch_size, node_count, _ = node_features.shape
        hidden = self.node_input(node_features.reshape(batch_size * node_count, -1))
        flat_types = node_types.reshape(-1)
        if edge_subtypes is None:
            edge_subtypes = torch.ones_like(edge_types)
        for layer in self.layers:
            hidden = layer(
                hidden,
                flat_types,
                edge_index,
                edge_types,
                edge_subtype=edge_subtypes,
                edge_family=edge_types,
            )
        hidden = hidden.reshape(batch_size, node_count, -1)
        graph_embedding = hidden.mean(dim=1)
        slots = []
        for slot in range(4):
            u = candidate_endpoints[..., 2 * slot]
            v = candidate_endpoints[..., 2 * slot + 1]
            valid = (u >= 0) & (v >= 0)
            u_safe = u.clamp(min=0)
            v_safe = v.clamp(min=0)
            batch = torch.arange(batch_size, device=hidden.device)[:, None]
            hu = hidden[batch, u_safe]
            hv = hidden[batch, v_safe]
            pair = torch.cat((hu, hv, torch.abs(hu - hv), hu * hv), dim=-1)
            slots.append(self.endpoint_projection(pair) * valid.unsqueeze(-1))
        graph_context = graph_embedding[:, None, :].expand(-1, candidate_features.size(1), -1)
        combined = torch.cat((candidate_features, *slots, graph_context), dim=-1)
        return self.scorer(combined).squeeze(-1)


def _make_batch(dataset: HGTEditDataset, indices: np.ndarray, device: torch.device) -> tuple:
    batch_size = len(indices)
    node_count = dataset.node_types.shape[0]
    edge_parts = []
    type_parts = []
    subtype_parts = []
    for batch_index, example_index in enumerate(indices.tolist()):
        edge_parts.append(
            torch.from_numpy(dataset.edge_indices[example_index]).long()
            + batch_index * node_count
        )
        type_parts.append(torch.from_numpy(dataset.edge_types[example_index]).long())
        subtype = (
            np.ones(dataset.edge_types[example_index].shape[0], dtype=np.int64)
            if dataset.edge_subtypes is None
            else dataset.edge_subtypes[example_index]
        )
        subtype_parts.append(torch.from_numpy(subtype).long())
    role = (
        np.empty((batch_size, node_count, 0), dtype=np.float32)
        if dataset.role_features is None
        else dataset.role_features[indices]
    )
    return (
        torch.from_numpy(dataset.node_features[indices]).float().to(device),
        torch.from_numpy(np.tile(dataset.node_types, (batch_size, 1))).long().to(device),
        torch.cat(edge_parts, dim=1).to(device),
        torch.cat(type_parts).to(device),
        torch.from_numpy(dataset.candidate_features[indices]).float().to(device),
        torch.from_numpy(dataset.candidate_endpoints[indices]).long().to(device),
        torch.cat(subtype_parts).to(device),
        torch.from_numpy(role).float().to(device),
        torch.from_numpy(dataset.targets[indices]).long().to(device),
    )


def _ranking_metrics(logits: torch.Tensor, targets: torch.Tensor) -> Mapping[str, float]:
    order = logits.argsort(dim=1, descending=True)
    rows = torch.arange(logits.size(0), device=logits.device)
    positive = logits[rows, targets]
    masked = logits.clone()
    masked[rows, targets] = -torch.inf
    return {
        "top1": float((order[:, 0] == targets).float().mean().cpu()),
        "top5": float((order[:, : min(5, logits.size(1))] == targets[:, None]).any(dim=1).float().mean().cpu()),
        "margin": float((positive - masked.max(dim=1).values).mean().cpu()),
    }


def train_hgt_listwise_edit_denoiser(
    dataset: HGTEditDataset,
    epochs: int = 20,
    batch_size: int = 4,
    learning_rate: float = 3.0e-4,
    weight_decay: float = 1.0e-4,
    device: str | torch.device = "cuda",
    seed: int = 0,
    training_objective: str = "inverse_ce",
    utility_temperature: float = 1.0,
    **model_kwargs,
) -> tuple[HGTListwiseEditDenoiser, Mapping[str, float]]:
    torch.manual_seed(int(seed))
    rng = np.random.default_rng(int(seed))
    device = torch.device(device)
    validation_trajectory = int(dataset.trajectory_ids.max())
    train_indices = np.flatnonzero(dataset.trajectory_ids != validation_trajectory)
    validation_indices = np.flatnonzero(dataset.trajectory_ids == validation_trajectory)
    model = HGTListwiseEditDenoiser(
        node_feature_dim=dataset.node_features.shape[-1],
        candidate_feature_dim=dataset.candidate_features.shape[-1],
        type_count=dataset.type_count,
        family_count=dataset.family_count,
        **model_kwargs,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    training_objective = str(training_objective).strip().lower()
    if training_objective not in {"inverse_ce", "utility_kl"}:
        raise ValueError("HGT listwise objective must be inverse_ce or utility_kl")
    final_loss = float("nan")
    for epoch in range(int(epochs)):
        model.train()
        shuffled = rng.permutation(train_indices)
        losses = []
        for start in range(0, len(shuffled), int(batch_size)):
            batch = _make_batch(dataset, shuffled[start : start + int(batch_size)], device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(*batch[:-1])
            if training_objective == "inverse_ce":
                loss = nn.functional.cross_entropy(logits, batch[-1])
            else:
                improvements = torch.from_numpy(
                    dataset.candidate_improvements[
                        shuffled[start : start + int(batch_size)]
                    ]
                ).float().to(device)
                teacher = torch.softmax(improvements / float(utility_temperature), dim=1)
                loss = -(teacher * torch.log_softmax(logits, dim=1)).sum(dim=1).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        final_loss = float(np.mean(losses))
        print(f"[HGT-LISTWISE-EPOCH] epoch={epoch} loss={final_loss:.6f}", flush=True)
    model.eval()
    all_logits = []
    all_targets = []
    with torch.no_grad():
        for start in range(0, len(validation_indices), int(batch_size)):
            batch = _make_batch(dataset, validation_indices[start : start + int(batch_size)], device)
            all_logits.append(model(*batch[:-1]))
            all_targets.append(batch[-1])
    logits = torch.cat(all_logits)
    targets = torch.cat(all_targets)
    metrics = dict(_ranking_metrics(logits, targets))
    validation_improvements = torch.from_numpy(
        dataset.candidate_improvements[validation_indices]
    ).float().to(device)
    selected = logits.argmax(dim=1)
    rows = torch.arange(logits.size(0), device=device)
    metrics.update(
        train_loss=final_loss,
        train_examples=float(len(train_indices)),
        validation_examples=float(len(validation_indices)),
        validation_cross_entropy=float(nn.functional.cross_entropy(logits, targets).cpu()),
        validation_selected_utility=float(
            validation_improvements[rows, selected].mean().cpu()
        ),
        validation_oracle_utility=float(
            validation_improvements.max(dim=1).values.mean().cpu()
        ),
        validation_utility_regret=float(
            (
                validation_improvements.max(dim=1).values
                - validation_improvements[rows, selected]
            ).mean().cpu()
        ),
        training_objective=training_objective,
        utility_temperature=float(utility_temperature),
    )
    return model, metrics


def score_graph_candidates(
    model: HGTListwiseEditDenoiser,
    graph: TypedGraph,
    candidates: Sequence[EditProposal],
    time_fraction: float,
    device: torch.device,
) -> torch.Tensor:
    edges, relations = graph_edge_tensors(graph)
    candidate_features = mixed_edit_candidate_matrix(graph, candidates, time_fraction)
    endpoints = candidate_endpoint_matrix(candidates)
    with torch.no_grad():
        return model(
            torch.from_numpy(graph_node_features(graph, time_fraction))[None].float().to(device),
            torch.from_numpy(graph.node_types)[None].long().to(device),
            torch.from_numpy(edges).long().to(device),
            torch.from_numpy(relations).long().to(device),
            torch.from_numpy(candidate_features)[None].float().to(device),
            torch.from_numpy(endpoints)[None].long().to(device),
        ).squeeze(0)


def evaluate_hgt_fixed_t(
    source: TypedGraph,
    model: HGTListwiseEditDenoiser,
    time_fractions: Sequence[float],
    trials_per_time: int,
    forward_steps: int,
    candidates_per_example: int,
    seed: int,
    actions: Sequence[str] = EDIT_ACTIONS,
    device: str | torch.device = "cuda",
) -> Mapping[str, Mapping[str, float]]:
    prior = MarginalGraphPrior.from_training_graph(source)
    schedule = MixedKernelSchedule()
    device = torch.device(device)
    model = model.to(device).eval()
    report = {}
    for time_index, fraction in enumerate(time_fractions):
        correct = top5 = action_correct = 0
        margins = []
        for trial in range(int(trials_per_time)):
            rng = np.random.default_rng(seed + 10_000_019 * time_index + 1_000_003 * trial)
            graph = source.clone()
            last = None
            steps = max(1, int(round(float(fraction) * forward_steps)))
            for step in range(1, steps + 1):
                last = sample_mixed_forward_transition(
                    graph, prior, rng, step / forward_steps, schedule, actions=actions
                )
                apply_edit(graph, last)
            positive = inverse_edit(last)
            candidates, target = sample_edit_candidates(
                graph, rng, candidates_per_example, actions=actions, positive=positive
            )
            logits = score_graph_candidates(model, graph, candidates, float(fraction), device)
            order = logits.argsort(descending=True)
            selected = int(order[0])
            correct += int(selected == target)
            top5 += int(target in order[: min(5, len(candidates))].tolist())
            action_correct += int(candidates[selected].action == positive.action)
            negative = torch.cat((logits[:target], logits[target + 1 :]))
            margins.append(float((logits[target] - negative.max()).cpu()))
        report[f"{float(fraction):.3f}"] = {
            "trials": float(trials_per_time),
            "top1": correct / float(trials_per_time),
            "top5": top5 / float(trials_per_time),
            "action_top1": action_correct / float(trials_per_time),
            "margin": float(np.mean(margins)),
        }
    return report


def evaluate_paired_free_x0_fixed_t(
    source: TypedGraph,
    hgt_model: HGTListwiseEditDenoiser,
    mlp_model: SwitchDenoiserMLP,
    time_fractions: Sequence[float],
    trials_per_time: int,
    forward_steps: int,
    candidates_per_example: int,
    seed: int,
    actions: Sequence[str] = EDIT_ACTIONS,
    device: str | torch.device = "cuda",
) -> Mapping[str, Mapping[str, Mapping[str, float]]]:
    """Evaluate both scorers on identical non-injected inference candidates."""

    prior = MarginalGraphPrior.from_training_graph(source)
    schedule = MixedKernelSchedule()
    device = torch.device(device)
    hgt_model = hgt_model.to(device).eval()
    mlp_model = mlp_model.to(device).eval()
    report = {}
    for time_index, fraction in enumerate(time_fractions):
        stats = {
            "hgt": {"top1": 0, "top5": 0, "selected_improvement": []},
            "mlp": {"top1": 0, "top5": 0, "selected_improvement": []},
        }
        improving_coverage = 0
        for trial in range(int(trials_per_time)):
            rng = np.random.default_rng(
                int(seed) + 10_000_019 * time_index + 1_000_003 * trial
            )
            graph = source.clone()
            steps = max(1, int(round(float(fraction) * forward_steps)))
            for step in range(1, steps + 1):
                proposal = sample_mixed_forward_transition(
                    graph, prior, rng, step / forward_steps, schedule, actions=actions
                )
                apply_edit(graph, proposal)
            candidates, _ = sample_edit_candidates(
                graph, rng, candidates_per_example, actions=actions
            )
            improvements = np.asarray(
                [edit_distance_improvement(graph, source, item) for item in candidates],
                dtype=np.float32,
            )
            target = int(improvements.argmax())
            improving_coverage += int(improvements[target] > 0)
            matrix = mixed_edit_candidate_matrix(graph, candidates, float(fraction))
            with torch.no_grad():
                hgt_logits = score_graph_candidates(
                    hgt_model, graph, candidates, float(fraction), device
                )
                mlp_logits = mlp_model(
                    torch.from_numpy(matrix)[None].float().to(device)
                ).squeeze(0)
            for name, logits in (("hgt", hgt_logits), ("mlp", mlp_logits)):
                order = logits.argsort(descending=True)
                selected = int(order[0])
                stats[name]["top1"] += int(selected == target)
                stats[name]["top5"] += int(
                    target in order[: min(5, len(candidates))].tolist()
                )
                stats[name]["selected_improvement"].append(float(improvements[selected]))
        row = {
            "trials": float(trials_per_time),
            "improving_candidate_rate": improving_coverage / float(trials_per_time),
        }
        for name in ("hgt", "mlp"):
            row[name] = {
                "top1": stats[name]["top1"] / float(trials_per_time),
                "top5": stats[name]["top5"] / float(trials_per_time),
                "selected_improvement": float(
                    np.mean(stats[name]["selected_improvement"])
                ),
            }
        report[f"{float(fraction):.3f}"] = row
    return report


def sample_hgt_listwise_edit_diffusion(
    prior: MarginalGraphPrior,
    model: HGTListwiseEditDenoiser,
    steps: int,
    candidates_per_step: int,
    seed: int,
    device: str | torch.device = "cuda",
    metrics_every: int = 100,
    actions: Sequence[str] = EDIT_ACTIONS,
) -> MixedEditSamplingResult:
    graph = prior.sample(seed=seed + 71)
    rng = np.random.default_rng(seed)
    device = torch.device(device)
    model = model.to(device).eval()
    trace = [{"step": 0.0, **graph_metrics(graph)}]
    action_counts = {action: 0 for action in (*EDIT_ACTIONS, "noop")}
    applied = proposed = 0
    for reverse_step in range(int(steps), 0, -1):
        fraction = reverse_step / float(steps)
        candidates, _ = sample_edit_candidates(graph, rng, candidates_per_step, actions=actions)
        proposed += len(candidates)
        logits = score_graph_candidates(model, graph, candidates, fraction, device)
        proposal = candidates[int(logits.argmax())]
        apply_edit(graph, proposal)
        action_counts[proposal.action] += 1
        applied += int(proposal.action != "noop")
        completed = steps - reverse_step + 1
        if completed == steps or (metrics_every > 0 and completed % metrics_every == 0):
            trace.append({"step": float(completed), **graph_metrics(graph)})
    graph.validate()
    return MixedEditSamplingResult(graph, trace, action_counts, proposed, applied)


def train_paired_mlp(dataset: HGTEditDataset, **kwargs) -> tuple[SwitchDenoiserMLP, Mapping]:
    model, report = train_switch_denoiser(dataset.as_listwise_dataset(), **kwargs)
    return model, report.to_json_dict()


def save_hgt_edit_checkpoint(
    path: str | Path,
    model: HGTListwiseEditDenoiser,
    prior: MarginalGraphPrior,
    report: Mapping,
    actions: Sequence[str],
) -> None:
    torch.save(
        {
            "format": "dihug-hgt-listwise-edit-v1",
            "model_config": model.config,
            "state_dict": model.state_dict(),
            "prior": prior.to_json_dict(),
            "train_report": dict(report),
            "actions": list(actions),
        },
        Path(path),
    )


def load_hgt_edit_checkpoint(path: str | Path, device: str | torch.device = "cpu"):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != "dihug-hgt-listwise-edit-v1":
        raise ValueError("unsupported HGT edit checkpoint")
    model = HGTListwiseEditDenoiser(**payload["model_config"])
    model.load_state_dict(payload["state_dict"])
    return model.to(device), MarginalGraphPrior.from_json_dict(payload["prior"]), payload
