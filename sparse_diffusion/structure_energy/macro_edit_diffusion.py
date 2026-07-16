from __future__ import annotations

from dataclasses import dataclass
from typing import List, Mapping, Sequence, Tuple

import numpy as np
import torch
from torch import nn

from sparse_diffusion.models.pyhgt_denoiser import HGTConv

from .global_context_joint_edit import GlobalContextJointEditResidual
from .features import FeatureSelection, GraphletFeatureExtractor, select_features
from .graph import TypedGraph
from .hgt_edit_denoiser import (
    candidate_endpoint_matrix,
    graph_edge_tensors,
    graph_node_features,
)
from .mixed_edit_diffusion import (
    EditProposal,
    MarginalGraphPrior,
    MixedEditSamplingResult,
    _family_names,
    apply_edit,
    inverse_edit,
    mixed_edit_candidate_matrix,
    sample_valid_edit,
)
from .switch_chain import graph_metrics
from .switch_diffusion import SwitchDenoiserMLP


RS_ACTIONS = ("relocation", "switch")


def proposal_nodes(proposal: EditProposal) -> frozenset[int]:
    return frozenset(
        int(node)
        for edge in tuple(proposal.old_edges) + tuple(proposal.new_edges)
        for node in edge
    )


def sample_conflict_free_rs_batch(
    graph: TypedGraph,
    rng: np.random.Generator,
    budget: int,
    max_attempts: int = 4096,
) -> Tuple[EditProposal, ...]:
    """Sample a node-disjoint, hence commuting, batch of legal R/S edits."""

    selected: List[EditProposal] = []
    used_nodes: set[int] = set()
    keys = set()
    attempts = 0
    while len(selected) < int(budget) and attempts < int(max_attempts):
        attempts += 1
        proposal = sample_valid_edit(graph, rng, actions=RS_ACTIONS)
        if proposal is None:
            continue
        nodes = proposal_nodes(proposal)
        key = (
            proposal.action,
            proposal.family,
            tuple(sorted(proposal.old_edges)),
            tuple(sorted(proposal.new_edges)),
        )
        if not nodes or nodes & used_nodes or key in keys:
            continue
        selected.append(proposal)
        used_nodes.update(nodes)
        keys.add(key)
    return tuple(selected)


def apply_commuting_edit_batch(
    graph: TypedGraph, proposals: Sequence[EditProposal]
) -> None:
    used: set[int] = set()
    for proposal in proposals:
        nodes = proposal_nodes(proposal)
        if nodes & used:
            raise ValueError("macro edit batch contains overlapping endpoints")
        used.update(nodes)
    for proposal in proposals:
        apply_edit(graph, proposal)


def _sample_macro_candidates(
    graph: TypedGraph,
    rng: np.random.Generator,
    count: int,
    positives: Sequence[EditProposal],
) -> Tuple[List[EditProposal], np.ndarray]:
    candidates: List[EditProposal] = []
    positive_keys = {
        (
            item.action,
            item.family,
            tuple(sorted(item.old_edges)),
            tuple(sorted(item.new_edges)),
        )
        for item in positives
    }
    keys = set()

    def add(item: EditProposal) -> None:
        key = (
            item.action,
            item.family,
            tuple(sorted(item.old_edges)),
            tuple(sorted(item.new_edges)),
        )
        if key not in keys:
            keys.add(key)
            candidates.append(item)

    for item in positives:
        add(item)
    attempts = 0
    while len(candidates) < int(count) and attempts < int(count) * 512:
        attempts += 1
        proposal = sample_valid_edit(graph, rng, actions=RS_ACTIONS)
        if proposal is not None:
            add(proposal)
    if len(candidates) != int(count):
        raise RuntimeError(f"sampled only {len(candidates)}/{count} macro candidates")
    order = rng.permutation(len(candidates))
    candidates = [candidates[index] for index in order.tolist()]
    mask = np.asarray(
        [
            (
                item.action,
                item.family,
                tuple(sorted(item.old_edges)),
                tuple(sorted(item.new_edges)),
            )
            in positive_keys
            for item in candidates
        ],
        dtype=np.float32,
    )
    if mask.sum() != len(positive_keys):
        raise RuntimeError("macro inverse candidates were lost")
    return candidates, mask


@dataclass(frozen=True)
class MacroEditDataset:
    node_features: np.ndarray
    edge_indices: Tuple[np.ndarray, ...]
    edge_types: Tuple[np.ndarray, ...]
    candidate_features: np.ndarray
    candidate_endpoints: np.ndarray
    candidate_family_ids: np.ndarray
    positive_masks: np.ndarray
    candidate_utilities: np.ndarray
    budgets: np.ndarray
    time_fractions: np.ndarray
    energy_targets: np.ndarray
    trajectory_ids: np.ndarray
    node_types: np.ndarray
    type_count: int
    family_count: int
    objective: str


def build_macro_rs_dataset(
    source: TypedGraph,
    *,
    trajectories: int,
    macro_steps: int,
    record_every: int,
    max_budget: int,
    candidates: int,
    seed: int,
    feature_profile: str = "legacy_local_v1",
    budget_schedule: str = "linear",
    record_schedule: str = "uniform",
) -> MacroEditDataset:
    if trajectories < 2 or macro_steps <= 0 or record_every <= 0:
        raise ValueError("invalid macro dataset budget")
    if budget_schedule not in {"linear", "constant"}:
        raise ValueError("macro budget schedule must be linear or constant")
    if record_schedule not in {"uniform", "quadratic"}:
        raise ValueError("macro record schedule must be uniform or quadratic")
    record_count = max(int(np.ceil(macro_steps / record_every)), 1)
    if record_schedule == "quadratic":
        record_steps = set(
            np.maximum(
                1,
                np.rint(
                    np.square(np.linspace(1.0 / record_count, 1.0, record_count))
                    * int(macro_steps)
                ).astype(np.int64),
            ).tolist()
        )
        record_steps.add(int(macro_steps))
    else:
        record_steps = set(range(int(record_every), int(macro_steps) + 1, int(record_every)))
        record_steps.add(int(macro_steps))
    family_to_id = {name: index for index, name in enumerate(_family_names(source))}
    rows = []
    for trajectory in range(int(trajectories)):
        rng = np.random.default_rng(int(seed) + 1_000_003 * trajectory)
        graph = source.clone()
        for step in range(1, int(macro_steps) + 1):
            fraction = float(step) / float(macro_steps)
            target_budget = (
                int(max_budget)
                if budget_schedule == "constant"
                else 1 + int(round((max_budget - 1) * fraction))
            )
            forward = sample_conflict_free_rs_batch(graph, rng, target_budget)
            if not forward:
                continue
            apply_commuting_edit_batch(graph, forward)
            if step not in record_steps:
                continue
            positives = tuple(inverse_edit(item) for item in forward)
            proposal_list, positive_mask = _sample_macro_candidates(
                graph, rng, candidates, positives
            )
            edge_index, edge_types = graph_edge_tensors(graph)
            node_state = graph_node_features(graph, fraction)[:, :-1]
            rows.append(
                (
                    node_state,
                    edge_index,
                    edge_types,
                    mixed_edit_candidate_matrix(
                        graph,
                        proposal_list,
                        fraction,
                        feature_profile=feature_profile,
                    ),
                    candidate_endpoint_matrix(proposal_list),
                    np.asarray(
                        [family_to_id[item.family] for item in proposal_list],
                        dtype=np.int64,
                    ),
                    positive_mask,
                    len(positives),
                    fraction,
                    trajectory,
                )
            )
    if not rows:
        raise RuntimeError("macro dataset contains no examples")
    return MacroEditDataset(
        node_features=np.stack([row[0] for row in rows]).astype(np.float32),
        edge_indices=tuple(row[1] for row in rows),
        edge_types=tuple(row[2] for row in rows),
        candidate_features=np.stack([row[3] for row in rows]).astype(np.float32),
        candidate_endpoints=np.stack([row[4] for row in rows]).astype(np.int64),
        candidate_family_ids=np.stack([row[5] for row in rows]).astype(np.int64),
        positive_masks=np.stack([row[6] for row in rows]).astype(np.float32),
        candidate_utilities=np.stack([row[6] for row in rows]).astype(np.float32),
        budgets=np.asarray([row[7] for row in rows], dtype=np.int64),
        time_fractions=np.asarray([row[8] for row in rows], dtype=np.float32),
        energy_targets=np.asarray([row[8] for row in rows], dtype=np.float32),
        trajectory_ids=np.asarray([row[9] for row in rows], dtype=np.int64),
        node_types=source.node_types.copy(),
        type_count=len(source.type_names),
        family_count=len(source.family_specs),
        objective="inverse",
    )


@dataclass(frozen=True)
class MacroStructureTarget:
    """Anonymous typed graphlet target learned from the training graph."""

    selection: FeatureSelection
    target_vector: np.ndarray
    null_distance: float

    def distance(self, graph: TypedGraph) -> float:
        extractor = GraphletFeatureExtractor(graph, self.selection.keys)
        vector = self.selection.vector(extractor.extract(graph))
        return float(np.mean(np.square(vector - self.target_vector)))


def fit_macro_structure_target(
    source: TypedGraph,
    null_graphs: Sequence[TypedGraph],
    *,
    max_features: int = 48,
) -> MacroStructureTarget:
    """Select schema-derived graphlets that distinguish data from deep RS noise."""

    if not null_graphs:
        raise ValueError("macro structure target needs at least one null graph")
    real = GraphletFeatureExtractor(source).extract(source)
    null_values = [GraphletFeatureExtractor(graph).extract(graph) for graph in null_graphs]
    selection = select_features(
        real,
        null_values,
        max_features=max_features,
        min_support=1.0,
        drop_fixed_degree_wedges=True,
        reserve_prefix_counts={"triangle/": 16, "gwesp/": 16, "mix/": 8},
    )
    target_vector = selection.vector(real)
    distances = [
        float(np.mean(np.square(selection.vector(values) - target_vector)))
        for values in null_values
    ]
    return MacroStructureTarget(
        selection=selection,
        target_vector=target_vector,
        null_distance=max(float(np.mean(distances)), 1.0e-8),
    )


def macro_structure_candidate_utilities(
    graph: TypedGraph,
    candidates: Sequence[EditProposal],
    target: MacroStructureTarget,
) -> Tuple[np.ndarray, float]:
    """Return target-distance reductions; clean edges are never compared."""

    before = target.distance(graph)
    values = []
    for proposal in candidates:
        candidate_graph = graph.clone()
        apply_edit(candidate_graph, proposal)
        values.append(before - target.distance(candidate_graph))
    return np.asarray(values, dtype=np.float32), before


def _conflict_free_positive_budget(
    proposals: Sequence[EditProposal], utilities: np.ndarray, max_budget: int
) -> int:
    used: set[int] = set()
    count = 0
    for index in np.argsort(np.asarray(utilities))[::-1].tolist():
        if float(utilities[index]) <= 0.0:
            break
        nodes = proposal_nodes(proposals[index])
        if nodes & used:
            continue
        used.update(nodes)
        count += 1
        if count >= int(max_budget):
            break
    return max(count, 1)


def build_macro_structure_utility_dataset(
    source: TypedGraph,
    *,
    trajectories: int,
    macro_steps: int,
    record_every: int,
    max_budget: int,
    candidates: int,
    seed: int,
    feature_profile: str = "legacy_local_v1",
    utility_temperature: float = 0.05,
    max_structure_features: int = 48,
) -> Tuple[MacroEditDataset, MacroStructureTarget]:
    """Build deep-noise, free-candidate supervision from anonymous structure."""

    if trajectories < 2 or macro_steps <= 0 or record_every <= 0:
        raise ValueError("invalid macro utility dataset budget")
    if utility_temperature <= 0.0:
        raise ValueError("utility temperature must be positive")
    family_to_id = {name: index for index, name in enumerate(_family_names(source))}
    states = []
    final_graphs = []
    for trajectory in range(int(trajectories)):
        rng = np.random.default_rng(int(seed) + 1_000_003 * trajectory)
        graph = source.clone()
        for step in range(1, int(macro_steps) + 1):
            forward = sample_conflict_free_rs_batch(graph, rng, max_budget)
            if forward:
                apply_commuting_edit_batch(graph, forward)
            if step % int(record_every) == 0:
                states.append((trajectory, step, graph.clone()))
        final_graphs.append(graph.clone())
    target = fit_macro_structure_target(
        source, final_graphs, max_features=max_structure_features
    )
    rows = []
    state_rng = np.random.default_rng(int(seed) + 77_777)
    for trajectory, step, graph in states:
        fraction = float(step) / float(macro_steps)
        proposal_list, _ = _sample_macro_candidates(
            graph, state_rng, candidates, positives=()
        )
        utilities, distance = macro_structure_candidate_utilities(
            graph, proposal_list, target
        )
        scaled = (utilities - float(utilities.max())) / float(utility_temperature)
        teacher = np.exp(np.clip(scaled, -40.0, 0.0)).astype(np.float32)
        edge_index, edge_types = graph_edge_tensors(graph)
        rows.append(
            (
                graph_node_features(graph, fraction)[:, :-1],
                edge_index,
                edge_types,
                mixed_edit_candidate_matrix(
                    graph,
                    proposal_list,
                    fraction,
                    feature_profile=feature_profile,
                ),
                candidate_endpoint_matrix(proposal_list),
                np.asarray(
                    [family_to_id[item.family] for item in proposal_list],
                    dtype=np.int64,
                ),
                teacher,
                utilities,
                _conflict_free_positive_budget(
                    proposal_list, utilities, max_budget
                ),
                fraction,
                min(distance / target.null_distance, 1.0),
                trajectory,
            )
        )
    if not rows:
        raise RuntimeError("macro utility dataset contains no examples")
    return (
        MacroEditDataset(
            node_features=np.stack([row[0] for row in rows]).astype(np.float32),
            edge_indices=tuple(row[1] for row in rows),
            edge_types=tuple(row[2] for row in rows),
            candidate_features=np.stack([row[3] for row in rows]).astype(np.float32),
            candidate_endpoints=np.stack([row[4] for row in rows]).astype(np.int64),
            candidate_family_ids=np.stack([row[5] for row in rows]).astype(np.int64),
            positive_masks=np.stack([row[6] for row in rows]).astype(np.float32),
            candidate_utilities=np.stack([row[7] for row in rows]).astype(np.float32),
            budgets=np.asarray([row[8] for row in rows], dtype=np.int64),
            time_fractions=np.asarray([row[9] for row in rows], dtype=np.float32),
            energy_targets=np.asarray([row[10] for row in rows], dtype=np.float32),
            trajectory_ids=np.asarray([row[11] for row in rows], dtype=np.int64),
            node_types=source.node_types.copy(),
            type_count=len(source.type_names),
            family_count=len(source.family_specs),
            objective="structure_utility",
        ),
        target,
    )


class MacroRSEditDenoiser(nn.Module):
    """Shared global HGT with legacy edit, graph-energy and budget heads."""

    def __init__(
        self,
        legacy_scorer: SwitchDenoiserMLP,
        *,
        node_feature_dim: int,
        candidate_feature_dim: int,
        type_count: int,
        family_count: int,
        hidden_dim: int = 128,
        heads: int = 4,
        layers: int = 2,
        max_budget: int = 4,
        edge_slot_dim: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_dim % heads:
            raise ValueError("hidden dimension must be divisible by heads")
        self.max_budget = int(max_budget)
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
        self.family_slot = nn.Embedding(max(family_count, 1), edge_slot_dim)
        self.state_slot = nn.Embedding(3, edge_slot_dim)
        self.joint = GlobalContextJointEditResidual(
            legacy_scorer,
            node_hidden_dim=hidden_dim,
            candidate_feature_dim=candidate_feature_dim,
            edge_slot_dim=edge_slot_dim,
            hidden_dim=hidden_dim,
        )
        for parameter in self.joint.legacy_scorer.parameters():
            parameter.requires_grad_(False)
        self.type_count = int(type_count)
        self.graph_pool = nn.Sequential(
            nn.Linear(hidden_dim * type_count, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.energy_head = nn.Linear(hidden_dim, 1)
        self.budget_head = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, max_budget),
        )
        nn.init.zeros_(self.budget_head[-1].weight)
        nn.init.zeros_(self.budget_head[-1].bias)
        with torch.no_grad():
            self.budget_head[-1].bias[0] = 1.0

    def encode(
        self,
        node_features: torch.Tensor,
        node_types: torch.Tensor,
        edge_index: torch.Tensor,
        edge_types: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, node_count, _ = node_features.shape
        hidden = self.node_input(node_features.reshape(batch_size * node_count, -1))
        flat_types = node_types.reshape(-1)
        for layer in self.layers:
            hidden = layer(hidden, flat_types, edge_index, edge_types)
        hidden = hidden.reshape(batch_size, node_count, -1)
        pooled = []
        for type_id in range(self.type_count):
            mask = node_types == int(type_id)
            denom = mask.sum(dim=1, keepdim=True).clamp(min=1)
            pooled.append((hidden * mask.unsqueeze(-1)).sum(dim=1) / denom)
        return hidden, self.graph_pool(torch.cat(pooled, dim=-1))

    def forward(
        self,
        node_features: torch.Tensor,
        node_types: torch.Tensor,
        edge_index: torch.Tensor,
        edge_types: torch.Tensor,
        candidate_features: torch.Tensor,
        candidate_endpoints: torch.Tensor,
        candidate_family_ids: torch.Tensor,
        time_fractions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden, graph_hidden = self.encode(
            node_features, node_types, edge_index, edge_types
        )
        batch_size, candidate_count = candidate_family_ids.shape
        family = self.family_slot(candidate_family_ids.clamp(min=0))
        slot_family = family[:, :, None, :].expand(-1, -1, 4, -1)
        state_ids = torch.tensor([1, 1, 2, 2], device=family.device)
        slot_state = self.state_slot(state_ids)[None, None].expand(
            batch_size, candidate_count, -1, -1
        )
        slot_hidden = slot_family + slot_state
        logits = self.joint(
            candidate_features,
            node_hidden=hidden,
            candidate_endpoints=candidate_endpoints,
            edge_slot_hidden=slot_hidden,
        )
        energy = self.energy_head(graph_hidden).squeeze(-1)
        budget = self.budget_head(
            torch.cat((graph_hidden, time_fractions[:, None]), dim=-1)
        )
        return logits, energy, budget


def _make_batch(
    dataset: MacroEditDataset, indices: np.ndarray, device: torch.device
) -> tuple:
    batch_size = len(indices)
    node_count = dataset.node_types.shape[0]
    edge_parts = []
    type_parts = []
    for batch_index, example_index in enumerate(indices.tolist()):
        edge_parts.append(
            torch.from_numpy(dataset.edge_indices[example_index]).long()
            + batch_index * node_count
        )
        type_parts.append(torch.from_numpy(dataset.edge_types[example_index]).long())
    return (
        torch.from_numpy(dataset.node_features[indices]).float().to(device),
        torch.from_numpy(np.tile(dataset.node_types, (batch_size, 1))).long().to(device),
        torch.cat(edge_parts, dim=1).to(device),
        torch.cat(type_parts).to(device),
        torch.from_numpy(dataset.candidate_features[indices]).float().to(device),
        torch.from_numpy(dataset.candidate_endpoints[indices]).long().to(device),
        torch.from_numpy(dataset.candidate_family_ids[indices]).long().to(device),
        torch.from_numpy(dataset.time_fractions[indices]).float().to(device),
        torch.from_numpy(dataset.positive_masks[indices]).float().to(device),
        torch.from_numpy(dataset.budgets[indices]).long().to(device),
        torch.from_numpy(dataset.candidate_utilities[indices]).float().to(device),
        torch.from_numpy(dataset.energy_targets[indices]).float().to(device),
    )


def train_macro_rs_denoiser(
    dataset: MacroEditDataset,
    legacy_scorer: SwitchDenoiserMLP,
    *,
    epochs: int = 20,
    batch_size: int = 2,
    learning_rate: float = 3.0e-4,
    device: str | torch.device = "cuda",
    seed: int = 0,
    max_budget: int = 4,
) -> Tuple[MacroRSEditDenoiser, Mapping[str, float]]:
    torch.manual_seed(int(seed))
    rng = np.random.default_rng(int(seed))
    device = torch.device(device)
    validation_trajectory = int(dataset.trajectory_ids.max())
    train_ids = np.flatnonzero(dataset.trajectory_ids != validation_trajectory)
    val_ids = np.flatnonzero(dataset.trajectory_ids == validation_trajectory)
    model = MacroRSEditDenoiser(
        legacy_scorer,
        node_feature_dim=dataset.node_features.shape[-1],
        candidate_feature_dim=dataset.candidate_features.shape[-1],
        type_count=dataset.type_count,
        family_count=dataset.family_count,
        max_budget=max_budget,
    ).to(device)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=learning_rate,
        weight_decay=1e-4,
    )
    final = {}
    for epoch in range(int(epochs)):
        model.train()
        totals = []
        shuffled = rng.permutation(train_ids)
        for start in range(0, len(shuffled), int(batch_size)):
            ids = shuffled[start : start + int(batch_size)]
            batch = _make_batch(dataset, ids, device)
            optimizer.zero_grad(set_to_none=True)
            logits, energy, budget_logits = model(*batch[:8])
            positive = batch[8]
            target_distribution = positive / positive.sum(dim=1, keepdim=True).clamp_min(
                1.0e-12
            )
            edit_loss = -(
                target_distribution * torch.log_softmax(logits, dim=1)
            ).sum(dim=1).mean()
            # Energy sees no explicit time input; predicting corruption level
            # therefore requires global structural evidence.
            energy_loss = nn.functional.mse_loss(torch.sigmoid(energy), batch[11])
            budget_loss = nn.functional.cross_entropy(
                budget_logits, (batch[9] - 1).clamp(0, max_budget - 1)
            )
            loss = edit_loss + 0.25 * energy_loss + 0.25 * budget_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            totals.append((float(loss.detach()), float(edit_loss.detach()), float(energy_loss.detach()), float(budget_loss.detach())))
        values = np.asarray(totals)
        final = {
            "loss": float(values[:, 0].mean()),
            "edit_loss": float(values[:, 1].mean()),
            "energy_loss": float(values[:, 2].mean()),
            "budget_loss": float(values[:, 3].mean()),
        }
        print(
            "[MACRO-RS-EPOCH] "
            f"epoch={epoch} loss={final['loss']:.6f} "
            f"edit={final['edit_loss']:.6f} energy={final['energy_loss']:.6f} "
            f"budget={final['budget_loss']:.6f}",
            flush=True,
        )
    model.eval()
    positive_hits = budget_hits = 0
    energy_error = 0.0
    selected_utilities = []
    oracle_utilities = []
    energy_values = []
    time_values = []
    with torch.no_grad():
        for index in val_ids.tolist():
            batch = _make_batch(dataset, np.asarray([index]), device)
            logits, energy, budget_logits = model(*batch[:8])
            selected = int(logits.argmax())
            oracle = int(batch[10][0].argmax())
            positive_hits += int(
                selected == oracle
                if dataset.objective == "structure_utility"
                else float(batch[8][0, selected]) > 0.0
            )
            budget_hits += int(int(budget_logits.argmax()) + 1 == int(batch[9][0]))
            energy_value = float(torch.sigmoid(energy)[0])
            energy_target = float(batch[11][0])
            energy_error += abs(energy_value - energy_target)
            energy_values.append(energy_value)
            time_values.append(energy_target)
            selected_utilities.append(float(batch[10][0, selected]))
            oracle_utilities.append(float(batch[10][0, oracle]))
    count = max(len(val_ids), 1)
    report = {
        **final,
        "train_examples": int(len(train_ids)),
        "validation_examples": int(len(val_ids)),
        "validation_positive_top1": positive_hits / count,
        "validation_budget_accuracy": budget_hits / count,
        "validation_energy_mae": energy_error / count,
        "validation_selected_utility": float(np.mean(selected_utilities)),
        "validation_oracle_utility": float(np.mean(oracle_utilities)),
        "validation_utility_regret": float(
            np.mean(oracle_utilities) - np.mean(selected_utilities)
        ),
        "validation_energy_time_correlation": (
            float(np.corrcoef(energy_values, time_values)[0, 1])
            if len(energy_values) > 1
            and np.std(energy_values) > 0
            and np.std(time_values) > 0
            else 0.0
        ),
    }
    return model, report


def _model_inputs_for_graph(
    graph: TypedGraph,
    candidates: Sequence[EditProposal],
    fraction: float,
    feature_profile: str,
    device: torch.device,
) -> tuple:
    edges, edge_types = graph_edge_tensors(graph)
    family_to_id = {name: index for index, name in enumerate(_family_names(graph))}
    return (
        torch.from_numpy(graph_node_features(graph, fraction)[:, :-1])[None].float().to(device),
        torch.from_numpy(graph.node_types)[None].long().to(device),
        torch.from_numpy(edges).long().to(device),
        torch.from_numpy(edge_types).long().to(device),
        torch.from_numpy(
            mixed_edit_candidate_matrix(
                graph, candidates, fraction, feature_profile=feature_profile
            )
        )[None].float().to(device),
        torch.from_numpy(candidate_endpoint_matrix(candidates))[None].long().to(device),
        torch.tensor(
            [[family_to_id[item.family] for item in candidates]],
            dtype=torch.long,
            device=device,
        ),
        torch.tensor([fraction], dtype=torch.float, device=device),
    )


def score_macro_graph_energy(
    graph: TypedGraph,
    model: MacroRSEditDenoiser,
    *,
    device: torch.device,
) -> torch.Tensor:
    edges, edge_types = graph_edge_tensors(graph)
    node_features = torch.from_numpy(graph_node_features(graph, 0.0)[:, :-1])[
        None
    ].float().to(device)
    node_types = torch.from_numpy(graph.node_types)[None].long().to(device)
    edge_index = torch.from_numpy(edges).long().to(device)
    relations = torch.from_numpy(edge_types).long().to(device)
    _, graph_hidden = model.encode(
        node_features, node_types, edge_index, relations
    )
    return model.energy_head(graph_hidden).squeeze(-1)


def sample_macro_rs_diffusion(
    initial: TypedGraph,
    model: MacroRSEditDenoiser,
    *,
    macro_steps: int,
    candidates_per_step: int,
    seed: int,
    device: str | torch.device = "cuda",
    feature_profile: str = "legacy_local_v1",
    metrics_every: int = 10,
    energy_guidance_scale: float = 0.0,
) -> MixedEditSamplingResult:
    graph = initial.clone()
    device = torch.device(device)
    model = model.to(device).eval()
    rng = np.random.default_rng(int(seed))
    trace = [{"step": 0.0, **graph_metrics(graph)}]
    counts = {"noop": 0, "relocation": 0, "switch": 0, "birth": 0, "death": 0}
    proposed = applied = 0
    for reverse_index in range(int(macro_steps)):
        fraction = 1.0 - float(reverse_index) / max(float(macro_steps), 1.0)
        candidates, _ = _sample_macro_candidates(
            graph, rng, candidates_per_step, positives=()
        )
        inputs = _model_inputs_for_graph(
            graph, candidates, fraction, feature_profile, device
        )
        with torch.no_grad():
            logits, current_energy, budget_logits = model(*inputs)
            if float(energy_guidance_scale) != 0.0:
                energy_improvements = []
                for proposal in candidates:
                    candidate_graph = graph.clone()
                    apply_edit(candidate_graph, proposal)
                    candidate_energy = score_macro_graph_energy(
                        candidate_graph, model, device=device
                    )
                    energy_improvements.append(current_energy - candidate_energy)
                logits = logits + float(energy_guidance_scale) * torch.cat(
                    energy_improvements
                )[None]
        budget = int(budget_logits[0].argmax()) + 1
        used: set[int] = set()
        selected = []
        for index in logits[0].argsort(descending=True).tolist():
            proposal = candidates[int(index)]
            nodes = proposal_nodes(proposal)
            if nodes & used:
                continue
            selected.append(proposal)
            used.update(nodes)
            if len(selected) == budget:
                break
        apply_commuting_edit_batch(graph, selected)
        for proposal in selected:
            counts[proposal.action] += 1
        proposed += len(candidates)
        applied += len(selected)
        completed = reverse_index + 1
        if completed == int(macro_steps) or completed % int(metrics_every) == 0:
            trace.append({"step": float(completed), **graph_metrics(graph)})
    return MixedEditSamplingResult(
        graph=graph,
        trace=trace,
        action_counts=counts,
        proposed=proposed,
        applied=applied,
    )
