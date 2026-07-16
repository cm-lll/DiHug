from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from .graph import TypedGraph
from .mixed_edit_diffusion import (
    EditProposal,
    MarginalGraphPrior,
    MixedEditSamplingResult,
    align_mixed_edit_features_for_model,
    apply_edit,
    mixed_edit_candidate_matrix,
    sample_edit_candidates,
    sample_valid_edit,
)
from .switch_chain import graph_metrics
from .switch_diffusion import SwitchDenoiserMLP


@dataclass(frozen=True)
class ExchangeableRoleDegreePrior:
    """Type-exchangeable joint relation-degree rows learned during training."""

    type_names: Tuple[str, ...]
    family_names: Tuple[str, ...]
    rows_by_type: Tuple[np.ndarray, ...]

    @classmethod
    def from_training_graph(cls, graph: TypedGraph) -> "ExchangeableRoleDegreePrior":
        family_names = tuple(sorted(graph.family_specs))
        rows = np.stack(
            [graph.family_degrees(family) for family in family_names], axis=1
        ).astype(np.int64)
        return cls(
            type_names=tuple(graph.type_names),
            family_names=family_names,
            rows_by_type=tuple(
                rows[graph.node_types == type_id].copy()
                for type_id in range(len(graph.type_names))
            ),
        )
    def sample_assignment(self, node_types: np.ndarray, seed: int) -> np.ndarray:
        rng = np.random.default_rng(int(seed))
        sampled = np.empty(
            (len(node_types), len(self.family_names)), dtype=np.int64
        )
        for type_id, source_rows in enumerate(self.rows_by_type):
            nodes = np.flatnonzero(node_types == int(type_id))
            if len(nodes) != len(source_rows):
                raise ValueError("role prior and sampling graph have different type counts")
            sampled[nodes] = source_rows[rng.permutation(len(source_rows))]
        return sampled

    def to_json_dict(self) -> dict:
        return {
            "format": "dihug-exchangeable-role-degree-prior-v1",
            "type_names": list(self.type_names),
            "family_names": list(self.family_names),
            "rows_by_type": [rows.tolist() for rows in self.rows_by_type],
            "inference_contract": {
                "training_adjacency": False,
                "node_identity_alignment": False,
                "type_exchangeable_role_rows": True,
            },
        }

    @classmethod
    def from_json_dict(cls, payload: Mapping) -> "ExchangeableRoleDegreePrior":
        if payload.get("format") != "dihug-exchangeable-role-degree-prior-v1":
            raise ValueError("unsupported role degree prior format")
        contract = payload.get("inference_contract", {})
        if contract.get("training_adjacency") is not False:
            raise ValueError("role prior may not contain training adjacency")
        if contract.get("node_identity_alignment") is not False:
            raise ValueError("role prior may not preserve training node identities")
        return cls(
            type_names=tuple(str(value) for value in payload["type_names"]),
            family_names=tuple(str(value) for value in payload["family_names"]),
            rows_by_type=tuple(
                np.asarray(rows, dtype=np.int64) for rows in payload["rows_by_type"]
            ),
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_json_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "ExchangeableRoleDegreePrior":
        return cls.from_json_dict(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass(frozen=True)
class RoleTargetProposalState:
    """Per-step sampling weights shared by targeted relocation candidates."""

    family_probabilities: np.ndarray
    residuals: Mapping[str, np.ndarray]
    old_edge_weights: Mapping[str, np.ndarray]
    source_weights: Mapping[str, np.ndarray]
    target_weights: Mapping[str, np.ndarray]


def relation_degree_error(
    graph: TypedGraph,
    target_rows: np.ndarray,
    family_names: Sequence[str],
) -> float:
    current = np.stack(
        [graph.family_degrees(family) for family in family_names], axis=1
    ).astype(np.float64)
    return float(np.square(current - target_rows).sum())


def proposal_relation_degree_gain(
    graph: TypedGraph,
    proposal: EditProposal,
    target_rows: np.ndarray,
    family_to_id: Mapping[str, int],
) -> float:
    """Exact reduction in squared relation-degree error from one edit."""

    if proposal.action == "noop":
        return 0.0
    family_id = int(family_to_id[proposal.family])
    current = graph.family_degrees(proposal.family).astype(np.float64)
    delta = {}
    for left, right in proposal.old_edges:
        delta[int(left)] = delta.get(int(left), 0) - 1
        delta[int(right)] = delta.get(int(right), 0) - 1
    for left, right in proposal.new_edges:
        delta[int(left)] = delta.get(int(left), 0) + 1
        delta[int(right)] = delta.get(int(right), 0) + 1
    gain = 0.0
    for node, change in delta.items():
        before = current[node] - float(target_rows[node, family_id])
        after = before + float(change)
        gain += before * before - after * after
    return float(gain)


def combine_structure_and_role_scores(
    structure_logits: torch.Tensor,
    degree_gains: torch.Tensor,
    *,
    time_fraction: float,
    role_weight: float,
    gain_scale: float,
    schedule_power: float = 1.0,
    schedule_floor: float = 0.0,
) -> torch.Tensor:
    if gain_scale <= 0.0 or schedule_power <= 0.0:
        raise ValueError("gain scale and schedule power must be positive")
    if not 0.0 <= float(schedule_floor) <= 1.0:
        raise ValueError("schedule floor must lie in [0,1]")
    annealed = float(time_fraction) ** float(schedule_power)
    multiplier = float(schedule_floor) + (1.0 - float(schedule_floor)) * annealed
    coefficient = float(role_weight) * multiplier
    return structure_logits + coefficient * degree_gains / float(gain_scale)


def calibrate_role_weight(
    structure_logits: np.ndarray,
    degree_gains: np.ndarray,
    *,
    gain_scale: float,
    target_std_ratio: float,
    gain_quantile: float = 0.9,
) -> float:
    if gain_scale <= 0.0 or target_std_ratio <= 0.0:
        raise ValueError("gain scale and target standard-deviation ratio must be positive")
    if not 0.0 < float(gain_quantile) <= 1.0:
        raise ValueError("gain quantile must lie in (0,1]")
    logits = np.asarray(structure_logits, dtype=np.float64).reshape(-1)
    gains = np.abs(np.asarray(degree_gains, dtype=np.float64).reshape(-1))
    gains = gains[gains > 0.0]
    if logits.size < 2 or gains.size == 0:
        raise ValueError("automatic role calibration needs variable logits and gains")
    logit_scale = max(float(logits.std()), 1.0e-6)
    gain_value = max(float(np.quantile(gains, float(gain_quantile))), 1.0e-6)
    return (
        float(target_std_ratio) * logit_scale * float(gain_scale) / gain_value
    )


def role_error_per_node(role_error: float, node_count: int) -> float:
    if int(node_count) <= 0:
        raise ValueError("node count must be positive")
    return float(role_error) / float(node_count)


def role_alignment_complete(
    role_error: float,
    node_count: int,
    threshold_per_node: float,
) -> bool:
    """Return whether the checkpoint-conditioned role constraint is satisfied."""

    if float(threshold_per_node) <= 0.0:
        return False
    return role_error_per_node(role_error, node_count) <= float(threshold_per_node)


def denoising_time_fraction(completed: int, horizon_steps: int) -> float:
    if int(horizon_steps) <= 0:
        raise ValueError("time horizon must be positive")
    return max(0.0, 1.0 - (int(completed) - 1) / float(horizon_steps))


def _edit_key(proposal: EditProposal) -> tuple:
    return (
        proposal.action,
        proposal.family,
        tuple(sorted(proposal.old_edges)),
        tuple(sorted(proposal.new_edges)),
    )


def propose_wedge_seeded_switch(
    graph: TypedGraph,
    rng: np.random.Generator,
    max_attempts: int = 512,
    degree_bias_power: float = 0.0,
) -> Optional[EditProposal]:
    """Propose a legal 2-switch containing one currently open typed wedge."""

    domain_to_family = {
        tuple(sorted((int(spec.src_type), int(spec.dst_type)))): family
        for family, spec in graph.family_specs.items()
    }
    if float(degree_bias_power) < 0.0:
        raise ValueError("degree bias power must be non-negative")
    centers = np.asarray(
        [node for node, neighbors in enumerate(graph.adjacency) if len(neighbors) >= 2],
        dtype=np.int64,
    )
    if centers.size == 0:
        return None
    degrees = graph.degrees().astype(np.float64)

    def selection_probabilities(nodes: np.ndarray) -> Optional[np.ndarray]:
        if float(degree_bias_power) == 0.0:
            return None
        denominator = np.maximum(
            degrees[nodes] * np.maximum(degrees[nodes] - 1.0, 0.0) / 2.0,
            1.0,
        )
        weights = np.power(denominator, -float(degree_bias_power))
        return weights / weights.sum()

    center_probabilities = selection_probabilities(centers)
    for _ in range(int(max_attempts)):
        center = int(rng.choice(centers, p=center_probabilities))
        neighbors = np.asarray(sorted(graph.adjacency[center]), dtype=np.int64)
        left, right = rng.choice(
            neighbors,
            size=2,
            replace=False,
            p=selection_probabilities(neighbors),
        ).tolist()
        if graph.has_edge(int(left), int(right)):
            continue
        domain = tuple(
            sorted(
                (
                    int(graph.node_types[int(left)]),
                    int(graph.node_types[int(right)]),
                )
            )
        )
        family = domain_to_family.get(domain)
        if family is None:
            continue
        try:
            desired = graph.orient_edge(family, (int(left), int(right)))
        except ValueError:
            continue
        source, target = desired
        source_neighbors = [
            node
            for node in graph.adjacency[source]
            if int(node) != center and graph.family_of(source, node) == family
        ]
        target_neighbors = [
            node
            for node in graph.adjacency[target]
            if int(node) != center and graph.family_of(target, node) == family
        ]
        if not source_neighbors or not target_neighbors:
            continue
        source_other = int(rng.choice(np.asarray(source_neighbors)))
        target_other = int(rng.choice(np.asarray(target_neighbors)))
        if len({source, target, source_other, target_other}) < 4:
            continue
        try:
            old_edges = (
                graph.orient_edge(family, (source, source_other)),
                graph.orient_edge(family, (target_other, target)),
            )
            complement = graph.orient_edge(
                family, (target_other, source_other)
            )
        except ValueError:
            continue
        new_edges = (desired, complement)
        if old_edges[0] == old_edges[1] or new_edges[0] == new_edges[1]:
            continue
        old_canonical = {graph.canonical_edge(*edge) for edge in old_edges}
        legal = True
        seen = set()
        for edge in new_edges:
            canonical = graph.canonical_edge(*edge)
            if canonical in seen:
                legal = False
                break
            seen.add(canonical)
            if graph.has_edge(*edge) and canonical not in old_canonical:
                legal = False
                break
        if legal:
            return EditProposal("switch", family, old_edges, new_edges)
    return None


def sample_switch_refinement_candidates(
    graph: TypedGraph,
    rng: np.random.Generator,
    count: int,
    closure_fraction: float = 0.0,
    degree_bias_power: float = 0.0,
) -> list[EditProposal]:
    if not 0.0 <= float(closure_fraction) <= 1.0:
        raise ValueError("closure candidate fraction must lie in [0,1]")
    if float(closure_fraction) == 0.0:
        return sample_edit_candidates(
            graph, rng, int(count), actions=("switch",)
        )[0]
    candidates = [EditProposal("noop", sorted(graph.family_specs)[0], (), ())]
    keys = {_edit_key(candidates[0])}
    requested = min(int(count) - 1, int(round((int(count) - 1) * closure_fraction)))
    attempts = 0
    while len(candidates) - 1 < requested and attempts < max(512, requested * 64):
        attempts += 1
        proposal = propose_wedge_seeded_switch(
            graph, rng, degree_bias_power=degree_bias_power
        )
        if proposal is not None and _edit_key(proposal) not in keys:
            keys.add(_edit_key(proposal))
            candidates.append(proposal)
    attempts = 0
    while len(candidates) < int(count) and attempts < int(count) * 512:
        attempts += 1
        proposal = sample_valid_edit(graph, rng, actions=("switch",))
        if proposal is not None and _edit_key(proposal) not in keys:
            keys.add(_edit_key(proposal))
            candidates.append(proposal)
    if len(candidates) != int(count):
        raise RuntimeError(
            f"sampled only {len(candidates)}/{count} switch refinement candidates"
        )
    return candidates


def build_role_target_proposal_state(
    graph: TypedGraph,
    target_rows: np.ndarray,
    family_names: Sequence[str],
) -> RoleTargetProposalState:
    family_to_id = {family: index for index, family in enumerate(family_names)}
    residuals = {}
    old_edge_weights = {}
    source_weights = {}
    target_weights = {}
    family_mass = []
    for family in family_names:
        residual = (
            np.asarray(target_rows[:, family_to_id[family]], dtype=np.float64)
            - graph.family_degrees(family).astype(np.float64)
        )
        residuals[family] = residual
        family_mass.append(float(np.maximum(residual, 0.0).sum()))
        old_edge_weights[family] = np.asarray(
            [
                max(-residual[int(left)], 0.0)
                + max(-residual[int(right)], 0.0)
                for left, right in graph.family_edges[family]
            ],
            dtype=np.float64,
        )
        spec = graph.family_specs[family]
        source_weights[family] = np.maximum(
            residual[np.asarray(spec.src_nodes, dtype=np.int64)], 0.0
        )
        target_weights[family] = np.maximum(
            residual[np.asarray(spec.dst_nodes, dtype=np.int64)], 0.0
        )
    mass = np.asarray(family_mass, dtype=np.float64)
    probabilities = mass / mass.sum() if float(mass.sum()) > 0.0 else mass
    return RoleTargetProposalState(
        family_probabilities=probabilities,
        residuals=residuals,
        old_edge_weights=old_edge_weights,
        source_weights=source_weights,
        target_weights=target_weights,
    )


def propose_role_targeted_relocation(
    graph: TypedGraph,
    target_rows: np.ndarray,
    family_names: Sequence[str],
    rng: np.random.Generator,
    max_attempts: int = 512,
    proposal_state: Optional[RoleTargetProposalState] = None,
) -> Optional[EditProposal]:
    """Move one family edge from role-surplus to role-deficit endpoints."""

    state = proposal_state or build_role_target_proposal_state(
        graph, target_rows, family_names
    )
    if float(state.family_probabilities.sum()) <= 0.0:
        return None
    for _ in range(int(max_attempts)):
        family = str(
            rng.choice(
                np.asarray(family_names), p=state.family_probabilities
            )
        )
        spec = graph.family_specs[family]
        residual = state.residuals[family]
        edges = graph.family_edges[family]
        if not edges:
            continue
        old_weights = state.old_edge_weights[family]
        if float(old_weights.sum()) <= 0.0:
            continue
        old_edge = edges[int(rng.choice(len(edges), p=old_weights / old_weights.sum()))]
        source_nodes = np.asarray(spec.src_nodes, dtype=np.int64)
        target_nodes = np.asarray(spec.dst_nodes, dtype=np.int64)
        source_weights = state.source_weights[family]
        target_weights = state.target_weights[family]
        old_source, old_target = graph.orient_edge(family, old_edge)
        source = (
            int(rng.choice(source_nodes, p=source_weights / source_weights.sum()))
            if float(source_weights.sum()) > 0.0
            else int(old_source)
        )
        target = (
            int(rng.choice(target_nodes, p=target_weights / target_weights.sum()))
            if float(target_weights.sum()) > 0.0
            else int(old_target)
        )
        if source == target:
            continue
        try:
            new_edge = graph.orient_edge(family, (source, target))
        except ValueError:
            continue
        if new_edge == old_edge or graph.has_edge(*new_edge):
            continue
        return EditProposal("relocation", family, (old_edge,), (new_edge,))
    return None


def sample_role_alignment_candidates(
    graph: TypedGraph,
    target_rows: np.ndarray,
    family_names: Sequence[str],
    rng: np.random.Generator,
    count: int,
    targeted_fraction: float = 0.0,
    actions: Sequence[str] = ("relocation", "switch"),
) -> list[EditProposal]:
    if not 0.0 <= float(targeted_fraction) <= 1.0:
        raise ValueError("targeted role candidate fraction must lie in [0,1]")
    if float(targeted_fraction) == 0.0:
        return sample_edit_candidates(graph, rng, int(count), actions=actions)[0]
    candidates = [EditProposal("noop", sorted(graph.family_specs)[0], (), ())]
    keys = {_edit_key(candidates[0])}
    requested = min(int(count) - 1, int(round((int(count) - 1) * targeted_fraction)))
    proposal_state = build_role_target_proposal_state(
        graph, target_rows, family_names
    )
    attempts = 0
    while len(candidates) - 1 < requested and attempts < max(512, requested * 64):
        attempts += 1
        proposal = propose_role_targeted_relocation(
            graph,
            target_rows,
            family_names,
            rng,
            proposal_state=proposal_state,
        )
        if proposal is not None and _edit_key(proposal) not in keys:
            keys.add(_edit_key(proposal))
            candidates.append(proposal)
    attempts = 0
    while len(candidates) < int(count) and attempts < int(count) * 512:
        attempts += 1
        proposal = sample_valid_edit(graph, rng, actions=actions)
        if proposal is not None and _edit_key(proposal) not in keys:
            keys.add(_edit_key(proposal))
            candidates.append(proposal)
    if len(candidates) != int(count):
        raise RuntimeError(
            f"sampled only {len(candidates)}/{count} role alignment candidates"
        )
    return candidates


def sample_role_guided_edit_diffusion(
    prior: MarginalGraphPrior,
    model: SwitchDenoiserMLP,
    role_prior: ExchangeableRoleDegreePrior,
    *,
    steps: int,
    candidates_per_step: int,
    seed: int,
    device: str | torch.device = "cpu",
    metrics_every: int = 100,
    actions: Sequence[str] = ("relocation", "switch"),
    initial_graph: Optional[TypedGraph] = None,
    feature_profile: str = "typed_closure_compact_v2",
    role_weight: float = 0.1,
    gain_scale: float = 10.0,
    schedule_power: float = 1.0,
    schedule_floor: float = 0.0,
    role_error_per_node_threshold: float = 0.0,
    model_time_horizon_steps: Optional[int] = None,
    model_time_start_step: int = 0,
    closure_candidate_fraction: float = 0.0,
    closure_degree_bias_power: float = 0.0,
    auto_role_balance: float = 0.0,
    role_calibration_batches: int = 32,
    targeted_role_candidate_fraction: float = 0.0,
    context_model: Optional[torch.nn.Module] = None,
    context_refresh_steps: int = 20,
) -> MixedEditSamplingResult:
    graph = prior.sample(seed + 71) if initial_graph is None else initial_graph.clone()
    if tuple(graph.type_names) != role_prior.type_names:
        raise ValueError("role prior and graph schemas disagree")
    family_names = tuple(sorted(graph.family_specs))
    if family_names != role_prior.family_names:
        raise ValueError("role prior and graph families disagree")
    target_rows = role_prior.sample_assignment(graph.node_types, seed + 91_919)
    family_to_id = {family: index for index, family in enumerate(family_names)}
    rng = np.random.default_rng(int(seed))
    device = torch.device(device)
    model = model.to(device).eval()
    if context_model is not None:
        if int(context_refresh_steps) <= 0:
            raise ValueError("context refresh interval must be positive")
        context_model = context_model.to(device).eval()
    effective_role_weight = float(role_weight)
    if float(auto_role_balance) > 0.0:
        if int(role_calibration_batches) <= 0:
            raise ValueError("role calibration batch count must be positive")
        calibration_rng = np.random.default_rng(int(seed) + 4_100_009)
        calibration_logits = []
        calibration_gains = []
        with torch.no_grad():
            for _ in range(int(role_calibration_batches)):
                candidates, _ = sample_edit_candidates(
                    graph,
                    calibration_rng,
                    candidates_per_step,
                    actions=actions,
                )
                matrix = mixed_edit_candidate_matrix(
                    graph,
                    candidates,
                    1.0,
                    feature_profile=feature_profile,
                )
                matrix = align_mixed_edit_features_for_model(
                    matrix, model, family_count=len(family_names)
                )
                logits = model(
                    torch.from_numpy(matrix)[None].float().to(device)
                ).squeeze(0)
                gains = [
                    proposal_relation_degree_gain(
                        graph, proposal, target_rows, family_to_id
                    )
                    for proposal in candidates
                ]
                calibration_logits.extend(logits.cpu().tolist())
                calibration_gains.extend(gains)
        effective_role_weight = calibrate_role_weight(
            np.asarray(calibration_logits),
            np.asarray(calibration_gains),
            gain_scale=gain_scale,
            target_std_ratio=auto_role_balance,
        )
    current_role_error = relation_degree_error(graph, target_rows, family_names)
    switch_refinement = role_alignment_complete(
        current_role_error,
        graph.num_nodes,
        role_error_per_node_threshold,
    )
    transition_step = 0 if switch_refinement else None
    trace = [
        {
            "step": 0.0,
            "relation_degree_error": current_role_error,
            "relation_degree_error_per_node": role_error_per_node(
                current_role_error, graph.num_nodes
            ),
            "phase": "switch_refinement" if switch_refinement else "role_alignment",
            "role_phase_transition_step": transition_step,
            "effective_role_weight": effective_role_weight,
            **graph_metrics(graph),
        }
    ]
    action_counts = {"relocation": 0, "switch": 0, "noop": 0}
    proposed = 0
    applied = 0
    time_horizon = (
        int(steps)
        if model_time_horizon_steps is None
        else int(model_time_horizon_steps)
    )
    if time_horizon <= 0:
        raise ValueError("model time horizon must be positive")
    if int(model_time_start_step) < 0:
        raise ValueError("model time start step must be non-negative")
    context_hidden = None
    last_context_refresh = -int(context_refresh_steps)
    with torch.no_grad():
        for completed in range(1, int(steps) + 1):
            time_fraction = denoising_time_fraction(
                int(model_time_start_step) + completed, time_horizon
            )
            if not switch_refinement and role_alignment_complete(
                current_role_error,
                graph.num_nodes,
                role_error_per_node_threshold,
            ):
                switch_refinement = True
                transition_step = completed
            if switch_refinement:
                candidates = sample_switch_refinement_candidates(
                    graph,
                    rng,
                    candidates_per_step,
                    closure_fraction=closure_candidate_fraction,
                    degree_bias_power=closure_degree_bias_power,
                )
            else:
                candidates = sample_role_alignment_candidates(
                    graph,
                    target_rows,
                    family_names,
                    rng,
                    candidates_per_step,
                    targeted_fraction=targeted_role_candidate_fraction,
                    actions=actions,
                )
            proposed += len(candidates)
            if context_model is not None and switch_refinement:
                if (
                    context_hidden is None
                    or completed - last_context_refresh >= int(context_refresh_steps)
                ):
                    context_hidden = context_model.encode_graph(
                        graph, time_fraction, device
                    )
                    last_context_refresh = completed
                structure_logits, _ = context_model.score_graph_candidates(
                    graph,
                    candidates,
                    time_fraction,
                    feature_profile,
                    device,
                    encoded=context_hidden,
                )
            else:
                matrix = mixed_edit_candidate_matrix(
                    graph,
                    candidates,
                    time_fraction,
                    feature_profile=feature_profile,
                )
                matrix = align_mixed_edit_features_for_model(
                    matrix, model, family_count=len(family_names)
                )
                structure_logits = model(
                    torch.from_numpy(matrix)[None].float().to(device)
                ).squeeze(0)
            degree_gains = torch.tensor(
                [
                    proposal_relation_degree_gain(
                        graph, proposal, target_rows, family_to_id
                    )
                    for proposal in candidates
                ],
                dtype=structure_logits.dtype,
                device=device,
            )
            scores = combine_structure_and_role_scores(
                structure_logits,
                degree_gains,
                time_fraction=time_fraction,
                role_weight=effective_role_weight,
                gain_scale=gain_scale,
                schedule_power=schedule_power,
                schedule_floor=schedule_floor,
            )
            selected_index = int(scores.argmax().item())
            selected = candidates[selected_index]
            apply_edit(graph, selected)
            current_role_error = max(
                0.0,
                current_role_error - float(degree_gains[selected_index].item()),
            )
            action_counts[selected.action] += 1
            applied += int(selected.action != "noop")
            if completed == int(steps) or (
                int(metrics_every) > 0 and completed % int(metrics_every) == 0
            ):
                trace.append(
                    {
                        "step": float(completed),
                        "relation_degree_error": current_role_error,
                        "relation_degree_error_per_node": role_error_per_node(
                            current_role_error, graph.num_nodes
                        ),
                        "phase": (
                            "switch_refinement"
                            if switch_refinement
                            else "role_alignment"
                        ),
                        "role_phase_transition_step": transition_step,
                        "effective_role_weight": effective_role_weight,
                        **graph_metrics(graph),
                    }
                )
    recomputed_role_error = relation_degree_error(graph, target_rows, family_names)
    if not np.isclose(current_role_error, recomputed_role_error):
        raise RuntimeError("incremental relation-degree error drifted from graph state")
    graph.validate()
    return MixedEditSamplingResult(
        graph=graph,
        trace=trace,
        action_counts=action_counts,
        proposed=proposed,
        applied=applied,
    )
