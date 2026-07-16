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
    with torch.no_grad():
        for completed in range(1, int(steps) + 1):
            time_fraction = denoising_time_fraction(completed, time_horizon)
            if not switch_refinement and role_alignment_complete(
                current_role_error,
                graph.num_nodes,
                role_error_per_node_threshold,
            ):
                switch_refinement = True
                transition_step = completed
            step_actions = ("switch",) if switch_refinement else actions
            candidates, _ = sample_edit_candidates(
                graph, rng, candidates_per_step, actions=step_actions
            )
            proposed += len(candidates)
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
                role_weight=role_weight,
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
