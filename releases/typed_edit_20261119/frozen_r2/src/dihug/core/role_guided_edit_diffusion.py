from __future__ import annotations

import json
import hashlib
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

    @staticmethod
    def _apportion_degrees(
        weights: np.ndarray,
        total: int,
        capacity: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Project anonymous role weights onto an exact endpoint degree sum.

        The projection is identity when ``total`` equals the learned degree sum.
        Otherwise it uses capped largest-remainder apportionment with random
        tie-breaking.  It never associates a learned row with an original node
        identity; the rows have already been permuted within type.
        """

        weights = np.asarray(weights, dtype=np.float64).reshape(-1)
        total = int(total)
        capacity = int(capacity)
        if total < 0 or capacity < 0 or total > len(weights) * capacity:
            raise ValueError(
                f"unrealizable endpoint degree sum: total={total}, "
                f"nodes={len(weights)}, capacity={capacity}"
            )
        if total == 0:
            return np.zeros(len(weights), dtype=np.int64)
        if np.any(weights < 0.0) or not np.isfinite(weights).all():
            raise ValueError("role weights must be finite and non-negative")

        if float(weights.sum()) == 0.0:
            scaled = np.full(len(weights), total / max(len(weights), 1), dtype=np.float64)
        else:
            scaled = weights * (float(total) / float(weights.sum()))
        base = np.minimum(np.floor(scaled).astype(np.int64), capacity)
        remaining = total - int(base.sum())
        if remaining < 0:
            raise RuntimeError("capped degree apportionment exceeded its target")

        tie_break = rng.random(len(weights))
        while remaining:
            eligible = np.flatnonzero(base < capacity)
            if eligible.size == 0:
                raise RuntimeError("degree apportionment exhausted endpoint capacity")
            residual = scaled[eligible] - base[eligible]
            order = eligible[np.lexsort((tie_break[eligible], -residual))]
            take = min(remaining, len(order))
            base[order[:take]] += 1
            remaining -= take
        return base

    def sample_assignment_for_graph(
        self,
        graph: TypedGraph,
        seed: int,
    ) -> np.ndarray:
        """Sample anonymous roles conditioned only on this graph's family counts.

        This is needed when ``M_f`` is drawn from the learned Binomial marginal:
        relocation/switch edits preserve ``M_f``, so the role target must have
        the same endpoint sums.  Conditioning uses family counts and schema only,
        never the sampled adjacency or clean node identities.
        """

        if tuple(graph.type_names) != self.type_names:
            raise ValueError("role prior and graph type schemas disagree")
        family_names = tuple(sorted(graph.family_specs))
        if family_names != self.family_names:
            raise ValueError("role prior and graph family schemas disagree")

        sampled = self.sample_assignment(graph.node_types, seed)
        conditioned = np.zeros_like(sampled)
        rng = np.random.default_rng(int(seed) + 2_654_435_761)
        for family_id, family in enumerate(family_names):
            spec = graph.family_specs[family]
            edge_count = len(graph.family_edges[family])
            src_nodes = np.asarray(spec.src_nodes, dtype=np.int64)
            if spec.bipartite:
                dst_nodes = np.asarray(spec.dst_nodes, dtype=np.int64)
                conditioned[src_nodes, family_id] = self._apportion_degrees(
                    sampled[src_nodes, family_id], edge_count, len(dst_nodes), rng
                )
                conditioned[dst_nodes, family_id] = self._apportion_degrees(
                    sampled[dst_nodes, family_id], edge_count, len(src_nodes), rng
                )
            else:
                conditioned[src_nodes, family_id] = self._apportion_degrees(
                    sampled[src_nodes, family_id],
                    2 * edge_count,
                    max(len(src_nodes) - 1, 0),
                    rng,
                )
        return conditioned

    def family_counts(
        self, family_endpoints: Mapping[str, Tuple[int, int]]
    ) -> Mapping[str, int]:
        """Recover family cardinalities from the anonymous degree-row multiset."""

        counts = {}
        for family_id, family in enumerate(self.family_names):
            src_type, dst_type = family_endpoints[family]
            src_sum = int(self.rows_by_type[int(src_type)][:, family_id].sum())
            if int(src_type) == int(dst_type):
                if src_sum % 2:
                    raise ValueError(f"odd degree sum for homogeneous family {family}")
                counts[family] = src_sum // 2
                continue
            dst_sum = int(self.rows_by_type[int(dst_type)][:, family_id].sum())
            if src_sum != dst_sum:
                raise ValueError(
                    f"endpoint degree sums disagree for {family}: {src_sum} != {dst_sum}"
                )
            counts[family] = src_sum
        return counts

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


def role_alignment_plateaued(
    reference_error: float,
    current_error: float,
    noop_count: int,
    observation_count: int,
    *,
    min_relative_improvement: float,
    min_noop_fraction: float,
) -> bool:
    """Detect a stalled role phase without consulting the clean graph."""

    if int(observation_count) <= 0:
        return False
    if float(min_relative_improvement) < 0.0:
        raise ValueError("plateau improvement threshold must be non-negative")
    if not 0.0 <= float(min_noop_fraction) <= 1.0:
        raise ValueError("plateau noop threshold must lie in [0, 1]")
    improvement = max(float(reference_error) - float(current_error), 0.0)
    relative_improvement = improvement / max(float(reference_error), 1.0)
    noop_fraction = int(noop_count) / float(observation_count)
    return (
        relative_improvement <= float(min_relative_improvement)
        and noop_fraction >= float(min_noop_fraction)
    )


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


def validate_stage1_positive_gain_guard(enabled, *, phase_policy, selection_mode,
                                       decoder, sampling_backend, arm='M'):
    """The PR2G contract is opt-in CPU auto/model/MAP, not a new general decoder."""
    if not isinstance(enabled, bool):
        raise TypeError('stage1_positive_gain_guard must be bool')
    if enabled and (phase_policy != 'auto' or selection_mode != 'model' or
                    decoder != 'map' or sampling_backend != 'cpu' or arm != 'M'):
        raise ValueError('stage1_positive_gain_guard requires CPU auto/model/MAP arm M')


def guarded_stage1_index(scores, degree_gain_values, original_selected):
    """Filter exact integer gains, preserving original score/index and fallback.

The already computed float64 reductions are integral on the bounded integer
degree domains. No epsilon, float32 sign decision or score modification is used.
"""
    integers = []
    for value in degree_gain_values:
        if not np.isfinite(value) or float(value) != int(value):
            raise ValueError('positive-gain guard requires exact integral degree gains')
        integers.append(int(value))
    positive = [i for i,gain in enumerate(integers) if gain > 0]
    if not positive:
        return int(original_selected), positive, False, True
    subset = scores[positive]
    return positive[int(subset.argmax().item())], positive, True, False


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


def positive_role_zero_count(
    graph: TypedGraph,
    target_rows: np.ndarray,
    family_names: Sequence[str],
) -> int:
    """Count nodes left isolated despite a positive sampled role target."""

    target = np.asarray(target_rows)
    if target.shape != (graph.num_nodes, len(family_names)):
        raise ValueError("role target shape disagrees with graph schema")
    family_degrees = graph.all_family_degrees()
    current_total = np.zeros(graph.num_nodes, dtype=np.int64)
    for family in family_names:
        current_total += np.asarray(family_degrees[family], dtype=np.int64)
    return int(np.count_nonzero((target.sum(axis=1) > 0) & (current_total == 0)))


def proposal_positive_role_coverage_gain(
    graph: TypedGraph,
    proposal: EditProposal,
    target_rows: np.ndarray,
    family_names: Sequence[str],
) -> float:
    """Return the reduction in uncovered positive-role nodes after an edit."""

    target = np.asarray(target_rows)
    if target.shape != (graph.num_nodes, len(family_names)):
        raise ValueError("role target shape disagrees with graph schema")
    family_degrees = graph.all_family_degrees()
    current_total = np.zeros(graph.num_nodes, dtype=np.int64)
    for family in family_names:
        current_total += np.asarray(family_degrees[family], dtype=np.int64)
    delta: dict[int, int] = {}
    for left, right in proposal.old_edges:
        delta[int(left)] = delta.get(int(left), 0) - 1
        delta[int(right)] = delta.get(int(right), 0) - 1
    for left, right in proposal.new_edges:
        delta[int(left)] = delta.get(int(left), 0) + 1
        delta[int(right)] = delta.get(int(right), 0) + 1
    gain = 0
    positive_target = target.sum(axis=1) > 0
    for node, change in delta.items():
        before = bool(positive_target[node] and current_total[node] == 0)
        after = bool(positive_target[node] and current_total[node] + change == 0)
        gain += int(before) - int(after)
    return float(gain)


def prefer_positive_role_coverage(
    scores: torch.Tensor,
    degree_gains: torch.Tensor,
    coverage_gains: torch.Tensor,
) -> torch.Tensor:
    """Lexicographically prefer edits that improve role error and coverage."""

    if scores.shape != degree_gains.shape or scores.shape != coverage_gains.shape:
        raise ValueError("score and gain tensors must have identical shapes")
    eligible = (degree_gains > 0) & (coverage_gains > 0)
    span = (scores.max() - scores.min()).detach().abs() + 1.0
    return scores + eligible.to(scores.dtype) * span


def denoising_time_fraction(completed: int, horizon_steps: int) -> float:
    if int(horizon_steps) <= 0:
        raise ValueError("time horizon must be positive")
    return max(0.0, 1.0 - (int(completed) - 1) / float(horizon_steps))


def break_role_score_ties(scores, rng):
    """Consume the legacy random draws, but perturb only exact maxima.

    A categorical decoder must also keep the original non-tied scores.
    This helper is only for the MAP tie-selection branch.
    """
    jitter = torch.from_numpy(rng.random(len(scores)).astype(np.float32)).to(scores.device)
    maxima = scores == scores.max()
    # Work in index-selection space: no epsilon can reverse a real score gap.
    chosen = torch.where(maxima, jitter, torch.full_like(jitter, -1.0)).argmax()
    # Return a finite selection ranking, not perturbed logits/probabilities.
    result = torch.zeros_like(scores)
    result[chosen] = 1.0
    return result


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
    hub_bias_power: float = 0.0,
    diagnostics: Optional[dict] = None,
    stable_neighbor_order: bool = False,
) -> Optional[EditProposal]:
    """Propose a legal 2-switch containing one currently open typed wedge."""

    domain_to_family = {
        tuple(sorted((int(spec.src_type), int(spec.dst_type)))): family
        for family, spec in graph.family_specs.items()
    }
    if float(degree_bias_power) < 0.0:
        raise ValueError("degree bias power must be non-negative")
    if float(hub_bias_power) < 0.0:
        raise ValueError("hub bias power must be non-negative")
    centers = np.asarray(
        [node for node, neighbors in enumerate(graph.adjacency) if len(neighbors) >= 2],
        dtype=np.int64,
    )
    if centers.size == 0:
        return None
    degrees = graph.degrees().astype(np.float64)

    def selection_probabilities(nodes: np.ndarray) -> Optional[np.ndarray]:
        exponent = float(hub_bias_power) - float(degree_bias_power)
        if exponent == 0.0:
            return None
        denominator = np.maximum(
            degrees[nodes] * np.maximum(degrees[nodes] - 1.0, 0.0) / 2.0,
            1.0,
        )
        weights = np.power(denominator, exponent)
        return weights / weights.sum()

    center_probabilities = selection_probabilities(centers)
    for _ in range(int(max_attempts)):
        if diagnostics is not None:
            diagnostics['wedge_inner_trials'] = diagnostics.get('wedge_inner_trials', 0) + 1
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
        if stable_neighbor_order:
            source_neighbors.sort()
            target_neighbors.sort()
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
    hub_bias_power: float = 0.0,
    diagnostics: Optional[dict] = None,
    stable_neighbor_order: bool = False,
) -> list[EditProposal]:
    if int(count) < 2:
        raise ValueError('candidate count must be at least two')
    if not 0.0 <= float(closure_fraction) <= 1.0:
        raise ValueError("closure candidate fraction must lie in [0,1]")
    if float(closure_fraction) == 0.0:
        return sample_edit_candidates(
            graph, rng, int(count), actions=("switch",), diagnostics=diagnostics
        )[0]
    candidates = [EditProposal("noop", sorted(graph.family_specs)[0], (), ())]
    keys = {_edit_key(candidates[0])}
    requested = min(int(count) - 1, int(round((int(count) - 1) * closure_fraction)))
    attempts = 0
    while len(candidates) - 1 < requested and attempts < max(512, requested * 64):
        attempts += 1
        proposal = propose_wedge_seeded_switch(
            graph,
            rng,
            degree_bias_power=degree_bias_power,
            hub_bias_power=hub_bias_power,
            diagnostics=diagnostics,
            stable_neighbor_order=stable_neighbor_order,
        )
        if diagnostics is not None:
            diagnostics['guided_outer_trials'] = attempts
            key = 'unreturned_proposals' if proposal is None else ('duplicate_proposals' if _edit_key(proposal) in keys else 'unique_proposals')
            diagnostics[key] = diagnostics.get(key, 0) + 1
        if proposal is not None and _edit_key(proposal) not in keys:
            keys.add(_edit_key(proposal))
            candidates.append(proposal)
    attempts = 0
    while len(candidates) < int(count) and attempts < int(count) * 512:
        attempts += 1
        proposal = sample_valid_edit(graph, rng, actions=("switch",), diagnostics=diagnostics)
        if diagnostics is not None:
            diagnostics['fallback_outer_trials'] = attempts
            key = 'unreturned_proposals' if proposal is None else ('duplicate_proposals' if _edit_key(proposal) in keys else 'unique_proposals')
            diagnostics[key] = diagnostics.get(key, 0) + 1
        if proposal is not None and _edit_key(proposal) not in keys:
            keys.add(_edit_key(proposal))
            candidates.append(proposal)
    if len(candidates) != int(count):
        if diagnostics is not None: diagnostics['actual_candidates'] = len(candidates)
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
    diagnostics: Optional[dict] = None,
) -> list[EditProposal]:
    if int(count) < 2:
        raise ValueError('candidate count must be at least two')
    if not 0.0 <= float(targeted_fraction) <= 1.0:
        raise ValueError("targeted role candidate fraction must lie in [0,1]")
    if float(targeted_fraction) == 0.0:
        return sample_edit_candidates(graph, rng, int(count), actions=actions, diagnostics=diagnostics)[0]
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
        if diagnostics is not None:
            diagnostics['guided_outer_trials'] = attempts
            key = 'unreturned_proposals' if proposal is None else ('duplicate_proposals' if _edit_key(proposal) in keys else 'unique_proposals')
            diagnostics[key] = diagnostics.get(key, 0) + 1
        if proposal is not None and _edit_key(proposal) not in keys:
            keys.add(_edit_key(proposal))
            candidates.append(proposal)
    attempts = 0
    while len(candidates) < int(count) and attempts < int(count) * 512:
        attempts += 1
        proposal = sample_valid_edit(graph, rng, actions=actions, diagnostics=diagnostics)
        if diagnostics is not None:
            diagnostics['fallback_outer_trials'] = attempts
            key = 'unreturned_proposals' if proposal is None else ('duplicate_proposals' if _edit_key(proposal) in keys else 'unique_proposals')
            diagnostics[key] = diagnostics.get(key, 0) + 1
        if proposal is not None and _edit_key(proposal) not in keys:
            keys.add(_edit_key(proposal))
            candidates.append(proposal)
    if len(candidates) != int(count):
        if diagnostics is not None: diagnostics['actual_candidates'] = len(candidates)
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
    relative_role_error_threshold: Optional[float] = None,
    require_positive_role_coverage: bool = False,
    model_time_horizon_steps: Optional[int] = None,
    model_time_start_step: int = 0,
    closure_candidate_fraction: float = 0.0,
    closure_degree_bias_power: float = 0.0,
    closure_hub_bias_power: float = 0.0,
    auto_role_balance: float = 0.0,
    role_calibration_batches: int = 32,
    targeted_role_candidate_fraction: float = 0.0,
    sampling_backend: str = "cpu",
    gpu_proposal_pool_size: int = 0,
    gpu_proposal_chunk_size: int = 1024,
    gpu_candidate_mode: str = "legacy_pool",
    gpu_metric_edge_chunk_size: int = 2048,
    gpu_phase_check_interval: int = 32,
    gpu_use_triton: bool = True,
    phase_policy: str = "auto",
    fixed_role_steps: Optional[int] = None,
    role_plateau_window_steps: int = 0,
    role_plateau_min_relative_improvement: float = 0.0,
    role_plateau_min_noop_fraction: float = 1.0,
    selection_mode: str = "model",
    decoder: str = "map",
    decoder_temperature: float = 1.0,
    stage1_positive_gain_guard: bool = False,
    runtime=None,
) -> MixedEditSamplingResult:
    validate_stage1_positive_gain_guard(stage1_positive_gain_guard,
        phase_policy=str(phase_policy).strip().lower(), selection_mode=str(selection_mode).strip().lower(),
        decoder=str(decoder).strip().lower(), sampling_backend=str(sampling_backend).strip().lower(),
        arm='M' if runtime is None else runtime.arm)
    if runtime is not None and runtime.restore:
        from .sampling_session import restore_graph
        graph = restore_graph(runtime.restore['graph'])
    else:
        graph = prior.sample(seed + 71) if initial_graph is None else initial_graph.clone()
    if tuple(graph.type_names) != role_prior.type_names:
        raise ValueError("role prior and graph schemas disagree")
    family_names = tuple(sorted(graph.family_specs))
    if family_names != role_prior.family_names:
        raise ValueError("role prior and graph families disagree")
    target_rows = (role_prior.sample_assignment_for_graph(graph, seed + 91_919)
                   if runtime is None or runtime.target_rows is None else runtime.target_rows.copy())
    if runtime is not None:
        from .target_feasibility import validate_role_target
        runtime.target_feasibility = validate_role_target(graph,target_rows,family_names)
    role_assignment_seed = int(seed) + 91_919
    role_assignment_hash = hashlib.sha256(
        "|".join(family_names).encode("utf-8")
        + np.ascontiguousarray(target_rows).tobytes()
    ).hexdigest()
    sampling_backend = str(sampling_backend).strip().lower()
    if sampling_backend not in {"cpu", "gpu"}:
        raise ValueError("sampling backend must be cpu or gpu")
    if relative_role_error_threshold is not None and not (
        0.0 <= float(relative_role_error_threshold) <= 1.0
    ):
        raise ValueError("relative role-error threshold must be in [0, 1]")
    phase_policy = str(phase_policy).strip().lower()
    if phase_policy not in {"auto", "fixed", "relocation_only", "switch_only"}:
        raise ValueError(
            "phase policy must be auto, fixed, relocation_only, or switch_only"
        )
    if phase_policy == "fixed":
        if fixed_role_steps is None or not 0 <= int(fixed_role_steps) <= int(steps):
            raise ValueError("fixed phase policy needs fixed_role_steps in [0, steps]")
    elif fixed_role_steps is not None:
        raise ValueError("fixed_role_steps is only valid with phase_policy='fixed'")
    if int(role_plateau_window_steps) < 0:
        raise ValueError("role plateau window must be non-negative")
    if int(role_plateau_window_steps) > 0 and phase_policy != "auto":
        raise ValueError("role plateau fallback requires phase_policy='auto'")
    if float(role_plateau_min_relative_improvement) < 0.0:
        raise ValueError("role plateau improvement threshold must be non-negative")
    if not 0.0 <= float(role_plateau_min_noop_fraction) <= 1.0:
        raise ValueError("role plateau noop threshold must lie in [0, 1]")
    selection_mode = str(selection_mode).strip().lower()
    if selection_mode not in {"model", "role_only", "random"}:
        raise ValueError("selection mode must be model, role_only, or random")
    decoder = str(decoder).strip().lower()
    if decoder not in {"map", "categorical"}:
        raise ValueError("decoder must be map or categorical")
    if float(decoder_temperature) <= 0.0:
        raise ValueError("decoder temperature must be positive")
    if selection_mode == "random" and decoder != "map":
        raise ValueError("random selection mode does not use a categorical decoder")
    if sampling_backend == "gpu":
        if (
            tuple(actions) != ("relocation", "switch")
            or phase_policy not in {"auto", "fixed"}
            or selection_mode not in {"model", "role_only"}
        ):
            raise ValueError(
            "GPU-resident backend currently requires the model-selected "
                "automatic or fixed relocation/switch policy"
            )
        from .gpu_role_guided_edit import sample_gpu_role_guided_edit_diffusion

        result = sample_gpu_role_guided_edit_diffusion(
            graph,
            model,
            target_rows,
            steps=steps,
            candidates_per_step=candidates_per_step,
            seed=seed,
            device=device,
            metrics_every=metrics_every,
            role_weight=role_weight,
            gain_scale=gain_scale,
            schedule_power=schedule_power,
            schedule_floor=schedule_floor,
            role_error_per_node_threshold=role_error_per_node_threshold,
            relative_role_error_threshold=relative_role_error_threshold,
            require_positive_role_coverage=require_positive_role_coverage,
            model_time_horizon_steps=(
                int(steps)
                if model_time_horizon_steps is None
                else int(model_time_horizon_steps)
            ),
            model_time_start_step=model_time_start_step,
            closure_candidate_fraction=closure_candidate_fraction,
            closure_degree_bias_power=closure_degree_bias_power,
            closure_hub_bias_power=closure_hub_bias_power,
            targeted_role_candidate_fraction=targeted_role_candidate_fraction,
            auto_role_balance=auto_role_balance,
            role_calibration_batches=role_calibration_batches,
            proposal_pool_size=gpu_proposal_pool_size,
            proposal_chunk_size=gpu_proposal_chunk_size,
            candidate_mode=gpu_candidate_mode,
            metric_edge_chunk_size=gpu_metric_edge_chunk_size,
            phase_check_interval=gpu_phase_check_interval,
            use_triton=gpu_use_triton,
            fixed_role_steps=fixed_role_steps,
            role_plateau_window_steps=role_plateau_window_steps,
            role_plateau_min_relative_improvement=(
                role_plateau_min_relative_improvement
            ),
            role_plateau_min_noop_fraction=role_plateau_min_noop_fraction,
            use_model_score=selection_mode == "model",
            decoder=decoder,
            decoder_temperature=decoder_temperature,
        )
        result.trace[0]["role_assignment_seed"] = role_assignment_seed
        result.trace[0]["role_assignment_hash"] = role_assignment_hash
        return result
    family_to_id = {family: index for index, family in enumerate(family_names)}
    rng = np.random.default_rng(int(seed))
    device = torch.device(device)
    model = model.to(device).eval()
    effective_role_weight = float(role_weight)
    if float(auto_role_balance) > 0.0 and not (runtime is not None and runtime.restore):
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
    relative_role_error_target = (
        current_role_error * float(relative_role_error_threshold)
        if relative_role_error_threshold is not None
        else None
    )

    def role_alignment_is_complete(error: float) -> bool:
        if relative_role_error_target is not None:
            error_complete = float(error) <= float(relative_role_error_target)
        else:
            error_complete = role_alignment_complete(
                error,
                graph.num_nodes,
                role_error_per_node_threshold,
            )
        if not error_complete:
            return False
        return not require_positive_role_coverage or positive_role_zero_count(
            graph, target_rows, family_names
        ) == 0

    switch_refinement = phase_policy == "switch_only" or (
        phase_policy == "auto"
        and role_alignment_is_complete(current_role_error)
    ) or (
        phase_policy == "fixed" and int(fixed_role_steps or 0) == 0
    )
    transition_step = 0 if switch_refinement else None
    transition_reason = "initial" if switch_refinement else None
    plateau_reference_step = 0
    plateau_reference_error = current_role_error
    plateau_noops = 0
    plateau_observations = 0
    trace = [
        {
            "step": 0.0,
            "relation_degree_error": current_role_error,
            "relation_degree_error_per_node": role_error_per_node(
                current_role_error, graph.num_nodes
            ),
            "relation_degree_error_threshold": (
                relative_role_error_target
                if relative_role_error_target is not None
                else float(role_error_per_node_threshold) * graph.num_nodes
            ),
            "relative_role_error_threshold": relative_role_error_threshold,
            "positive_role_zero_count": positive_role_zero_count(
                graph, target_rows, family_names
            ),
            "require_positive_role_coverage": bool(require_positive_role_coverage),
            "phase": "switch_refinement" if switch_refinement else "role_alignment",
            "role_phase_transition_step": transition_step,
            "role_phase_transition_reason": transition_reason,
            "effective_role_weight": effective_role_weight,
            "role_assignment_seed": role_assignment_seed,
            "role_assignment_hash": role_assignment_hash,
            **graph_metrics(graph),
        }
    ]
    action_counts = {"relocation": 0, "switch": 0, "noop": 0}
    proposed = 0
    applied = 0
    time_horizon = (
        max(int(steps), 1)
        if model_time_horizon_steps is None
        else int(model_time_horizon_steps)
    )
    if time_horizon <= 0:
        raise ValueError("model time horizon must be positive")
    if int(model_time_start_step) < 0:
        raise ValueError("model time start step must be non-negative")
    start_completed = 0
    if runtime is not None and runtime.restore:
        state = runtime.restore['state']
        effective_role_weight = state['effective_role_weight']
        current_role_error = state['current_role_error']
        relative_role_error_target = state['relative_role_error_target']
        switch_refinement = state['switch_refinement']
        transition_step,transition_reason = state['transition_step'],state['transition_reason']
        plateau_reference_step,plateau_reference_error = state['plateau_reference_step'],state['plateau_reference_error']
        plateau_noops,plateau_observations = state['plateau_noops'],state['plateau_observations']
        trace,action_counts = state['trace'],state['action_counts']
        proposed,applied,time_horizon = state['proposed'],state['applied'],state['time_horizon']
        start_completed = runtime.restore['completed']
        rng.bit_generator.state = runtime.restore['numpy_generator']
        if not np.isclose(current_role_error,relation_degree_error(graph,target_rows,family_names)):
            raise ValueError('snapshot graph/degree error mismatch')
    with torch.no_grad():
        for completed in range(start_completed+1, int(steps) + 1):
            time_fraction = denoising_time_fraction(
                int(model_time_start_step) + completed, time_horizon
            )
            if (
                phase_policy == "auto"
                and not switch_refinement
                and role_alignment_is_complete(current_role_error)
            ):
                switch_refinement = True
                transition_step = completed
                transition_reason = "threshold"
            if (
                phase_policy == "fixed"
                and not switch_refinement
                and completed > int(fixed_role_steps or 0)
            ):
                switch_refinement = True
                transition_step = int(fixed_role_steps or 0)
                transition_reason = "fixed"
            if runtime is not None and runtime.before_attempt(graph,target_rows,locals()):
                break
            proposal_rng,selection_rng = (rng,rng) if runtime is None else runtime.streams(
                rng,seed,int(model_time_start_step)+completed)
            diagnostics = None if runtime is None else runtime.diagnostics
            if switch_refinement:
                try:
                    candidates = sample_switch_refinement_candidates(
                        graph,proposal_rng,candidates_per_step,
                        closure_fraction=closure_candidate_fraction,
                        degree_bias_power=closure_degree_bias_power,
                        hub_bias_power=closure_hub_bias_power,diagnostics=diagnostics,
                        stable_neighbor_order=runtime is not None)
                except Exception as exc:
                    if runtime is not None: runtime.failure(exc)
                    raise
            else:
                try:
                    candidates = sample_role_alignment_candidates(
                    graph,
                    target_rows,
                    family_names,
                    proposal_rng,
                    candidates_per_step,
                    targeted_fraction=targeted_role_candidate_fraction,
                    actions=("relocation",) if phase_policy == "relocation_only" else actions,
                    diagnostics=diagnostics)
                except Exception as exc:
                    if runtime is not None: runtime.failure(exc)
                    raise
            proposed += len(candidates)
            if selection_mode in {"random", "role_only"} or (runtime is not None and phase_policy == 'switch_only' and runtime.arm in ('R','C')):
                structure_logits = torch.zeros(
                    len(candidates), dtype=torch.float32, device=device
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
            degree_gain_values = [
                    proposal_relation_degree_gain(
                        graph, proposal, target_rows, family_to_id
                    )
                    for proposal in candidates
                ]
            degree_gains = torch.tensor(
                degree_gain_values,
                dtype=structure_logits.dtype,
                device=device,
            )
            scores = None
            if runtime is not None and phase_policy == 'switch_only' and runtime.arm == 'C':
                from .attribution_motifs import rule_scores
                scores = torch.tensor(rule_scores(graph,candidates,runtime.templates,runtime.reference_counts),dtype=torch.float64)
                selected_index = int(scores.argmax())
            elif selection_mode == "random" or (runtime is not None and phase_policy == 'switch_only' and runtime.arm == 'R'):
                selected_index = int(selection_rng.integers(len(candidates)))
            else:
                scores = combine_structure_and_role_scores(
                    structure_logits,
                    degree_gains,
                    time_fraction=time_fraction,
                    role_weight=effective_role_weight,
                    gain_scale=gain_scale,
                    schedule_power=schedule_power,
                    schedule_floor=schedule_floor,
                )
                if require_positive_role_coverage and not switch_refinement:
                    coverage_gains = torch.tensor(
                        [
                            proposal_positive_role_coverage_gain(
                                graph, proposal, target_rows, family_names
                            )
                            for proposal in candidates
                        ],
                        dtype=scores.dtype,
                        device=device,
                    )
                    scores = prefer_positive_role_coverage(
                        scores, degree_gains, coverage_gains
                    )
                if decoder == "categorical":
                    probabilities = torch.softmax(
                        scores / float(decoder_temperature), dim=0
                    ).detach().cpu().numpy()
                    selected_index = int(selection_rng.choice(len(candidates), p=probabilities))
                else:
                    selected_index = int((break_role_score_ties(scores,selection_rng)
                                          if selection_mode=='role_only' else scores).argmax().item())
            original_argmax = None if scores is None else int(scores.argmax().item())
            positive_indices = [i for i,g in enumerate(degree_gain_values) if g > 0]
            guard_applied = fallback_to_legacy = False
            if stage1_positive_gain_guard and not switch_refinement:
                selected_index,positive_indices,guard_applied,fallback_to_legacy = guarded_stage1_index(
                    scores,degree_gain_values,selected_index)
            selected = candidates[selected_index]
            apply_edit(graph, selected)
            current_role_error = max(
                0.0,
                current_role_error - float(degree_gains[selected_index].item()),
            )
            action_counts[selected.action] += 1
            applied += int(selected.action != "noop")
            if not switch_refinement:
                plateau_observations += 1
                plateau_noops += int(selected.action == "noop")
                if (
                    phase_policy == "auto"
                    and int(role_plateau_window_steps) > 0
                    and completed - plateau_reference_step
                    >= int(role_plateau_window_steps)
                ):
                    coverage_complete = (
                        not require_positive_role_coverage
                        or positive_role_zero_count(
                            graph, target_rows, family_names
                        )
                        == 0
                    )
                    if coverage_complete and role_alignment_plateaued(
                        plateau_reference_error,
                        current_role_error,
                        plateau_noops,
                        plateau_observations,
                        min_relative_improvement=(
                            role_plateau_min_relative_improvement
                        ),
                        min_noop_fraction=role_plateau_min_noop_fraction,
                    ):
                        switch_refinement = True
                        transition_step = completed
                        transition_reason = "plateau"
                    else:
                        plateau_reference_step = completed
                        plateau_reference_error = current_role_error
                        plateau_noops = 0
                        plateau_observations = 0
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
                        "role_phase_transition_reason": transition_reason,
                        "positive_role_zero_count": positive_role_zero_count(
                            graph, target_rows, family_names
                        ),
                        "effective_role_weight": effective_role_weight,
                        **graph_metrics(graph),
                    }
                )
            if runtime is not None:
                current_degrees = np.stack([graph.family_degrees(f) for f in family_names],axis=1)
                if runtime.after_attempt(graph,target_rows,locals()): break
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
