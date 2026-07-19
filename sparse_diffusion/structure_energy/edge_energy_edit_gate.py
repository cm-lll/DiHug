from __future__ import annotations

from dataclasses import dataclass
from typing import List, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from .graph import TypedGraph
from .hgt_edit_denoiser import graph_edge_tensors, graph_node_features
from .latent_edit_representation import (
    RepresentationSnapshot,
    RepresentationWarmupDataset,
    WarmupHGTEncoder,
    _normalize_role_rows,
    _role_normalizer,
)
from .mixed_edit_diffusion import (
    EditProposal,
    edit_distance_improvement,
    mixed_edit_candidate_matrix,
    sample_edit_candidates,
)
from .switch_diffusion import SwitchDenoiserMLP


_ACTION_TO_ID = {"noop": 0, "relocation": 1, "switch": 2}


@dataclass(frozen=True)
class EdgeEnergyExample:
    level: float
    base_logits: np.ndarray
    edge_delta: np.ndarray
    action_ids: np.ndarray
    utilities: np.ndarray


class EdgeEnergyFusion(nn.Module):
    """Add a calibrated clean-edge energy delta to a frozen edit scorer."""

    def __init__(self) -> None:
        super().__init__()
        # One coefficient per state-changing terminal action. Projection after
        # every update keeps the learned term aligned with clean-edge evidence.
        self.action_weights = nn.Parameter(torch.zeros(2))

    def forward(
        self,
        base_logits: torch.Tensor,
        normalized_edge_delta: torch.Tensor,
        action_ids: torch.Tensor,
    ) -> torch.Tensor:
        if base_logits.shape != normalized_edge_delta.shape:
            raise ValueError("base logits and edge delta must have equal shapes")
        if action_ids.shape != base_logits.shape:
            raise ValueError("action ids must match candidate score shape")
        safe_ids = (action_ids - 1).clamp(min=0, max=1)
        weights = self.action_weights[safe_ids]
        active = (action_ids > 0).to(base_logits.dtype)
        return base_logits + active * weights * normalized_edge_delta

    @torch.no_grad()
    def project_nonnegative(self) -> None:
        self.action_weights.clamp_(min=0.0)


class EdgeEnergyInferenceResidual(nn.Module):
    """Frozen HGT clean-edge residual for role-guided edit inference."""

    supports_role_alignment = True

    def __init__(
        self,
        representation: WarmupHGTEncoder,
        *,
        family_names: Sequence[str],
        role_means: np.ndarray,
        role_scales: np.ndarray,
        action_scales: Sequence[float],
        action_weights: Sequence[float],
    ) -> None:
        super().__init__()
        self.representation = representation
        self.family_names = tuple(family_names)
        self.register_buffer(
            "role_means", torch.as_tensor(role_means, dtype=torch.float32)
        )
        self.register_buffer(
            "role_scales", torch.as_tensor(role_scales, dtype=torch.float32)
        )
        self.register_buffer(
            "action_scales", torch.as_tensor(action_scales, dtype=torch.float32)
        )
        self.register_buffer(
            "action_weights", torch.as_tensor(action_weights, dtype=torch.float32)
        )

    def encode_graph(
        self,
        graph: TypedGraph,
        time_fraction: float,
        device: torch.device,
        *,
        role_rows: np.ndarray,
    ) -> torch.Tensor:
        node_types = torch.from_numpy(graph.node_types).long().to(device)
        transformed = torch.log1p(
            torch.as_tensor(role_rows, dtype=torch.float32, device=device)
        )
        role_condition = (
            transformed - self.role_means[node_types]
        ) / self.role_scales[node_types]
        edge_index, edge_types = graph_edge_tensors(graph)
        return self.representation.encode(
            torch.from_numpy(graph_node_features(graph, time_fraction))
            .float()
            .to(device),
            node_types,
            torch.from_numpy(edge_index).long().to(device),
            torch.from_numpy(edge_types).long().to(device),
            role_condition,
        )

    def score_graph_candidates(
        self,
        graph: TypedGraph,
        candidates: Sequence[EditProposal],
        time_fraction: float,
        device: torch.device,
        *,
        encoded: torch.Tensor,
    ) -> torch.Tensor:
        raw = candidate_edge_energy_delta(
            graph,
            candidates,
            representation=self.representation,
            hidden=encoded,
            family_names=self.family_names,
            time_fraction=time_fraction,
            device=device,
        )
        result = torch.zeros(len(candidates), dtype=torch.float32, device=device)
        for index, proposal in enumerate(candidates):
            action_id = _ACTION_TO_ID[proposal.action]
            if action_id <= 0:
                continue
            scale = self.action_scales[action_id - 1].clamp_min(1.0e-3)
            result[index] = (
                self.action_weights[action_id - 1]
                * float(raw[index])
                / scale
            )
        return result


def _encode_snapshot(
    model: WarmupHGTEncoder,
    snapshot: RepresentationSnapshot,
    role_condition: torch.Tensor,
    node_types: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    edge_index, edge_types = graph_edge_tensors(snapshot.graph)
    return model.encode(
        torch.from_numpy(
            graph_node_features(snapshot.graph, snapshot.time_fraction)
        ).float().to(device),
        node_types,
        torch.from_numpy(edge_index).long().to(device),
        torch.from_numpy(edge_types).long().to(device),
        role_condition,
    )


def candidate_edge_energy_delta(
    graph: TypedGraph,
    candidates: Sequence[EditProposal],
    *,
    representation: WarmupHGTEncoder,
    hidden: torch.Tensor,
    family_names: Sequence[str],
    time_fraction: float,
    device: torch.device,
) -> np.ndarray:
    """Return mean(new clean-edge logit) minus mean(old clean-edge logit).

    The current graph is encoded once. Old and new edge slots are decoded as
    queries in one batch, so evaluating C complete edits does not require C HGT
    forward passes.
    """

    family_to_id = {name: index for index, name in enumerate(family_names)}
    query_edges: List[tuple[int, int]] = []
    query_families: List[int] = []
    current_exists: List[float] = []
    owners: List[int] = []
    signs: List[float] = []
    for candidate_index, proposal in enumerate(candidates):
        if proposal.action == "noop":
            continue
        family_id = family_to_id[proposal.family]
        for edge in proposal.old_edges:
            query_edges.append(graph.orient_edge(proposal.family, edge))
            query_families.append(family_id)
            current_exists.append(1.0)
            owners.append(candidate_index)
            signs.append(-1.0 / max(len(proposal.old_edges), 1))
        for edge in proposal.new_edges:
            query_edges.append(graph.orient_edge(proposal.family, edge))
            query_families.append(family_id)
            current_exists.append(0.0)
            owners.append(candidate_index)
            signs.append(1.0 / max(len(proposal.new_edges), 1))
    deltas = np.zeros(len(candidates), dtype=np.float32)
    if not query_edges:
        return deltas
    edge_tensor = torch.as_tensor(query_edges, dtype=torch.long, device=device).T
    family_tensor = torch.as_tensor(
        query_families, dtype=torch.long, device=device
    )
    exists_tensor = torch.as_tensor(
        current_exists, dtype=torch.float32, device=device
    )
    with torch.no_grad():
        logits = representation.edge_logits(
            hidden,
            edge_tensor,
            family_tensor,
            exists_tensor,
            float(time_fraction),
        ).cpu().numpy()
    np.add.at(
        deltas,
        np.asarray(owners, dtype=np.int64),
        np.asarray(signs, dtype=np.float32) * np.asarray(logits, dtype=np.float32),
    )
    return deltas


def _prepare_snapshot_examples(
    snapshot: RepresentationSnapshot,
    *,
    representation: WarmupHGTEncoder,
    base_scorer: SwitchDenoiserMLP,
    family_names: Sequence[str],
    feature_profile: str,
    role_means: np.ndarray,
    role_scales: np.ndarray,
    node_types: torch.Tensor,
    candidates_per_example: int,
    candidate_sets: int,
    seed: int,
    device: torch.device,
) -> List[EdgeEnergyExample]:
    role_array = _normalize_role_rows(
        snapshot.role_rows,
        snapshot.graph.node_types,
        role_means,
        role_scales,
    )
    role_condition = torch.from_numpy(role_array).float().to(device)
    with torch.no_grad():
        hidden = _encode_snapshot(
            representation,
            snapshot,
            role_condition,
            node_types,
            device,
        )
    rows = []
    for candidate_set in range(int(candidate_sets)):
        rng = np.random.default_rng(int(seed) + 97_409 * candidate_set)
        candidates, _ = sample_edit_candidates(
            snapshot.graph,
            rng,
            int(candidates_per_example),
            actions=("relocation", "switch"),
        )
        features = mixed_edit_candidate_matrix(
            snapshot.graph,
            candidates,
            snapshot.time_fraction,
            feature_profile=feature_profile,
        ).astype(np.float32)
        with torch.no_grad():
            base_logits = (
                base_scorer(torch.from_numpy(features)[None].to(device))[0]
                .cpu()
                .numpy()
            )
        edge_delta = candidate_edge_energy_delta(
            snapshot.graph,
            candidates,
            representation=representation,
            hidden=hidden,
            family_names=family_names,
            time_fraction=snapshot.time_fraction,
            device=device,
        )
        utilities = np.asarray(
            [
                edit_distance_improvement(snapshot.graph, snapshot.clean, proposal)
                for proposal in candidates
            ],
            dtype=np.float32,
        )
        rows.append(
            EdgeEnergyExample(
                level=float(snapshot.time_fraction),
                base_logits=np.asarray(base_logits, dtype=np.float32),
                edge_delta=edge_delta,
                action_ids=np.asarray(
                    [_ACTION_TO_ID[proposal.action] for proposal in candidates],
                    dtype=np.int64,
                ),
                utilities=utilities,
            )
        )
    return rows


def _action_scales(examples: Sequence[EdgeEnergyExample]) -> np.ndarray:
    scales = np.ones(2, dtype=np.float32)
    for action_id in (1, 2):
        values = np.concatenate(
            [
                example.edge_delta[example.action_ids == action_id]
                for example in examples
            ]
        )
        rms = float(np.sqrt(np.mean(np.square(values)))) if len(values) else 1.0
        scales[action_id - 1] = max(rms, 1.0e-3)
    return scales


def _normalized_delta(
    example: EdgeEnergyExample, scales: np.ndarray
) -> np.ndarray:
    result = np.asarray(example.edge_delta, dtype=np.float32).copy()
    for action_id in (1, 2):
        mask = example.action_ids == action_id
        result[mask] /= float(scales[action_id - 1])
    return result


def _mean_metrics(
    examples: Sequence[EdgeEnergyExample],
    fusion: EdgeEnergyFusion,
    scales: np.ndarray,
    device: torch.device,
) -> Mapping[str, float]:
    rows = []
    fusion.eval()
    with torch.no_grad():
        for example in examples:
            base = torch.from_numpy(example.base_logits)[None].to(device)
            delta_np = _normalized_delta(example, scales)
            delta = torch.from_numpy(delta_np)[None].to(device)
            actions = torch.from_numpy(example.action_ids)[None].to(device)
            fused = fusion(base, delta, actions)
            utility = np.asarray(example.utilities, dtype=np.float64)
            base_probability = torch.softmax(base, dim=1)[0].cpu().numpy()
            fused_probability = torch.softmax(fused, dim=1)[0].cpu().numpy()
            edge_probability = torch.softmax(delta, dim=1)[0].cpu().numpy()
            correlation = 0.0
            if np.std(delta_np) > 1.0e-8 and np.std(utility) > 1.0e-8:
                correlation = float(np.corrcoef(delta_np, utility)[0, 1])
            rows.append(
                {
                    "base_selected": float(utility[int(base.argmax(dim=1))]),
                    "fused_selected": float(utility[int(fused.argmax(dim=1))]),
                    "edge_selected": float(utility[int(delta.argmax(dim=1))]),
                    "oracle": float(utility.max()),
                    "base_expected": float(np.dot(base_probability, utility)),
                    "fused_expected": float(np.dot(fused_probability, utility)),
                    "edge_expected": float(np.dot(edge_probability, utility)),
                    "edge_utility_correlation": correlation,
                }
            )
    result = {
        key: float(np.mean([row[key] for row in rows]))
        for key in rows[0]
    }
    result["selected_utility_gain"] = (
        result["fused_selected"] - result["base_selected"]
    )
    result["expected_utility_gain"] = (
        result["fused_expected"] - result["base_expected"]
    )
    return result


def train_edge_energy_fusion_gate(
    dataset: RepresentationWarmupDataset,
    representation: WarmupHGTEncoder,
    base_scorer: SwitchDenoiserMLP,
    *,
    feature_profile: str = "typed_closure_compact_v2",
    candidates_per_example: int = 16,
    train_candidate_sets: int = 4,
    validation_candidate_sets: int = 8,
    epochs: int = 100,
    learning_rate: float = 0.05,
    target_temperature: float = 0.5,
    l2_weight: float = 0.01,
    device: str | torch.device = "cuda",
    seed: int = 0,
) -> tuple[EdgeEnergyFusion, Mapping[str, object]]:
    """Fit only two non-negative coefficients on frozen HGT/MLP scores."""

    device = torch.device(device)
    torch.manual_seed(int(seed))
    representation = representation.to(device).eval()
    base_scorer = base_scorer.to(device).eval()
    for model in (representation, base_scorer):
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    node_types = torch.from_numpy(dataset.node_types).long().to(device)
    role_means, role_scales = _role_normalizer(
        dataset.snapshots[0].role_rows,
        dataset.node_types,
        dataset.type_count,
    )
    validation_trajectory = max(item.trajectory_id for item in dataset.snapshots)
    train_examples: List[EdgeEnergyExample] = []
    validation_examples: List[EdgeEnergyExample] = []
    for snapshot_index, snapshot in enumerate(dataset.snapshots):
        is_validation = snapshot.trajectory_id == validation_trajectory
        target = validation_examples if is_validation else train_examples
        target.extend(
            _prepare_snapshot_examples(
                snapshot,
                representation=representation,
                base_scorer=base_scorer,
                family_names=dataset.family_names,
                feature_profile=feature_profile,
                role_means=role_means,
                role_scales=role_scales,
                node_types=node_types,
                candidates_per_example=int(candidates_per_example),
                candidate_sets=(
                    int(validation_candidate_sets)
                    if is_validation
                    else int(train_candidate_sets)
                ),
                seed=int(seed) + 1_000_003 * snapshot_index,
                device=device,
            )
        )
    scales = _action_scales(train_examples)
    fusion = EdgeEnergyFusion().to(device)
    optimizer = torch.optim.AdamW(
        fusion.parameters(), lr=float(learning_rate), weight_decay=0.0
    )
    rng = np.random.default_rng(int(seed))
    final_loss = 0.0
    for epoch in range(int(epochs)):
        fusion.train()
        losses = []
        for example_index in rng.permutation(len(train_examples)).tolist():
            example = train_examples[int(example_index)]
            base = torch.from_numpy(example.base_logits)[None].to(device)
            delta = torch.from_numpy(_normalized_delta(example, scales))[None].to(device)
            actions = torch.from_numpy(example.action_ids)[None].to(device)
            utilities = torch.from_numpy(example.utilities)[None].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = fusion(base, delta, actions)
            teacher = torch.softmax(utilities / float(target_temperature), dim=1)
            listwise = -(teacher * torch.log_softmax(logits, dim=1)).sum(dim=1).mean()
            regularizer = float(l2_weight) * fusion.action_weights.square().sum()
            loss = listwise + regularizer
            loss.backward()
            optimizer.step()
            fusion.project_nonnegative()
            losses.append(float(loss.detach().cpu()))
        final_loss = float(np.mean(losses))
        if epoch in {0, int(epochs) - 1}:
            weights = fusion.action_weights.detach().cpu().tolist()
            print(
                "[EDGE-ENERGY-EPOCH] "
                f"epoch={epoch} loss={final_loss:.6f} "
                f"lambda_relocation={weights[0]:.6f} "
                f"lambda_switch={weights[1]:.6f}",
                flush=True,
            )

    by_noise = {}
    for level in sorted({item.level for item in validation_examples}):
        selected = [item for item in validation_examples if item.level == level]
        by_noise[str(level)] = dict(_mean_metrics(selected, fusion, scales, device))
    overall = dict(_mean_metrics(validation_examples, fusion, scales, device))
    report = {
        "format": "dihug-edge-energy-edit-gate-v1",
        "train_examples": len(train_examples),
        "validation_examples": len(validation_examples),
        "candidates_per_example": int(candidates_per_example),
        "feature_profile": str(feature_profile),
        "action_scales": {
            "relocation": float(scales[0]),
            "switch": float(scales[1]),
        },
        "action_weights": {
            "relocation": float(fusion.action_weights[0].detach().cpu()),
            "switch": float(fusion.action_weights[1].detach().cpu()),
        },
        "family_names": list(dataset.family_names),
        "role_normalizer": {
            "means": role_means.tolist(),
            "scales": role_scales.tolist(),
        },
        "train_loss": final_loss,
        "overall": overall,
        "by_noise": by_noise,
    }
    return fusion, report
