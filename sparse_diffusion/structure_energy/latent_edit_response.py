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
    _cosine_distance,
    _normalize_role_rows,
    _role_normalizer,
)
from .mixed_edit_diffusion import (
    EditProposal,
    apply_edit,
    edit_distance_improvement,
    mixed_edit_candidate_matrix,
    sample_edit_candidates,
)
from .switch_diffusion import SwitchDenoiserMLP


@dataclass(frozen=True)
class LatentResponseExample:
    level: float
    features: np.ndarray
    endpoint_hidden: np.ndarray
    endpoint_prototype: np.ndarray
    endpoint_role_delta: np.ndarray
    endpoint_mask: np.ndarray
    utilities: np.ndarray
    response_index: int
    response_target: float


class LocalEditResponseScorer(nn.Module):
    def __init__(
        self,
        *,
        feature_dim: int,
        hidden_dim: int,
        role_dim: int,
        scorer_hidden_dim: int = 128,
        scorer_layers: int = 2,
        dropout: float = 0.1,
        response_scale: float = 1000.0,
    ) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.role_dim = int(role_dim)
        self.response_scale = float(response_scale)
        if self.response_scale <= 0.0:
            raise ValueError("response scale must be positive")
        response_width = max(int(hidden_dim), 64)
        self.feature_projection = nn.Sequential(
            nn.Linear(self.feature_dim, response_width),
            nn.SiLU(),
        )
        self.response = nn.Sequential(
            nn.Linear(
                2 * self.hidden_dim + self.role_dim + response_width,
                response_width,
            ),
            nn.LayerNorm(response_width),
            nn.SiLU(),
            nn.Linear(response_width, 1),
        )
        final_response = self.response[-1]
        assert isinstance(final_response, nn.Linear)
        nn.init.zeros_(final_response.weight)
        nn.init.zeros_(final_response.bias)
        self.base_scorer = SwitchDenoiserMLP(
            self.feature_dim,
            hidden_dim=int(scorer_hidden_dim),
            layers=int(scorer_layers),
            dropout=float(dropout),
        )
        self.latent_scorer = SwitchDenoiserMLP(
            self.feature_dim + 1,
            hidden_dim=int(scorer_hidden_dim),
            layers=int(scorer_layers),
            dropout=float(dropout),
        )

    def forward(
        self,
        features: torch.Tensor,
        endpoint_hidden: torch.Tensor,
        endpoint_prototype: torch.Tensor,
        endpoint_role_delta: torch.Tensor,
        endpoint_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if features.ndim != 3:
            raise ValueError("features must have shape [B,C,D]")
        projected = self.feature_projection(features)[:, :, None, :].expand(
            -1, -1, endpoint_hidden.shape[2], -1
        )
        response_input = torch.cat(
            (
                endpoint_hidden,
                endpoint_prototype,
                endpoint_role_delta,
                projected,
            ),
            dim=-1,
        )
        endpoint_gain_scaled = self.response(response_input).squeeze(-1)
        mask = endpoint_mask.to(endpoint_gain_scaled.dtype)
        latent_gain_scaled = (endpoint_gain_scaled * mask).sum(dim=-1) / mask.sum(
            dim=-1
        ).clamp_min(1.0)
        latent_gain = latent_gain_scaled / self.response_scale
        base_score = self.base_scorer(features)
        latent_score = self.latent_scorer(
            torch.cat((features, latent_gain_scaled[..., None]), dim=-1)
        )
        return base_score, latent_score, endpoint_gain_scaled, latent_gain


def _proposal_endpoint_data(
    graph: TypedGraph,
    proposals: Sequence[EditProposal],
    family_names: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    family_to_id = {name: index for index, name in enumerate(family_names)}
    node_ids = np.full((len(proposals), 4), -1, dtype=np.int64)
    masks = np.zeros((len(proposals), 4), dtype=np.bool_)
    role_delta = np.zeros(
        (len(proposals), 4, len(family_names)), dtype=np.float32
    )
    scale = max(float(np.sqrt(graph.num_nodes)), 1.0)
    for proposal_index, proposal in enumerate(proposals):
        affected = sorted(
            {
                int(node)
                for edge in tuple(proposal.old_edges) + tuple(proposal.new_edges)
                for node in edge
            }
        )
        if len(affected) > 4:
            raise AssertionError("an edit affects more than four endpoints")
        positions = {node: index for index, node in enumerate(affected)}
        node_ids[proposal_index, : len(affected)] = affected
        masks[proposal_index, : len(affected)] = True
        if proposal.action == "noop":
            continue
        family_id = family_to_id[proposal.family]
        for edge in proposal.old_edges:
            for node in edge:
                role_delta[proposal_index, positions[int(node)], family_id] -= 1.0
        for edge in proposal.new_edges:
            for node in edge:
                role_delta[proposal_index, positions[int(node)], family_id] += 1.0
    role_delta /= scale
    return node_ids, masks, role_delta


def _encode_graph(
    model: WarmupHGTEncoder,
    graph: TypedGraph,
    time_fraction: float,
    role_condition: torch.Tensor,
    node_types: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    edge_index, edge_types = graph_edge_tensors(graph)
    return model.encode(
        torch.from_numpy(graph_node_features(graph, time_fraction))
        .float()
        .to(device),
        node_types,
        torch.from_numpy(edge_index).long().to(device),
        torch.from_numpy(edge_types).long().to(device),
        role_condition,
    )


def _prepare_example(
    snapshot: RepresentationSnapshot,
    *,
    representation: WarmupHGTEncoder,
    family_names: Sequence[str],
    role_means: np.ndarray,
    role_scales: np.ndarray,
    node_types: torch.Tensor,
    candidates_per_example: int,
    seed: int,
    device: torch.device,
) -> LatentResponseExample:
    rng = np.random.default_rng(int(seed))
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
        feature_profile="typed_closure_compact_v2",
    )
    utilities = np.asarray(
        [
            edit_distance_improvement(snapshot.graph, snapshot.clean, proposal)
            for proposal in candidates
        ],
        dtype=np.float32,
    )
    role_array = _normalize_role_rows(
        snapshot.role_rows,
        snapshot.graph.node_types,
        role_means,
        role_scales,
    )
    role_condition = torch.from_numpy(role_array).float().to(device)
    with torch.no_grad():
        hidden = _encode_graph(
            representation,
            snapshot.graph,
            snapshot.time_fraction,
            role_condition,
            node_types,
            device,
        )
        prototypes = representation.role_prototypes(role_condition, node_types)
        assignments = _cosine_distance(prototypes, hidden[:, None, :]).argmin(dim=1)
        chosen_prototypes = prototypes[
            torch.arange(prototypes.shape[0], device=device), assignments
        ]
    node_ids, endpoint_mask, role_delta = _proposal_endpoint_data(
        snapshot.graph, candidates, family_names
    )
    safe_ids = np.maximum(node_ids, 0)
    endpoint_hidden = hidden[torch.from_numpy(safe_ids).long().to(device)].cpu().numpy()
    endpoint_prototype = chosen_prototypes[
        torch.from_numpy(safe_ids).long().to(device)
    ].cpu().numpy()
    endpoint_hidden[~endpoint_mask] = 0.0
    endpoint_prototype[~endpoint_mask] = 0.0

    non_noop = np.flatnonzero(endpoint_mask.any(axis=1))
    if not len(non_noop):
        raise RuntimeError("candidate set contains no state-changing edit")
    response_index = int(rng.choice(non_noop))
    post_graph = snapshot.graph.clone()
    apply_edit(post_graph, candidates[response_index])
    with torch.no_grad():
        post_hidden = _encode_graph(
            representation,
            post_graph,
            snapshot.time_fraction,
            role_condition,
            node_types,
            device,
        )
    response_nodes = node_ids[response_index]
    response_mask = endpoint_mask[response_index]
    valid_nodes = response_nodes[response_mask]
    valid_tensor = torch.from_numpy(valid_nodes).long().to(device)
    before_energy = _cosine_distance(
        hidden[valid_tensor], chosen_prototypes[valid_tensor]
    ).mean()
    after_energy = _cosine_distance(
        post_hidden[valid_tensor], chosen_prototypes[valid_tensor]
    ).mean()
    response_target = float((before_energy - after_energy).cpu())
    return LatentResponseExample(
        level=float(snapshot.time_fraction),
        features=np.asarray(features, dtype=np.float32),
        endpoint_hidden=np.asarray(endpoint_hidden, dtype=np.float32),
        endpoint_prototype=np.asarray(endpoint_prototype, dtype=np.float32),
        endpoint_role_delta=np.asarray(role_delta, dtype=np.float32),
        endpoint_mask=endpoint_mask,
        utilities=utilities,
        response_index=response_index,
        response_target=response_target,
    )


def _tensor(value: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(value).to(device)


def train_local_edit_response_gate(
    dataset: RepresentationWarmupDataset,
    representation: WarmupHGTEncoder,
    *,
    candidates_per_example: int = 16,
    validation_candidate_sets: int = 4,
    epochs: int = 20,
    learning_rate: float = 5.0e-4,
    response_loss_weight: float = 1.0,
    target_temperature: float = 0.5,
    device: str | torch.device = "cuda",
    seed: int = 0,
) -> tuple[LocalEditResponseScorer, Mapping[str, object]]:
    device = torch.device(device)
    torch.manual_seed(int(seed))
    representation = representation.to(device).eval()
    for parameter in representation.parameters():
        parameter.requires_grad_(False)
    node_types = torch.from_numpy(dataset.node_types).long().to(device)
    role_means, role_scales = _role_normalizer(
        dataset.snapshots[0].role_rows,
        dataset.node_types,
        dataset.type_count,
    )
    validation_trajectory = max(item.trajectory_id for item in dataset.snapshots)
    train_snapshots = [
        item for item in dataset.snapshots if item.trajectory_id != validation_trajectory
    ]
    validation_snapshots = [
        item for item in dataset.snapshots if item.trajectory_id == validation_trajectory
    ]
    train_examples = [
        _prepare_example(
            snapshot,
            representation=representation,
            family_names=dataset.family_names,
            role_means=role_means,
            role_scales=role_scales,
            node_types=node_types,
            candidates_per_example=int(candidates_per_example),
            seed=int(seed) + 1_000_003 * index,
            device=device,
        )
        for index, snapshot in enumerate(train_snapshots)
    ]
    validation_examples: List[LatentResponseExample] = []
    for snapshot_index, snapshot in enumerate(validation_snapshots):
        for candidate_set in range(int(validation_candidate_sets)):
            validation_examples.append(
                _prepare_example(
                    snapshot,
                    representation=representation,
                    family_names=dataset.family_names,
                    role_means=role_means,
                    role_scales=role_scales,
                    node_types=node_types,
                    candidates_per_example=int(candidates_per_example),
                    seed=(
                        int(seed)
                        + 90_000_119
                        + 1_000_003 * snapshot_index
                        + 97_409 * candidate_set
                    ),
                    device=device,
                )
            )
    sample = train_examples[0]
    model = LocalEditResponseScorer(
        feature_dim=sample.features.shape[1],
        hidden_dim=sample.endpoint_hidden.shape[-1],
        role_dim=sample.endpoint_role_delta.shape[-1],
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate))
    rng = np.random.default_rng(int(seed))
    final_losses = {}
    for epoch in range(int(epochs)):
        model.train()
        rows = []
        for example_index in rng.permutation(len(train_examples)).tolist():
            example = train_examples[int(example_index)]
            features = _tensor(example.features, device).float()[None]
            hidden = _tensor(example.endpoint_hidden, device).float()[None]
            prototypes = _tensor(example.endpoint_prototype, device).float()[None]
            role_delta = _tensor(example.endpoint_role_delta, device).float()[None]
            mask = _tensor(example.endpoint_mask, device).bool()[None]
            utilities = _tensor(example.utilities, device).float()[None]
            optimizer.zero_grad(set_to_none=True)
            base_score, latent_score, _, latent_gain = model(
                features, hidden, prototypes, role_delta, mask
            )
            target_distribution = torch.softmax(
                utilities / float(target_temperature), dim=1
            )
            base_loss = -(
                target_distribution * torch.log_softmax(base_score, dim=1)
            ).sum(dim=1).mean()
            latent_loss = -(
                target_distribution * torch.log_softmax(latent_score, dim=1)
            ).sum(dim=1).mean()
            response_index = int(example.response_index)
            response_loss = nn.functional.mse_loss(
                latent_gain[0, response_index] * model.response_scale,
                torch.tensor(
                    example.response_target * model.response_scale,
                    device=device,
                ),
            )
            loss = base_loss + latent_loss + float(response_loss_weight) * response_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            rows.append(
                (
                    float(loss.detach().cpu()),
                    float(base_loss.detach().cpu()),
                    float(latent_loss.detach().cpu()),
                    float(response_loss.detach().cpu()),
                )
            )
        means = np.asarray(rows).mean(axis=0)
        final_losses = {
            "total": float(means[0]),
            "base": float(means[1]),
            "latent": float(means[2]),
            "response": float(means[3]),
        }
        print(
            "[LATENT-RESPONSE-EPOCH] "
            f"epoch={epoch} total={means[0]:.6f} base={means[1]:.6f} "
            f"latent={means[2]:.6f} response={means[3]:.6f}",
            flush=True,
        )

    model.eval()
    evaluation_rows = []
    with torch.no_grad():
        for example in validation_examples:
            features = _tensor(example.features, device).float()[None]
            hidden = _tensor(example.endpoint_hidden, device).float()[None]
            prototypes = _tensor(example.endpoint_prototype, device).float()[None]
            role_delta = _tensor(example.endpoint_role_delta, device).float()[None]
            mask = _tensor(example.endpoint_mask, device).bool()[None]
            base_score, latent_score, _, latent_gain = model(
                features, hidden, prototypes, role_delta, mask
            )
            utilities = np.asarray(example.utilities, dtype=np.float64)
            base_probability = torch.softmax(base_score, dim=1)[0].cpu().numpy()
            latent_probability = torch.softmax(latent_score, dim=1)[0].cpu().numpy()
            response_index = int(example.response_index)
            prediction = float(latent_gain[0, response_index].cpu())
            target = float(example.response_target)
            response_mse = float(np.square(prediction - target))
            zero_mse = float(np.square(target))
            gain = latent_gain[0].cpu().numpy()
            gain_std = float(np.std(gain))
            utility_std = float(np.std(utilities))
            correlation = 0.0
            if gain_std > 1.0e-8 and utility_std > 1.0e-8:
                correlation = float(np.corrcoef(gain, utilities)[0, 1])
            evaluation_rows.append(
                {
                    "level": float(example.level),
                    "base_selected": float(utilities[int(base_score.argmax(dim=1))]),
                    "latent_selected": float(utilities[int(latent_score.argmax(dim=1))]),
                    "oracle": float(utilities.max()),
                    "base_expected": float(np.dot(base_probability, utilities)),
                    "latent_expected": float(np.dot(latent_probability, utilities)),
                    "response_mse": response_mse,
                    "zero_response_mse": zero_mse,
                    "gain_utility_correlation": correlation,
                }
            )
    by_noise = {}
    for level in sorted({row["level"] for row in evaluation_rows}):
        selected = [row for row in evaluation_rows if row["level"] == level]
        by_noise[str(level)] = {
            key: float(np.mean([row[key] for row in selected]))
            for key in (
                "base_selected",
                "latent_selected",
                "oracle",
                "base_expected",
                "latent_expected",
                "response_mse",
                "zero_response_mse",
                "gain_utility_correlation",
            )
        }
        by_noise[str(level)]["selected_utility_gain"] = (
            by_noise[str(level)]["latent_selected"]
            - by_noise[str(level)]["base_selected"]
        )
        by_noise[str(level)]["expected_utility_gain"] = (
            by_noise[str(level)]["latent_expected"]
            - by_noise[str(level)]["base_expected"]
        )
        zero = by_noise[str(level)]["zero_response_mse"]
        by_noise[str(level)]["response_mse_reduction"] = (
            zero - by_noise[str(level)]["response_mse"]
        ) / max(zero, 1.0e-12)
    report = {
        "format": "dihug-local-edit-response-gate-v1",
        "train_examples": len(train_examples),
        "validation_examples": len(validation_examples),
        "candidates_per_example": int(candidates_per_example),
        "train_losses": final_losses,
        "by_noise": by_noise,
    }
    return model, report
