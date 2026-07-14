from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .checkpoint import StructureEnergyCheckpoint
from .features import FeatureSelection, GraphletFeatureExtractor
from .graph import TypedGraph
from .roles import permuted_witness, sample_role_plan, validate_role_realization
from .switch_accelerator import (
    BatchedClosedTriangleKernel,
    tensor_candidate_features,
)
from .switch_chain import SwitchProposal, graph_metrics, propose_switch


@dataclass(frozen=True)
class SwitchDiffusionDataset:
    candidate_features: np.ndarray
    candidate_improvements: np.ndarray
    targets: np.ndarray
    trajectory_ids: np.ndarray

    def __post_init__(self) -> None:
        features = np.asarray(self.candidate_features)
        targets = np.asarray(self.targets)
        improvements = np.asarray(self.candidate_improvements)
        trajectory_ids = np.asarray(self.trajectory_ids)
        if features.ndim != 3:
            raise ValueError("switch candidate features must have shape [N, C, D]")
        if targets.shape != (features.shape[0],):
            raise ValueError("switch targets have the wrong shape")
        if improvements.shape != features.shape[:2]:
            raise ValueError("switch candidate improvements have the wrong shape")
        if trajectory_ids.shape != targets.shape:
            raise ValueError("switch trajectory IDs have the wrong shape")


@dataclass(frozen=True)
class SwitchDiffusionTrainReport:
    training_objective: str
    utility_schedule_power: float
    epochs: int
    train_examples: int
    validation_examples: int
    train_loss: float
    validation_loss: float
    validation_inverse_cross_entropy: float
    validation_top1: float
    validation_top5: float
    validation_margin: float
    validation_oracle_top1: float
    validation_model_oracle_top1: float
    validation_utility_cross_entropy: float
    validation_selected_utility: float
    learned_condition_rank: int
    learned_condition_rmse: float

    def to_json_dict(self) -> dict:
        return {
            "training_objective": str(self.training_objective),
            "utility_schedule_power": float(self.utility_schedule_power),
            "epochs": int(self.epochs),
            "train_examples": int(self.train_examples),
            "validation_examples": int(self.validation_examples),
            "train_loss": float(self.train_loss),
            "validation_loss": float(self.validation_loss),
            "validation_inverse_cross_entropy": float(
                self.validation_inverse_cross_entropy
            ),
            "validation_top1": float(self.validation_top1),
            "validation_top5": float(self.validation_top5),
            "validation_margin": float(self.validation_margin),
            "validation_oracle_top1": float(self.validation_oracle_top1),
            "validation_model_oracle_top1": float(
                self.validation_model_oracle_top1
            ),
            "validation_utility_cross_entropy": float(
                self.validation_utility_cross_entropy
            ),
            "validation_selected_utility": float(
                self.validation_selected_utility
            ),
            "learned_condition_rank": int(self.learned_condition_rank),
            "learned_condition_rmse": float(self.learned_condition_rmse),
        }


@dataclass(frozen=True)
class SwitchSamplingResult:
    graph: TypedGraph
    trace: Sequence[Mapping[str, float]]
    proposed: int
    applied: int
    selection_mode: str
    oracle_agreement: float


class SwitchDenoiserMLP(nn.Module):
    """Listwise switch scorer with a replaceable graph-encoder boundary."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        layers: int = 3,
        dropout: float = 0.1,
        learned_condition_feature_count: int = 0,
        learned_condition_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if int(layers) < 1:
            raise ValueError("switch scorer needs at least one hidden layer")
        self.input_dim = int(input_dim)
        self.learned_condition_feature_count = int(
            learned_condition_feature_count
        )
        self.learned_condition_temperature = float(
            learned_condition_temperature
        )
        if self.learned_condition_feature_count < 0:
            raise ValueError("learned condition feature count must be non-negative")
        if self.learned_condition_temperature <= 0.0:
            raise ValueError("learned condition temperature must be positive")
        modules: List[nn.Module] = []
        width = int(input_dim)
        for _ in range(int(layers)):
            modules.extend(
                [
                    nn.Linear(width, int(hidden_dim)),
                    nn.LayerNorm(int(hidden_dim)),
                    nn.SiLU(),
                    nn.Dropout(float(dropout)),
                ]
            )
            width = int(hidden_dim)
        modules.append(nn.Linear(width, 1))
        self.network = nn.Sequential(*modules)
        if self.learned_condition_feature_count > 0:
            self.learned_target = nn.Parameter(
                torch.zeros(self.learned_condition_feature_count)
            )
            final = self.network[-1]
            assert isinstance(final, nn.Linear)
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)
        else:
            self.register_parameter("learned_target", None)

    def forward(self, candidate_features: torch.Tensor) -> torch.Tensor:
        if candidate_features.ndim != 3:
            raise ValueError("candidate features must have shape [B, C, D]")
        score = self.network(candidate_features).squeeze(-1)
        if self.learned_target is not None:
            width = self.learned_condition_feature_count
            delta = candidate_features[..., :width]
            current = candidate_features[..., width : 2 * width]
            utility = (
                2.0 * (delta * (self.learned_target - current)).sum(dim=-1)
                - (delta * delta).sum(dim=-1)
            )
            score = score + utility / self.learned_condition_temperature
        return score

    def set_learned_target(self, value: np.ndarray) -> None:
        if self.learned_target is None:
            raise ValueError("switch scorer has no learned condition parameter")
        tensor = torch.as_tensor(
            value,
            dtype=self.learned_target.dtype,
            device=self.learned_target.device,
        )
        if tensor.shape != self.learned_target.shape:
            raise ValueError("learned condition has the wrong shape")
        with torch.no_grad():
            self.learned_target.copy_(tensor)


def _proposal_key(proposal: SwitchProposal) -> tuple:
    return (
        str(proposal.family),
        tuple(sorted(tuple(sorted(edge)) for edge in proposal.old_edges)),
        tuple(sorted(tuple(sorted(edge)) for edge in proposal.new_edges)),
    )


def inverse_switch(proposal: SwitchProposal) -> SwitchProposal:
    return SwitchProposal(
        family=proposal.family,
        indices=proposal.indices,
        old_edges=proposal.new_edges,
        new_edges=proposal.old_edges,
    )


def apply_switch(graph: TypedGraph, proposal: SwitchProposal) -> None:
    graph.replace_edges(
        proposal.family,
        proposal.indices,
        proposal.old_edges,
        proposal.new_edges,
    )


def sample_valid_switch(
    graph: TypedGraph,
    rng: np.random.Generator,
    family: Optional[str] = None,
    max_attempts: int = 256,
) -> Optional[SwitchProposal]:
    families = [
        name for name, edges in graph.family_edges.items() if len(edges) >= 2
    ]
    if family is not None:
        families = [str(family)] if str(family) in families else []
    if not families:
        return None
    weights = np.asarray(
        [len(graph.family_edges[name]) for name in families], dtype=np.float64
    )
    weights /= weights.sum()
    for _ in range(int(max_attempts)):
        selected = (
            str(family)
            if family is not None
            else str(rng.choice(families, p=weights))
        )
        proposal = propose_switch(graph, selected, rng)
        if proposal is not None:
            return proposal
    return None


def sample_switch_candidates(
    graph: TypedGraph,
    rng: np.random.Generator,
    count: int,
    positive: Optional[SwitchProposal] = None,
    max_attempt_factor: int = 64,
) -> tuple[List[SwitchProposal], int]:
    if int(count) < 2:
        raise ValueError("candidate set must contain at least two switches")
    candidates: List[SwitchProposal] = []
    keys = set()
    if positive is not None:
        candidates.append(positive)
        keys.add(_proposal_key(positive))
    attempts = 0
    limit = int(count) * int(max_attempt_factor)
    while len(candidates) < int(count) and attempts < limit:
        attempts += 1
        proposal = sample_valid_switch(graph, rng)
        if proposal is None:
            continue
        key = _proposal_key(proposal)
        if key in keys:
            continue
        keys.add(key)
        candidates.append(proposal)
    if len(candidates) != int(count):
        raise RuntimeError(
            f"sampled only {len(candidates)}/{count} unique legal switches"
        )
    target = 0
    if positive is not None:
        order = rng.permutation(len(candidates))
        candidates = [candidates[index] for index in order.tolist()]
        target = int(np.flatnonzero(order == 0)[0])
    return candidates, target


def proposal_delta(
    graph: TypedGraph,
    extractor: GraphletFeatureExtractor,
    proposal: SwitchProposal,
) -> Counter:
    """Return an exact feature delta while restoring the input graph."""

    delta = extractor.switch_delta(
        graph,
        proposal.family,
        proposal.indices,
        proposal.old_edges,
        proposal.new_edges,
    )
    graph.replace_edges(
        proposal.family,
        proposal.indices,
        proposal.new_edges,
        proposal.old_edges,
    )
    return delta


def switch_feature_vector(
    graph: TypedGraph,
    extractor: GraphletFeatureExtractor,
    selection: FeatureSelection,
    current_features: Mapping[str, int | float],
    target_vector: Optional[np.ndarray],
    proposal: SwitchProposal,
    family_to_id: Mapping[str, int],
    time_fraction: float,
    conditioning_mode: str = "target_residual",
) -> np.ndarray:
    delta = selection.delta_vector(proposal_delta(graph, extractor, proposal))
    current = selection.vector(current_features)
    conditioning_mode = str(conditioning_mode).strip().lower()
    if conditioning_mode == "target_residual":
        if target_vector is None:
            raise ValueError("target_residual conditioning requires a target vector")
        context = np.asarray(target_vector, dtype=np.float64) - current
        interaction = delta * np.sign(context)
    elif conditioning_mode == "state_only":
        # The reverse kernel observes the current anonymous graph state but no
        # statistic of the clean training graph. The inverse-switch labels are
        # the only denoising supervision.
        context = current
        interaction = delta * current
    else:
        raise ValueError(f"unsupported switch conditioning mode {conditioning_mode!r}")
    family = np.zeros(len(family_to_id), dtype=np.float64)
    family[int(family_to_id[proposal.family])] = 1.0
    scalar = np.asarray(
        [
            float(time_fraction),
            float(np.dot(delta, context) / max(len(delta), 1)),
            float(np.linalg.norm(delta)),
        ],
        dtype=np.float64,
    )
    return np.concatenate((delta, context, interaction, family, scalar)).astype(
        np.float32
    )


def candidate_feature_matrix(
    graph: TypedGraph,
    extractor: GraphletFeatureExtractor,
    selection: FeatureSelection,
    current_features: Mapping[str, int | float],
    target_vector: Optional[np.ndarray],
    candidates: Sequence[SwitchProposal],
    family_to_id: Mapping[str, int],
    time_fraction: float,
    conditioning_mode: str = "target_residual",
) -> np.ndarray:
    return np.stack(
        [
            switch_feature_vector(
                graph,
                extractor,
                selection,
                current_features,
                target_vector,
                proposal,
                family_to_id,
                time_fraction,
                conditioning_mode,
            )
            for proposal in candidates
        ]
    )


def candidate_target_improvements(
    candidate_features: np.ndarray,
    feature_count: int,
    residual: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Exact decrease in squared normalized target error for each switch."""

    matrix = np.asarray(candidate_features, dtype=np.float64)
    width = int(feature_count)
    if matrix.ndim != 2 or matrix.shape[1] < 2 * width:
        raise ValueError("candidate feature matrix is incompatible with selection")
    delta = matrix[:, :width]
    if residual is None:
        residual_matrix = matrix[:, width : 2 * width]
    else:
        residual_vector = np.asarray(residual, dtype=np.float64)
        if residual_vector.shape != (width,):
            raise ValueError("target residual has the wrong shape")
        residual_matrix = np.broadcast_to(residual_vector, delta.shape)
    return 2.0 * np.einsum("ij,ij->i", delta, residual_matrix) - np.einsum(
        "ij,ij->i", delta, delta
    )


def switch_diffusion_time_fraction(
    micro_step: int,
    total_micro_steps: int,
    diffusion_steps: int = 0,
    switches_per_diffusion_step: int = 0,
) -> float:
    """Map a micro switch to its discrete diffusion-time layer."""

    micro_step = int(micro_step)
    total_micro_steps = int(total_micro_steps)
    diffusion_steps = int(diffusion_steps)
    switches_per_diffusion_step = int(switches_per_diffusion_step)
    if micro_step < 1 or micro_step > total_micro_steps:
        raise ValueError("micro step is outside the switch trajectory")
    if diffusion_steps == 0 and switches_per_diffusion_step == 0:
        return float(micro_step) / float(total_micro_steps)
    if diffusion_steps <= 0 or switches_per_diffusion_step <= 0:
        raise ValueError(
            "diffusion_steps and switches_per_diffusion_step must be set together"
        )
    expected = diffusion_steps * switches_per_diffusion_step
    if expected != total_micro_steps:
        raise ValueError(
            "macro diffusion schedule does not match total switch count: "
            f"{diffusion_steps}*{switches_per_diffusion_step}!={total_micro_steps}"
        )
    diffusion_step = (micro_step - 1) // switches_per_diffusion_step + 1
    return float(diffusion_step) / float(diffusion_steps)


def build_switch_diffusion_dataset(
    source: TypedGraph,
    selection: FeatureSelection,
    trajectories: int,
    forward_steps: int,
    record_every: int,
    candidates_per_example: int,
    seed: int,
    conditioning_mode: str = "target_residual",
    diffusion_steps: int = 0,
    switches_per_diffusion_step: int = 0,
) -> tuple[SwitchDiffusionDataset, np.ndarray]:
    """Generate exact forward/reverse supervision from anonymous trajectories."""

    if int(forward_steps) <= 0 or int(record_every) <= 0:
        raise ValueError("forward_steps and record_every must be positive")
    target_values = GraphletFeatureExtractor(
        source, tracked_keys=selection.keys
    ).extract(source)
    target_vector = selection.vector(target_values)
    family_to_id = {name: index for index, name in enumerate(sorted(source.family_specs))}
    examples: List[np.ndarray] = []
    improvements: List[np.ndarray] = []
    targets: List[int] = []
    trajectory_ids: List[int] = []
    for trajectory_id in range(int(trajectories)):
        trajectory_seed = int(seed) + 1_000_003 * trajectory_id
        rng = np.random.default_rng(trajectory_seed)
        plan = sample_role_plan(source, trajectory_seed + 17)
        graph = permuted_witness(source, plan)
        extractor = GraphletFeatureExtractor(graph, tracked_keys=selection.keys)
        current_features = extractor.extract(graph)
        for step in range(1, int(forward_steps) + 1):
            forward = sample_valid_switch(graph, rng)
            if forward is None:
                continue
            delta = extractor.switch_delta(
                graph,
                forward.family,
                forward.indices,
                forward.old_edges,
                forward.new_edges,
            )
            extractor.apply_delta(current_features, delta)
            if step % int(record_every) != 0:
                continue
            positive = inverse_switch(forward)
            candidates, target = sample_switch_candidates(
                graph,
                rng,
                candidates_per_example,
                positive=positive,
            )
            matrix = candidate_feature_matrix(
                graph,
                extractor,
                selection,
                current_features,
                target_vector,
                candidates,
                family_to_id,
                switch_diffusion_time_fraction(
                    step,
                    forward_steps,
                    diffusion_steps,
                    switches_per_diffusion_step,
                ),
                conditioning_mode,
            )
            examples.append(matrix)
            improvements.append(
                candidate_target_improvements(
                    matrix,
                    len(selection.keys),
                    residual=(
                        target_vector - selection.vector(current_features)
                        if str(conditioning_mode).strip().lower() == "state_only"
                        else None
                    ),
                )
            )
            targets.append(target)
            trajectory_ids.append(trajectory_id)
    if not examples:
        raise RuntimeError("forward switch process produced no training examples")
    return (
        SwitchDiffusionDataset(
            candidate_features=np.stack(examples),
            candidate_improvements=np.stack(improvements),
            targets=np.asarray(targets, dtype=np.int64),
            trajectory_ids=np.asarray(trajectory_ids, dtype=np.int64),
        ),
        target_vector,
    )


def _evaluate_switch_model(
    model: SwitchDenoiserMLP,
    loader: DataLoader,
    device: torch.device,
    objective: str,
    inverse_loss_weight: float,
    utility_loss_weight: float,
    utility_temperature: float,
    utility_schedule_power: float,
) -> tuple[float, float, float, float, float, float, float, float]:
    model.eval()
    losses = []
    top1 = []
    top5 = []
    margins = []
    utility_losses = []
    model_oracle = []
    selected_utility = []
    objective_values = []
    criterion = nn.CrossEntropyLoss(reduction="none")
    with torch.no_grad():
        for features, target, improvements in loader:
            features = features.to(device)
            target = target.to(device)
            improvements = improvements.to(device)
            logits = model(features)
            objective_losses = _switch_training_losses(
                logits,
                target,
                improvements,
                features,
                objective,
                inverse_loss_weight,
                utility_loss_weight,
                utility_temperature,
                utility_schedule_power,
            )
            objective_values.extend(objective_losses.cpu().tolist())
            losses.extend(criterion(logits, target).cpu().tolist())
            top1.extend((logits.argmax(dim=1) == target).float().cpu().tolist())
            width = min(5, logits.shape[1])
            top = logits.topk(width, dim=1).indices
            top5.extend((top == target[:, None]).any(dim=1).float().cpu().tolist())
            positive = logits.gather(1, target[:, None]).squeeze(1)
            masked = logits.clone()
            masked.scatter_(1, target[:, None], float("-inf"))
            margins.extend((positive - masked.max(dim=1).values).cpu().tolist())
            teacher = torch.softmax(
                improvements / float(utility_temperature), dim=1
            )
            utility_losses.extend(
                (-(teacher * torch.log_softmax(logits, dim=1)).sum(dim=1))
                .cpu()
                .tolist()
            )
            prediction = logits.argmax(dim=1)
            oracle = improvements.argmax(dim=1)
            model_oracle.extend((prediction == oracle).float().cpu().tolist())
            selected_utility.extend(
                improvements.gather(1, prediction[:, None]).squeeze(1).cpu().tolist()
            )
    inverse_mean = float(np.mean(losses))
    utility_mean = float(np.mean(utility_losses))
    objective_mean = float(np.mean(objective_values))
    return (
        objective_mean,
        inverse_mean,
        float(np.mean(top1)),
        float(np.mean(top5)),
        float(np.mean(margins)),
        utility_mean,
        float(np.mean(model_oracle)),
        float(np.mean(selected_utility)),
    )


def _switch_training_losses(
    logits: torch.Tensor,
    targets: torch.Tensor,
    improvements: torch.Tensor,
    candidate_features: torch.Tensor,
    objective: str,
    inverse_loss_weight: float,
    utility_loss_weight: float,
    utility_temperature: float,
    utility_schedule_power: float,
) -> torch.Tensor:
    """Train an inverse kernel with exact or utility-equivalent positives.

    The utility teacher is constructed only while building training data. It is
    never part of the scorer input or the sampling checkpoint.
    """

    objective = str(objective).strip().lower()
    if objective not in {
        "inverse_ce",
        "utility_kl",
        "hybrid",
        "scheduled_hybrid",
    }:
        raise ValueError(f"unsupported switch training objective {objective!r}")
    inverse = nn.functional.cross_entropy(logits, targets, reduction="none")
    teacher = torch.softmax(improvements / float(utility_temperature), dim=1)
    utility = -(teacher * nn.functional.log_softmax(logits, dim=1)).sum(dim=1)
    if objective == "inverse_ce":
        return inverse
    if objective == "utility_kl":
        return utility
    if objective == "hybrid":
        return (
            float(inverse_loss_weight) * inverse
            + float(utility_loss_weight) * utility
        )
    # Forward time is the third scalar from the end of every candidate feature.
    # At high noise, many switches are structurally equivalent reverse moves;
    # near the clean endpoint, the observed inverse action becomes identifiable.
    time_fraction = candidate_features[:, 0, -3].clamp(0.0, 1.0)
    inverse_coefficient = float(inverse_loss_weight) * (1.0 - time_fraction)
    utility_coefficient = float(utility_loss_weight) * time_fraction.pow(
        float(utility_schedule_power)
    )
    normalizer = (inverse_coefficient + utility_coefficient).clamp_min(1.0e-8)
    return (
        inverse_coefficient * inverse + utility_coefficient * utility
    ) / normalizer


def _switch_training_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    improvements: torch.Tensor,
    candidate_features: torch.Tensor,
    objective: str,
    inverse_loss_weight: float,
    utility_loss_weight: float,
    utility_temperature: float,
    utility_schedule_power: float,
) -> torch.Tensor:
    return _switch_training_losses(
        logits,
        targets,
        improvements,
        candidate_features,
        objective,
        inverse_loss_weight,
        utility_loss_weight,
        utility_temperature,
        utility_schedule_power,
    ).mean()


def fit_learned_switch_condition(
    candidate_features: np.ndarray,
    candidate_improvements: np.ndarray,
    feature_count: int,
    ridge: float = 1.0e-6,
) -> tuple[np.ndarray, int, float]:
    """Fit the anonymous graphlet condition by convex ridge regression.

    Squared-error utility for normalized state ``z`` and switch delta ``d`` is
    exactly ``2 d·(tau-z) - ||d||²``. Thus ``tau`` is learned from training
    actions and stored in model parameters; it is not a sampler-time target.
    """

    matrix = np.asarray(candidate_features, dtype=np.float64)
    utility = np.asarray(candidate_improvements, dtype=np.float64)
    width = int(feature_count)
    if matrix.ndim != 3 or utility.shape != matrix.shape[:2]:
        raise ValueError("learned switch condition data has incompatible shapes")
    if width <= 0 or matrix.shape[2] < 2 * width:
        raise ValueError("invalid learned switch condition feature count")
    delta = matrix[:, :, :width].reshape(-1, width)
    current = matrix[:, :, width : 2 * width].reshape(-1, width)
    response = (
        utility.reshape(-1)
        + 2.0 * np.einsum("ij,ij->i", delta, current)
        + np.einsum("ij,ij->i", delta, delta)
    )
    design = 2.0 * delta
    estimate = np.linalg.solve(
        design.T @ design + float(ridge) * np.eye(width),
        design.T @ response,
    )
    prediction = (
        design @ estimate
        - 2.0 * np.einsum("ij,ij->i", delta, current)
        - np.einsum("ij,ij->i", delta, delta)
    )
    rmse = float(np.sqrt(np.mean((prediction - utility.reshape(-1)) ** 2)))
    return estimate, int(np.linalg.matrix_rank(design)), rmse


def train_switch_denoiser(
    dataset: SwitchDiffusionDataset,
    hidden_dim: int = 256,
    layers: int = 3,
    dropout: float = 0.1,
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 1.0e-3,
    weight_decay: float = 1.0e-4,
    validation_trajectory: Optional[int] = None,
    training_objective: str = "inverse_ce",
    inverse_loss_weight: float = 0.2,
    utility_loss_weight: float = 1.0,
    utility_temperature: float = 1.0,
    utility_schedule_power: float = 1.0,
    learned_condition_feature_count: int = 0,
    learned_condition_ridge: float = 1.0e-6,
    device: str | torch.device = "cpu",
    seed: int = 0,
) -> tuple[SwitchDenoiserMLP, SwitchDiffusionTrainReport]:
    torch.manual_seed(int(seed))
    training_objective = str(training_objective).strip().lower()
    if training_objective not in {
        "inverse_ce",
        "utility_kl",
        "hybrid",
        "scheduled_hybrid",
    }:
        raise ValueError(
            f"unsupported switch training objective {training_objective!r}"
        )
    if float(utility_temperature) <= 0.0:
        raise ValueError("utility temperature must be positive")
    if float(utility_schedule_power) <= 0.0:
        raise ValueError("utility schedule power must be positive")
    if float(inverse_loss_weight) < 0.0 or float(utility_loss_weight) < 0.0:
        raise ValueError("switch loss weights must be non-negative")
    if training_objective == "hybrid" and (
        float(inverse_loss_weight) + float(utility_loss_weight) <= 0.0
    ):
        raise ValueError("hybrid switch loss needs a positive weight")
    device = torch.device(device)
    trajectory_ids = np.asarray(dataset.trajectory_ids)
    if validation_trajectory is None:
        validation_trajectory = int(trajectory_ids.max())
    validation_mask = trajectory_ids == int(validation_trajectory)
    if not validation_mask.any() or validation_mask.all():
        raise ValueError("validation split must leave train and validation examples")
    features = torch.from_numpy(dataset.candidate_features).float()
    targets = torch.from_numpy(dataset.targets).long()
    improvements = torch.from_numpy(dataset.candidate_improvements).float()
    train_indices = torch.from_numpy(np.flatnonzero(~validation_mask)).long()
    validation_indices = torch.from_numpy(np.flatnonzero(validation_mask)).long()
    train_loader = DataLoader(
        TensorDataset(
            features[train_indices],
            targets[train_indices],
            improvements[train_indices],
        ),
        batch_size=int(batch_size),
        shuffle=True,
        generator=torch.Generator().manual_seed(int(seed)),
    )
    validation_loader = DataLoader(
        TensorDataset(
            features[validation_indices],
            targets[validation_indices],
            improvements[validation_indices],
        ),
        batch_size=int(batch_size),
        shuffle=False,
    )
    model = SwitchDenoiserMLP(
        input_dim=int(features.shape[-1]),
        hidden_dim=hidden_dim,
        layers=layers,
        dropout=dropout,
        learned_condition_feature_count=learned_condition_feature_count,
        learned_condition_temperature=utility_temperature,
    ).to(device)
    learned_condition_rank = 0
    learned_condition_rmse = float("nan")
    if int(learned_condition_feature_count) > 0:
        if training_objective != "utility_kl":
            raise ValueError(
                "learned switch condition requires utility_kl training"
            )
        estimate, learned_condition_rank, learned_condition_rmse = (
            fit_learned_switch_condition(
                np.asarray(dataset.candidate_features)[~validation_mask],
                np.asarray(dataset.candidate_improvements)[~validation_mask],
                int(learned_condition_feature_count),
                ridge=learned_condition_ridge,
            )
        )
        model.set_learned_target(estimate)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay)
    )
    final_train_loss = float("nan")
    for _ in range(int(epochs)):
        model.train()
        losses = []
        for batch_features, batch_target, batch_improvements in train_loader:
            batch_features = batch_features.to(device)
            batch_target = batch_target.to(device)
            batch_improvements = batch_improvements.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = _switch_training_loss(
                model(batch_features),
                batch_target,
                batch_improvements,
                batch_features,
                training_objective,
                inverse_loss_weight,
                utility_loss_weight,
                utility_temperature,
                utility_schedule_power,
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        final_train_loss = float(np.mean(losses))
    (
        validation_loss,
        validation_inverse_loss,
        top1,
        top5,
        margin,
        validation_utility_loss,
        validation_model_oracle,
        validation_selected_utility,
    ) = _evaluate_switch_model(
        model,
        validation_loader,
        device,
        training_objective,
        inverse_loss_weight,
        utility_loss_weight,
        utility_temperature,
        utility_schedule_power,
    )
    validation_targets = np.asarray(dataset.targets)[validation_mask]
    validation_oracle = np.asarray(dataset.candidate_improvements)[
        validation_mask
    ].argmax(axis=1)
    report = SwitchDiffusionTrainReport(
        training_objective=training_objective,
        utility_schedule_power=float(utility_schedule_power),
        epochs=int(epochs),
        train_examples=int(train_indices.numel()),
        validation_examples=int(validation_indices.numel()),
        train_loss=final_train_loss,
        validation_loss=validation_loss,
        validation_inverse_cross_entropy=validation_inverse_loss,
        validation_top1=top1,
        validation_top5=top5,
        validation_margin=margin,
        validation_oracle_top1=float(
            np.mean(validation_oracle == validation_targets)
        ),
        validation_model_oracle_top1=validation_model_oracle,
        validation_utility_cross_entropy=validation_utility_loss,
        validation_selected_utility=validation_selected_utility,
        learned_condition_rank=learned_condition_rank,
        learned_condition_rmse=learned_condition_rmse,
    )
    return model, report


def randomize_role_realization(
    graph: TypedGraph,
    steps: int,
    seed: int,
) -> TypedGraph:
    randomized = graph.clone()
    rng = np.random.default_rng(int(seed))
    for _ in range(int(steps)):
        proposal = sample_valid_switch(randomized, rng)
        if proposal is not None:
            apply_switch(randomized, proposal)
    return randomized


def sample_switch_diffusion(
    initial_graph: TypedGraph,
    model: SwitchDenoiserMLP,
    selection: FeatureSelection,
    target_vector: Optional[np.ndarray],
    steps: int,
    candidates_per_step: int,
    seed: int,
    device: str | torch.device = "cpu",
    temperature: float = 0.1,
    metrics_every: int = 100,
    selection_mode: str = "model",
    conditioning_mode: str = "target_residual",
    switch_kernel: str = "auto",
    learned_condition_noop: bool = False,
    learned_condition_noop_patience: int = 0,
    diffusion_steps: int = 0,
    switches_per_diffusion_step: int = 0,
    verbose: bool = False,
) -> SwitchSamplingResult:
    selection_mode = str(selection_mode).strip().lower()
    if selection_mode not in {"model", "random", "oracle"}:
        raise ValueError("selection_mode must be model, random, or oracle")
    if selection_mode == "oracle" and target_vector is None:
        raise ValueError("oracle sampling requires a clean-graph target vector")
    graph = initial_graph.clone()
    expected_degrees = graph.all_family_degrees()
    family_to_id = {name: index for index, name in enumerate(sorted(graph.family_specs))}
    extractor = GraphletFeatureExtractor(graph, tracked_keys=selection.keys)
    current_features = extractor.extract(graph)
    rng = np.random.default_rng(int(seed))
    device = torch.device(device)
    torch_generator = torch.Generator(device=device).manual_seed(int(seed))
    model = model.to(device).eval()
    if learned_condition_noop and model.learned_target is None:
        raise ValueError(
            "learned-condition no-op guard requires a learned condition model"
        )
    if int(learned_condition_noop_patience) < 0:
        raise ValueError("learned-condition no-op patience must be non-negative")
    switch_kernel = str(switch_kernel).strip().lower()
    if switch_kernel not in {"auto", "python", "tensor"}:
        raise ValueError("switch_kernel must be auto, python, or tensor")
    use_tensor_kernel = (
        switch_kernel == "tensor"
        or (
            switch_kernel == "auto"
            and device.type == "cuda"
            and BatchedClosedTriangleKernel.supports(selection)
        )
    )
    tensor_kernel = None
    current_tensor = None
    target_tensor = None
    if use_tensor_kernel:
        tensor_kernel = BatchedClosedTriangleKernel(graph, selection, device)
        current_tensor = torch.as_tensor(
            selection.vector(current_features), dtype=torch.float64, device=device
        )
        target_tensor = (
            None
            if target_vector is None
            else torch.as_tensor(target_vector, dtype=torch.float64, device=device)
        )
        if verbose:
            print(
                "[SWITCH-TENSOR-KERNEL] "
                f"device={device} features={len(selection.keys)} "
                f"nodes={graph.num_nodes}",
                flush=True,
            )
    trace: List[Mapping[str, float]] = []
    applied = 0
    oracle_matches = 0
    noop_streak = 0
    evaluated_steps = 0
    for reverse_step in range(int(steps), 0, -1):
        evaluated_steps += 1
        candidates, _ = sample_switch_candidates(
            graph, rng, candidates_per_step, positive=None
        )
        time_fraction = switch_diffusion_time_fraction(
            reverse_step,
            int(steps),
            diffusion_steps,
            switches_per_diffusion_step,
        )
        diffusion_step = (
            int(np.ceil(time_fraction * int(diffusion_steps)))
            if int(diffusion_steps) > 0
            else reverse_step
        )
        if tensor_kernel is not None:
            assert current_tensor is not None
            delta_tensor = tensor_kernel.delta_matrix(candidates)
            family_ids = torch.as_tensor(
                [family_to_id[item.family] for item in candidates],
                dtype=torch.long,
                device=device,
            )
            feature_tensor = tensor_candidate_features(
                delta_tensor,
                current_tensor,
                family_ids,
                len(family_to_id),
                time_fraction,
                conditioning_mode,
                target_tensor,
            ).to(torch.float32)
            if target_tensor is not None:
                residual_tensor = target_tensor - current_tensor
                improvements_tensor = (
                    2.0 * (delta_tensor * residual_tensor[None]).sum(dim=1)
                    - (delta_tensor * delta_tensor).sum(dim=1)
                )
                oracle = int(improvements_tensor.argmax().item())
            else:
                oracle = -1
        else:
            matrix = candidate_feature_matrix(
                graph,
                extractor,
                selection,
                current_features,
                target_vector,
                candidates,
                family_to_id,
                time_fraction,
                conditioning_mode,
            )
            if target_vector is not None:
                improvements = candidate_target_improvements(
                    matrix,
                    len(selection.keys),
                    residual=(
                        np.asarray(target_vector, dtype=np.float64)
                        - selection.vector(current_features)
                        if str(conditioning_mode).strip().lower() == "state_only"
                        else None
                    ),
                )
                oracle = int(np.argmax(improvements))
            else:
                oracle = -1
        if selection_mode == "oracle":
            chosen = oracle
        elif selection_mode == "random":
            chosen = int(rng.integers(len(candidates)))
        else:
            with torch.no_grad():
                if tensor_kernel is not None:
                    logits = model(feature_tensor[None]).squeeze(0)
                else:
                    logits = model(torch.from_numpy(matrix)[None].to(device)).squeeze(0)
            if float(temperature) <= 0.0:
                chosen = int(logits.argmax().item())
            else:
                probabilities = torch.softmax(logits / float(temperature), dim=0)
                chosen = int(
                    torch.multinomial(
                        probabilities,
                        num_samples=1,
                        generator=torch_generator,
                    ).item()
                )
        if learned_condition_noop:
            assert model.learned_target is not None
            learned_target = model.learned_target.detach().to(device=device)
            if tensor_kernel is not None:
                learned_utility = (
                    2.0
                    * (
                        delta_tensor
                        * (learned_target.to(delta_tensor.dtype) - current_tensor)[None]
                    ).sum(dim=1)
                    - (delta_tensor * delta_tensor).sum(dim=1)
                )
                chosen_utility = float(learned_utility[chosen].item())
            else:
                learned_target_numpy = learned_target.cpu().numpy()
                delta_matrix = np.asarray(matrix[:, : len(selection.keys)])
                current_vector = selection.vector(current_features)
                learned_utility = (
                    2.0
                    * np.einsum(
                        "ij,ij->i",
                        delta_matrix,
                        learned_target_numpy[None] - current_vector[None],
                    )
                    - np.einsum("ij,ij->i", delta_matrix, delta_matrix)
                )
                chosen_utility = float(learned_utility[chosen])
            if chosen_utility <= 0.0:
                noop_streak += 1
                completed = int(steps) - reverse_step + 1
                if completed == 1 or completed == int(steps) or (
                    int(metrics_every) > 0
                    and completed % int(metrics_every) == 0
                ):
                    metrics = graph_metrics(graph)
                    trace.append(
                        {
                            "step": float(completed),
                            "diffusion_step": float(diffusion_step),
                            "target_feature_mae": float("nan"),
                            "oracle_agreement": float("nan"),
                            "learned_condition_noop": 1.0,
                            "applied_switches": float(applied),
                            **metrics,
                        }
                    )
                    if verbose:
                        print(
                            "[SWITCH-DIFFUSION-TRACE] "
                            f"mode={selection_mode} step={completed}/{steps} "
                            f"tri={metrics['triangles']:.0f} "
                            f"clustering={metrics['clustering']:.6f} "
                            "learned_condition_noop=1",
                            flush=True,
                        )
                if (
                    int(learned_condition_noop_patience) > 0
                    and noop_streak >= int(learned_condition_noop_patience)
                ):
                    if verbose:
                        print(
                            "[SWITCH-DIFFUSION-STOP] "
                            f"step={completed} noop_streak={noop_streak}",
                            flush=True,
                        )
                    break
                continue
        noop_streak = 0
        oracle_matches += int(oracle >= 0 and chosen == oracle)
        proposal = candidates[chosen]
        if tensor_kernel is not None:
            apply_switch(graph, proposal)
            tensor_kernel.commit(proposal)
            current_tensor = current_tensor + delta_tensor[chosen]
        else:
            delta = extractor.switch_delta(
                graph,
                proposal.family,
                proposal.indices,
                proposal.old_edges,
                proposal.new_edges,
            )
            extractor.apply_delta(current_features, delta)
        applied += 1
        completed = int(steps) - reverse_step + 1
        if completed == 1 or completed == int(steps) or (
            int(metrics_every) > 0 and completed % int(metrics_every) == 0
        ):
            metrics = graph_metrics(graph)
            current_vector = (
                current_tensor.detach().cpu().numpy()
                if current_tensor is not None
                else selection.vector(current_features)
            )
            target_mae = (
                float(
                    np.mean(
                        np.abs(
                            np.asarray(target_vector, dtype=np.float64)
                            - current_vector
                        )
                    )
                )
                if target_vector is not None
                else float("nan")
            )
            trace_row = {
                "step": float(completed),
                "diffusion_step": float(diffusion_step),
                "target_feature_mae": target_mae,
                "oracle_agreement": (
                    float(oracle_matches) / float(completed)
                    if target_vector is not None
                    else float("nan")
                ),
                "applied_switches": float(applied),
                **metrics,
            }
            trace.append(trace_row)
            if verbose:
                print(
                    "[SWITCH-DIFFUSION-TRACE] "
                    f"mode={selection_mode} step={completed}/{steps} "
                    f"tri={metrics['triangles']:.0f} "
                    f"clustering={metrics['clustering']:.6f} "
                    f"feature_mae={target_mae:.6f} "
                    f"oracle_agreement={trace_row['oracle_agreement']:.4f}",
                    flush=True,
                )
    for family, expected in expected_degrees.items():
        if not np.array_equal(graph.family_degrees(family), expected):
            raise AssertionError(f"switch diffusion changed degree plan for {family}")
    graph.validate()
    return SwitchSamplingResult(
        graph=graph,
        trace=tuple(trace),
        proposed=int(evaluated_steps) * int(candidates_per_step),
        applied=applied,
        selection_mode=selection_mode,
        oracle_agreement=(
            float(oracle_matches) / max(float(applied), 1.0)
            if target_vector is not None
            else float("nan")
        ),
    )


def save_switch_checkpoint(
    path: str | Path,
    model: SwitchDenoiserMLP,
    structure_checkpoint: StructureEnergyCheckpoint,
    target_vector: Optional[np.ndarray],
    hidden_dim: int,
    layers: int,
    dropout: float,
    report: SwitchDiffusionTrainReport,
    conditioning_mode: str = "target_residual",
) -> None:
    conditioning_mode = str(conditioning_mode).strip().lower()
    if conditioning_mode not in {"target_residual", "state_only"}:
        raise ValueError(f"unsupported switch conditioning mode {conditioning_mode!r}")
    if conditioning_mode == "target_residual" and target_vector is None:
        raise ValueError("target_residual checkpoint requires a target vector")
    payload = {
        "format": "dihug-switch-diffusion-v2",
        "structure_checkpoint": structure_checkpoint.to_json_dict(),
        "conditioning_mode": conditioning_mode,
        "inference_contract": {
            "clean_graphlet_target": conditioning_mode == "target_residual",
            "training_adjacency": False,
            "anonymous_relation_degree_rows": True,
            "node_type_counts": True,
            "family_schema": True,
            "learned_graphlet_condition": (
                model.learned_condition_feature_count > 0
            ),
        },
        "input_dim": int(model.input_dim),
        "hidden_dim": int(hidden_dim),
        "layers": int(layers),
        "dropout": float(dropout),
        "learned_condition_feature_count": int(
            model.learned_condition_feature_count
        ),
        "learned_condition_temperature": float(
            model.learned_condition_temperature
        ),
        "state_dict": model.cpu().state_dict(),
        "train_report": report.to_json_dict(),
    }
    if conditioning_mode == "target_residual":
        payload["target_vector"] = np.asarray(
            target_vector, dtype=np.float64
        ).tolist()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_switch_checkpoint(
    path: str | Path, device: str | torch.device = "cpu"
) -> tuple[
    SwitchDenoiserMLP,
    StructureEnergyCheckpoint,
    Optional[np.ndarray],
    Mapping,
    str,
]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    checkpoint_format = payload.get("format")
    if checkpoint_format not in {
        "dihug-switch-diffusion-v1",
        "dihug-switch-diffusion-v2",
    }:
        raise ValueError("unsupported switch-diffusion checkpoint")
    conditioning_mode = str(
        payload.get("conditioning_mode", "target_residual")
    ).strip().lower()
    model = SwitchDenoiserMLP(
        input_dim=int(payload["input_dim"]),
        hidden_dim=int(payload["hidden_dim"]),
        layers=int(payload["layers"]),
        dropout=float(payload["dropout"]),
        learned_condition_feature_count=int(
            payload.get("learned_condition_feature_count", 0)
        ),
        learned_condition_temperature=float(
            payload.get("learned_condition_temperature", 1.0)
        ),
    )
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    structure = StructureEnergyCheckpoint.from_json_dict(
        payload["structure_checkpoint"]
    )
    target_payload = payload.get("target_vector")
    target = (
        None
        if target_payload is None
        else np.asarray(target_payload, dtype=np.float64)
    )
    if conditioning_mode == "target_residual" and target is None:
        raise ValueError("target-conditioned checkpoint has no target vector")
    return (
        model,
        structure,
        target,
        payload.get("train_report", {}),
        conditioning_mode,
    )


def save_sampling_result(path: str | Path, result: SwitchSamplingResult) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "graph": result.graph.to_json_dict(),
        "trace": list(result.trace),
        "proposed": int(result.proposed),
        "applied": int(result.applied),
        "selection_mode": str(result.selection_mode),
        "oracle_agreement": float(result.oracle_agreement),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
