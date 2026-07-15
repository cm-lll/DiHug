from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .graph import TypedGraph
from .mixed_edit_diffusion import (
    EDIT_ACTIONS,
    SCORER_ACTIONS,
    EditProposal,
    MarginalGraphPrior,
    MixedEditSamplingResult,
    MixedKernelSchedule,
    apply_edit,
    inverse_edit,
    mixed_edit_candidate_matrix,
    sample_mixed_forward_transition,
    sample_valid_edit,
)
from .switch_chain import graph_metrics
from .switch_diffusion import SwitchDiffusionDataset


def mixed_edit_state_dim(type_count: int, family_count: int) -> int:
    return 3 + 4 * int(type_count) + 5 * int(family_count)


def _proposal_key(proposal: EditProposal) -> tuple:
    return (
        proposal.action,
        "" if proposal.action == "noop" else proposal.family,
        tuple(sorted(proposal.old_edges)),
        tuple(sorted(proposal.new_edges)),
    )


def sample_factorized_edit_candidates(
    graph: TypedGraph,
    rng: np.random.Generator,
    count: int,
    actions: Sequence[str] = EDIT_ACTIONS,
    positive: Optional[EditProposal] = None,
) -> Tuple[List[EditProposal], int]:
    """Cover action-family groups, then add endpoint negatives."""

    if int(count) < 2:
        raise ValueError("candidate count must be at least two")
    candidates: List[EditProposal] = []
    keys = set()

    def add(proposal: EditProposal) -> bool:
        key = _proposal_key(proposal)
        if key in keys:
            return False
        keys.add(key)
        candidates.append(proposal)
        return True

    if positive is not None:
        add(positive)
    add(EditProposal("noop", sorted(graph.family_specs)[0], (), ()))

    # Conditional endpoint learning needs multiple candidates in the positive
    # action-family group. This does not alter the forward-kernel likelihood.
    if positive is not None and positive.action != "noop":
        desired = min(max(int(count) // 4, 4), int(count) - 1)
        attempts = 0
        while len(candidates) < desired + 1 and attempts < desired * 128:
            attempts += 1
            proposal = sample_valid_edit(
                graph,
                rng,
                actions=(positive.action,),
                family=positive.family,
            )
            if proposal is not None:
                add(proposal)

    groups = [
        (action, family)
        for action in actions
        for family in sorted(graph.family_specs)
    ]
    for index in rng.permutation(len(groups)).tolist():
        if len(candidates) >= int(count):
            break
        action, family = groups[index]
        proposal = sample_valid_edit(
            graph, rng, actions=(action,), family=family
        )
        if proposal is not None:
            add(proposal)

    attempts = 0
    while len(candidates) < int(count) and attempts < int(count) * 512:
        attempts += 1
        proposal = sample_valid_edit(graph, rng, actions=actions)
        if proposal is not None:
            add(proposal)
    if len(candidates) != int(count):
        raise RuntimeError(
            f"sampled only {len(candidates)}/{count} factorized candidates"
        )
    target = 0
    if positive is not None:
        order = rng.permutation(len(candidates))
        candidates = [candidates[index] for index in order.tolist()]
        target = int(np.flatnonzero(order == 0)[0])
    return candidates, target


def build_factorized_edit_dataset(
    source: TypedGraph,
    trajectories: int,
    forward_steps: int,
    record_every: int,
    candidates_per_example: int,
    seed: int,
    actions: Sequence[str] = EDIT_ACTIONS,
    schedule: Optional[MixedKernelSchedule] = None,
) -> SwitchDiffusionDataset:
    if int(trajectories) < 2:
        raise ValueError("factorized training needs at least two trajectories")
    prior = MarginalGraphPrior.from_training_graph(source)
    schedule = schedule or MixedKernelSchedule()
    examples = []
    targets = []
    trajectory_ids = []
    for trajectory_id in range(int(trajectories)):
        rng = np.random.default_rng(int(seed) + 1_000_003 * trajectory_id)
        graph = source.clone()
        for step in range(1, int(forward_steps) + 1):
            forward = sample_mixed_forward_transition(
                graph,
                prior,
                rng,
                float(step) / float(forward_steps),
                schedule,
                actions=actions,
            )
            apply_edit(graph, forward)
            if step % int(record_every) != 0:
                continue
            positive = inverse_edit(forward)
            candidates, target = sample_factorized_edit_candidates(
                graph,
                rng,
                candidates_per_example,
                actions=actions,
                positive=positive,
            )
            examples.append(
                mixed_edit_candidate_matrix(
                    graph, candidates, float(step) / float(forward_steps)
                )
            )
            targets.append(target)
            trajectory_ids.append(trajectory_id)
    if not examples:
        raise RuntimeError("factorized forward process produced no examples")
    shape = (len(examples), int(candidates_per_example))
    return SwitchDiffusionDataset(
        candidate_features=np.stack(examples),
        candidate_improvements=np.zeros(shape, dtype=np.float32),
        targets=np.asarray(targets, dtype=np.int64),
        trajectory_ids=np.asarray(trajectory_ids, dtype=np.int64),
    )


class FactorizedEditDenoiser(nn.Module):
    """Calibrated action/family kernel plus conditional endpoint ranking."""

    def __init__(
        self,
        input_dim: int,
        state_dim: int,
        family_count: int,
        hidden_dim: int = 256,
        layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.state_dim = int(state_dim)
        self.family_count = int(family_count)
        self.action_count = len(SCORER_ACTIONS)
        if self.input_dim <= self.state_dim:
            raise ValueError("candidate input must include local and state features")
        state_layers: List[nn.Module] = []
        width = self.state_dim
        for _ in range(max(int(layers), 1)):
            state_layers.extend(
                [
                    nn.Linear(width, int(hidden_dim)),
                    nn.LayerNorm(int(hidden_dim)),
                    nn.SiLU(),
                    nn.Dropout(float(dropout)),
                ]
            )
            width = int(hidden_dim)
        self.state_encoder = nn.Sequential(*state_layers)
        self.action_head = nn.Linear(width, self.action_count)
        self.family_head = nn.Linear(width, self.action_count * self.family_count)
        self.endpoint_head = nn.Sequential(
            nn.Linear(self.input_dim - self.state_dim + width, int(hidden_dim)),
            nn.LayerNorm(int(hidden_dim)),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), 1),
        )

    def components(
        self, candidate_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if candidate_features.ndim != 3:
            raise ValueError("candidate features must have shape [B,C,D]")
        state = candidate_features[:, 0, -self.state_dim :]
        hidden = self.state_encoder(state)
        action_logits = self.action_head(hidden)
        family_logits = self.family_head(hidden).reshape(
            -1, self.action_count, self.family_count
        )
        local = candidate_features[..., : -self.state_dim]
        expanded = hidden[:, None, :].expand(-1, local.shape[1], -1)
        endpoint_logits = self.endpoint_head(
            torch.cat((local, expanded), dim=-1)
        ).squeeze(-1)
        return action_logits, family_logits, endpoint_logits

    def forward(self, candidate_features: torch.Tensor) -> torch.Tensor:
        action_logits, family_logits, endpoint_logits = self.components(
            candidate_features
        )
        action_ids = candidate_features[..., : self.action_count].argmax(dim=-1)
        family_start = self.action_count
        family_ids = candidate_features[
            ..., family_start : family_start + self.family_count
        ].argmax(dim=-1)
        action_log_probability = torch.log_softmax(action_logits, dim=-1)
        family_log_probability = torch.log_softmax(family_logits, dim=-1)
        scores = torch.empty_like(endpoint_logits)
        for batch_index in range(candidate_features.shape[0]):
            for action_id in torch.unique(action_ids[batch_index]).tolist():
                action_id = int(action_id)
                action_mask = action_ids[batch_index] == action_id
                if action_id == 0:
                    group_indices = torch.nonzero(
                        action_mask, as_tuple=False
                    ).flatten()
                    group_endpoint = torch.log_softmax(
                        endpoint_logits[batch_index, group_indices], dim=0
                    )
                    scores[batch_index, group_indices] = (
                        action_log_probability[batch_index, action_id]
                        + group_endpoint
                    )
                    continue
                represented_families = torch.unique(
                    family_ids[batch_index, action_mask]
                )
                for family_id in represented_families.tolist():
                    family_id = int(family_id)
                    group_mask = action_mask & (
                        family_ids[batch_index] == family_id
                    )
                    group_indices = torch.nonzero(
                        group_mask, as_tuple=False
                    ).flatten()
                    group_endpoint = torch.log_softmax(
                        endpoint_logits[batch_index, group_indices], dim=0
                    )
                    scores[batch_index, group_indices] = (
                        action_log_probability[batch_index, action_id]
                        + family_log_probability[
                            batch_index, action_id, family_id
                        ]
                        + group_endpoint
                    )
        return scores


def _target_labels(
    features: torch.Tensor, targets: torch.Tensor, family_count: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    selected = features[
        torch.arange(features.shape[0], device=features.device), targets
    ]
    action_ids = selected[:, : len(SCORER_ACTIONS)].argmax(dim=-1)
    start = len(SCORER_ACTIONS)
    family_ids = selected[:, start : start + int(family_count)].argmax(dim=-1)
    return action_ids, family_ids


def _factorized_loss(
    model: FactorizedEditDenoiser,
    features: torch.Tensor,
    targets: torch.Tensor,
    balanced_action_weights: torch.Tensor,
    balanced_action_aux_weight: float,
) -> Tuple[torch.Tensor, Mapping[str, torch.Tensor]]:
    action_logits, family_logits, _ = model.components(features)
    candidate_log_probability = model(features)
    target_actions, target_families = _target_labels(
        features, targets, model.family_count
    )
    action_loss = nn.functional.cross_entropy(action_logits, target_actions)
    balanced_action_loss = nn.functional.cross_entropy(
        action_logits, target_actions, weight=balanced_action_weights
    )
    active = target_actions != 0
    if active.any():
        selected_family_logits = family_logits[
            torch.arange(features.shape[0], device=features.device), target_actions
        ]
        family_loss = nn.functional.cross_entropy(
            selected_family_logits[active], target_families[active]
        )
    else:
        family_loss = action_loss.new_zeros(())
    endpoint_loss = nn.functional.nll_loss(candidate_log_probability, targets)
    # The endpoint NLL already contains action/family terms. Separate heads are
    # retained as explicit calibrated supervision, with a small representation
    # auxiliary for rare actions.
    total = (
        endpoint_loss
        + 0.5 * action_loss
        + 0.5 * family_loss
        + float(balanced_action_aux_weight) * balanced_action_loss
    )
    return total, {
        "action": action_loss,
        "balanced_action": balanced_action_loss,
        "family": family_loss,
        "candidate": endpoint_loss,
    }


def train_factorized_edit_denoiser(
    dataset: SwitchDiffusionDataset,
    type_count: int,
    family_count: int,
    hidden_dim: int = 256,
    layers: int = 3,
    dropout: float = 0.1,
    epochs: int = 50,
    batch_size: int = 64,
    learning_rate: float = 1.0e-3,
    weight_decay: float = 1.0e-4,
    balanced_action_aux_weight: float = 0.1,
    device: str | torch.device = "cpu",
    seed: int = 0,
) -> Tuple[FactorizedEditDenoiser, Mapping]:
    torch.manual_seed(int(seed))
    device = torch.device(device)
    features = torch.from_numpy(dataset.candidate_features).float()
    targets = torch.from_numpy(dataset.targets).long()
    trajectory_ids = np.asarray(dataset.trajectory_ids)
    validation_trajectory = int(trajectory_ids.max())
    validation_mask = trajectory_ids == validation_trajectory
    if not validation_mask.any() or validation_mask.all():
        raise ValueError("validation split must leave train and validation examples")
    train_indices = torch.from_numpy(np.flatnonzero(~validation_mask)).long()
    validation_indices = torch.from_numpy(np.flatnonzero(validation_mask)).long()
    train_loader = DataLoader(
        TensorDataset(features[train_indices], targets[train_indices]),
        batch_size=int(batch_size),
        shuffle=True,
        generator=torch.Generator().manual_seed(int(seed)),
    )
    validation_loader = DataLoader(
        TensorDataset(features[validation_indices], targets[validation_indices]),
        batch_size=int(batch_size),
        shuffle=False,
    )
    model = FactorizedEditDenoiser(
        input_dim=int(features.shape[-1]),
        state_dim=mixed_edit_state_dim(type_count, family_count),
        family_count=family_count,
        hidden_dim=hidden_dim,
        layers=layers,
        dropout=dropout,
    ).to(device)
    train_action_ids, _ = _target_labels(
        features[train_indices], targets[train_indices], family_count
    )
    counts = torch.bincount(
        train_action_ids, minlength=len(SCORER_ACTIONS)
    ).float()
    action_weights = counts.sum().sqrt() / counts.clamp_min(1.0).sqrt()
    action_weights = (action_weights / action_weights.mean()).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay)
    )
    final_train_loss = float("nan")
    for _ in range(int(epochs)):
        model.train()
        losses = []
        for batch_features, batch_targets in train_loader:
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss, _ = _factorized_loss(
                model,
                batch_features,
                batch_targets,
                action_weights,
                balanced_action_aux_weight,
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        final_train_loss = float(np.mean(losses))

    model.eval()
    metrics: Dict[str, List[float]] = {
        "loss": [],
        "candidate_top1": [],
        "candidate_top5": [],
        "action_top1": [],
        "family_top1": [],
    }
    per_action_correct = np.zeros(len(SCORER_ACTIONS), dtype=np.int64)
    per_action_total = np.zeros(len(SCORER_ACTIONS), dtype=np.int64)
    with torch.no_grad():
        for batch_features, batch_targets in validation_loader:
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)
            loss, _ = _factorized_loss(
                model,
                batch_features,
                batch_targets,
                action_weights,
                balanced_action_aux_weight,
            )
            scores = model(batch_features)
            action_logits, family_logits, _ = model.components(batch_features)
            target_actions, target_families = _target_labels(
                batch_features, batch_targets, family_count
            )
            predicted_actions = action_logits.argmax(dim=-1)
            metrics["loss"].append(float(loss.cpu()))
            metrics["candidate_top1"].extend(
                (scores.argmax(dim=-1) == batch_targets).float().cpu().tolist()
            )
            width = min(5, scores.shape[1])
            metrics["candidate_top5"].extend(
                (scores.topk(width, dim=-1).indices == batch_targets[:, None])
                .any(dim=-1)
                .float()
                .cpu()
                .tolist()
            )
            metrics["action_top1"].extend(
                (predicted_actions == target_actions).float().cpu().tolist()
            )
            active = target_actions != 0
            if active.any():
                selected_family = family_logits[
                    torch.arange(batch_features.shape[0], device=device), target_actions
                ].argmax(dim=-1)
                metrics["family_top1"].extend(
                    (selected_family[active] == target_families[active])
                    .float()
                    .cpu()
                    .tolist()
                )
            for action_id in range(len(SCORER_ACTIONS)):
                mask = target_actions == action_id
                per_action_total[action_id] += int(mask.sum().item())
                per_action_correct[action_id] += int(
                    (predicted_actions[mask] == action_id).sum().item()
                )
    report = {
        "format": "dihug-factorized-edit-train-v1",
        "epochs": int(epochs),
        "train_examples": int(train_indices.numel()),
        "validation_examples": int(validation_indices.numel()),
        "train_loss": final_train_loss,
        "validation_loss": float(np.mean(metrics["loss"])),
        "validation_candidate_top1": float(np.mean(metrics["candidate_top1"])),
        "validation_candidate_top5": float(np.mean(metrics["candidate_top5"])),
        "validation_action_top1": float(np.mean(metrics["action_top1"])),
        "validation_family_top1": float(np.mean(metrics["family_top1"])),
        "validation_action_recall": {
            action: float(per_action_correct[index])
            / max(int(per_action_total[index]), 1)
            for index, action in enumerate(SCORER_ACTIONS)
        },
        "validation_action_support": {
            action: int(per_action_total[index])
            for index, action in enumerate(SCORER_ACTIONS)
        },
        "balanced_action_aux_weight": float(balanced_action_aux_weight),
    }
    return model, report


def sample_factorized_edit_diffusion(
    prior: MarginalGraphPrior,
    model: FactorizedEditDenoiser,
    steps: int,
    candidates_per_step: int,
    seed: int,
    actions: Sequence[str] = EDIT_ACTIONS,
    device: str | torch.device = "cpu",
    temperature: float = 0.0,
    metrics_every: int = 1000,
) -> MixedEditSamplingResult:
    graph = prior.sample(seed=int(seed) + 71)
    rng = np.random.default_rng(int(seed))
    device = torch.device(device)
    generator = torch.Generator(device=device).manual_seed(int(seed))
    model = model.to(device).eval()
    trace = [{"step": 0.0, **graph_metrics(graph)}]
    action_counts = {action: 0 for action in SCORER_ACTIONS}
    proposed = 0
    applied = 0
    for reverse_step in range(int(steps), 0, -1):
        time_fraction = float(reverse_step) / float(steps)
        candidates, _ = sample_factorized_edit_candidates(
            graph,
            rng,
            candidates_per_step,
            actions=actions,
        )
        proposed += len(candidates)
        matrix = mixed_edit_candidate_matrix(graph, candidates, time_fraction)
        with torch.no_grad():
            logits = model(torch.from_numpy(matrix)[None].to(device)).squeeze(0)
        if float(temperature) <= 0.0:
            chosen = int(logits.argmax().item())
        else:
            probability = torch.softmax(logits / float(temperature), dim=-1)
            chosen = int(
                torch.multinomial(probability, 1, generator=generator).item()
            )
        proposal = candidates[chosen]
        apply_edit(graph, proposal)
        action_counts[proposal.action] += 1
        applied += int(proposal.action != "noop")
        completed = int(steps) - reverse_step + 1
        if completed == int(steps) or (
            int(metrics_every) > 0 and completed % int(metrics_every) == 0
        ):
            trace.append({"step": float(completed), **graph_metrics(graph)})
    graph.validate()
    return MixedEditSamplingResult(
        graph=graph,
        trace=trace,
        action_counts=action_counts,
        proposed=proposed,
        applied=applied,
    )


def evaluate_factorized_fixed_t(
    source: TypedGraph,
    model: FactorizedEditDenoiser,
    time_fractions: Sequence[float],
    trials_per_time: int,
    forward_steps: int,
    candidates_per_example: int,
    seed: int,
    actions: Sequence[str] = EDIT_ACTIONS,
    device: str | torch.device = "cpu",
    schedule: Optional[MixedKernelSchedule] = None,
) -> Mapping[str, Mapping]:
    prior = MarginalGraphPrior.from_training_graph(source)
    schedule = schedule or MixedKernelSchedule()
    device = torch.device(device)
    model = model.to(device).eval()
    output = {}
    for time_index, fraction in enumerate(time_fractions):
        candidate_correct = 0
        action_correct = 0
        action_total = np.zeros(len(SCORER_ACTIONS), dtype=np.int64)
        action_hits = np.zeros(len(SCORER_ACTIONS), dtype=np.int64)
        margins = []
        for trial in range(int(trials_per_time)):
            rng = np.random.default_rng(
                int(seed) + 10_000_019 * time_index + 1_000_003 * trial
            )
            graph = source.clone()
            last = EditProposal("noop", sorted(graph.family_specs)[0], (), ())
            steps = max(1, int(round(float(fraction) * int(forward_steps))))
            for step in range(1, steps + 1):
                last = sample_mixed_forward_transition(
                    graph,
                    prior,
                    rng,
                    float(step) / float(forward_steps),
                    schedule,
                    actions=actions,
                )
                apply_edit(graph, last)
            positive = inverse_edit(last)
            candidates, target = sample_factorized_edit_candidates(
                graph,
                rng,
                candidates_per_example,
                actions=actions,
                positive=positive,
            )
            matrix = torch.from_numpy(
                mixed_edit_candidate_matrix(graph, candidates, float(fraction))
            )[None].to(device)
            with torch.no_grad():
                scores = model(matrix).squeeze(0)
                action_logits, _, _ = model.components(matrix)
            selected = int(scores.argmax().item())
            predicted_action = int(action_logits.argmax(dim=-1).item())
            target_action = SCORER_ACTIONS.index(positive.action)
            candidate_correct += int(selected == target)
            action_correct += int(predicted_action == target_action)
            action_total[target_action] += 1
            action_hits[target_action] += int(predicted_action == target_action)
            negative = torch.cat((scores[:target], scores[target + 1 :]))
            margins.append(float((scores[target] - negative.max()).item()))
        output[f"{float(fraction):.3f}"] = {
            "trials": int(trials_per_time),
            "candidate_top1": candidate_correct / max(int(trials_per_time), 1),
            "action_top1": action_correct / max(int(trials_per_time), 1),
            "margin": float(np.mean(margins)),
            "action_recall": {
                action: float(action_hits[index]) / max(int(action_total[index]), 1)
                for index, action in enumerate(SCORER_ACTIONS)
            },
            "action_support": {
                action: int(action_total[index])
                for index, action in enumerate(SCORER_ACTIONS)
            },
        }
    return output


def save_factorized_edit_checkpoint(
    path: str | Path,
    model: FactorizedEditDenoiser,
    prior: MarginalGraphPrior,
    hidden_dim: int,
    layers: int,
    dropout: float,
    actions: Sequence[str],
    report: Mapping,
) -> None:
    payload = {
        "format": "dihug-factorized-edit-diffusion-v1",
        "prior": prior.to_json_dict(),
        "input_dim": model.input_dim,
        "state_dim": model.state_dim,
        "family_count": model.family_count,
        "hidden_dim": int(hidden_dim),
        "layers": int(layers),
        "dropout": float(dropout),
        "actions": list(actions),
        "state_dict": model.state_dict(),
        "train_report": dict(report),
        "inference_contract": prior.to_json_dict()["inference_contract"],
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_factorized_edit_checkpoint(
    path: str | Path, device: str | torch.device = "cpu"
) -> Tuple[FactorizedEditDenoiser, MarginalGraphPrior, Mapping]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != "dihug-factorized-edit-diffusion-v1":
        raise ValueError("unsupported factorized edit checkpoint")
    prior = MarginalGraphPrior.from_json_dict(payload["prior"])
    model = FactorizedEditDenoiser(
        input_dim=int(payload["input_dim"]),
        state_dim=int(payload["state_dim"]),
        family_count=int(payload["family_count"]),
        hidden_dim=int(payload["hidden_dim"]),
        layers=int(payload["layers"]),
        dropout=float(payload["dropout"]),
    )
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    metadata = {
        "actions": tuple(payload["actions"]),
        "train_report": payload.get("train_report", {}),
    }
    return model, prior, metadata
