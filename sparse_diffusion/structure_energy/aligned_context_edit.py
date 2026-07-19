from __future__ import annotations

import copy
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import nn

from .hgt_edit_denoiser import HGTEditDataset, _make_batch
from .mixed_edit_diffusion import EDIT_ACTIONS, MarginalGraphPrior
from .role_guided_edit_diffusion import ExchangeableRoleDegreePrior
from .rooted_context_edit import RootedContextEditResidual
from .switch_diffusion import SwitchDenoiserMLP


def _reset_parameters(module: nn.Module) -> None:
    reset = getattr(module, "reset_parameters", None)
    if callable(reset):
        reset()


def _teacher_distribution(
    improvements: torch.Tensor, temperature: float
) -> torch.Tensor:
    centered = improvements - improvements.mean(dim=1, keepdim=True)
    scale = improvements.std(dim=1, keepdim=True).clamp(min=1.0e-6)
    return torch.softmax(centered / scale / float(temperature), dim=1)


def _role_terms(
    dataset: HGTEditDataset,
    indices: np.ndarray,
    device: torch.device,
    *,
    role_weight: float,
    gain_scale: float,
    schedule_power: float,
    schedule_floor: float,
    teacher_weight: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return inference-matched role bias and a scale-free teacher signal."""

    shape = (len(indices), dataset.candidate_features.shape[1])
    if dataset.candidate_role_gains is None:
        zeros = torch.zeros(shape, dtype=torch.float32, device=device)
        return zeros, zeros
    gains = torch.from_numpy(dataset.candidate_role_gains[indices]).float().to(device)
    if dataset.time_fractions is None:
        fractions = torch.from_numpy(
            dataset.node_features[indices, 0, -1]
        ).float().to(device)
    else:
        fractions = torch.from_numpy(dataset.time_fractions[indices]).float().to(device)
    multiplier = float(schedule_floor) + (
        1.0 - float(schedule_floor)
    ) * fractions.clamp(0.0, 1.0).pow(float(schedule_power))
    role_bias = (
        float(role_weight)
        * multiplier[:, None]
        * gains
        / float(gain_scale)
    )
    centered = gains - gains.mean(dim=1, keepdim=True)
    normalized = centered / gains.std(dim=1, keepdim=True).clamp(min=1.0e-6)
    role_teacher = float(teacher_weight) * multiplier[:, None] * normalized
    return role_bias, role_teacher


def train_aligned_context_edit_denoiser(
    dataset: HGTEditDataset,
    legacy_scorer: SwitchDenoiserMLP,
    *,
    epochs: int = 20,
    batch_size: int = 2,
    learning_rate: float = 3.0e-4,
    legacy_learning_rate_scale: float = 1.0,
    weight_decay: float = 1.0e-4,
    legacy_init: str = "random",
    legacy_unfreeze_epoch: int = 0,
    training_objective: str = "inverse_ce",
    utility_temperature: float = 1.0,
    trust_weight: float = 0.0,
    role_weight: float = 0.05,
    role_gain_scale: float = 10.0,
    role_schedule_power: float = 1.0,
    role_schedule_floor: float = 1.0,
    role_teacher_weight: float = 0.5,
    residual_min_utility_gain: float = 0.05,
    device: str | torch.device = "cuda",
    seed: int = 0,
    **model_kwargs,
) -> tuple[RootedContextEditResidual, Mapping[str, float]]:
    """Train HSD context and the original scorer on the original edit states.

    The dataset, candidate sets and labels are exactly those used by the paired
    explicit MLP. Only the scorer representation and optimization strategy vary.
    """

    legacy_init = str(legacy_init).strip().lower()
    if legacy_init not in {"pretrained", "random"}:
        raise ValueError("legacy_init must be pretrained or random")
    training_objective = str(training_objective).strip().lower()
    if training_objective not in {"inverse_ce", "utility_kl", "utility_role_kl"}:
        raise ValueError(
            "training_objective must be inverse_ce, utility_kl, or utility_role_kl"
        )
    if training_objective == "utility_kl" and not np.any(
        dataset.candidate_improvements
    ):
        raise ValueError("utility training needs a dataset with nonzero utilities")
    if training_objective == "utility_role_kl" and not (
        np.any(dataset.candidate_improvements)
        or (
            dataset.candidate_role_gains is not None
            and np.any(dataset.candidate_role_gains)
        )
    ):
        raise ValueError(
            "role-aware utility training needs nonzero structure utility or role gain"
        )
    if float(role_gain_scale) <= 0.0 or float(role_schedule_power) <= 0.0:
        raise ValueError("role gain scale and schedule power must be positive")
    if not 0.0 <= float(role_schedule_floor) <= 1.0:
        raise ValueError("role schedule floor must lie in [0,1]")
    if float(residual_min_utility_gain) < 0.0:
        raise ValueError("minimum residual utility gain must be non-negative")

    torch.manual_seed(int(seed))
    rng = np.random.default_rng(int(seed))
    device = torch.device(device)
    validation_trajectory = int(dataset.trajectory_ids.max())
    train_indices = np.flatnonzero(dataset.trajectory_ids != validation_trajectory)
    validation_indices = np.flatnonzero(
        dataset.trajectory_ids == validation_trajectory
    )
    scorer = copy.deepcopy(legacy_scorer)
    if legacy_init == "random":
        scorer.apply(_reset_parameters)
    model = RootedContextEditResidual(
        scorer,
        node_feature_dim=dataset.node_features.shape[-1],
        candidate_feature_dim=dataset.candidate_features.shape[-1],
        type_count=dataset.type_count,
        family_count=dataset.family_count,
        **model_kwargs,
    ).to(device)
    model.set_residual_scale(1.0)

    legacy_parameters = list(model.joint.legacy_scorer.parameters())
    legacy_ids = {id(parameter) for parameter in legacy_parameters}
    context_parameters = [
        parameter for parameter in model.parameters() if id(parameter) not in legacy_ids
    ]
    for parameter in legacy_parameters:
        parameter.requires_grad_(int(legacy_unfreeze_epoch) == 0)
    optimizer = torch.optim.AdamW(
        [
            {"params": context_parameters, "lr": float(learning_rate)},
            {
                "params": legacy_parameters,
                "lr": float(learning_rate) * float(legacy_learning_rate_scale),
            },
        ],
        weight_decay=float(weight_decay),
    )

    final_loss = float("nan")
    for epoch in range(int(epochs)):
        if epoch == int(legacy_unfreeze_epoch):
            for parameter in legacy_parameters:
                parameter.requires_grad_(True)
        model.train()
        shuffled = rng.permutation(train_indices)
        epoch_losses = []
        epoch_hits = 0
        epoch_count = 0
        for start in range(0, len(shuffled), int(batch_size)):
            indices = shuffled[start : start + int(batch_size)]
            batch = _make_batch(dataset, indices, device)
            optimizer.zero_grad(set_to_none=True)
            role_bias, role_teacher = _role_terms(
                dataset,
                indices,
                device,
                role_weight=role_weight,
                gain_scale=role_gain_scale,
                schedule_power=role_schedule_power,
                schedule_floor=role_schedule_floor,
                teacher_weight=role_teacher_weight,
            )
            logits = model(*batch[:-1]) + role_bias
            if training_objective == "inverse_ce":
                task = nn.functional.cross_entropy(logits, batch[-1])
            else:
                improvements = torch.from_numpy(
                    dataset.candidate_improvements[indices]
                ).float().to(device)
                teacher_signal = improvements + (
                    role_teacher if training_objective == "utility_role_kl" else 0.0
                )
                teacher = _teacher_distribution(teacher_signal, utility_temperature)
                task = -(
                    teacher * torch.log_softmax(logits, dim=1)
                ).sum(dim=1).mean()
            if float(trust_weight) > 0.0:
                with torch.no_grad():
                    base_distribution = torch.softmax(
                        model.joint.base_score(batch[4]) + role_bias, dim=1
                    )
                trust = nn.functional.kl_div(
                    torch.log_softmax(logits, dim=1),
                    base_distribution,
                    reduction="batchmean",
                )
            else:
                trust = logits.sum() * 0.0
            loss = task + float(trust_weight) * trust
            loss.backward()
            trainable = [
                parameter for parameter in model.parameters() if parameter.requires_grad
            ]
            nn.utils.clip_grad_norm_(trainable, 5.0)
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
            epoch_hits += int((logits.argmax(dim=1) == batch[-1]).sum().item())
            epoch_count += int(batch[-1].numel())
        final_loss = float(np.mean(epoch_losses))
        print(
            "[ALIGNED-CONTEXT-EPOCH] "
            f"epoch={epoch} loss={final_loss:.6f} "
            f"top1={epoch_hits / max(epoch_count, 1):.4f}",
            flush=True,
        )

    model.eval()
    logits_rows = []
    target_rows = []
    utility_rows = []
    role_teacher_rows = []
    role_bias_rows = []
    base_rows = []
    with torch.no_grad():
        for start in range(0, len(validation_indices), int(batch_size)):
            indices = validation_indices[start : start + int(batch_size)]
            batch = _make_batch(dataset, indices, device)
            role_bias, role_teacher = _role_terms(
                dataset,
                indices,
                device,
                role_weight=role_weight,
                gain_scale=role_gain_scale,
                schedule_power=role_schedule_power,
                schedule_floor=role_schedule_floor,
                teacher_weight=role_teacher_weight,
            )
            logits_rows.append(model(*batch[:-1]).cpu())
            base_rows.append(model.joint.base_score(batch[4]).cpu())
            role_bias_rows.append(role_bias.cpu())
            role_teacher_rows.append(role_teacher.cpu())
            target_rows.append(batch[-1].cpu())
            utility_rows.append(
                torch.from_numpy(dataset.candidate_improvements[indices]).float()
            )
    fused_structure_logits = torch.cat(logits_rows)
    base_logits = torch.cat(base_rows)
    role_bias = torch.cat(role_bias_rows)
    role_teacher = torch.cat(role_teacher_rows)
    targets = torch.cat(target_rows)
    utilities = torch.cat(utility_rows)
    teacher_signal = utilities + (
        role_teacher if training_objective == "utility_role_kl" else 0.0
    )
    residual = fused_structure_logits - base_logits
    scale_utilities = {}
    for scale in (0.0, 0.25, 0.5, 0.75, 1.0):
        candidate_logits = base_logits + float(scale) * residual + role_bias
        chosen = candidate_logits.argmax(dim=1)
        scale_utilities[float(scale)] = float(
            teacher_signal[
                torch.arange(chosen.numel()), chosen
            ].mean()
        )
    best_scale = max(scale_utilities, key=scale_utilities.get)
    residual_utility_gain = (
        float(scale_utilities[best_scale]) - float(scale_utilities[0.0])
    )
    selected_scale = (
        float(best_scale)
        if residual_utility_gain >= float(residual_min_utility_gain)
        else 0.0
    )
    model.set_residual_scale(selected_scale)
    logits = base_logits + float(selected_scale) * residual + role_bias
    base_logits = base_logits + role_bias
    rows = torch.arange(targets.numel())
    selected = logits.argmax(dim=1)
    base_selected = base_logits.argmax(dim=1)
    positive = logits[rows, targets]
    negative = logits.clone()
    negative[rows, targets] = -torch.inf
    report = {
        "training_objective": training_objective,
        "legacy_init": legacy_init,
        "legacy_unfreeze_epoch": int(legacy_unfreeze_epoch),
        "train_loss": final_loss,
        "train_examples": int(len(train_indices)),
        "validation_examples": int(len(validation_indices)),
        "validation_top1": float((selected == targets).float().mean()),
        "validation_top5": float(
            (
                logits.topk(min(5, logits.shape[1]), dim=1).indices
                == targets[:, None]
            )
            .any(dim=1)
            .float()
            .mean()
        ),
        "validation_margin": float((positive - negative.max(dim=1).values).mean()),
        "validation_base_top1": float(
            (base_selected == targets).float().mean()
        ),
        "validation_selected_utility": float(teacher_signal[rows, selected].mean()),
        "validation_base_utility": float(teacher_signal[rows, base_selected].mean()),
        "validation_oracle_utility": float(teacher_signal.max(dim=1).values.mean()),
        "residual_scale": float(selected_scale),
        "best_residual_scale_before_gate": float(best_scale),
        "residual_utility_gain": float(residual_utility_gain),
        "residual_min_utility_gain": float(residual_min_utility_gain),
        "scale_utilities": scale_utilities,
        "role_weight": float(role_weight),
        "role_gain_scale": float(role_gain_scale),
        "role_schedule_power": float(role_schedule_power),
        "role_schedule_floor": float(role_schedule_floor),
        "role_teacher_weight": float(role_teacher_weight),
    }
    return model, report


def save_aligned_context_checkpoint(
    path: str | Path,
    model: RootedContextEditResidual,
    prior: MarginalGraphPrior,
    role_prior: ExchangeableRoleDegreePrior,
    report: Mapping[str, float],
    *,
    feature_profile: str,
    actions: Sequence[str] = EDIT_ACTIONS,
    hidden_dim: int = 256,
    layers: int = 3,
    dropout: float = 0.1,
    training_noise: Mapping[str, object] | None = None,
) -> None:
    """Write a self-contained bundle accepted by the production sampler."""

    legacy = model.joint.legacy_scorer
    legacy_payload = {
        "format": "dihug-mixed-edit-diffusion-v1",
        "prior": prior.to_json_dict(),
        "input_dim": int(legacy.input_dim),
        "hidden_dim": int(hidden_dim),
        "layers": int(layers),
        "dropout": float(dropout),
        "state_dict": legacy.state_dict(),
        "train_report": dict(report),
        "actions": list(actions),
        "feature_version": "mixed_edit_v1",
        "feature_profile": str(feature_profile),
        "training_noise": dict(
            training_noise or {"forward_mode": "mixed_edit_stepwise"}
        ),
        "inference_contract": prior.to_json_dict()["inference_contract"],
    }
    payload = {
        "format": "dihug-rooted-context-edit-residual-v1",
        "model_config": model.config,
        "state_dict": model.state_dict(),
        "report": dict(report),
        "legacy_checkpoint": None,
        "legacy_checkpoint_payload": legacy_payload,
        "exchangeable_role_prior": role_prior.to_json_dict(),
        "subtype_schema": None,
        "inference_contract": {
            "training_adjacency": False,
            "graph_metric_target": False,
            "rooted_closure_target_at_inference": False,
            "persistent_edge_hidden_state": False,
            "candidate_context": "typed_endpoint_node_states",
        },
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
