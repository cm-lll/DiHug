from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence, Tuple

import numpy as np
import torch
from torch import nn

from .graph import TypedGraph
from .hgt_edit_denoiser import candidate_endpoint_matrix, graph_node_features
from .mixed_edit_diffusion import (
    EditProposal,
    align_mixed_edit_features_for_model,
    apply_edit,
    inverse_edit,
    mixed_edit_candidate_matrix,
    sample_edit_candidates,
    sample_valid_edit,
)
from .roles import permuted_witness, sample_role_plan
from .role_guided_edit_diffusion import sample_switch_refinement_candidates
from .rooted_context_edit import (
    RootedClosureSignature,
    RootedContextEditResidual,
    rooted_closure_signature,
    rooted_closure_utility,
)
from .switch_diffusion import SwitchDenoiserMLP


DEFAULT_MACRO_ACTIONS = ("switch",)


@dataclass(frozen=True)
class SequentialMacroExample:
    """One reverse macro transition with a single frozen node context.

    ``graph`` is the state after the forward micro transitions. Applying
    ``inverse_edits`` in order reconstructs the macro-start state exactly.
    """

    graph: TypedGraph
    inverse_edits: Tuple[EditProposal, ...]
    time_fraction: float
    trajectory_id: int
    macro_step: int


@dataclass(frozen=True)
class SequentialMacroDataset:
    examples: Tuple[SequentialMacroExample, ...]
    node_feature_dim: int
    candidate_feature_dim: int
    type_count: int
    family_count: int
    node_types: np.ndarray
    macro_steps: int
    micro_steps: int
    feature_profile: str
    actions: Tuple[str, ...]
    rooted_target: RootedClosureSignature


def _edge_state(graph: TypedGraph) -> Mapping[str, frozenset[tuple[int, int]]]:
    return {
        family: frozenset(tuple(map(int, edge)) for edge in graph.family_edges[family])
        for family in graph.family_specs
    }


def build_sequential_macro_dataset(
    source: TypedGraph,
    legacy_scorer: SwitchDenoiserMLP,
    *,
    trajectories: int = 4,
    macro_steps: int = 100,
    micro_steps: int = 20,
    record_every: int = 1,
    seed: int = 0,
    feature_profile: str = "typed_closure_compact_v2",
    actions: Sequence[str] = DEFAULT_MACRO_ACTIONS,
) -> SequentialMacroDataset:
    """Build sequential macro transitions for cached-context denoising.

    Forward edits are applied one by one. Each recorded example stores the
    macro-end graph and the exact inverse sequence in reverse order. This is
    deliberately different from the historical commuting-batch experiment:
    graph validity is checked after every micro transition.
    """

    if int(trajectories) < 2:
        raise ValueError("sequential macro training needs at least two trajectories")
    if int(macro_steps) <= 0 or int(micro_steps) <= 0 or int(record_every) <= 0:
        raise ValueError("macro_steps, micro_steps and record_every must be positive")
    actions = tuple(str(action) for action in actions)
    if not actions or set(actions) - {"relocation", "switch"}:
        raise ValueError("terminal macro actions must be relocation and/or switch")

    examples = []
    for trajectory_id in range(int(trajectories)):
        trajectory_seed = int(seed) + 1_000_003 * trajectory_id
        rng = np.random.default_rng(trajectory_seed)
        graph = permuted_witness(
            source, sample_role_plan(source, trajectory_seed + 17)
        )
        for macro_step in range(1, int(macro_steps) + 1):
            before = _edge_state(graph)
            forward = []
            for _ in range(int(micro_steps)):
                proposal = sample_valid_edit(graph, rng, actions=actions)
                if proposal is None:
                    raise RuntimeError("failed to sample a legal macro micro-edit")
                apply_edit(graph, proposal)
                forward.append(proposal)
            if macro_step % int(record_every) != 0 and macro_step != int(macro_steps):
                continue
            inverses = tuple(inverse_edit(item) for item in reversed(forward))
            restored = graph.clone()
            for proposal in inverses:
                apply_edit(restored, proposal)
            if _edge_state(restored) != before:
                raise RuntimeError("sequential macro inverse did not restore its start state")
            examples.append(
                SequentialMacroExample(
                    graph=graph.clone(),
                    inverse_edits=inverses,
                    time_fraction=float(macro_step) / float(macro_steps),
                    trajectory_id=trajectory_id,
                    macro_step=macro_step,
                )
            )
    if not examples:
        raise RuntimeError("sequential macro dataset contains no examples")

    probe_graph = examples[0].graph
    probe_rng = np.random.default_rng(int(seed) + 91_919)
    probe_candidates, _ = sample_edit_candidates(
        probe_graph,
        probe_rng,
        count=2,
        actions=actions,
        positive=examples[0].inverse_edits[0],
    )
    probe = mixed_edit_candidate_matrix(
        probe_graph,
        probe_candidates,
        examples[0].time_fraction,
        feature_profile=feature_profile,
    )
    probe = align_mixed_edit_features_for_model(
        probe, legacy_scorer, family_count=len(source.family_specs)
    )
    return SequentialMacroDataset(
        examples=tuple(examples),
        node_feature_dim=int(graph_node_features(source, 0.0).shape[-1]),
        candidate_feature_dim=int(probe.shape[-1]),
        type_count=len(source.type_names),
        family_count=len(source.family_specs),
        node_types=source.node_types.copy(),
        macro_steps=int(macro_steps),
        micro_steps=int(micro_steps),
        feature_profile=str(feature_profile),
        actions=actions,
        rooted_target=rooted_closure_signature(source),
    )


def _candidate_tensors(
    graph: TypedGraph,
    candidates: Sequence[EditProposal],
    time_fraction: float,
    feature_profile: str,
    model: RootedContextEditResidual,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    matrix = mixed_edit_candidate_matrix(
        graph, candidates, time_fraction, feature_profile=feature_profile
    )
    matrix = align_mixed_edit_features_for_model(
        matrix,
        model.joint.legacy_scorer,
        family_count=len(graph.family_specs),
    )
    endpoints = candidate_endpoint_matrix(candidates)
    return (
        torch.from_numpy(matrix)[None].float().to(device),
        torch.from_numpy(endpoints)[None].long().to(device),
    )


def _run_macro_example(
    model: RootedContextEditResidual,
    example: SequentialMacroExample,
    *,
    candidates_per_micro: int,
    feature_profile: str,
    actions: Sequence[str],
    rng: np.random.Generator,
    device: torch.device,
    train: bool,
    base_aux_weight: float,
    trust_weight: float,
) -> tuple[torch.Tensor, int, int, int]:
    graph = example.graph.clone()
    hidden = model.encode_graph(graph, example.time_fraction, device)
    losses = []
    fused_hits = 0
    base_hits = 0
    micro_count = 0
    for positive in example.inverse_edits:
        candidates, target = sample_edit_candidates(
            graph,
            rng,
            count=int(candidates_per_micro),
            actions=actions,
            positive=positive,
        )
        features, endpoints = _candidate_tensors(
            graph,
            candidates,
            example.time_fraction,
            feature_profile,
            model,
            device,
        )
        logits = model.score_encoded(hidden, features, endpoints)
        base_logits = model.joint.legacy_scorer(features)
        target_tensor = torch.tensor([target], dtype=torch.long, device=device)
        fused_hits += int(int(logits.argmax(dim=1).item()) == target)
        base_hits += int(int(base_logits.argmax(dim=1).item()) == target)
        micro_count += 1
        if train:
            task = nn.functional.cross_entropy(logits, target_tensor)
            base_task = nn.functional.cross_entropy(base_logits, target_tensor)
            trust = nn.functional.kl_div(
                torch.log_softmax(logits, dim=1),
                torch.softmax(base_logits.detach(), dim=1),
                reduction="batchmean",
            )
            losses.append(
                task
                + float(base_aux_weight) * base_task
                + float(trust_weight) * trust
            )
        apply_edit(graph, positive)
    if not losses:
        loss = hidden.sum() * 0.0
    else:
        loss = torch.stack(losses).mean()
    return loss, fused_hits, base_hits, micro_count


def _utility_teacher(
    utilities: torch.Tensor, temperature: float
) -> torch.Tensor:
    centered = utilities - utilities.mean(dim=1, keepdim=True)
    scale = utilities.std(dim=1, keepdim=True).clamp(min=1.0e-6)
    return torch.softmax(centered / scale / float(temperature), dim=1)


def _run_utility_macro_example(
    model: RootedContextEditResidual,
    example: SequentialMacroExample,
    target: RootedClosureSignature,
    *,
    candidates_per_micro: int,
    feature_profile: str,
    rng: np.random.Generator,
    device: torch.device,
    train: bool,
    teacher_temperature: float,
    trust_weight: float,
) -> tuple[torch.Tensor, float, float, float, int]:
    """Teacher-force a structural reverse macro under one node encoding."""

    graph = example.graph.clone()
    hidden = model.encode_graph(graph, example.time_fraction, device)
    losses = []
    fused_utility = 0.0
    base_utility = 0.0
    oracle_utility = 0.0
    micro_count = 0
    for _ in range(len(example.inverse_edits)):
        candidates = sample_switch_refinement_candidates(
            graph,
            rng,
            int(candidates_per_micro),
            closure_fraction=1.0,
        )
        current = rooted_closure_signature(graph, target.boundaries)
        utility_values = np.asarray(
            [
                rooted_closure_utility(graph, proposal, target, current=current)
                for proposal in candidates
            ],
            dtype=np.float32,
        )
        features, endpoints = _candidate_tensors(
            graph,
            candidates,
            example.time_fraction,
            feature_profile,
            model,
            device,
        )
        logits = model.score_encoded(hidden, features, endpoints)
        base_logits = model.joint.legacy_scorer(features)
        utilities = torch.from_numpy(utility_values)[None].to(device)
        oracle_index = int(np.argmax(utility_values))
        fused_index = int(logits.argmax(dim=1).item())
        base_index = int(base_logits.argmax(dim=1).item())
        fused_utility += float(utility_values[fused_index])
        base_utility += float(utility_values[base_index])
        oracle_utility += float(utility_values[oracle_index])
        micro_count += 1
        if train:
            teacher = _utility_teacher(utilities, teacher_temperature)
            task = -(teacher * torch.log_softmax(logits, dim=1)).sum(dim=1).mean()
            trust = nn.functional.kl_div(
                torch.log_softmax(logits, dim=1),
                torch.softmax(base_logits.detach(), dim=1),
                reduction="batchmean",
            )
            losses.append(task + float(trust_weight) * trust)
        apply_edit(graph, candidates[oracle_index])
    loss = hidden.sum() * 0.0 if not losses else torch.stack(losses).mean()
    return loss, fused_utility, base_utility, oracle_utility, micro_count


def train_sequential_macro_context(
    dataset: SequentialMacroDataset,
    legacy_scorer: SwitchDenoiserMLP,
    *,
    epochs: int = 10,
    candidates_per_micro: int = 16,
    learning_rate: float = 3.0e-4,
    weight_decay: float = 1.0e-4,
    base_aux_weight: float = 0.25,
    trust_weight: float = 0.02,
    finetune_legacy: bool = True,
    supervision: str = "exact_inverse",
    teacher_temperature: float = 0.5,
    device: str | torch.device = "cuda",
    seed: int = 0,
    **model_kwargs,
) -> tuple[RootedContextEditResidual, Mapping[str, float]]:
    """Train one node encoding against a sequence of micro inverse edits."""

    if int(candidates_per_micro) < 2:
        raise ValueError("candidates_per_micro must be at least two")
    supervision = str(supervision).strip().lower()
    if supervision not in {"exact_inverse", "rooted_utility"}:
        raise ValueError("supervision must be exact_inverse or rooted_utility")
    if supervision == "rooted_utility" and dataset.actions != ("switch",):
        raise ValueError("rooted utility macro supervision is switch-only")
    torch.manual_seed(int(seed))
    rng = np.random.default_rng(int(seed))
    device = torch.device(device)
    validation_trajectory = max(item.trajectory_id for item in dataset.examples)
    train_examples = tuple(
        item for item in dataset.examples if item.trajectory_id != validation_trajectory
    )
    validation_examples = tuple(
        item for item in dataset.examples if item.trajectory_id == validation_trajectory
    )
    if not train_examples or not validation_examples:
        raise ValueError("macro dataset must contain train and validation trajectories")

    model = RootedContextEditResidual(
        copy.deepcopy(legacy_scorer),
        node_feature_dim=dataset.node_feature_dim,
        candidate_feature_dim=dataset.candidate_feature_dim,
        type_count=dataset.type_count,
        family_count=dataset.family_count,
        **model_kwargs,
    ).to(device)
    for parameter in model.joint.legacy_scorer.parameters():
        parameter.requires_grad_(bool(finetune_legacy))
    model.set_residual_scale(1.0)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=float(learning_rate), weight_decay=float(weight_decay)
    )
    final_loss = float("nan")
    for epoch in range(int(epochs)):
        model.train()
        epoch_losses = []
        fused_hits = base_hits = micro_total = 0
        fused_utility = base_utility = oracle_utility = 0.0
        for index in rng.permutation(len(train_examples)).tolist():
            example = train_examples[int(index)]
            optimizer.zero_grad(set_to_none=True)
            if supervision == "rooted_utility":
                loss, fused_value, base_value, oracle_value, count = (
                    _run_utility_macro_example(
                        model,
                        example,
                        dataset.rooted_target,
                        candidates_per_micro=candidates_per_micro,
                        feature_profile=dataset.feature_profile,
                        rng=rng,
                        device=device,
                        train=True,
                        teacher_temperature=teacher_temperature,
                        trust_weight=trust_weight,
                    )
                )
                fused = base = 0
                fused_utility += fused_value
                base_utility += base_value
                oracle_utility += oracle_value
            else:
                loss, fused, base, count = _run_macro_example(
                    model,
                    example,
                    candidates_per_micro=candidates_per_micro,
                    feature_profile=dataset.feature_profile,
                    actions=dataset.actions,
                    rng=rng,
                    device=device,
                    train=True,
                    base_aux_weight=base_aux_weight,
                    trust_weight=trust_weight,
                )
            loss.backward()
            nn.utils.clip_grad_norm_(trainable, 5.0)
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
            fused_hits += fused
            base_hits += base
            micro_total += count
        final_loss = float(np.mean(epoch_losses))
        diagnostic = (
            f"fused_utility={fused_utility / max(micro_total, 1):.6g} "
            f"base_utility={base_utility / max(micro_total, 1):.6g} "
            f"oracle_utility={oracle_utility / max(micro_total, 1):.6g}"
            if supervision == "rooted_utility"
            else (
                f"fused_top1={fused_hits / max(micro_total, 1):.4f} "
                f"base_top1={base_hits / max(micro_total, 1):.4f}"
            )
        )
        print(
            "[SEQUENTIAL-MACRO-EPOCH] "
            f"epoch={epoch} loss={final_loss:.6f} {diagnostic}",
            flush=True,
        )

    model.eval()
    scales = (0.0, 0.25, 0.5, 0.75, 1.0)
    scale_hits: dict[float, int] = {scale: 0 for scale in scales}
    scale_utilities: dict[float, float] = {scale: 0.0 for scale in scales}
    oracle_validation_utility = 0.0
    validation_total = 0
    validation_rng = np.random.default_rng(int(seed) + 808_081)
    with torch.no_grad():
        for example in validation_examples:
            graph = example.graph.clone()
            hidden = model.encode_graph(graph, example.time_fraction, device)
            for positive in example.inverse_edits:
                if supervision == "rooted_utility":
                    candidates = sample_switch_refinement_candidates(
                        graph,
                        validation_rng,
                        int(candidates_per_micro),
                        closure_fraction=1.0,
                    )
                    current = rooted_closure_signature(
                        graph, dataset.rooted_target.boundaries
                    )
                    utility_values = np.asarray(
                        [
                            rooted_closure_utility(
                                graph,
                                proposal,
                                dataset.rooted_target,
                                current=current,
                            )
                            for proposal in candidates
                        ],
                        dtype=np.float32,
                    )
                    target = int(np.argmax(utility_values))
                else:
                    candidates, target = sample_edit_candidates(
                        graph,
                        validation_rng,
                        count=int(candidates_per_micro),
                        actions=dataset.actions,
                        positive=positive,
                    )
                features, endpoints = _candidate_tensors(
                    graph,
                    candidates,
                    example.time_fraction,
                    dataset.feature_profile,
                    model,
                    device,
                )
                fused = model.score_encoded(hidden, features, endpoints)
                base = model.joint.legacy_scorer(features)
                for scale in scale_hits:
                    blended = base + float(scale) * (fused - base)
                    selected = int(blended.argmax(dim=1).item())
                    scale_hits[scale] += int(selected == target)
                    if supervision == "rooted_utility":
                        scale_utilities[scale] += float(utility_values[selected])
                if supervision == "rooted_utility":
                    oracle_validation_utility += float(utility_values[target])
                validation_total += 1
                apply_edit(
                    graph,
                    candidates[target] if supervision == "rooted_utility" else positive,
                )
    scale_accuracy = {
        scale: hits / max(validation_total, 1) for scale, hits in scale_hits.items()
    }
    scale_mean_utility = {
        scale: value / max(validation_total, 1)
        for scale, value in scale_utilities.items()
    }
    selected_scale = (
        max(scale_mean_utility, key=scale_mean_utility.get)
        if supervision == "rooted_utility"
        else max(scale_accuracy, key=scale_accuracy.get)
    )
    model.set_residual_scale(float(selected_scale))
    report = {
        "train_loss": final_loss,
        "train_macro_examples": float(len(train_examples)),
        "validation_macro_examples": float(len(validation_examples)),
        "train_micro_examples_per_epoch": float(
            sum(len(item.inverse_edits) for item in train_examples)
        ),
        "validation_micro_examples": float(validation_total),
        "macro_steps": float(dataset.macro_steps),
        "micro_steps_per_macro": float(dataset.micro_steps),
        "context_refresh_steps": float(dataset.micro_steps),
        "finetune_legacy": float(bool(finetune_legacy)),
        "supervision": supervision,
        "base_validation_top1": float(scale_accuracy[0.0]),
        "context_validation_top1": float(scale_accuracy[selected_scale]),
        "top1_gain": float(scale_accuracy[selected_scale] - scale_accuracy[0.0]),
        "residual_scale": float(selected_scale),
        "scale_accuracy": {
            str(scale): float(value) for scale, value in scale_accuracy.items()
        },
        "base_validation_utility": float(scale_mean_utility[0.0]),
        "context_validation_utility": float(scale_mean_utility[selected_scale]),
        "oracle_validation_utility": float(
            oracle_validation_utility / max(validation_total, 1)
        ),
        "utility_gain": float(
            scale_mean_utility[selected_scale] - scale_mean_utility[0.0]
        ),
        "scale_utility": {
            str(scale): float(value) for scale, value in scale_mean_utility.items()
        },
        "node_context_only": 1.0,
        "sequential_micro_transitions": 1.0,
        "simultaneous_batch": 0.0,
    }
    return model, report


def save_sequential_macro_context_checkpoint(
    path: str | Path,
    model: RootedContextEditResidual,
    report: Mapping[str, float],
    legacy_checkpoint: str | Path,
    *,
    role_prior: Mapping | None = None,
    subtype_schema: Mapping | None = None,
) -> None:
    from .rooted_context_edit import save_rooted_context_checkpoint

    save_rooted_context_checkpoint(
        path,
        model,
        {
            **dict(report),
            "training_protocol": "sequential-macro-node-context-v1",
        },
        legacy_checkpoint,
        role_prior=role_prior,
        subtype_schema=subtype_schema,
    )
