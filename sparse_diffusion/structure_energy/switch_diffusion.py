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
    epochs: int
    train_examples: int
    validation_examples: int
    train_loss: float
    validation_loss: float
    validation_top1: float
    validation_top5: float
    validation_margin: float
    validation_oracle_top1: float

    def to_json_dict(self) -> dict:
        return {
            "epochs": int(self.epochs),
            "train_examples": int(self.train_examples),
            "validation_examples": int(self.validation_examples),
            "train_loss": float(self.train_loss),
            "validation_loss": float(self.validation_loss),
            "validation_top1": float(self.validation_top1),
            "validation_top5": float(self.validation_top5),
            "validation_margin": float(self.validation_margin),
            "validation_oracle_top1": float(self.validation_oracle_top1),
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
    ) -> None:
        super().__init__()
        if int(layers) < 1:
            raise ValueError("switch scorer needs at least one hidden layer")
        self.input_dim = int(input_dim)
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

    def forward(self, candidate_features: torch.Tensor) -> torch.Tensor:
        if candidate_features.ndim != 3:
            raise ValueError("candidate features must have shape [B, C, D]")
        return self.network(candidate_features).squeeze(-1)


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
    target_vector: np.ndarray,
    proposal: SwitchProposal,
    family_to_id: Mapping[str, int],
    time_fraction: float,
) -> np.ndarray:
    delta = selection.delta_vector(proposal_delta(graph, extractor, proposal))
    current = selection.vector(current_features)
    residual = np.asarray(target_vector, dtype=np.float64) - current
    aligned = delta * np.sign(residual)
    family = np.zeros(len(family_to_id), dtype=np.float64)
    family[int(family_to_id[proposal.family])] = 1.0
    scalar = np.asarray(
        [
            float(time_fraction),
            float(np.dot(delta, residual) / max(len(delta), 1)),
            float(np.linalg.norm(delta)),
        ],
        dtype=np.float64,
    )
    return np.concatenate((delta, residual, aligned, family, scalar)).astype(
        np.float32
    )


def candidate_feature_matrix(
    graph: TypedGraph,
    extractor: GraphletFeatureExtractor,
    selection: FeatureSelection,
    current_features: Mapping[str, int | float],
    target_vector: np.ndarray,
    candidates: Sequence[SwitchProposal],
    family_to_id: Mapping[str, int],
    time_fraction: float,
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
            )
            for proposal in candidates
        ]
    )


def candidate_target_improvements(
    candidate_features: np.ndarray,
    feature_count: int,
) -> np.ndarray:
    """Exact decrease in squared normalized target error for each switch."""

    matrix = np.asarray(candidate_features, dtype=np.float64)
    width = int(feature_count)
    if matrix.ndim != 2 or matrix.shape[1] < 2 * width:
        raise ValueError("candidate feature matrix is incompatible with selection")
    delta = matrix[:, :width]
    residual = matrix[:, width : 2 * width]
    return 2.0 * np.einsum("ij,ij->i", delta, residual) - np.einsum(
        "ij,ij->i", delta, delta
    )


def build_switch_diffusion_dataset(
    source: TypedGraph,
    selection: FeatureSelection,
    trajectories: int,
    forward_steps: int,
    record_every: int,
    candidates_per_example: int,
    seed: int,
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
                float(step) / float(forward_steps),
            )
            examples.append(matrix)
            improvements.append(
                candidate_target_improvements(matrix, len(selection.keys))
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
) -> tuple[float, float, float, float]:
    model.eval()
    losses = []
    top1 = []
    top5 = []
    margins = []
    criterion = nn.CrossEntropyLoss(reduction="none")
    with torch.no_grad():
        for features, target in loader:
            features = features.to(device)
            target = target.to(device)
            logits = model(features)
            losses.extend(criterion(logits, target).cpu().tolist())
            top1.extend((logits.argmax(dim=1) == target).float().cpu().tolist())
            width = min(5, logits.shape[1])
            top = logits.topk(width, dim=1).indices
            top5.extend((top == target[:, None]).any(dim=1).float().cpu().tolist())
            positive = logits.gather(1, target[:, None]).squeeze(1)
            masked = logits.clone()
            masked.scatter_(1, target[:, None], float("-inf"))
            margins.extend((positive - masked.max(dim=1).values).cpu().tolist())
    return (
        float(np.mean(losses)),
        float(np.mean(top1)),
        float(np.mean(top5)),
        float(np.mean(margins)),
    )


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
    device: str | torch.device = "cpu",
    seed: int = 0,
) -> tuple[SwitchDenoiserMLP, SwitchDiffusionTrainReport]:
    torch.manual_seed(int(seed))
    device = torch.device(device)
    trajectory_ids = np.asarray(dataset.trajectory_ids)
    if validation_trajectory is None:
        validation_trajectory = int(trajectory_ids.max())
    validation_mask = trajectory_ids == int(validation_trajectory)
    if not validation_mask.any() or validation_mask.all():
        raise ValueError("validation split must leave train and validation examples")
    features = torch.from_numpy(dataset.candidate_features).float()
    targets = torch.from_numpy(dataset.targets).long()
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
    model = SwitchDenoiserMLP(
        input_dim=int(features.shape[-1]),
        hidden_dim=hidden_dim,
        layers=layers,
        dropout=dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay)
    )
    criterion = nn.CrossEntropyLoss()
    final_train_loss = float("nan")
    for _ in range(int(epochs)):
        model.train()
        losses = []
        for batch_features, batch_target in train_loader:
            batch_features = batch_features.to(device)
            batch_target = batch_target.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_features), batch_target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        final_train_loss = float(np.mean(losses))
    validation_loss, top1, top5, margin = _evaluate_switch_model(
        model, validation_loader, device
    )
    validation_targets = np.asarray(dataset.targets)[validation_mask]
    validation_oracle = np.asarray(dataset.candidate_improvements)[
        validation_mask
    ].argmax(axis=1)
    report = SwitchDiffusionTrainReport(
        epochs=int(epochs),
        train_examples=int(train_indices.numel()),
        validation_examples=int(validation_indices.numel()),
        train_loss=final_train_loss,
        validation_loss=validation_loss,
        validation_top1=top1,
        validation_top5=top5,
        validation_margin=margin,
        validation_oracle_top1=float(
            np.mean(validation_oracle == validation_targets)
        ),
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
    target_vector: np.ndarray,
    steps: int,
    candidates_per_step: int,
    seed: int,
    device: str | torch.device = "cpu",
    temperature: float = 0.1,
    metrics_every: int = 100,
    selection_mode: str = "model",
    verbose: bool = False,
) -> SwitchSamplingResult:
    selection_mode = str(selection_mode).strip().lower()
    if selection_mode not in {"model", "random", "oracle"}:
        raise ValueError("selection_mode must be model, random, or oracle")
    graph = initial_graph.clone()
    expected_degrees = graph.all_family_degrees()
    family_to_id = {name: index for index, name in enumerate(sorted(graph.family_specs))}
    extractor = GraphletFeatureExtractor(graph, tracked_keys=selection.keys)
    current_features = extractor.extract(graph)
    rng = np.random.default_rng(int(seed))
    device = torch.device(device)
    torch_generator = torch.Generator(device=device).manual_seed(int(seed))
    model = model.to(device).eval()
    trace: List[Mapping[str, float]] = []
    applied = 0
    oracle_matches = 0
    for reverse_step in range(int(steps), 0, -1):
        candidates, _ = sample_switch_candidates(
            graph, rng, candidates_per_step, positive=None
        )
        matrix = candidate_feature_matrix(
            graph,
            extractor,
            selection,
            current_features,
            target_vector,
            candidates,
            family_to_id,
            float(reverse_step) / max(float(steps), 1.0),
        )
        improvements = candidate_target_improvements(matrix, len(selection.keys))
        oracle = int(np.argmax(improvements))
        if selection_mode == "oracle":
            chosen = oracle
        elif selection_mode == "random":
            chosen = int(rng.integers(len(candidates)))
        else:
            with torch.no_grad():
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
        oracle_matches += int(chosen == oracle)
        proposal = candidates[chosen]
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
            target_mae = float(
                np.mean(
                    np.abs(
                        np.asarray(target_vector, dtype=np.float64)
                        - selection.vector(current_features)
                    )
                )
            )
            trace_row = {
                "step": float(completed),
                "target_feature_mae": target_mae,
                "oracle_agreement": float(oracle_matches) / float(completed),
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
        proposed=int(steps) * int(candidates_per_step),
        applied=applied,
        selection_mode=selection_mode,
        oracle_agreement=float(oracle_matches) / max(float(applied), 1.0),
    )


def save_switch_checkpoint(
    path: str | Path,
    model: SwitchDenoiserMLP,
    structure_checkpoint: StructureEnergyCheckpoint,
    target_vector: np.ndarray,
    hidden_dim: int,
    layers: int,
    dropout: float,
    report: SwitchDiffusionTrainReport,
) -> None:
    payload = {
        "format": "dihug-switch-diffusion-v1",
        "structure_checkpoint": structure_checkpoint.to_json_dict(),
        "target_vector": np.asarray(target_vector, dtype=np.float64).tolist(),
        "input_dim": int(model.input_dim),
        "hidden_dim": int(hidden_dim),
        "layers": int(layers),
        "dropout": float(dropout),
        "state_dict": model.cpu().state_dict(),
        "train_report": report.to_json_dict(),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_switch_checkpoint(
    path: str | Path, device: str | torch.device = "cpu"
) -> tuple[
    SwitchDenoiserMLP,
    StructureEnergyCheckpoint,
    np.ndarray,
    Mapping,
]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != "dihug-switch-diffusion-v1":
        raise ValueError("unsupported switch-diffusion checkpoint")
    model = SwitchDenoiserMLP(
        input_dim=int(payload["input_dim"]),
        hidden_dim=int(payload["hidden_dim"]),
        layers=int(payload["layers"]),
        dropout=float(payload["dropout"]),
    )
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    structure = StructureEnergyCheckpoint.from_json_dict(
        payload["structure_checkpoint"]
    )
    target = np.asarray(payload["target_vector"], dtype=np.float64)
    return model, structure, target, payload.get("train_report", {})


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
