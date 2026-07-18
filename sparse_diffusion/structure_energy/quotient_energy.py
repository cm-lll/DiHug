from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn

from sparse_diffusion.models.pyhgt_denoiser import HGTConv

from .graph import TypedGraph
from .hgt_edit_denoiser import graph_edge_tensors, graph_node_features
from .mixed_edit_diffusion import (
    EditProposal,
    MixedEditSamplingResult,
    apply_edit,
    permute_typed_graph_within_types,
    sample_edit_candidates,
    sample_valid_edit,
)
from .switch_chain import graph_metrics


@dataclass(frozen=True)
class QuotientEnergyExample:
    graph: TypedGraph
    clean: bool
    negative_kind: str
    noise_fraction: float
    trajectory_id: int


@dataclass(frozen=True)
class QuotientEnergyDataset:
    examples: Tuple[QuotientEnergyExample, ...]
    type_count: int
    family_count: int

    def __post_init__(self) -> None:
        if not self.examples:
            raise ValueError("quotient energy dataset is empty")
        if len({example.trajectory_id for example in self.examples}) < 2:
            raise ValueError("quotient energy dataset needs at least two trajectories")


class TypedQuotientEnergy(nn.Module):
    """Typed-permutation-invariant graph energy.

    Node encodings are equivariant; means are pooled separately by node type.
    No node identifiers, clean adjacency, target metric, or noise time enters the
    model. Lower values denote more data-like graph states.
    """

    def __init__(
        self,
        node_feature_dim: int,
        type_count: int,
        family_count: int,
        hidden_dim: int = 96,
        heads: int = 4,
        layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if int(hidden_dim) % int(heads):
            raise ValueError("energy hidden dimension must be divisible by heads")
        self.config = {
            "node_feature_dim": int(node_feature_dim),
            "type_count": int(type_count),
            "family_count": int(family_count),
            "hidden_dim": int(hidden_dim),
            "heads": int(heads),
            "layers": int(layers),
            "dropout": float(dropout),
        }
        self.node_input = nn.Sequential(
            nn.Linear(int(node_feature_dim), int(hidden_dim)),
            nn.LayerNorm(int(hidden_dim)),
            nn.SiLU(),
        )
        self.layers = nn.ModuleList(
            HGTConv(
                int(hidden_dim),
                int(hidden_dim),
                num_types=int(type_count),
                num_relations=int(family_count),
                n_heads=int(heads),
                dropout=float(dropout),
                use_norm=True,
            )
            for _ in range(int(layers))
        )
        self.energy_head = nn.Sequential(
            nn.Linear(int(type_count) * int(hidden_dim), int(hidden_dim)),
            nn.LayerNorm(int(hidden_dim)),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), 1),
        )

    def forward(
        self,
        node_features: torch.Tensor,
        node_types: torch.Tensor,
        edge_index: torch.Tensor,
        edge_types: torch.Tensor,
        graph_ids: torch.Tensor,
        graph_count: int,
    ) -> torch.Tensor:
        hidden = self.node_input(node_features)
        for layer in self.layers:
            hidden = layer(hidden, node_types, edge_index, edge_types)
        pooled = []
        for graph_id in range(int(graph_count)):
            by_type = []
            graph_mask = graph_ids == graph_id
            for type_id in range(int(self.config["type_count"])):
                mask = graph_mask & (node_types == type_id)
                if mask.any():
                    by_type.append(hidden[mask].mean(dim=0))
                else:
                    by_type.append(torch.zeros_like(hidden[0]))
            pooled.append(torch.cat(by_type, dim=0))
        return self.energy_head(torch.stack(pooled)).squeeze(-1)


def _graph_batch(
    graphs: Sequence[TypedGraph], device: torch.device
) -> Tuple[torch.Tensor, ...]:
    node_features = []
    node_types = []
    graph_ids = []
    edge_indices = []
    edge_types = []
    offset = 0
    for graph_id, graph in enumerate(graphs):
        # Drop time: otherwise NCE can solve the task without reading structure.
        node_features.append(graph_node_features(graph, 0.0)[:, :-1])
        node_types.append(graph.node_types)
        graph_ids.append(np.full(graph.num_nodes, graph_id, dtype=np.int64))
        edges, relations = graph_edge_tensors(graph)
        edge_indices.append(edges + offset)
        edge_types.append(relations)
        offset += graph.num_nodes
    return (
        torch.from_numpy(np.concatenate(node_features)).float().to(device),
        torch.from_numpy(np.concatenate(node_types)).long().to(device),
        torch.from_numpy(np.concatenate(edge_indices, axis=1)).long().to(device),
        torch.from_numpy(np.concatenate(edge_types)).long().to(device),
        torch.from_numpy(np.concatenate(graph_ids)).long().to(device),
    )


def graph_energies(
    model: TypedQuotientEnergy,
    graphs: Sequence[TypedGraph],
    device: str | torch.device,
) -> torch.Tensor:
    if not graphs:
        raise ValueError("cannot score an empty graph batch")
    device = torch.device(device)
    tensors = _graph_batch(graphs, device)
    return model(*tensors, graph_count=len(graphs))


def _corrupt_graph(
    graph: TypedGraph,
    action: str,
    steps: int,
    rng: np.random.Generator,
) -> TypedGraph:
    corrupted = graph.clone()
    applied = 0
    attempts = 0
    while applied < int(steps) and attempts < max(256, int(steps) * 64):
        attempts += 1
        proposal = sample_valid_edit(corrupted, rng, actions=(str(action),))
        if proposal is None:
            continue
        apply_edit(corrupted, proposal)
        applied += 1
    if applied < int(steps):
        raise RuntimeError(f"applied only {applied}/{steps} {action} corruptions")
    return corrupted


def build_quotient_energy_dataset(
    source: TypedGraph,
    prior_sampler: Callable[[int], TypedGraph],
    trajectories: int,
    noise_fractions: Sequence[float],
    seed: int,
) -> QuotientEnergyDataset:
    if int(trajectories) < 2:
        raise ValueError("quotient energy training needs at least two trajectories")
    fractions = tuple(float(value) for value in noise_fractions)
    if not fractions or any(value <= 0.0 for value in fractions):
        raise ValueError("noise fractions must be positive")
    step_schedule = {1}
    step_schedule.update(
        max(1, int(round(value * source.num_edges))) for value in fractions
    )
    normalized_schedule = tuple(
        (steps, float(steps) / float(max(source.num_edges, 1)))
        for steps in sorted(step_schedule)
    )
    examples: List[QuotientEnergyExample] = []
    for trajectory_id in range(int(trajectories)):
        rng = np.random.default_rng(int(seed) + 1_000_003 * trajectory_id)
        clean = permute_typed_graph_within_types(
            source, int(seed) + 20_000_033 + trajectory_id
        )
        examples.append(
            QuotientEnergyExample(clean, True, "clean_orbit", 0.0, trajectory_id)
        )
        prior = prior_sampler(int(seed) + 30_000_059 + trajectory_id)
        examples.append(
            QuotientEnergyExample(prior, False, "prior", 1.0, trajectory_id)
        )
        for steps, fraction in normalized_schedule:
            examples.append(
                QuotientEnergyExample(
                    _corrupt_graph(clean, "relocation", steps, rng),
                    False,
                    "relocation",
                    fraction,
                    trajectory_id,
                )
            )
            examples.append(
                QuotientEnergyExample(
                    _corrupt_graph(clean, "switch", steps, rng),
                    False,
                    "switch",
                    fraction,
                    trajectory_id,
                )
            )
    return QuotientEnergyDataset(
        tuple(examples), len(source.type_names), len(source.family_specs)
    )


def _ranking_report(
    model: TypedQuotientEnergy,
    examples: Sequence[QuotientEnergyExample],
    device: torch.device,
) -> Mapping[str, float]:
    model.eval()
    with torch.no_grad():
        energies = graph_energies(model, [item.graph for item in examples], device)
    values = energies.detach().cpu().numpy()
    clean_values = values[[item.clean for item in examples]]
    report: Dict[str, float] = {
        "clean_energy_mean": float(clean_values.mean()),
        "clean_energy_std": float(clean_values.std()),
    }
    correct = []
    clean_reference = float(clean_values.mean())
    for kind in sorted({item.negative_kind for item in examples if not item.clean}):
        kind_mask = [
            not item.clean and item.negative_kind == kind for item in examples
        ]
        kind_values = values[kind_mask]
        accuracy = float(np.mean(clean_reference < kind_values))
        report[f"{kind}_ranking_accuracy"] = accuracy
        report[f"{kind}_energy_margin"] = float(
            kind_values.mean() - clean_reference
        )
        correct.extend((clean_reference < kind_values).tolist())
        fractions = sorted(
            {
                item.noise_fraction
                for item in examples
                if not item.clean and item.negative_kind == kind
            }
        )
        for fraction in fractions:
            fraction_values = values[
                [
                    not item.clean
                    and item.negative_kind == kind
                    and item.noise_fraction == fraction
                    for item in examples
                ]
            ]
            report[f"{kind}_ranking_accuracy_at_{fraction:.3f}"] = float(
                np.mean(clean_reference < fraction_values)
            )
            report[f"{kind}_energy_margin_at_{fraction:.3f}"] = float(
                fraction_values.mean() - clean_reference
            )
    report["overall_ranking_accuracy"] = float(np.mean(correct))
    return report


def train_quotient_energy(
    dataset: QuotientEnergyDataset,
    epochs: int,
    hidden_dim: int = 96,
    heads: int = 4,
    layers: int = 2,
    dropout: float = 0.1,
    learning_rate: float = 3.0e-4,
    weight_decay: float = 1.0e-4,
    margin: float = 1.0,
    device: str | torch.device = "cuda",
    seed: int = 0,
    initial_model: Optional[TypedQuotientEnergy] = None,
) -> Tuple[TypedQuotientEnergy, Mapping[str, float]]:
    torch.manual_seed(int(seed))
    device = torch.device(device)
    first = dataset.examples[0].graph
    if initial_model is None:
        model = TypedQuotientEnergy(
            node_feature_dim=graph_node_features(first, 0.0).shape[1] - 1,
            type_count=dataset.type_count,
            family_count=dataset.family_count,
            hidden_dim=hidden_dim,
            heads=heads,
            layers=layers,
            dropout=dropout,
        ).to(device)
    else:
        model = initial_model.to(device)
        if int(model.config["type_count"]) != int(dataset.type_count) or int(
            model.config["family_count"]
        ) != int(dataset.family_count):
            raise ValueError("initial energy model and dataset schemas differ")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay)
    )
    validation_id = max(item.trajectory_id for item in dataset.examples)
    train_ids = sorted(
        {item.trajectory_id for item in dataset.examples if item.trajectory_id != validation_id}
    )
    rng = np.random.default_rng(int(seed))
    final_loss = float("nan")
    for epoch in range(int(epochs)):
        model.train()
        losses = []
        for trajectory_id in rng.permutation(train_ids).tolist():
            group = [
                item for item in dataset.examples if item.trajectory_id == trajectory_id
            ]
            clean = [item for item in group if item.clean]
            negatives = [item for item in group if not item.clean]
            if len(clean) != 1 or not negatives:
                raise AssertionError("each energy trajectory needs one clean graph")
            optimizer.zero_grad(set_to_none=True)
            energies = graph_energies(
                model, [clean[0].graph] + [item.graph for item in negatives], device
            )
            ranking = nn.functional.softplus(
                energies[0] - energies[1:] + float(margin)
            ).mean()
            centered = energies.square().mean() * 1.0e-4
            loss = ranking + centered
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        final_loss = float(np.mean(losses))
        if epoch == 0 or epoch == int(epochs) - 1 or (epoch + 1) % 5 == 0:
            print(
                f"[QUOTIENT-ENERGY-EPOCH] epoch={epoch} loss={final_loss:.6f}",
                flush=True,
            )
    validation = [
        item for item in dataset.examples if item.trajectory_id == validation_id
    ]
    report = dict(_ranking_report(model, validation, device))
    report.update(
        train_loss=final_loss,
        epochs=int(epochs),
        train_trajectories=len(train_ids),
        validation_trajectory=int(validation_id),
    )
    return model, report


def add_on_policy_replay_negatives(
    dataset: QuotientEnergyDataset,
    model: TypedQuotientEnergy,
    prior_sampler: Callable[[int], TypedGraph],
    steps: int,
    candidates_per_step: int,
    seed: int,
    device: str | torch.device = "cuda",
) -> QuotientEnergyDataset:
    """Mine low-energy states without consulting clean metrics or adjacency."""

    trajectory_ids = sorted({item.trajectory_id for item in dataset.examples})
    replay: List[QuotientEnergyExample] = []
    for trajectory_id in trajectory_ids:
        initial = prior_sampler(int(seed) + 40_000_081 + trajectory_id)
        result = sample_quotient_energy_edits(
            initial,
            model,
            steps=int(steps),
            candidates_per_step=int(candidates_per_step),
            seed=int(seed) + 50_000_111 + trajectory_id,
            device=device,
            metrics_every=0,
        )
        replay.append(
            QuotientEnergyExample(
                result.graph,
                False,
                "model_replay",
                1.0,
                trajectory_id,
            )
        )
    return QuotientEnergyDataset(
        tuple(dataset.examples) + tuple(replay),
        dataset.type_count,
        dataset.family_count,
    )


def calibrate_minimum_energy_drop(
    model: TypedQuotientEnergy,
    clean_graph: TypedGraph,
    trials_per_action: int,
    seed: int,
    device: str | torch.device = "cuda",
    quantile: float = 1.0,
) -> Mapping[str, float]:
    """Calibrate a target-free trust threshold on held-out clean-orbit neighbors."""

    if not 0.5 <= float(quantile) <= 1.0:
        raise ValueError("energy-drop calibration quantile must be in [0.5, 1]")
    rng = np.random.default_rng(int(seed))
    clean = permute_typed_graph_within_types(clean_graph, int(seed) + 7_919)
    neighbors = []
    actions = []
    for action in ("relocation", "switch"):
        for _ in range(int(trials_per_action)):
            proposal = sample_valid_edit(clean, rng, actions=(action,))
            if proposal is None:
                continue
            neighbor = clean.clone()
            apply_edit(neighbor, proposal)
            neighbors.append(neighbor)
            actions.append(action)
    if not neighbors:
        raise RuntimeError("energy-drop calibration found no legal neighbors")
    model.eval()
    with torch.no_grad():
        energies = graph_energies(model, [clean] + neighbors, device).detach().cpu().numpy()
    drops = energies[0] - energies[1:]
    threshold = max(float(np.quantile(drops, float(quantile))), 0.0) + 1.0e-6
    report: Dict[str, float] = {
        "recommended_minimum_energy_drop": threshold,
        "calibration_drop_quantile": float(quantile),
        "calibration_trials": float(len(drops)),
        "calibration_max_spurious_drop": float(drops.max()),
    }
    for action in sorted(set(actions)):
        selected = drops[np.asarray(actions) == action]
        report[f"calibration_{action}_max_spurious_drop"] = float(selected.max())
    return report


def sample_quotient_energy_edits(
    initial_graph: TypedGraph,
    model: TypedQuotientEnergy,
    steps: int,
    candidates_per_step: int,
    seed: int,
    actions: Sequence[str] = ("relocation", "switch"),
    device: str | torch.device = "cuda",
    metrics_every: int = 100,
    minimum_energy_drop: float = 0.0,
    noop_patience: int = 0,
) -> MixedEditSamplingResult:
    graph = initial_graph.clone()
    rng = np.random.default_rng(int(seed))
    device = torch.device(device)
    model = model.to(device).eval()
    trace = [{"step": 0.0, **graph_metrics(graph)}]
    action_counts = {action: 0 for action in ("birth", "death", "relocation", "switch")}
    action_counts["noop"] = 0
    applied = 0
    proposed = 0
    noop_streak = 0
    for step in range(1, int(steps) + 1):
        candidates, _ = sample_edit_candidates(
            graph, rng, int(candidates_per_step), actions=actions
        )
        candidate_graphs = []
        for proposal in candidates:
            candidate = graph.clone()
            apply_edit(candidate, proposal)
            candidate_graphs.append(candidate)
        with torch.no_grad():
            energies = graph_energies(model, candidate_graphs, device)
        noop_index = next(
            index for index, proposal in enumerate(candidates) if proposal.action == "noop"
        )
        selected = int(energies.argmin().item())
        if float(energies[noop_index] - energies[selected]) <= float(minimum_energy_drop):
            selected = noop_index
        proposal = candidates[selected]
        apply_edit(graph, proposal)
        action_counts[proposal.action] += 1
        applied += int(proposal.action != "noop")
        proposed += len(candidates)
        noop_streak = noop_streak + 1 if proposal.action == "noop" else 0
        if step == int(steps) or (int(metrics_every) > 0 and step % int(metrics_every) == 0):
            trace.append(
                {
                    "step": float(step),
                    "energy": float(energies[selected].detach().cpu()),
                    **graph_metrics(graph),
                }
            )
        if int(noop_patience) > 0 and noop_streak >= int(noop_patience):
            if not trace or trace[-1]["step"] != float(step):
                trace.append(
                    {
                        "step": float(step),
                        "energy": float(energies[selected].detach().cpu()),
                        "early_stop_noop_streak": float(noop_streak),
                        **graph_metrics(graph),
                    }
                )
            break
    graph.validate()
    return MixedEditSamplingResult(graph, trace, action_counts, proposed, applied)


def save_quotient_energy_checkpoint(
    path: str | Path,
    model: TypedQuotientEnergy,
    report: Mapping[str, float],
) -> None:
    payload = {
        "format": "dihug-typed-quotient-energy-v1",
        "model_config": dict(model.config),
        "state_dict": model.state_dict(),
        "train_report": dict(report),
        "inference_contract": {
            "training_adjacency": False,
            "target_metrics": False,
            "node_identifiers": False,
        },
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_quotient_energy_checkpoint(
    path: str | Path, device: str | torch.device = "cpu"
) -> Tuple[TypedQuotientEnergy, Mapping]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != "dihug-typed-quotient-energy-v1":
        raise ValueError("unsupported quotient energy checkpoint")
    model = TypedQuotientEnergy(**payload["model_config"])
    model.load_state_dict(payload["state_dict"])
    return model.to(device), payload


def write_quotient_report(path: str | Path, payload: Mapping) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True), encoding="utf-8")
