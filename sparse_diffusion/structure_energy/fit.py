from __future__ import annotations

from dataclasses import dataclass
from typing import List, Mapping, Optional, Sequence

import numpy as np
from scipy.optimize import minimize

from .features import FeatureSelection, GraphletFeatureExtractor
from .graph import TypedGraph
from .switch_chain import LinearGraphletEnergy, run_switch_chain


@dataclass(frozen=True)
class FitReport:
    success: bool
    message: str
    iterations: int
    objective: float
    train_pair_accuracy: float
    heldout_pair_accuracy: float
    train_margin: float
    heldout_margin: float

    def to_json_dict(self) -> dict:
        return {
            "success": bool(self.success),
            "message": self.message,
            "iterations": int(self.iterations),
            "objective": float(self.objective),
            "train_pair_accuracy": float(self.train_pair_accuracy),
            "heldout_pair_accuracy": float(self.heldout_pair_accuracy),
            "train_margin": float(self.train_margin),
            "heldout_margin": float(self.heldout_margin),
        }


@dataclass(frozen=True)
class MomentFitRound:
    round: int
    moment_mae: float
    moment_max_error: float
    gradient_norm: float
    mean_acceptance: float
    mean_valid_rate: float
    weight_norm: float
    mean_triangles: float
    mean_clustering: float

    def to_json_dict(self) -> dict:
        return {
            "round": int(self.round),
            "moment_mae": float(self.moment_mae),
            "moment_max_error": float(self.moment_max_error),
            "gradient_norm": float(self.gradient_norm),
            "mean_acceptance": float(self.mean_acceptance),
            "mean_valid_rate": float(self.mean_valid_rate),
            "weight_norm": float(self.weight_norm),
            "mean_triangles": float(self.mean_triangles),
            "mean_clustering": float(self.mean_clustering),
        }


@dataclass(frozen=True)
class MomentFitReport:
    converged: bool
    rounds: int
    target_vector: np.ndarray
    final_model_vector: np.ndarray
    trajectory: Sequence[MomentFitRound]

    def to_json_dict(self) -> dict:
        return {
            "converged": bool(self.converged),
            "rounds": int(self.rounds),
            "target_vector": self.target_vector.tolist(),
            "final_model_vector": self.final_model_vector.tolist(),
            "trajectory": [item.to_json_dict() for item in self.trajectory],
        }


def _pair_metrics(delta: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    if delta.shape[0] == 0:
        return 0.0, 0.0
    margin = delta @ weights
    return float(np.mean(margin > 0.0)), float(np.mean(margin))


def fit_pairwise_nce(
    real: Mapping[str, int | float],
    null_samples: Sequence[Mapping[str, int | float]],
    selection: FeatureSelection,
    heldout_samples: Optional[Sequence[Mapping[str, int | float]]] = None,
    l2: float = 0.1,
    heldout_fraction: float = 0.25,
    max_weight: float = 8.0,
    seed: int = 0,
) -> tuple[LinearGraphletEnergy, FitReport]:
    if len(null_samples) < 2:
        raise ValueError("at least two null samples are required for NCE")
    real_vector = selection.vector(real)
    null_matrix = np.stack([selection.vector(sample) for sample in null_samples])
    if heldout_samples is not None:
        if not heldout_samples:
            raise ValueError("explicit NCE heldout samples are empty")
        train_delta = real_vector[None, :] - null_matrix
        heldout_matrix = np.stack(
            [selection.vector(sample) for sample in heldout_samples]
        )
        heldout_delta = real_vector[None, :] - heldout_matrix
    else:
        rng = np.random.default_rng(int(seed))
        order = rng.permutation(len(null_samples))
        heldout_count = max(1, int(round(len(order) * float(heldout_fraction))))
        heldout_idx = order[:heldout_count]
        train_idx = order[heldout_count:]
        if train_idx.size == 0:
            raise ValueError("heldout split left no NCE training samples")
        train_delta = real_vector[None, :] - null_matrix[train_idx]
        heldout_delta = real_vector[None, :] - null_matrix[heldout_idx]

    def objective(weights: np.ndarray) -> tuple[float, np.ndarray]:
        margins = train_delta @ weights
        loss = np.logaddexp(0.0, -margins).mean()
        loss += 0.5 * float(l2) * float(np.dot(weights, weights))
        negative_sigmoid = 1.0 / (1.0 + np.exp(np.clip(margins, -60.0, 60.0)))
        gradient = -(negative_sigmoid[:, None] * train_delta).mean(axis=0)
        gradient += float(l2) * weights
        return float(loss), gradient

    initial = np.zeros(len(selection.keys), dtype=np.float64)
    result = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        jac=True,
        bounds=[(-float(max_weight), float(max_weight))] * len(selection.keys),
    )
    weights = np.asarray(result.x, dtype=np.float64)
    train_accuracy, train_margin = _pair_metrics(train_delta, weights)
    heldout_accuracy, heldout_margin = _pair_metrics(heldout_delta, weights)
    energy = LinearGraphletEnergy(selection, weights)
    report = FitReport(
        success=bool(result.success),
        message=str(result.message),
        iterations=int(result.nit),
        objective=float(result.fun),
        train_pair_accuracy=train_accuracy,
        heldout_pair_accuracy=heldout_accuracy,
        train_margin=train_margin,
        heldout_margin=heldout_margin,
    )
    return energy, report


def fit_moment_matching_energy(
    target: Mapping[str, int | float],
    selection: FeatureSelection,
    initial_graphs: Sequence[TypedGraph],
    rounds: int = 24,
    steps_per_round: int = 5_000,
    samples_per_round: int = 4,
    learning_rate: float = 0.2,
    l2: float = 0.01,
    max_weight: float = 8.0,
    tolerance: float = 0.10,
    seed: int = 0,
    initial_weights: Optional[np.ndarray] = None,
    verify_every: int = 0,
    curveball_probability: float = 0.0,
) -> tuple[LinearGraphletEnergy, MomentFitReport, List[TypedGraph]]:
    """Fit a fixed-degree maximum-entropy model by persistent MCMC.

    For frozen ``w``, the population negative log likelihood is convex and has
    gradient ``E_w[z(G)] - z(G_real)``. The chains below estimate that
    expectation; Adam is used only as a stochastic optimizer. The returned
    energy is frozen and must be validated with fresh chains.
    """

    if not initial_graphs:
        raise ValueError("moment fitting needs at least one persistent chain")
    if int(rounds) <= 0 or int(steps_per_round) <= 0:
        raise ValueError("moment fitting rounds and steps must be positive")
    if int(samples_per_round) <= 0:
        raise ValueError("samples_per_round must be positive")

    target_vector = selection.vector(target)
    if initial_weights is None:
        weights = np.zeros(len(selection.keys), dtype=np.float64)
    else:
        weights = np.asarray(initial_weights, dtype=np.float64).copy()
        if weights.shape != target_vector.shape:
            raise ValueError("initial moment weights have the wrong shape")
    weights = np.clip(weights, -float(max_weight), float(max_weight))

    graphs = [graph.clone() for graph in initial_graphs]
    first_moment = np.zeros_like(weights)
    second_moment = np.zeros_like(weights)
    beta1, beta2 = 0.9, 0.999
    epsilon = 1.0e-8
    trajectory: List[MomentFitRound] = []
    final_model_vector = np.zeros_like(target_vector)
    converged = False

    thin = max(int(steps_per_round) // int(samples_per_round), 1)
    burn_in = thin
    for round_id in range(int(rounds)):
        energy = LinearGraphletEnergy(selection, weights)
        sampled_vectors = []
        acceptance_rates = []
        valid_rates = []
        sampled_triangles = []
        sampled_clustering = []
        updated_graphs = []
        for chain_id, graph in enumerate(graphs):
            result = run_switch_chain(
                graph,
                GraphletFeatureExtractor(graph, tracked_keys=selection.keys),
                steps=int(steps_per_round),
                burn_in=burn_in,
                thin=thin,
                seed=int(seed) + 100_003 * round_id + 1009 * chain_id,
                energy=energy,
                verify_every=int(verify_every),
                curveball_probability=float(curveball_probability),
            )
            updated_graphs.append(result.graph)
            acceptance_rates.append(result.acceptance_rate)
            valid_rates.append(result.valid_rate)
            sampled_vectors.extend(
                selection.vector(sample.features) for sample in result.samples
            )
            sampled_triangles.extend(
                float(sample.metrics["triangles"]) for sample in result.samples
            )
            sampled_clustering.extend(
                float(sample.metrics["clustering"]) for sample in result.samples
            )
        graphs = updated_graphs
        if not sampled_vectors:
            sampled_vectors = [
                selection.vector(
                    GraphletFeatureExtractor(
                        graph, tracked_keys=selection.keys
                    ).extract(graph)
                )
                for graph in graphs
            ]
        final_model_vector = np.mean(np.stack(sampled_vectors), axis=0)

        residual = target_vector - final_model_vector
        relative = np.abs(residual) / np.maximum(np.abs(target_vector), 1.0)
        moment_mae = float(relative.mean())
        moment_max_error = float(relative.max(initial=0.0))
        gradient = residual - float(l2) * weights
        gradient_norm = float(np.linalg.norm(gradient))
        trajectory.append(
            MomentFitRound(
                round=round_id,
                moment_mae=moment_mae,
                moment_max_error=moment_max_error,
                gradient_norm=gradient_norm,
                mean_acceptance=float(np.mean(acceptance_rates)),
                mean_valid_rate=float(np.mean(valid_rates)),
                weight_norm=float(np.linalg.norm(weights)),
                mean_triangles=float(np.mean(sampled_triangles))
                if sampled_triangles
                else 0.0,
                mean_clustering=float(np.mean(sampled_clustering))
                if sampled_clustering
                else 0.0,
            )
        )
        print(
            "[MOMENT-FIT] "
            f"round={round_id} mae={moment_mae:.4f} "
            f"max={moment_max_error:.4f} grad={gradient_norm:.3f} "
            f"accept={float(np.mean(acceptance_rates)):.3f} "
            f"tri={float(np.mean(sampled_triangles)) if sampled_triangles else 0.0:.1f} "
            f"cluster={float(np.mean(sampled_clustering)) if sampled_clustering else 0.0:.4f} "
            f"weight_norm={float(np.linalg.norm(weights)):.3f}",
            flush=True,
        )
        if moment_mae <= float(tolerance):
            converged = True
            break

        step_index = round_id + 1
        first_moment = beta1 * first_moment + (1.0 - beta1) * gradient
        second_moment = beta2 * second_moment + (1.0 - beta2) * np.square(gradient)
        corrected_first = first_moment / (1.0 - beta1**step_index)
        corrected_second = second_moment / (1.0 - beta2**step_index)
        weights += float(learning_rate) * corrected_first / (
            np.sqrt(corrected_second) + epsilon
        )
        weights = np.clip(weights, -float(max_weight), float(max_weight))

    report = MomentFitReport(
        converged=converged,
        rounds=len(trajectory),
        target_vector=target_vector,
        final_model_vector=final_model_vector,
        trajectory=tuple(trajectory),
    )
    return LinearGraphletEnergy(selection, weights), report, graphs
