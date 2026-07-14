from __future__ import annotations

from dataclasses import dataclass
from typing import List, Mapping, Optional, Sequence

import numpy as np
from scipy.optimize import minimize

from .diagnostics import effective_sample_size, half_drift_z_score, split_r_hat
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
    weight_version: int
    samples_carried_in: int
    sample_batches: int
    samples_per_chain: int
    feature_ess_min: float
    feature_ess_q10: float
    feature_split_rhat_max: float
    feature_split_rhat_q90: float
    feature_relative_drift_max: float
    feature_relative_drift_q90: float
    feature_drift_z_max: float
    feature_drift_z_q90: float
    expectation_converged: bool
    update_method: str
    step_size: float
    update_direction_norm: float
    actual_update_norm: float
    approximate_kl: float
    fisher_condition: float
    covariance_rank: int
    weight_at_bound_fraction: float
    continuation_stage: int
    continuation_stages: int
    continuation_fraction: float
    stage_moment_mae: float

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
            "weight_version": int(self.weight_version),
            "samples_carried_in": int(self.samples_carried_in),
            "sample_batches": int(self.sample_batches),
            "samples_per_chain": int(self.samples_per_chain),
            "feature_ess_min": float(self.feature_ess_min),
            "feature_ess_q10": float(self.feature_ess_q10),
            "feature_split_rhat_max": float(self.feature_split_rhat_max),
            "feature_split_rhat_q90": float(self.feature_split_rhat_q90),
            "feature_relative_drift_max": float(self.feature_relative_drift_max),
            "feature_relative_drift_q90": float(self.feature_relative_drift_q90),
            "feature_drift_z_max": float(self.feature_drift_z_max),
            "feature_drift_z_q90": float(self.feature_drift_z_q90),
            "expectation_converged": bool(self.expectation_converged),
            "update_method": self.update_method,
            "step_size": float(self.step_size),
            "update_direction_norm": float(self.update_direction_norm),
            "actual_update_norm": float(self.actual_update_norm),
            "approximate_kl": float(self.approximate_kl),
            "fisher_condition": float(self.fisher_condition),
            "covariance_rank": int(self.covariance_rank),
            "weight_at_bound_fraction": float(self.weight_at_bound_fraction),
            "continuation_stage": int(self.continuation_stage),
            "continuation_stages": int(self.continuation_stages),
            "continuation_fraction": float(self.continuation_fraction),
            "stage_moment_mae": float(self.stage_moment_mae),
        }


@dataclass(frozen=True)
class MomentFitReport:
    converged: bool
    rounds: int
    target_vector: np.ndarray
    final_model_vector: np.ndarray
    trajectory: Sequence[MomentFitRound]
    successful_updates: int
    reached_continuation_fraction: float

    def to_json_dict(self) -> dict:
        return {
            "converged": bool(self.converged),
            "rounds": int(self.rounds),
            "target_vector": self.target_vector.tolist(),
            "final_model_vector": self.final_model_vector.tolist(),
            "trajectory": [item.to_json_dict() for item in self.trajectory],
            "successful_updates": int(self.successful_updates),
            "reached_continuation_fraction": float(
                self.reached_continuation_fraction
            ),
        }


def _continuation_schedule(stages: int, start_fraction: float) -> np.ndarray:
    """Return monotone moment targets from the null model to the real graph."""

    stages = int(stages)
    if stages <= 1:
        return np.asarray([1.0], dtype=np.float64)
    if not 0.0 < float(start_fraction) < 1.0:
        raise ValueError("continuation_start_fraction must be in (0, 1)")
    levels = np.geomspace(float(start_fraction), 1.0, stages)
    levels[-1] = 1.0
    return levels


def _relative_moment_error(
    model: np.ndarray, target: np.ndarray, anchor: np.ndarray
) -> tuple[float, float]:
    """Measure error relative to the structural displacement from the null."""

    residual = np.asarray(target) - np.asarray(model)
    scale = np.maximum(np.abs(np.asarray(target) - np.asarray(anchor)), 1.0)
    relative = np.abs(residual) / scale
    return float(relative.mean()), float(relative.max(initial=0.0))


def _empirical_quantile(
    values: np.ndarray, quantile: float, *, conservative_upper: bool
) -> float:
    """Return a finite-sample quantile without interpolating through infinity."""

    ordered = np.sort(np.asarray(values, dtype=np.float64))
    if ordered.size == 0:
        return float("nan")
    position = float(quantile) * float(ordered.size - 1)
    index = int(np.ceil(position) if conservative_upper else np.floor(position))
    return float(ordered[min(max(index, 0), ordered.size - 1)])


def _feature_chain_diagnostics(
    chains: Sequence[Sequence[np.ndarray]], target: np.ndarray, quantile: float
) -> dict:
    """Summarize frozen-weight mixing over every selected feature."""

    if not chains or any(not chain for chain in chains):
        return {
            "ess_min": 0.0,
            "ess_q10": 0.0,
            "split_rhat_max": float("inf"),
            "split_rhat_q90": float("inf"),
            "drift_max": float("inf"),
            "drift_q90": float("inf"),
            "drift_z_max": float("inf"),
            "drift_z_q90": float("inf"),
        }
    width = int(target.size)
    ess_values = []
    rhat_values = []
    drift_values = []
    drift_z_values = []
    for feature_id in range(width):
        traces = [
            [float(vector[feature_id]) for vector in chain]
            for chain in chains
        ]
        ess_values.extend(effective_sample_size(trace) for trace in traces)
        feature_rhat = split_r_hat(traces)
        rhat_values.append(feature_rhat)
        early = []
        late = []
        for trace in traces:
            half = len(trace) // 2
            if half < 2:
                continue
            early.extend(trace[:half])
            late.extend(trace[-half:])
        if not early or not late:
            drift_values.append(float("inf"))
            drift_z_values.append(float("inf"))
        else:
            difference = abs(float(np.mean(late)) - float(np.mean(early)))
            scale = max(abs(float(target[feature_id])), 1.0)
            drift_values.append(difference / scale)
            drift_z_values.append(half_drift_z_score(traces))
    rhat_array = np.asarray(rhat_values, dtype=np.float64)
    drift_array = np.asarray(drift_values, dtype=np.float64)
    drift_z_array = np.asarray(drift_z_values, dtype=np.float64)
    ess_array = np.asarray(ess_values, dtype=np.float64)
    upper = float(quantile)
    lower = 1.0 - upper
    return {
        "ess_min": float(np.min(ess_array)),
        "ess_q10": _empirical_quantile(
            ess_array, lower, conservative_upper=False
        ),
        "split_rhat_max": float(np.max(rhat_array, initial=0.0)),
        "split_rhat_q90": _empirical_quantile(
            rhat_array, upper, conservative_upper=True
        ),
        "drift_max": float(np.max(drift_array, initial=0.0)),
        "drift_q90": _empirical_quantile(
            drift_array, upper, conservative_upper=True
        ),
        "drift_z_max": float(np.max(drift_z_array, initial=0.0)),
        "drift_z_q90": _empirical_quantile(
            drift_z_array, upper, conservative_upper=True
        ),
    }


def _feature_covariance(
    samples: np.ndarray, shrinkage: float
) -> tuple[np.ndarray, int]:
    """Estimate a stable Fisher matrix from normalized sufficient statistics."""

    matrix = np.asarray(samples, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("feature samples must be a two-dimensional matrix")
    width = int(matrix.shape[1])
    if matrix.shape[0] < 2:
        return np.zeros((width, width), dtype=np.float64), 0
    covariance = np.atleast_2d(np.cov(matrix, rowvar=False, ddof=1)).astype(
        np.float64, copy=False
    )
    if covariance.shape != (width, width):
        covariance = covariance.reshape(width, width)
    covariance = 0.5 * (covariance + covariance.T)
    diagonal = np.diag(np.diag(covariance))
    covariance = (1.0 - float(shrinkage)) * covariance + float(
        shrinkage
    ) * diagonal
    covariance = np.nan_to_num(covariance, nan=0.0, posinf=0.0, neginf=0.0)
    return covariance, int(np.linalg.matrix_rank(covariance, tol=1.0e-10))


def _natural_gradient_direction(
    gradient: np.ndarray,
    covariance: np.ndarray,
    damping: float,
    l2: float,
) -> tuple[np.ndarray, float]:
    """Solve the damped Fisher system for an exponential-family ascent step."""

    width = int(gradient.size)
    system = np.asarray(covariance, dtype=np.float64) + (
        float(damping) + float(l2)
    ) * np.eye(width, dtype=np.float64)
    condition = float(np.linalg.cond(system))
    try:
        direction = np.linalg.solve(system, gradient)
    except np.linalg.LinAlgError:
        direction = np.linalg.pinv(system, rcond=1.0e-10) @ gradient
    return np.asarray(direction, dtype=np.float64), condition


def _trust_region_update(
    direction: np.ndarray,
    step_size: float,
    covariance: np.ndarray,
    max_norm: float,
    kl_radius: float,
) -> tuple[np.ndarray, float]:
    """Scale an update to satisfy Euclidean and local-KL trust regions."""

    update = float(step_size) * np.asarray(direction, dtype=np.float64)
    norm = float(np.linalg.norm(update))
    if float(max_norm) > 0.0 and norm > float(max_norm):
        update *= float(max_norm) / norm
    approximate_kl = max(
        0.0, float(0.5 * update @ np.asarray(covariance) @ update)
    )
    if float(kl_radius) > 0.0 and approximate_kl > float(kl_radius):
        scale = np.sqrt(float(kl_radius) / max(approximate_kl, 1.0e-300))
        update *= scale
        approximate_kl = max(
            0.0, float(0.5 * update @ np.asarray(covariance) @ update)
        )
    return update, approximate_kl


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
    max_batches_per_round: int = 16,
    min_samples_per_chain: int = 16,
    min_ess_per_chain: float = 8.0,
    rhat_threshold: float = 1.10,
    drift_z_threshold: float = 2.0,
    warmup_steps_per_round: int = 0,
    step_decay: float = 0.60,
    update_norm_clip: float = 0.5,
    diagnostic_quantile: float = 0.90,
    max_retained_samples_per_chain: int = 256,
    update_method: str = "diagonal",
    natural_gradient_damping: float = 0.05,
    covariance_shrinkage: float = 0.10,
    kl_trust_radius: float = 0.0,
    target_anchor_vector: Optional[np.ndarray] = None,
    continuation_stages: int = 0,
    continuation_start_fraction: float = 0.05,
    continuation_stage_tolerance: float = 0.25,
) -> tuple[LinearGraphletEnergy, MomentFitReport, List[TypedGraph]]:
    """Fit a fixed-degree maximum-entropy model by persistent MCMC.

    For frozen ``w``, the population negative log likelihood is convex and has
    gradient ``E_w[z(G)] - z(G_real)``. The implementation performs the
    equivalent log-likelihood ascent update ``z(G_real) - E_w[z(G)]``. Samples
    are retained while ``w`` is unchanged and cleared only after an accepted
    update, so every diagnostic is computed from one frozen target
    distribution. The returned energy is frozen and must be validated with
    fresh chains.
    """

    if not initial_graphs:
        raise ValueError("moment fitting needs at least one persistent chain")
    if int(rounds) <= 0 or int(steps_per_round) <= 0:
        raise ValueError("moment fitting rounds and steps must be positive")
    if int(samples_per_round) <= 0:
        raise ValueError("samples_per_round must be positive")
    if int(max_batches_per_round) <= 0:
        raise ValueError("max_batches_per_round must be positive")
    if int(min_samples_per_chain) < 4:
        raise ValueError("min_samples_per_chain must be at least four")
    if not 0.5 < float(step_decay) <= 1.0:
        raise ValueError("step_decay must be in (0.5, 1]")
    if not 0.5 <= float(diagnostic_quantile) < 1.0:
        raise ValueError("diagnostic_quantile must be in [0.5, 1)")
    if int(max_retained_samples_per_chain) < int(min_samples_per_chain):
        raise ValueError(
            "max_retained_samples_per_chain must be at least "
            "min_samples_per_chain"
        )
    update_method = str(update_method).strip().lower()
    if update_method not in {"diagonal", "natural"}:
        raise ValueError("update_method must be 'diagonal' or 'natural'")
    if float(natural_gradient_damping) <= 0.0:
        raise ValueError("natural_gradient_damping must be positive")
    if not 0.0 <= float(covariance_shrinkage) <= 1.0:
        raise ValueError("covariance_shrinkage must be in [0, 1]")
    if float(kl_trust_radius) < 0.0:
        raise ValueError("kl_trust_radius must be nonnegative")

    target_vector = selection.vector(target)
    if target_anchor_vector is None:
        target_anchor = np.zeros_like(target_vector)
    else:
        target_anchor = np.asarray(target_anchor_vector, dtype=np.float64).copy()
        if target_anchor.shape != target_vector.shape:
            raise ValueError("target_anchor_vector has the wrong shape")
    if not np.all(np.isfinite(target_anchor)):
        raise ValueError("target_anchor_vector must be finite")
    if float(continuation_stage_tolerance) <= 0.0:
        raise ValueError("continuation_stage_tolerance must be positive")
    continuation_levels = _continuation_schedule(
        continuation_stages, continuation_start_fraction
    )
    continuation_stage = 0
    if initial_weights is None:
        weights = np.zeros(len(selection.keys), dtype=np.float64)
    else:
        weights = np.asarray(initial_weights, dtype=np.float64).copy()
        if weights.shape != target_vector.shape:
            raise ValueError("initial moment weights have the wrong shape")
    weights = np.clip(weights, -float(max_weight), float(max_weight))

    graphs = [graph.clone() for graph in initial_graphs]
    trajectory: List[MomentFitRound] = []
    final_model_vector = np.zeros_like(target_vector)
    converged = False
    successful_updates = 0
    vector_chains: List[List[np.ndarray]] = [[] for _ in graphs]
    metric_chains: List[List[Mapping[str, float]]] = [[] for _ in graphs]
    weights_need_warmup = True

    thin = max(int(steps_per_round) // int(samples_per_round), 1)
    burn_in = thin
    for round_id in range(int(rounds)):
        continuation_fraction = float(continuation_levels[continuation_stage])
        stage_target_vector = target_anchor + continuation_fraction * (
            target_vector - target_anchor
        )
        energy = LinearGraphletEnergy(selection, weights)
        samples_carried_in = min((len(chain) for chain in vector_chains), default=0)
        acceptance_rates = []
        valid_rates = []
        if weights_need_warmup and int(warmup_steps_per_round) > 0:
            warmed = []
            for chain_id, graph in enumerate(graphs):
                result = run_switch_chain(
                    graph,
                    GraphletFeatureExtractor(graph, tracked_keys=selection.keys),
                    steps=int(warmup_steps_per_round),
                    burn_in=int(warmup_steps_per_round) + 1,
                    thin=max(int(warmup_steps_per_round), 1),
                    seed=int(seed) + 10_000_019 * round_id + 1009 * chain_id,
                    energy=energy,
                    verify_every=int(verify_every),
                    curveball_probability=float(curveball_probability),
                )
                warmed.append(result.graph)
            graphs = warmed
        weights_need_warmup = False

        expectation_converged = False
        feature_ess = 0.0
        feature_ess_q10 = 0.0
        feature_rhat = float("inf")
        feature_rhat_q90 = float("inf")
        feature_drift = float("inf")
        feature_drift_q90 = float("inf")
        feature_drift_z = float("inf")
        feature_drift_z_q90 = float("inf")
        batches_used = 0
        for batch_id in range(int(max_batches_per_round)):
            updated_graphs = []
            for chain_id, graph in enumerate(graphs):
                is_continuation = len(vector_chains[chain_id]) > 0
                result = run_switch_chain(
                    graph,
                    GraphletFeatureExtractor(graph, tracked_keys=selection.keys),
                    steps=int(steps_per_round),
                    burn_in=0
                    if is_continuation or int(warmup_steps_per_round) > 0
                    else burn_in,
                    thin=thin,
                    seed=(
                        int(seed)
                        + 100_003 * round_id
                        + 10_007 * batch_id
                        + 1009 * chain_id
                    ),
                    energy=energy,
                    verify_every=int(verify_every),
                    curveball_probability=float(curveball_probability),
                )
                updated_graphs.append(result.graph)
                acceptance_rates.append(result.acceptance_rate)
                valid_rates.append(result.valid_rate)
                vector_chains[chain_id].extend(
                    selection.vector(sample.features) for sample in result.samples
                )
                metric_chains[chain_id].extend(
                    sample.metrics for sample in result.samples
                )
                retained = int(max_retained_samples_per_chain)
                if len(vector_chains[chain_id]) > retained:
                    del vector_chains[chain_id][:-retained]
                    del metric_chains[chain_id][:-retained]
            graphs = updated_graphs
            batches_used = batch_id + 1
            samples_ready = min(len(chain) for chain in vector_chains)
            if samples_ready < int(min_samples_per_chain):
                continue
            diagnostics = _feature_chain_diagnostics(
                vector_chains, stage_target_vector, diagnostic_quantile
            )
            feature_ess = diagnostics["ess_min"]
            feature_ess_q10 = diagnostics["ess_q10"]
            feature_rhat = diagnostics["split_rhat_max"]
            feature_rhat_q90 = diagnostics["split_rhat_q90"]
            feature_drift = diagnostics["drift_max"]
            feature_drift_q90 = diagnostics["drift_q90"]
            feature_drift_z = diagnostics["drift_z_max"]
            feature_drift_z_q90 = diagnostics["drift_z_q90"]
            expectation_converged = bool(
                feature_ess_q10 >= float(min_ess_per_chain)
                and feature_rhat_q90 <= float(rhat_threshold)
                and feature_drift_z_q90 <= float(drift_z_threshold)
            )
            if expectation_converged:
                break

        sampled_vectors = [vector for chain in vector_chains for vector in chain]
        sampled_triangles = [
            float(sample["triangles"])
            for chain in metric_chains
            for sample in chain
        ]
        sampled_clustering = [
            float(sample["clustering"])
            for chain in metric_chains
            for sample in chain
        ]
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

        moment_mae, moment_max_error = _relative_moment_error(
            final_model_vector, target_vector, target_anchor
        )
        stage_moment_mae, _ = _relative_moment_error(
            final_model_vector, stage_target_vector, target_anchor
        )
        if expectation_converged:
            while (
                stage_moment_mae <= float(continuation_stage_tolerance)
                and continuation_stage + 1 < len(continuation_levels)
            ):
                continuation_stage += 1
                continuation_fraction = float(
                    continuation_levels[continuation_stage]
                )
                stage_target_vector = target_anchor + continuation_fraction * (
                    target_vector - target_anchor
                )
                stage_moment_mae, _ = _relative_moment_error(
                    final_model_vector, stage_target_vector, target_anchor
                )
        residual = stage_target_vector - final_model_vector
        gradient = residual - float(l2) * weights
        gradient_norm = float(np.linalg.norm(gradient))
        sample_matrix = np.stack(sampled_vectors)
        covariance, covariance_rank = _feature_covariance(
            sample_matrix, covariance_shrinkage
        )
        if update_method == "natural":
            update_direction, fisher_condition = _natural_gradient_direction(
                gradient,
                covariance,
                damping=natural_gradient_damping,
                l2=l2,
            )
        else:
            preconditioner = 1.0 / np.maximum(np.abs(target_vector), 1.0)
            update_direction = preconditioner * gradient
            fisher_condition = 1.0
        update_direction_norm = float(np.linalg.norm(update_direction))
        next_update = successful_updates + 1
        at_final_stage = continuation_stage + 1 == len(continuation_levels)
        active_tolerance = (
            float(tolerance)
            if at_final_stage
            else float(continuation_stage_tolerance)
        )
        can_update = bool(
            expectation_converged
            and stage_moment_mae > active_tolerance
            and round_id + 1 < int(rounds)
        )
        step_size = (
            float(learning_rate) / float(next_update) ** float(step_decay)
            if can_update
            else 0.0
        )
        effective_kl_radius = float(kl_trust_radius)
        if len(continuation_levels) > 1 and effective_kl_radius > 0.0:
            effective_kl_radius *= max(
                continuation_fraction, 1.0 / float(len(continuation_levels))
            )
        actual_update, approximate_kl = _trust_region_update(
            update_direction,
            step_size=step_size,
            covariance=covariance,
            max_norm=update_norm_clip,
            kl_radius=effective_kl_radius,
        )
        actual_update_norm = float(np.linalg.norm(actual_update))
        bound_fraction = float(
            np.mean(np.isclose(np.abs(weights), float(max_weight), atol=1.0e-8))
        )
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
                weight_version=successful_updates,
                samples_carried_in=samples_carried_in,
                sample_batches=batches_used,
                samples_per_chain=min((len(chain) for chain in vector_chains), default=0),
                feature_ess_min=feature_ess,
                feature_ess_q10=feature_ess_q10,
                feature_split_rhat_max=feature_rhat,
                feature_split_rhat_q90=feature_rhat_q90,
                feature_relative_drift_max=feature_drift,
                feature_relative_drift_q90=feature_drift_q90,
                feature_drift_z_max=feature_drift_z,
                feature_drift_z_q90=feature_drift_z_q90,
                expectation_converged=expectation_converged,
                update_method=update_method,
                step_size=step_size,
                update_direction_norm=update_direction_norm,
                actual_update_norm=actual_update_norm,
                approximate_kl=approximate_kl,
                fisher_condition=fisher_condition,
                covariance_rank=covariance_rank,
                weight_at_bound_fraction=bound_fraction,
                continuation_stage=continuation_stage,
                continuation_stages=len(continuation_levels),
                continuation_fraction=continuation_fraction,
                stage_moment_mae=stage_moment_mae,
            )
        )
        print(
            "[MOMENT-FIT] "
            f"round={round_id} mae={moment_mae:.4f} "
            f"stage_mae={stage_moment_mae:.4f} "
            f"stage={continuation_stage + 1}/{len(continuation_levels)} "
            f"rho={continuation_fraction:.4f} "
            f"max={moment_max_error:.4f} grad={gradient_norm:.3f} "
            f"accept={float(np.mean(acceptance_rates)):.3f} "
            f"tri={float(np.mean(sampled_triangles)) if sampled_triangles else 0.0:.1f} "
            f"cluster={float(np.mean(sampled_clustering)) if sampled_clustering else 0.0:.4f} "
            f"weight_norm={float(np.linalg.norm(weights)):.3f} "
            f"weight_version={successful_updates} carried={samples_carried_in} "
            f"batches={batches_used} samples={min((len(chain) for chain in vector_chains), default=0)} "
            f"ess_q10={feature_ess_q10:.2f} split_rhat_q90={feature_rhat_q90:.3f} "
            f"drift_q90={feature_drift_q90:.4f} "
            f"drift_z_q90={feature_drift_z_q90:.3f} "
            f"worst=({feature_ess:.2f},{feature_rhat:.3f},"
            f"{feature_drift:.4f},{feature_drift_z:.3f}) "
            f"expectation_ok={int(expectation_converged)} "
            f"method={update_method} step_size={step_size:.5f} "
            f"direction_norm={update_direction_norm:.3f} "
            f"update_norm={actual_update_norm:.3f} kl={approximate_kl:.4f} "
            f"fisher_cond={fisher_condition:.2e} cov_rank={covariance_rank} "
            f"bound={bound_fraction:.3f}",
            flush=True,
        )
        if (
            expectation_converged
            and at_final_stage
            and moment_mae <= float(tolerance)
        ):
            converged = True
            break
        if can_update:
            weights += actual_update
            weights = np.clip(weights, -float(max_weight), float(max_weight))
            successful_updates += 1
            vector_chains = [[] for _ in graphs]
            metric_chains = [[] for _ in graphs]
            weights_need_warmup = True
        elif not expectation_converged:
            print(
                "[MOMENT-FIT] update skipped: frozen-weight expectation "
                "did not pass ESS/split-Rhat/drift diagnostics",
                flush=True,
            )

    report = MomentFitReport(
        converged=converged,
        rounds=len(trajectory),
        target_vector=target_vector,
        final_model_vector=final_model_vector,
        trajectory=tuple(trajectory),
        successful_updates=successful_updates,
        reached_continuation_fraction=float(
            continuation_levels[continuation_stage]
        ),
    )
    return LinearGraphletEnergy(selection, weights), report, graphs
