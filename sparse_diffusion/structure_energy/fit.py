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
    step_size: float
    update_direction_norm: float
    weight_at_bound_fraction: float

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
            "step_size": float(self.step_size),
            "update_direction_norm": float(self.update_direction_norm),
            "weight_at_bound_fraction": float(self.weight_at_bound_fraction),
        }


@dataclass(frozen=True)
class MomentFitReport:
    converged: bool
    rounds: int
    target_vector: np.ndarray
    final_model_vector: np.ndarray
    trajectory: Sequence[MomentFitRound]
    successful_updates: int

    def to_json_dict(self) -> dict:
        return {
            "converged": bool(self.converged),
            "rounds": int(self.rounds),
            "target_vector": self.target_vector.tolist(),
            "final_model_vector": self.final_model_vector.tolist(),
            "trajectory": [item.to_json_dict() for item in self.trajectory],
            "successful_updates": int(self.successful_updates),
        }


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
        "ess_q10": float(np.quantile(ess_array, lower)),
        "split_rhat_max": float(np.max(rhat_array, initial=0.0)),
        "split_rhat_q90": float(np.quantile(rhat_array, upper)),
        "drift_max": float(np.max(drift_array, initial=0.0)),
        "drift_q90": float(np.quantile(drift_array, upper)),
        "drift_z_max": float(np.max(drift_z_array, initial=0.0)),
        "drift_z_q90": float(np.quantile(drift_z_array, upper)),
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
    max_batches_per_round: int = 16,
    min_samples_per_chain: int = 16,
    min_ess_per_chain: float = 8.0,
    rhat_threshold: float = 1.10,
    drift_z_threshold: float = 2.0,
    warmup_steps_per_round: int = 0,
    step_decay: float = 0.60,
    update_norm_clip: float = 5.0,
    diagnostic_quantile: float = 0.90,
) -> tuple[LinearGraphletEnergy, MomentFitReport, List[TypedGraph]]:
    """Fit a fixed-degree maximum-entropy model by persistent MCMC.

    For frozen ``w``, the population negative log likelihood is convex and has
    gradient ``E_w[z(G)] - z(G_real)``. The chains below estimate that
    expectation. A diagonally preconditioned Robbins-Monro update uses a
    diminishing step size only after the frozen-weight chains pass ESS,
    split-Rhat, and drift checks. The returned energy is frozen and must be
    validated with fresh chains.
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

    target_vector = selection.vector(target)
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

    thin = max(int(steps_per_round) // int(samples_per_round), 1)
    burn_in = thin
    for round_id in range(int(rounds)):
        energy = LinearGraphletEnergy(selection, weights)
        vector_chains: List[List[np.ndarray]] = [[] for _ in graphs]
        metric_chains: List[List[Mapping[str, float]]] = [[] for _ in graphs]
        acceptance_rates = []
        valid_rates = []
        if int(warmup_steps_per_round) > 0:
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
                result = run_switch_chain(
                    graph,
                    GraphletFeatureExtractor(graph, tracked_keys=selection.keys),
                    steps=int(steps_per_round),
                    burn_in=burn_in,
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
            graphs = updated_graphs
            batches_used = batch_id + 1
            samples_ready = min(len(chain) for chain in vector_chains)
            if samples_ready < int(min_samples_per_chain):
                continue
            diagnostics = _feature_chain_diagnostics(
                vector_chains, target_vector, diagnostic_quantile
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

        residual = target_vector - final_model_vector
        relative = np.abs(residual) / np.maximum(np.abs(target_vector), 1.0)
        moment_mae = float(relative.mean())
        moment_max_error = float(relative.max(initial=0.0))
        gradient = residual - float(l2) * weights
        gradient_norm = float(np.linalg.norm(gradient))
        preconditioner = 1.0 / np.maximum(np.abs(target_vector), 1.0)
        update_direction = preconditioner * gradient
        update_direction_norm = float(np.linalg.norm(update_direction))
        if float(update_norm_clip) > 0.0 and update_direction_norm > float(
            update_norm_clip
        ):
            update_direction *= float(update_norm_clip) / update_direction_norm
            update_direction_norm = float(update_norm_clip)
        next_update = successful_updates + 1
        step_size = (
            float(learning_rate) / float(next_update) ** float(step_decay)
            if expectation_converged
            else 0.0
        )
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
                step_size=step_size,
                update_direction_norm=update_direction_norm,
                weight_at_bound_fraction=bound_fraction,
            )
        )
        print(
            "[MOMENT-FIT] "
            f"round={round_id} mae={moment_mae:.4f} "
            f"max={moment_max_error:.4f} grad={gradient_norm:.3f} "
            f"accept={float(np.mean(acceptance_rates)):.3f} "
            f"tri={float(np.mean(sampled_triangles)) if sampled_triangles else 0.0:.1f} "
            f"cluster={float(np.mean(sampled_clustering)) if sampled_clustering else 0.0:.4f} "
            f"weight_norm={float(np.linalg.norm(weights)):.3f} "
            f"batches={batches_used} samples={min((len(chain) for chain in vector_chains), default=0)} "
            f"ess_q10={feature_ess_q10:.2f} split_rhat_q90={feature_rhat_q90:.3f} "
            f"drift_q90={feature_drift_q90:.4f} "
            f"drift_z_q90={feature_drift_z_q90:.3f} "
            f"worst=({feature_ess:.2f},{feature_rhat:.3f},"
            f"{feature_drift:.4f},{feature_drift_z:.3f}) "
            f"expectation_ok={int(expectation_converged)} "
            f"step_size={step_size:.5f} bound={bound_fraction:.3f}",
            flush=True,
        )
        if expectation_converged and moment_mae <= float(tolerance):
            converged = True
            break
        if expectation_converged and round_id + 1 < int(rounds):
            weights += step_size * update_direction
            weights = np.clip(weights, -float(max_weight), float(max_weight))
            successful_updates += 1
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
    )
    return LinearGraphletEnergy(selection, weights), report, graphs
