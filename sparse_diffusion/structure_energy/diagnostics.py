from __future__ import annotations

import math
from typing import Dict, Mapping, Optional, Sequence

import numpy as np


def integrated_autocorrelation_time(values: Sequence[float]) -> float:
    x = np.asarray(values, dtype=np.float64)
    if x.size < 4 or np.var(x) <= 1.0e-15:
        return 1.0
    x = x - x.mean()
    size = 1 << (2 * x.size - 1).bit_length()
    spectrum = np.fft.rfft(x, n=size)
    autocov = np.fft.irfft(spectrum * np.conjugate(spectrum), n=size)[: x.size]
    autocov /= np.arange(x.size, 0, -1)
    autocorr = autocov / autocov[0]
    total = 0.0
    # Geyer's initial positive sequence, paired to reduce noisy tail bias.
    for index in range(1, len(autocorr) - 1, 2):
        pair = float(autocorr[index] + autocorr[index + 1])
        if pair <= 0.0:
            break
        total += pair
    return max(1.0 + 2.0 * total, 1.0)


def effective_sample_size(values: Sequence[float]) -> float:
    values = list(values)
    return float(len(values)) / integrated_autocorrelation_time(values)


def r_hat(chains: Sequence[Sequence[float]]) -> float:
    if len(chains) < 2:
        return float("nan")
    length = min(len(chain) for chain in chains)
    if length < 4:
        return float("nan")
    matrix = np.asarray([chain[-length:] for chain in chains], dtype=np.float64)
    within = np.var(matrix, axis=1, ddof=1).mean()
    between = length * np.var(matrix.mean(axis=1), ddof=1)
    if within <= 1.0e-15:
        return 1.0 if between <= 1.0e-15 else float("inf")
    variance = ((length - 1.0) / length) * within + between / length
    return float(np.sqrt(max(variance / within, 1.0)))


def split_r_hat(chains: Sequence[Sequence[float]]) -> float:
    """Return split-Rhat so shared within-chain drift is not hidden.

    Ordinary Rhat can be close to one when all chains move in the same
    direction. Splitting every chain into an early and a late half turns that
    drift into between-chain variance while preserving the usual diagnostic's
    interpretation for stationary traces.
    """

    split = []
    for chain in chains:
        values = list(chain)
        half = len(values) // 2
        if half < 4:
            continue
        split.extend((values[:half], values[-half:]))
    return r_hat(split)


def relative_half_drift(
    chains: Sequence[Sequence[float]], reference: Optional[float] = None
) -> float:
    """Measure pooled early-to-late drift on a reference-relative scale."""

    early = []
    late = []
    for chain in chains:
        values = np.asarray(chain, dtype=np.float64)
        half = values.size // 2
        if half < 2:
            continue
        early.extend(values[:half].tolist())
        late.extend(values[-half:].tolist())
    if not early or not late:
        return float("inf")
    early_mean = float(np.mean(early))
    late_mean = float(np.mean(late))
    if reference is None or not math.isfinite(float(reference)):
        scale = max(abs(early_mean), abs(late_mean), 1.0e-12)
    else:
        scale = max(abs(float(reference)), 1.0e-12)
    return abs(late_mean - early_mean) / scale


def half_drift_z_score(chains: Sequence[Sequence[float]]) -> float:
    """Test early/late mean drift against autocorrelation-aware uncertainty."""

    early = []
    late = []
    early_ess = 0.0
    late_ess = 0.0
    for chain in chains:
        values = list(float(value) for value in chain)
        half = len(values) // 2
        if half < 2:
            continue
        early_trace = values[:half]
        late_trace = values[-half:]
        early.extend(early_trace)
        late.extend(late_trace)
        early_ess += effective_sample_size(early_trace)
        late_ess += effective_sample_size(late_trace)
    if not early or not late:
        return float("inf")
    difference = abs(float(np.mean(late)) - float(np.mean(early)))
    early_variance = float(np.var(early, ddof=1)) if len(early) > 1 else 0.0
    late_variance = float(np.var(late, ddof=1)) if len(late) > 1 else 0.0
    standard_error = math.sqrt(
        early_variance / max(early_ess, 1.0)
        + late_variance / max(late_ess, 1.0)
    )
    if standard_error <= 1.0e-12:
        return 0.0 if difference <= 1.0e-12 else float("inf")
    return difference / standard_error


def summarize_chain_metrics(
    chain_metrics: Sequence[Sequence[Mapping[str, float]]],
    reference: Optional[Mapping[str, float]] = None,
) -> Dict[str, dict]:
    names = sorted(
        set().union(
            *(set(sample) for chain in chain_metrics for sample in chain)
        )
    )
    report: Dict[str, dict] = {}
    for name in names:
        traces = [
            [float(sample[name]) for sample in chain if name in sample]
            for chain in chain_metrics
        ]
        traces = [trace for trace in traces if trace]
        flat = np.asarray([value for trace in traces for value in trace])
        report[name] = {
            "mean": float(flat.mean()) if flat.size else float("nan"),
            "std": float(flat.std(ddof=1)) if flat.size > 1 else 0.0,
            "ess_min": min((effective_sample_size(trace) for trace in traces), default=0.0),
            "r_hat": r_hat(traces),
            "split_r_hat": split_r_hat(traces),
            "relative_half_drift": relative_half_drift(
                traces, None if reference is None else reference.get(name)
            ),
            "half_drift_z": half_drift_z_score(traces),
            "samples_per_chain_min": min((len(trace) for trace in traces), default=0),
        }
    return report
