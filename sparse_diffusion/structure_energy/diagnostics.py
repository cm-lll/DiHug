from __future__ import annotations

from typing import Dict, Mapping, Sequence

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
    if within <= 1.0e-15:
        return 1.0
    between = length * np.var(matrix.mean(axis=1), ddof=1)
    variance = ((length - 1.0) / length) * within + between / length
    return float(np.sqrt(max(variance / within, 0.0)))


def summarize_chain_metrics(
    chain_metrics: Sequence[Sequence[Mapping[str, float]]],
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
        }
    return report
