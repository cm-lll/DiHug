from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.special import zeta

from .graph import Edge, TypedGraph
from .switch_chain import graph_metrics


Distribution = Mapping[str, float]


def _safe_assortativity(left: Sequence[float], right: Sequence[float]) -> float:
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    if (
        left_array.size < 2
        or right_array.size < 2
        or float(left_array.std()) == 0.0
        or float(right_array.std()) == 0.0
    ):
        return 0.0
    value = float(np.corrcoef(left_array, right_array)[0, 1])
    return value if math.isfinite(value) else 0.0


def _histogram(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
    values = np.asarray(values, dtype=np.int64).reshape(-1)
    if values.size == 0:
        return np.asarray([0], dtype=np.int64), np.asarray([1], dtype=np.int64), 1
    support, counts = np.unique(values, return_counts=True)
    return support.astype(np.int64), counts.astype(np.int64), int(values.size)


def degree_mmd_rbf(
    reference: np.ndarray,
    generated: np.ndarray,
    alpha: float = 4.0,
) -> float:
    """Exact unbiased degree MMD using the HGEN RBF kernel.

    The implementation groups repeated scalar degree values before evaluating
    the kernel, avoiding an O(N^2) dense matrix for large graphs.
    """

    x, cx, nx = _histogram(reference)
    y, cy, ny = _histogram(generated)
    if nx < 2 or ny < 2:
        return 0.0

    kxx = np.exp(-float(alpha) * np.square(x[:, None] - x[None, :]))
    kyy = np.exp(-float(alpha) * np.square(y[:, None] - y[None, :]))
    kxy = np.exp(-float(alpha) * np.square(x[:, None] - y[None, :]))
    within_x = float(cx @ kxx @ cx - cx.sum()) / float(nx * (nx - 1))
    within_y = float(cy @ kyy @ cy - cy.sum()) / float(ny * (ny - 1))
    cross = float(cx @ kxy @ cy) / float(nx * ny)
    return float(within_x + within_y - 2.0 * cross)


def degree_mmd_report(reference: np.ndarray, generated: np.ndarray) -> Dict[str, float]:
    """Return the unbiased estimate and its nonnegative reporting value.

    An unbiased finite-sample MMD-squared estimate can be slightly negative.
    The raw value is retained for reproducibility, while tables use the
    nonnegative value expected of the population discrepancy.
    """

    raw = degree_mmd_rbf(reference, generated)
    return {"degree_mmd": max(float(raw), 0.0), "degree_mmd_raw": float(raw)}


def powerlaw_tail_fit(degrees: np.ndarray, xmin: int = 1) -> Dict[str, float]:
    """Fit a discrete power-law tail and report its KS discrepancy.

    The alpha estimate uses the standard discrete continuity correction. The
    KS value is reported alongside alpha so a visually plausible exponent is
    not mistaken for a good tail fit.
    """

    values = np.asarray(degrees, dtype=np.int64).reshape(-1)
    values = values[values >= int(xmin)]
    if values.size < 2:
        return {"alpha": 0.0, "xmin": float(xmin), "ks": 0.0, "tail_size": float(values.size)}
    denominator = float(np.log(values / (float(xmin) - 0.5)).sum())
    alpha = 0.0 if denominator <= 0.0 else 1.0 + float(values.size) / denominator
    support = np.arange(int(xmin), int(values.max()) + 1, dtype=np.float64)
    empirical = np.searchsorted(np.sort(values), support, side="right") / float(values.size)
    normalizer = float(zeta(alpha, float(xmin))) if alpha > 1.0 else math.inf
    if not math.isfinite(normalizer) or normalizer <= 0.0:
        ks = 1.0
    else:
        mass = np.power(support, -alpha) / normalizer
        theoretical = np.cumsum(mass)
        ks = float(np.max(np.abs(empirical - theoretical)))
    return {
        "alpha": float(alpha),
        "xmin": float(xmin),
        "ks": ks,
        "tail_size": float(values.size),
    }


def _normalize(counter: Mapping[str, int | float]) -> Dict[str, float]:
    total = float(sum(counter.values()))
    if total <= 0.0:
        return {}
    return {str(key): float(value) / total for key, value in sorted(counter.items())}


def distribution_distances(
    reference: Distribution, generated: Distribution
) -> Dict[str, float]:
    keys = sorted(set(reference) | set(generated))
    if not keys:
        return {"tv": 0.0, "jsd": 0.0}
    p = np.asarray([float(reference.get(key, 0.0)) for key in keys], dtype=np.float64)
    q = np.asarray([float(generated.get(key, 0.0)) for key in keys], dtype=np.float64)
    p /= max(float(p.sum()), 1.0)
    q /= max(float(q.sum()), 1.0)
    midpoint = 0.5 * (p + q)

    def kl(left: np.ndarray, right: np.ndarray) -> float:
        mask = left > 0.0
        return float(np.sum(left[mask] * np.log(left[mask] / right[mask])))

    return {
        "tv": float(0.5 * np.abs(p - q).sum()),
        "jsd": float(0.5 * kl(p, midpoint) + 0.5 * kl(q, midpoint)),
    }


def family_statistics(graph: TypedGraph) -> Dict[str, dict]:
    total_degree = graph.degrees()
    output: Dict[str, dict] = {}
    for family in sorted(graph.family_specs):
        spec = graph.family_specs[family]
        family_degree = graph.family_degrees(family)
        domain_nodes = np.asarray(
            sorted(set(spec.src_nodes) | set(spec.dst_nodes)), dtype=np.int64
        )
        total_left = []
        total_right = []
        family_left = []
        family_right = []
        for u, v in graph.family_edges[family]:
            total_left.append(total_degree[u])
            total_right.append(total_degree[v])
            family_left.append(family_degree[u])
            family_right.append(family_degree[v])
        output[family] = {
            "edges": int(len(graph.family_edges[family])),
            "domain_nodes": int(domain_nodes.size),
            "degree_values": family_degree[domain_nodes].astype(int).tolist(),
            "total_degree_assortativity": _safe_assortativity(
                total_left, total_right
            ),
            "family_degree_assortativity": _safe_assortativity(
                family_left, family_right
            ),
        }
    return output


def _triangle_signature(
    graph: TypedGraph, nodes: Tuple[int, int, int]
) -> str:
    import itertools

    candidates = []
    for permutation in itertools.permutations(nodes):
        a, b, c = permutation
        node_types = tuple(graph.type_names[int(graph.node_types[node])] for node in (a, b, c))
        families = (
            str(graph.family_of(a, b)),
            str(graph.family_of(b, c)),
            str(graph.family_of(a, c)),
        )
        candidates.append("|".join(node_types + families))
    return min(candidates)


def typed_triangle_counts(graph: TypedGraph) -> Dict[str, int]:
    counts: Counter[str] = Counter()
    for u in range(graph.num_nodes):
        for v in graph.adjacency[u]:
            if v <= u:
                continue
            for w in graph.adjacency[u] & graph.adjacency[v]:
                if w <= v:
                    continue
                counts[_triangle_signature(graph, (u, v, w))] += 1
    return dict(sorted(counts.items()))


def typed_metapath_counts(
    graph: TypedGraph, max_length: int = 3
) -> Dict[str, Dict[str, int]]:
    """Count typed walks of lengths 1..max_length.

    Walks follow the undirected generation graph and may backtrack, matching
    random-walk semantics. Reverse signatures are merged because the graph and
    relation families are undirected.
    """

    if int(max_length) < 1:
        return {}
    states = [
        {(graph.type_names[int(graph.node_types[node])],): 1}
        for node in range(graph.num_nodes)
    ]
    output: Dict[str, Dict[str, int]] = {}
    for length in range(1, int(max_length) + 1):
        next_states = [defaultdict(int) for _ in range(graph.num_nodes)]
        counts: Counter[str] = Counter()
        for source in range(graph.num_nodes):
            for target in graph.adjacency[source]:
                family = graph.family_of(source, target)
                target_type = graph.type_names[int(graph.node_types[target])]
                for signature, multiplicity in states[source].items():
                    extended = signature + (str(family), target_type)
                    next_states[target][extended] += int(multiplicity)
                    reverse = tuple(reversed(extended))
                    canonical = min(extended, reverse)
                    counts["|".join(canonical)] += int(multiplicity)
        # Counts retain oriented walk instances while reverse type signatures are
        # pooled. This avoids an arbitrary half-count for palindromic backtracks.
        output[str(length)] = dict(sorted(counts.items()))
        states = next_states
    return output


def edge_fingerprint(graph: TypedGraph) -> str:
    payload = [
        (family, int(u), int(v))
        for family in sorted(graph.family_edges)
        for u, v in sorted(graph.family_edges[family])
    ]
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def compare_heterogeneous_graphs(
    reference: TypedGraph, generated: TypedGraph
) -> dict:
    if reference.type_names != generated.type_names:
        raise ValueError("reference and generated node-type schemas differ")
    if set(reference.family_specs) != set(generated.family_specs):
        raise ValueError("reference and generated relation-family schemas differ")

    reference_family = family_statistics(reference)
    generated_family = family_statistics(generated)
    family_rows = {}
    for family in sorted(reference_family):
        expected = reference_family[family]
        observed = generated_family[family]
        family_mmd = degree_mmd_report(
            np.asarray(expected["degree_values"]),
            np.asarray(observed["degree_values"]),
        )
        family_rows[family] = {
            "target_edges": expected["edges"],
            "generated_edges": observed["edges"],
            "edge_relative_error": abs(observed["edges"] - expected["edges"])
            / max(expected["edges"], 1),
            **family_mmd,
            "target_total_degree_assortativity": expected[
                "total_degree_assortativity"
            ],
            "generated_total_degree_assortativity": observed[
                "total_degree_assortativity"
            ],
            "target_family_degree_assortativity": expected[
                "family_degree_assortativity"
            ],
            "generated_family_degree_assortativity": observed[
                "family_degree_assortativity"
            ],
        }

    reference_triangles = typed_triangle_counts(reference)
    generated_triangles = typed_triangle_counts(generated)
    reference_paths = typed_metapath_counts(reference)
    generated_paths = typed_metapath_counts(generated)
    target_edges = set(reference.edge_family)
    sample_edges = set(generated.edge_family)
    overlap = len(target_edges & sample_edges)

    global_mmd = degree_mmd_report(reference.degrees(), generated.degrees())
    return {
        "format": "dihug-heterogeneous-structure-metrics-v2",
        "global": {
            "target": graph_metrics(reference),
            "generated": graph_metrics(generated),
            **global_mmd,
            "target_powerlaw": powerlaw_tail_fit(reference.degrees()),
            "generated_powerlaw": powerlaw_tail_fit(generated.degrees()),
        },
        "families": family_rows,
        "typed_triangles": {
            "target_counts": reference_triangles,
            "generated_counts": generated_triangles,
            "target_distribution": _normalize(reference_triangles),
            "generated_distribution": _normalize(generated_triangles),
            **distribution_distances(
                _normalize(reference_triangles), _normalize(generated_triangles)
            ),
        },
        "typed_metapaths": {
            length: {
                "target_counts": reference_paths[length],
                "generated_counts": generated_paths[length],
                "target_distribution": _normalize(reference_paths[length]),
                "generated_distribution": _normalize(generated_paths[length]),
                **distribution_distances(
                    _normalize(reference_paths[length]),
                    _normalize(generated_paths[length]),
                ),
            }
            for length in sorted(reference_paths, key=int)
        },
        "edge_identity_diagnostic": {
            "target_edge_recall": overlap / max(len(target_edges), 1),
            "generated_edge_novelty": 1.0 - overlap / max(len(sample_edges), 1),
            "exact_training_graph_match": target_edges == sample_edges,
        },
        "generated_fingerprint": edge_fingerprint(generated),
    }


def aggregate_comparisons(rows: Sequence[Mapping]) -> dict:
    if not rows:
        raise ValueError("at least one comparison row is required")

    def summarize(values: Iterable[float]) -> dict:
        array = np.asarray(list(values), dtype=np.float64)
        return {
            "mean": float(array.mean()),
            "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        }

    global_keys = sorted(rows[0]["global"]["generated"])
    families = sorted(rows[0]["families"])
    report = {
        "format": "dihug-heterogeneous-structure-aggregate-v2",
        "samples": len(rows),
        "global": {},
        "degree_mmd": summarize(row["global"]["degree_mmd"] for row in rows),
        "degree_mmd_raw": summarize(
            row["global"]["degree_mmd_raw"] for row in rows
        ),
        "powerlaw": {
            "target": dict(rows[0]["global"]["target_powerlaw"]),
            "generated_alpha": summarize(
                row["global"]["generated_powerlaw"]["alpha"] for row in rows
            ),
            "generated_ks": summarize(
                row["global"]["generated_powerlaw"]["ks"] for row in rows
            ),
        },
        "families": {},
        "typed_triangles": {
            key: summarize(row["typed_triangles"][key] for row in rows)
            for key in ("tv", "jsd")
        },
        "typed_metapaths": {
            length: {
                key: summarize(row["typed_metapaths"][length][key] for row in rows)
                for key in ("tv", "jsd")
            }
            for length in rows[0]["typed_metapaths"]
        },
        "edge_identity_diagnostic": {
            key: summarize(row["edge_identity_diagnostic"][key] for row in rows)
            for key in ("target_edge_recall", "generated_edge_novelty")
        },
        "uniqueness_percent": 100.0
        * len({row["generated_fingerprint"] for row in rows})
        / len(rows),
    }
    for key in global_keys:
        target = float(rows[0]["global"]["target"][key])
        generated = [float(row["global"]["generated"][key]) for row in rows]
        report["global"][key] = {
            "target": target,
            "generated": summarize(generated),
            "absolute_error": summarize(abs(value - target) for value in generated),
            "relative_error": summarize(
                abs(value - target) / max(abs(target), 1.0) for value in generated
            ),
        }
    for family in families:
        target_family = rows[0]["families"][family]
        family_report = {
            "target_edges": target_family["target_edges"],
            "target_total_degree_assortativity": target_family[
                "target_total_degree_assortativity"
            ],
            "target_family_degree_assortativity": target_family[
                "target_family_degree_assortativity"
            ],
        }
        family_report.update({
            key: summarize(row["families"][family][key] for row in rows)
            for key in (
                "generated_edges",
                "edge_relative_error",
                "degree_mmd",
                "degree_mmd_raw",
                "generated_total_degree_assortativity",
                "generated_family_degree_assortativity",
            )
        })
        report["families"][family] = family_report
    triangle_keys = sorted(
        set().union(*(row["typed_triangles"]["target_counts"] for row in rows))
        | set().union(*(row["typed_triangles"]["generated_counts"] for row in rows))
    )
    report["typed_triangles"]["patterns"] = {
        key: {
            "target": float(rows[0]["typed_triangles"]["target_counts"].get(key, 0)),
            "generated": summarize(
                row["typed_triangles"]["generated_counts"].get(key, 0)
                for row in rows
            ),
        }
        for key in triangle_keys
    }
    for length in report["typed_metapaths"]:
        path_keys = sorted(
            set().union(
                *(row["typed_metapaths"][length]["target_counts"] for row in rows)
            )
            | set().union(
                *(row["typed_metapaths"][length]["generated_counts"] for row in rows)
            )
        )
        report["typed_metapaths"][length]["patterns"] = {
            key: {
                "target": float(
                    rows[0]["typed_metapaths"][length]["target_counts"].get(key, 0)
                ),
                "generated": summarize(
                    row["typed_metapaths"][length]["generated_counts"].get(key, 0)
                    for row in rows
                ),
            }
            for key in path_keys
        }
    return report


def save_comparison(path: str | Path, payload: Mapping) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
