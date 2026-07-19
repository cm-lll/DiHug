import numpy as np

from sparse_diffusion.structure_energy.graph import FamilySpec, TypedGraph
from sparse_diffusion.structure_energy.heterogeneous_metrics import (
    aggregate_comparisons,
    compare_heterogeneous_graphs,
    degree_mmd_rbf,
    degree_mmd_report,
    powerlaw_tail_fit,
    typed_metapath_counts,
    typed_triangle_counts,
)


def _typed_triangle_graph(edges=None):
    node_types = np.asarray([0, 1, 2, 0], dtype=np.int64)
    nodes = {0: (0, 3), 1: (1,), 2: (2,)}
    specs = {
        "AB": FamilySpec("AB", 0, 1, nodes[0], nodes[1]),
        "BC": FamilySpec("BC", 1, 2, nodes[1], nodes[2]),
        "AC": FamilySpec("AC", 0, 2, nodes[0], nodes[2]),
    }
    if edges is None:
        edges = {"AB": [(0, 1)], "BC": [(1, 2)], "AC": [(0, 2)]}
    return TypedGraph(node_types, ("A", "B", "C"), specs, edges)


def test_histogram_mmd_matches_dense_hgen_formula():
    left = np.asarray([0, 1, 1, 3], dtype=np.float64)
    right = np.asarray([0, 2, 2, 3], dtype=np.float64)
    alpha = 4.0

    def kernel(x, y):
        return np.exp(-alpha * np.square(x[:, None] - y[None, :]))

    kxx = kernel(left, left)
    kyy = kernel(right, right)
    kxy = kernel(left, right)
    expected = (
        (kxx.sum() - np.trace(kxx)) / (len(left) * (len(left) - 1))
        + (kyy.sum() - np.trace(kyy)) / (len(right) * (len(right) - 1))
        - 2.0 * kxy.mean()
    )
    assert np.isclose(degree_mmd_rbf(left, right), expected)


def test_mmd_report_keeps_raw_unbiased_estimate_and_clips_table_value():
    values = np.asarray([0, 1, 1, 2, 3], dtype=np.float64)
    report = degree_mmd_report(values, values)
    assert report["degree_mmd_raw"] <= 0.0
    assert report["degree_mmd"] == 0.0


def test_typed_triangle_and_metapath_signatures_are_schema_aware():
    graph = _typed_triangle_graph()
    triangles = typed_triangle_counts(graph)
    paths = typed_metapath_counts(graph, max_length=3)
    assert sum(triangles.values()) == 1
    assert len(triangles) == 1
    assert sum(paths["1"].values()) == 6
    assert sum(paths["2"].values()) == 12
    assert sum(paths["3"].values()) == 24


def test_comparison_reports_family_and_identity_metrics():
    target = _typed_triangle_graph()
    generated = _typed_triangle_graph(
        {"AB": [(3, 1)], "BC": [(1, 2)], "AC": [(3, 2)]}
    )
    row = compare_heterogeneous_graphs(target, generated)
    assert set(row["families"]) == {"AB", "AC", "BC"}
    assert row["typed_triangles"]["tv"] == 0.0
    assert row["edge_identity_diagnostic"]["exact_training_graph_match"] is False
    assert row["edge_identity_diagnostic"]["generated_edge_novelty"] > 0.0
    aggregate = aggregate_comparisons([row, row])
    assert aggregate["samples"] == 2
    assert aggregate["uniqueness_percent"] == 50.0
    assert aggregate["global"]["triangles"]["target"] == 1.0
    assert aggregate["typed_triangles"]["patterns"]


def test_powerlaw_fit_reports_tail_quality_not_only_alpha():
    report = powerlaw_tail_fit(np.asarray([1, 1, 2, 2, 3, 5, 8, 13]))
    assert report["alpha"] > 1.0
    assert 0.0 <= report["ks"] <= 1.0
    assert report["tail_size"] == 8.0
