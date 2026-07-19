import numpy as np

from sparse_diffusion.structure_energy.graph import FamilySpec, TypedGraph
from sparse_diffusion.structure_energy.latent_edit_representation import (
    _average_precision,
    build_representation_warmup_dataset,
    sample_clean_edge_queries,
    train_role_conditioned_alignment,
    train_representation_warmup,
)


def _heterogeneous_graph() -> TypedGraph:
    node_types = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64)
    specs = {
        "A-A": FamilySpec("A-A", 0, 0, (0, 1, 2), (0, 1, 2)),
        "A-B": FamilySpec("A-B", 0, 1, (0, 1, 2), (3, 4, 5)),
    }
    return TypedGraph(
        node_types,
        ("A", "B"),
        specs,
        {
            "A-A": [(0, 1), (1, 2)],
            "A-B": [(0, 3), (1, 3), (1, 4), (2, 5)],
        },
    )


def test_warmup_dataset_reaches_independent_top_noise_and_keeps_anonymous_roles():
    source = _heterogeneous_graph()
    dataset = build_representation_warmup_dataset(
        source,
        trajectories=2,
        noise_levels=(0.5, 1.0),
        seed=11,
    )
    assert len(dataset.snapshots) == 4
    assert dataset.family_names == ("A-A", "A-B")
    assert dataset.snapshots[-1].time_fraction == 1.0
    clean_rows = np.stack(
        [source.family_degrees(family) for family in dataset.family_names], axis=1
    )
    for snapshot in dataset.snapshots:
        for type_id in range(len(source.type_names)):
            nodes = np.flatnonzero(source.node_types == type_id)
            expected = sorted(map(tuple, clean_rows[nodes].tolist()))
            actual = sorted(map(tuple, snapshot.role_rows[nodes].tolist()))
            assert actual == expected


def test_clean_query_sampler_balances_positive_and_negative_targets():
    dataset = build_representation_warmup_dataset(
        _heterogeneous_graph(),
        trajectories=2,
        noise_levels=(1.0,),
        seed=13,
    )
    query_edges, families, labels, current = sample_clean_edge_queries(
        dataset.snapshots[0],
        dataset.family_names,
        queries_per_family=8,
        seed=17,
    )
    assert query_edges.shape[0] == 2
    assert labels.shape == families.shape == current.shape
    for family_id in range(dataset.family_count):
        family_labels = labels[families == family_id]
        assert set(family_labels.tolist()) == {0.0, 1.0}
        assert np.isclose(family_labels.mean(), 0.5)


def test_average_precision_treats_tied_scores_as_one_threshold():
    labels = np.asarray([1.0, 1.0, 0.0, 0.0], dtype=np.float32)
    assert np.isclose(_average_precision(labels, np.zeros(4)), 0.5)


def test_representation_warmup_cpu_smoke_reports_all_gates():
    dataset = build_representation_warmup_dataset(
        _heterogeneous_graph(),
        trajectories=2,
        noise_levels=(0.5, 1.0),
        seed=19,
    )
    _, report = train_representation_warmup(
        dataset,
        epochs=1,
        queries_per_family=4,
        hidden_dim=16,
        heads=4,
        layers=1,
        dropout=0.0,
        device="cpu",
        seed=23,
    )
    for key in (
        "edge_macro_ap",
        "edge_ap_gain",
        "edge_current_state_ap",
        "edge_gain_over_current",
        "role_mae",
        "role_type_baseline_mae",
        "embedding_mean_feature_std",
    ):
        assert np.isfinite(report[key])
    assert set(report["by_noise"]) == {"0.5", "1.0"}
    for metrics in report["by_noise"].values():
        assert np.isfinite(metrics["edge_gain_over_current"])
        assert np.isfinite(metrics["role_mae_reduction"])


def test_role_conditioned_alignment_cpu_smoke_has_fixed_clean_targets():
    dataset = build_representation_warmup_dataset(
        _heterogeneous_graph(),
        trajectories=2,
        noise_levels=(0.5, 1.0),
        seed=29,
    )
    base, _ = train_representation_warmup(
        dataset,
        epochs=1,
        queries_per_family=4,
        hidden_dim=16,
        heads=4,
        layers=1,
        dropout=0.0,
        device="cpu",
        seed=31,
    )
    _, report = train_role_conditioned_alignment(
        dataset,
        base_model_state=base.state_dict(),
        epochs=1,
        queries_per_family=4,
        hidden_dim=16,
        heads=4,
        layers=1,
        dropout=0.0,
        prototype_count=2,
        device="cpu",
        seed=37,
    )
    assert report["loaded_base_keys"] > 0
    assert report["prototype_count"] == 2
    assert set(report["by_noise"]) == {"0.5", "1.0"}
    for metrics in report["by_noise"].values():
        assert np.isfinite(metrics["latent_clean_cosine"])
        assert np.isfinite(metrics["prototype_clean_cosine"])
        assert np.isfinite(metrics["role_shuffle_distance"])
        assert np.isfinite(metrics["edge_context_gain"])
        assert np.isfinite(metrics["latent_context_gain"])
