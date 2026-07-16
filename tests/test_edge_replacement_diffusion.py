import numpy as np
import torch

from sparse_diffusion.structure_energy.edge_replacement_diffusion import (
    apply_replacement_transition,
    build_family_domains,
    replacement_log_probability,
    relation_degree_role_features,
    sample_exchangeable_role_features,
    sample_checkpoint_role_features,
    sample_replacement_transition,
    select_x0_replacement,
    EdgeReplacementDenoiser,
)
from sparse_diffusion.structure_energy.graph import FamilySpec, TypedGraph


def _graph() -> TypedGraph:
    types = np.asarray([0] * 5 + [1] * 4)
    specs = {
        "AA": FamilySpec("AA", 0, 0, tuple(range(5)), tuple(range(5))),
        "AB": FamilySpec("AB", 0, 1, tuple(range(5)), tuple(range(5, 9))),
    }
    return TypedGraph(
        types,
        ("A", "B"),
        specs,
        {
            "AA": [(0, 1), (1, 2), (2, 3), (3, 4)],
            "AB": [(0, 5), (1, 6), (2, 7), (3, 8), (4, 5)],
        },
    )


def test_replacement_kernel_is_reversible_symmetric_and_count_preserving():
    graph = _graph()
    domains = build_family_domains(graph)
    before = {family: set(edges) for family, edges in graph.family_edges.items()}
    transition = sample_replacement_transition(
        graph, np.random.default_rng(7), rate=0.5, domains=domains
    )
    forward_log_q = replacement_log_probability(graph, transition, domains)
    counts = {family: len(edges) for family, edges in graph.family_edges.items()}
    apply_replacement_transition(graph, transition)
    assert {family: len(edges) for family, edges in graph.family_edges.items()} == counts
    reverse = transition.inverse()
    reverse_log_q = replacement_log_probability(graph, reverse, domains)
    assert np.isclose(forward_log_q, reverse_log_q)
    apply_replacement_transition(graph, reverse)
    assert all(set(graph.family_edges[name]) == before[name] for name in before)


def test_replacement_kernel_never_creates_duplicates_or_schema_violations():
    graph = _graph()
    domains = build_family_domains(graph)
    rng = np.random.default_rng(13)
    counts = {family: len(edges) for family, edges in graph.family_edges.items()}
    for _ in range(100):
        transition = sample_replacement_transition(graph, rng, rate=0.4, domains=domains)
        apply_replacement_transition(graph, transition)
        assert {family: len(edges) for family, edges in graph.family_edges.items()} == counts
        assert len(graph.edge_family) == sum(counts.values())


def test_replacement_denoiser_scores_present_and_null_edges():
    torch.manual_seed(3)
    model = EdgeReplacementDenoiser(
        node_feature_dim=6,
        type_count=2,
        family_count=2,
        hidden_dim=16,
        heads=4,
        layers=1,
        dropout=0.0,
    )
    hidden = torch.randn(9, 16)
    edges = torch.tensor([[0, 5], [1, 6], [2, 7]])
    deleted = model.score_edges(
        hidden,
        edges,
        family_id=1,
        action_id=0,
        time_fraction=0.7,
        structural_features=torch.randn(3, 7),
    )
    added = model.score_edges(
        hidden,
        edges,
        family_id=1,
        action_id=1,
        time_fraction=0.7,
        structural_features=torch.randn(3, 7),
    )
    assert deleted.shape == (3,)
    assert added.shape == (3,)
    assert not torch.equal(deleted, added)


def test_x0_replacement_removes_low_scores_and_adds_only_score_improvements():
    selected = select_x0_replacement(
        ((0, 1), (1, 2), (2, 3)),
        (0.9, -0.4, 0.2),
        ((0, 2), (0, 3), (1, 3)),
        (0.1, 1.2, -0.7),
        budget=2,
    )
    assert selected.removed == ((1, 2),)
    assert selected.added == ((0, 3),)


def test_x0_replacement_stops_when_no_absent_edge_improves_the_set():
    selected = select_x0_replacement(
        ((0, 1), (1, 2)),
        (0.8, 0.7),
        ((0, 2), (0, 3)),
        (0.6, 0.5),
        budget=2,
    )
    assert selected.removed == ()
    assert selected.added == ()


def test_exchangeable_role_assignment_preserves_rows_only_within_node_type():
    graph = _graph()
    original = relation_degree_role_features(graph)
    sampled = sample_exchangeable_role_features(graph, seed=17)
    for type_id in range(len(graph.type_names)):
        nodes = np.flatnonzero(graph.node_types == type_id)
        assert sorted(map(tuple, sampled[nodes])) == sorted(map(tuple, original[nodes]))


def test_checkpoint_role_sampling_needs_no_training_graph_adjacency():
    graph = _graph()
    model = EdgeReplacementDenoiser(
        node_feature_dim=6,
        type_count=2,
        family_count=2,
        hidden_dim=16,
        heads=4,
        layers=1,
        dropout=0.0,
        role_residual_features=True,
    )
    rows = relation_degree_role_features(graph)
    model.set_role_prior(rows, graph.node_types)
    sampled = sample_checkpoint_role_features(model, graph.node_types, seed=23)
    for type_id in range(len(graph.type_names)):
        nodes = np.flatnonzero(graph.node_types == type_id)
        assert sorted(map(tuple, sampled[nodes])) == sorted(map(tuple, rows[nodes]))
