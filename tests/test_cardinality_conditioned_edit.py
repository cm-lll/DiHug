import numpy as np

from sparse_diffusion.structure_energy.cardinality_conditioned_edit import (
    FamilyCountCondition,
    family_edge_counts,
)
from sparse_diffusion.structure_energy.graph import FamilySpec, TypedGraph
from sparse_diffusion.structure_energy.mixed_edit_diffusion import (
    MarginalGraphPrior,
    apply_edit,
    family_domain_size,
    sample_valid_edit,
)


def _toy_graph() -> TypedGraph:
    node_types = np.asarray([0] * 6 + [1] * 4, dtype=np.int64)
    specs = {
        "AA": FamilySpec("AA", 0, 0, tuple(range(6)), tuple(range(6))),
        "AB": FamilySpec("AB", 0, 1, tuple(range(6)), tuple(range(6, 10))),
    }
    return TypedGraph(
        node_types,
        ("A", "B"),
        specs,
        {
            "AA": [(0, 1), (2, 3), (4, 5)],
            "AB": [(0, 6), (0, 7), (1, 6), (1, 8), (2, 7), (2, 9)],
        },
    )


def test_fixed_family_count_initialization_is_uniformly_anonymous_and_exact():
    source = _toy_graph()
    condition = FamilyCountCondition.from_training_graph(source)
    first = condition.sample_graph(17, mode="fixed")
    second = condition.sample_graph(19, mode="fixed")
    assert family_edge_counts(first) == family_edge_counts(source)
    assert family_edge_counts(second) == family_edge_counts(source)
    assert first.to_json_dict() != second.to_json_dict()


def test_binomial_family_count_initialization_matches_marginal_contract():
    source = _toy_graph()
    condition = FamilyCountCondition.from_training_graph(source)
    marginal = MarginalGraphPrior.from_training_graph(source)
    draws = [condition.sample_graph(seed, mode="binomial") for seed in range(200)]
    for family in marginal.family_names:
        observed = np.mean([len(graph.family_edges[family]) for graph in draws])
        capacity = family_domain_size(source.family_specs[family])
        expected = capacity * marginal.family_probabilities[family]
        assert abs(observed - expected) < max(0.4, 0.2 * expected)


def test_relocation_switch_preserves_sampled_family_count_condition():
    source = _toy_graph()
    graph = FamilyCountCondition.from_training_graph(source).sample_graph(
        31, mode="fixed"
    )
    before = family_edge_counts(graph)
    rng = np.random.default_rng(37)
    for _ in range(100):
        proposal = sample_valid_edit(
            graph, rng, actions=("relocation", "switch")
        )
        if proposal is not None:
            apply_edit(graph, proposal)
    assert family_edge_counts(graph) == before
