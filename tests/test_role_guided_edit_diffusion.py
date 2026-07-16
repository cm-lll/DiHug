import numpy as np
import torch

from sparse_diffusion.structure_energy.graph import FamilySpec, TypedGraph
from sparse_diffusion.structure_energy.mixed_edit_diffusion import EditProposal, apply_edit
from sparse_diffusion.structure_energy.role_guided_edit_diffusion import (
    ExchangeableRoleDegreePrior,
    combine_structure_and_role_scores,
    denoising_time_fraction,
    proposal_relation_degree_gain,
    propose_wedge_seeded_switch,
    role_alignment_complete,
    role_error_per_node,
    relation_degree_error,
    sample_switch_refinement_candidates,
)


def _graph() -> TypedGraph:
    types = np.asarray([0] * 4 + [1] * 3)
    specs = {
        "AA": FamilySpec("AA", 0, 0, tuple(range(4)), tuple(range(4))),
        "AB": FamilySpec("AB", 0, 1, tuple(range(4)), tuple(range(4, 7))),
    }
    return TypedGraph(
        types,
        ("A", "B"),
        specs,
        {"AA": [(0, 1), (1, 2)], "AB": [(0, 4), (1, 5), (2, 6)]},
    )


def test_exchangeable_role_degree_prior_preserves_only_typewise_row_multisets():
    graph = _graph()
    prior = ExchangeableRoleDegreePrior.from_training_graph(graph)
    sampled = prior.sample_assignment(graph.node_types, seed=13)
    family_names = tuple(sorted(graph.family_specs))
    original = np.stack(
        [graph.family_degrees(family) for family in family_names], axis=1
    )
    for type_id in range(len(graph.type_names)):
        nodes = np.flatnonzero(graph.node_types == type_id)
        assert sorted(map(tuple, sampled[nodes])) == sorted(map(tuple, original[nodes]))


def test_proposal_role_degree_gain_matches_full_graph_recomputation():
    graph = _graph()
    family_names = tuple(sorted(graph.family_specs))
    target = np.stack(
        [graph.family_degrees(family) for family in family_names], axis=1
    )
    corrupted = graph.clone()
    corrupted.remove_edge("AB", (0, 4))
    corrupted.add_edge("AB", (3, 4))
    proposal = EditProposal("relocation", "AB", ((3, 4),), ((0, 4),))
    before = relation_degree_error(corrupted, target, family_names)
    incremental = proposal_relation_degree_gain(
        corrupted, proposal, target, {name: i for i, name in enumerate(family_names)}
    )
    apply_edit(corrupted, proposal)
    after = relation_degree_error(corrupted, target, family_names)
    assert incremental == before - after


def test_zero_role_weight_exactly_recovers_structure_logits():
    logits = torch.tensor([-1.0, 0.5, 2.0])
    gains = torch.tensor([100.0, -20.0, 3.0])
    combined = combine_structure_and_role_scores(
        logits,
        gains,
        time_fraction=0.8,
        role_weight=0.0,
        gain_scale=10.0,
    )
    assert torch.equal(combined, logits)


def test_unit_schedule_floor_keeps_role_energy_constant_at_zero_time():
    logits = torch.tensor([0.0, 0.0])
    gains = torch.tensor([10.0, -10.0])
    combined = combine_structure_and_role_scores(
        logits,
        gains,
        time_fraction=0.0,
        role_weight=0.1,
        gain_scale=10.0,
        schedule_floor=1.0,
    )
    assert torch.allclose(combined, torch.tensor([0.1, -0.1]))


def test_role_alignment_threshold_is_normalized_by_graph_size():
    assert role_error_per_node(80.0, 10) == 8.0
    assert role_alignment_complete(80.0, 10, 8.0)
    assert not role_alignment_complete(81.0, 10, 8.0)
    assert not role_alignment_complete(0.0, 10, 0.0)


def test_model_time_horizon_can_end_before_structural_refinement():
    assert denoising_time_fraction(1, 20) == 1.0
    assert np.isclose(denoising_time_fraction(20, 20), 0.05)
    assert denoising_time_fraction(21, 20) == 0.0
    assert denoising_time_fraction(40, 20) == 0.0


def test_wedge_seeded_switch_is_legal_and_preserves_family_degrees():
    graph = _graph()
    graph.add_edge("AB", (3, 4))
    before = graph.family_degrees("AB").copy()
    proposal = propose_wedge_seeded_switch(
        graph, np.random.default_rng(3), max_attempts=4096
    )
    assert proposal is not None
    assert proposal.action == "switch"
    before_triangles = sum(
        len(graph.adjacency[u] & graph.adjacency[v])
        for u, v in proposal.new_edges
    )
    assert before_triangles >= 1
    apply_edit(graph, proposal)
    assert np.array_equal(graph.family_degrees("AB"), before)
    graph.validate()


def test_closure_mixed_candidate_sampler_preserves_count_and_legality():
    graph = _graph()
    graph.add_edge("AB", (3, 4))
    candidates = sample_switch_refinement_candidates(
        graph,
        np.random.default_rng(7),
        count=4,
        closure_fraction=0.5,
        degree_bias_power=1.0,
    )
    assert len(candidates) == 4
    assert candidates[0].action == "noop"
    for proposal in candidates[1:]:
        candidate = graph.clone()
        apply_edit(candidate, proposal)
        candidate.validate()
