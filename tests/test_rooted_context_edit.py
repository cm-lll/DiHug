import numpy as np
import torch

from sparse_diffusion.structure_energy.global_context_joint_edit import (
    NodeContextJointEditResidual,
)
from sparse_diffusion.structure_energy.graph import FamilySpec, TypedGraph
from sparse_diffusion.structure_energy.mixed_edit_diffusion import EditProposal
from sparse_diffusion.structure_energy.rooted_context_edit import (
    RootedContextEditResidual,
    rooted_closure_signature,
    rooted_closure_utility,
    schema_supports_triangles,
)
from sparse_diffusion.structure_energy.switch_diffusion import SwitchDenoiserMLP


def _graph(edges):
    node_types = np.asarray([0, 0, 0, 0, 0, 0], dtype=np.int64)
    spec = FamilySpec("A-A", 0, 0, tuple(range(6)), tuple(range(6)))
    return TypedGraph(node_types, ("A",), {"A-A": spec}, {"A-A": edges})


def test_rooted_signature_is_invariant_to_within_type_relabeling():
    graph = _graph([(0, 1), (1, 2), (0, 2), (2, 3), (3, 4)])
    permutation = np.asarray([3, 4, 2, 0, 1, 5])
    relabeled = _graph([(int(permutation[u]), int(permutation[v])) for u, v in graph.family_edges["A-A"]])
    left = rooted_closure_signature(graph)
    right = rooted_closure_signature(relabeled)
    assert np.allclose(left.values, right.values)
    assert np.allclose(left.weights, right.weights)


def test_rooted_utility_rewards_closing_low_degree_triangle():
    target = _graph([(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5)])
    current = _graph([(0, 1), (1, 2), (3, 4), (4, 5), (0, 5), (2, 3)])
    signature = rooted_closure_signature(target)
    proposal = EditProposal(
        "switch", "A-A", ((0, 5), (2, 3)), ((0, 2), (3, 5))
    )
    assert rooted_closure_utility(current, proposal, signature) > 0.0


def test_incremental_rooted_utility_matches_full_recomputation():
    target = _graph([(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5)])
    current = _graph([(0, 1), (1, 2), (3, 4), (4, 5), (0, 5), (2, 3)])
    signature = rooted_closure_signature(target)
    proposal = EditProposal(
        "switch", "A-A", ((0, 5), (2, 3)), ((0, 2), (3, 5))
    )
    before = rooted_closure_signature(current)
    after_graph = current.clone()
    from sparse_diffusion.structure_energy.mixed_edit_diffusion import apply_edit

    apply_edit(after_graph, proposal)
    after = rooted_closure_signature(after_graph)
    expected = signature.squared_error(before) - signature.squared_error(after)
    assert np.isclose(rooted_closure_utility(current, proposal, signature), expected)


def test_zero_context_residual_is_exact_legacy_and_legacy_is_frozen():
    torch.manual_seed(7)
    legacy = SwitchDenoiserMLP(input_dim=9, hidden_dim=16, layers=2, dropout=0.0)
    model = RootedContextEditResidual(
        legacy,
        node_feature_dim=4,
        candidate_feature_dim=9,
        type_count=1,
        family_count=1,
        hidden_dim=16,
        heads=4,
        layers=1,
        dropout=0.0,
    ).eval()
    node_features = torch.randn(1, 6, 4)
    node_types = torch.zeros(1, 6, dtype=torch.long)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    edge_types = torch.zeros(4, dtype=torch.long)
    candidate_features = torch.randn(1, 3, 9)
    endpoints = torch.tensor(
        [[[0, 1, 2, 3, 0, 2, 1, 3], [1, 2, 3, 4, 1, 3, 2, 4], [0, 1, 2, 3, 0, 3, 1, 2]]]
    )
    with torch.no_grad():
        expected = legacy(candidate_features)
        actual = model(
            node_features,
            node_types,
            edge_index,
            edge_types,
            candidate_features,
            endpoints,
        )
    assert torch.equal(expected, actual)
    assert isinstance(model.joint, NodeContextJointEditResidual)
    assert model.joint.uses_persistent_edge_state is False
    assert all(not parameter.requires_grad for parameter in model.joint.legacy_scorer.parameters())


def test_residual_scale_zero_is_exact_legacy_after_context_changes():
    torch.manual_seed(11)
    legacy = SwitchDenoiserMLP(input_dim=9, hidden_dim=16, layers=2, dropout=0.0)
    model = RootedContextEditResidual(
        legacy,
        node_feature_dim=4,
        candidate_feature_dim=9,
        type_count=1,
        family_count=1,
        hidden_dim=16,
        heads=4,
        layers=1,
        dropout=0.0,
    ).eval()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if not name.startswith("joint.legacy_scorer"):
                parameter.normal_(mean=0.0, std=0.2)
    model.set_residual_scale(0.0)
    node_features = torch.randn(1, 6, 4)
    node_types = torch.zeros(1, 6, dtype=torch.long)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    edge_types = torch.zeros(4, dtype=torch.long)
    candidate_features = torch.randn(1, 3, 9)
    endpoints = torch.tensor(
        [[[0, 1, 2, 3, 0, 2, 1, 3], [1, 2, 3, 4, 1, 3, 2, 4], [0, 1, 2, 3, 0, 3, 1, 2]]]
    )
    with torch.no_grad():
        expected = legacy(candidate_features)
        actual = model(
            node_features,
            node_types,
            edge_index,
            edge_types,
            candidate_features,
            endpoints,
        )
    assert torch.equal(expected, actual)


def test_triangle_incapable_typed_schema_is_detected():
    node_types = np.asarray([0, 0, 1, 2], dtype=np.int64)
    specs = {
        "A-B": FamilySpec("A-B", 0, 1, (0, 1), (2,)),
        "A-C": FamilySpec("A-C", 0, 2, (0, 1), (3,)),
    }
    graph = TypedGraph(node_types, ("A", "B", "C"), specs, {name: [] for name in specs})
    assert not schema_supports_triangles(graph)
    assert schema_supports_triangles(_graph([]))
