import numpy as np
import torch

from sparse_diffusion.structure_energy.global_context_joint_edit import (
    NodeContextJointEditResidual,
)
from sparse_diffusion.structure_energy.graph import FamilySpec, TypedGraph
from sparse_diffusion.structure_energy.hgt_edit_denoiser import (
    build_hgt_edit_dataset,
    graph_edge_tensors,
    graph_node_features,
    graph_role_features,
    sample_typed_query_edges,
)
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


def test_sparse_query_edges_are_deterministic_schema_valid_null_links():
    graph = _graph([(0, 1), (1, 2), (2, 3)])
    left, left_types = sample_typed_query_edges(
        graph, seed=17, edges_per_family=4
    )
    right, right_types = sample_typed_query_edges(
        graph, seed=17, edges_per_family=4
    )
    assert np.array_equal(left, right)
    assert np.array_equal(left_types, right_types)
    existing = {tuple(sorted(edge)) for edge in graph.family_edges["A-A"]}
    for index in range(0, left.shape[1], 2):
        u, v = int(left[0, index]), int(left[1, index])
        assert u != v
        assert tuple(sorted((u, v))) not in existing
        assert left[:, index + 1].tolist() == [v, u]


def test_joint_role_query_loss_updates_hgt_and_mlp():
    torch.manual_seed(23)
    graph = _graph([(0, 1), (1, 2), (2, 3), (3, 4)])
    base_width = 4
    scorer = SwitchDenoiserMLP(
        input_dim=9, hidden_dim=16, layers=2, dropout=0.0
    )
    model = RootedContextEditResidual(
        scorer,
        node_feature_dim=base_width,
        candidate_feature_dim=9,
        type_count=1,
        family_count=1,
        hidden_dim=16,
        heads=4,
        layers=1,
        dropout=0.0,
        role_conditioning=True,
        query_edges_per_family=3,
    )
    for parameter in model.joint.legacy_scorer.parameters():
        parameter.requires_grad_(True)
    with torch.no_grad():
        model.joint.residual[-1].weight.fill_(0.01)
    target = np.asarray([[2], [2], [2], [1], [1], [0]], dtype=np.int64)
    node_features = graph_node_features(graph, 0.5)
    role_features = graph_role_features(graph, target)
    context_index, context_types = graph_edge_tensors(graph)
    query_index, query_types = sample_typed_query_edges(
        graph, seed=29, edges_per_family=3
    )
    edge_index = np.concatenate((context_index, query_index), axis=1)
    edge_types = np.concatenate((context_types, query_types))
    edge_subtypes = np.concatenate(
        (np.ones_like(context_types), np.zeros_like(query_types))
    )
    features = torch.randn(1, 3, 9)
    endpoints = torch.tensor(
        [[[0, 1, 2, 3, 0, 2, 1, 3], [1, 2, 3, 4, 1, 3, 2, 4], [0, 1, 2, 3, 0, 3, 1, 2]]]
    )
    logits = model(
        torch.from_numpy(node_features)[None],
        torch.zeros(1, graph.num_nodes, dtype=torch.long),
        torch.from_numpy(edge_index),
        torch.from_numpy(edge_types),
        features,
        endpoints,
        torch.from_numpy(edge_subtypes),
        torch.from_numpy(role_features)[None],
    )
    torch.nn.functional.cross_entropy(logits, torch.tensor([1])).backward()
    hgt_grad = sum(
        float(parameter.grad.abs().sum())
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
        and (name.startswith("node_input") or name.startswith("layers"))
    )
    mlp_grad = sum(
        float(parameter.grad.abs().sum())
        for parameter in model.joint.legacy_scorer.parameters()
        if parameter.grad is not None
    )
    assert hgt_grad > 0.0
    assert mlp_grad > 0.0


def test_closed_form_unified_dataset_aligns_structure_role_and_noop():
    graph = _graph(
        [(0, 1), (1, 2), (0, 2), (2, 3), (3, 4), (4, 5)]
    )
    dataset = build_hgt_edit_dataset(
        graph,
        trajectories=2,
        forward_steps=4,
        record_every=2,
        candidates_per_example=4,
        seed=31,
        supervision="rooted_utility",
        feature_profile="typed_closure_compact_v2",
        role_conditioning=True,
        query_edges_per_family=1,
        forward_mode="closed_form",
        candidate_mode="unified",
    )

    assert dataset.candidate_features.shape[:2] == (4, 4)
    assert dataset.candidate_role_gains.shape == (4, 4)
    np.testing.assert_allclose(dataset.time_fractions, [0.5, 1.0, 0.5, 1.0])
    assert np.isfinite(dataset.candidate_improvements).all()
    assert np.isfinite(dataset.candidate_role_gains).all()
    for row in range(dataset.candidate_endpoints.shape[0]):
        noop = np.flatnonzero(dataset.candidate_endpoints[row, :, 0] == -1)
        assert noop.tolist() == [0]
        assert dataset.candidate_role_gains[row, 0] == 0.0


def test_role_aware_utility_accepts_role_only_schema():
    import copy

    from sparse_diffusion.structure_energy.aligned_context_edit import (
        train_aligned_context_edit_denoiser,
    )

    graph = _graph([(0, 1), (1, 2), (2, 3), (3, 4)])
    dataset = build_hgt_edit_dataset(
        graph,
        trajectories=2,
        forward_steps=2,
        record_every=1,
        candidates_per_example=4,
        seed=37,
        supervision="rooted_utility",
        feature_profile="typed_closure_compact_v2",
        role_conditioning=True,
        forward_mode="closed_form",
        candidate_mode="unified",
    )
    dataset.candidate_improvements[:] = 0.0
    scorer = SwitchDenoiserMLP(
        input_dim=dataset.candidate_features.shape[-1],
        hidden_dim=16,
        layers=2,
        dropout=0.0,
    )
    reference = copy.deepcopy(scorer.state_dict())
    model, report = train_aligned_context_edit_denoiser(
        dataset,
        scorer,
        epochs=1,
        batch_size=1,
        learning_rate=1.0e-3,
        legacy_init="pretrained",
        legacy_unfreeze_epoch=0,
        training_objective="utility_role_kl",
        role_teacher_weight=1.0,
        residual_min_utility_gain=1.0e6,
        device="cpu",
        seed=41,
        hidden_dim=16,
        heads=4,
        layers=1,
        dropout=0.0,
        role_conditioning=True,
    )
    assert report["validation_oracle_utility"] > 0.0
    assert report["residual_scale"] == 0.0
    assert report["joint_update_accepted"] is False
    for name, value in model.joint.legacy_scorer.state_dict().items():
        assert torch.equal(value.cpu(), reference[name])
    assert any(
        parameter.grad is not None
        for parameter in model.joint.legacy_scorer.parameters()
    )
