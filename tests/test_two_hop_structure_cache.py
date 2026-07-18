import torch

from sparse_diffusion import utils
from sparse_diffusion.models.pyhgt_denoiser import PyHGTDenoiser


def _model():
    return PyHGTDenoiser(
        n_layers=1,
        input_dims=utils.PlaceHolder(X=4, E=5, y=2, charge=0),
        hidden_dims={
            "dx": 16,
            "de": 8,
            "dy": 8,
            "n_head": 4,
            "dim_ffX": 32,
            "dim_ffE": 16,
            "dim_ffy": 16,
        },
        output_dims=utils.PlaceHolder(X=4, E=5, y=1, charge=0),
        dropout=0.0,
        sn_hidden_dim=0,
        heterogeneous=True,
        num_node_types=2,
        num_node_subtypes=4,
        num_relation_types=4,
        edge_family_offsets={"aa": 1, "ab": 3},
        type_offsets={"a": 0, "b": 2},
        use_two_hop_structure=True,
        edge_only_model=True,
    )


def test_cached_two_hop_features_match_uncached_features():
    model = _model()
    context = torch.tensor(
        [[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]],
        dtype=torch.long,
    )
    query = torch.tensor(
        [[0, 0, 1, 1, 2], [1, 2, 2, 3, 3]],
        dtype=torch.long,
    )

    uncached = model._two_hop_structure_features(
        query_edge_index=query,
        context_edge_index=context,
        num_nodes=4,
    )
    lookup = model.build_two_hop_structure_lookup(
        context_edge_index=context,
        num_nodes=4,
    )
    cached = model._two_hop_structure_features(
        query_edge_index=query,
        context_edge_index=context,
        num_nodes=4,
        lookup=lookup,
    )

    assert lookup["adjacency_nnz"] == 6
    assert torch.equal(cached, uncached)
    assert not lookup["degree"].requires_grad
    assert not lookup["two_hop_values"].requires_grad


def test_cached_lookup_is_reusable_for_different_query_blocks():
    model = _model()
    context = torch.tensor(
        [[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]],
        dtype=torch.long,
    )
    lookup = model.build_two_hop_structure_lookup(context, num_nodes=4)

    for query in (
        torch.tensor([[0, 1], [2, 3]], dtype=torch.long),
        torch.tensor([[0, 0, 1], [1, 3, 2]], dtype=torch.long),
    ):
        expected = model._two_hop_structure_features(
            query_edge_index=query,
            context_edge_index=context,
            num_nodes=4,
        )
        actual = model._two_hop_structure_features(
            query_edge_index=query,
            context_edge_index=context,
            num_nodes=4,
            lookup=lookup,
        )
        assert torch.equal(actual, expected)


def test_typed_two_hop_features_distinguish_middle_type_and_leg_family():
    model = _model()
    model.use_typed_two_hop_structure = True
    model.two_hop_structure_feature_dim = 4 + 2 + 2 * 2
    context = torch.tensor(
        [[0, 1, 1, 2, 0, 3, 3, 4], [1, 0, 2, 1, 3, 0, 4, 3]],
        dtype=torch.long,
    )
    node_type = torch.tensor([0, 0, 1, 1, 1])
    family = torch.tensor([0, 0, 1, 1, 1, 1, 0, 0])
    lookup = model.build_two_hop_structure_lookup(
        context_edge_index=context,
        num_nodes=5,
        node_type_ids=node_type,
        edge_family_ids=family,
    )
    query = torch.tensor([[0, 0], [2, 4]], dtype=torch.long)
    typed = model._two_hop_structure_features(
        query_edge_index=query,
        context_edge_index=context,
        num_nodes=5,
        lookup=lookup,
    )

    assert typed.shape == (2, model.two_hop_structure_feature_dim)
    assert "typed_two_hop_values" in lookup
    # Both queries have common neighbors, but typed channels reveal that one
    # path closes through node 1/type 0 and another through node 3/type 1.
    assert typed[0, 4] > 0
    assert typed[1, 5] > 0
    assert typed[0, 5] == 0
    assert typed[1, 4] == 0


def test_cached_forward_matches_uncached_forward_and_backpropagates():
    torch.manual_seed(23)
    model = _model()
    model.eval()
    with torch.no_grad():
        model.two_hop_structure_head[-1].weight.normal_(std=0.05)
        model.two_hop_structure_head[-1].bias.normal_(std=0.05)

    x = torch.eye(4)
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]],
        dtype=torch.long,
    )
    edge_attr = torch.nn.functional.one_hot(
        torch.tensor([1, 1, 3, 3, 3, 3]), num_classes=5
    ).float()
    y = torch.tensor([[0.25, 0.75]])
    batch = torch.zeros(4, dtype=torch.long)
    node_type = torch.tensor([0, 0, 1, 1])
    node_subtype = torch.tensor([0, 1, 0, 1])
    relation = torch.tensor([0, 0, 1, 1, 3, 3])
    family = torch.tensor([0, 0, 1, 1, 1, 1])

    uncached = model(
        X=x,
        edge_attr=edge_attr,
        edge_index=edge_index,
        y=y,
        batch=batch,
        node_type_ids=node_type,
        node_subtype_ids=node_subtype,
        relation_type_ids=relation,
        edge_family_ids=family,
    )
    lookup = model.build_two_hop_structure_lookup(edge_index, num_nodes=4)
    cached = model(
        X=x,
        edge_attr=edge_attr,
        edge_index=edge_index,
        y=y,
        batch=batch,
        node_type_ids=node_type,
        node_subtype_ids=node_subtype,
        relation_type_ids=relation,
        edge_family_ids=family,
        two_hop_structure_lookup=lookup,
    )

    assert torch.equal(cached.edge_attr, uncached.edge_attr)
    cached.edge_attr.square().mean().backward()
    assert model.two_hop_structure_head[-1].weight.grad is not None


def test_explicit_time_factor_scales_only_two_hop_residual():
    torch.manual_seed(29)
    model = _model()
    model.eval()
    with torch.no_grad():
        model.two_hop_structure_head[-1].weight.normal_(std=0.05)
        model.two_hop_structure_head[-1].bias.normal_(std=0.05)

    x = torch.eye(4)
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]],
        dtype=torch.long,
    )
    edge_attr = torch.nn.functional.one_hot(
        torch.tensor([1, 1, 3, 3, 3, 3]), num_classes=5
    ).float()
    common = dict(
        X=x,
        edge_attr=edge_attr,
        edge_index=edge_index,
        y=torch.tensor([[0.25, 0.75]]),
        batch=torch.zeros(4, dtype=torch.long),
        node_type_ids=torch.tensor([0, 0, 1, 1]),
        node_subtype_ids=torch.tensor([0, 1, 0, 1]),
        relation_type_ids=torch.tensor([0, 0, 1, 1, 3, 3]),
        edge_family_ids=torch.tensor([0, 0, 1, 1, 1, 1]),
        two_hop_structure_lookup=model.build_two_hop_structure_lookup(
            edge_index, num_nodes=4
        ),
    )

    factor_zero = model(
        **common, two_hop_scale_factor=torch.tensor([[0.0]])
    ).edge_attr
    factor_one = model(
        **common, two_hop_scale_factor=torch.tensor([[1.0]])
    ).edge_attr
    factor_quarter = model(
        **common, two_hop_scale_factor=torch.tensor([[0.25]])
    ).edge_attr

    assert torch.allclose(
        factor_quarter - factor_zero,
        0.25 * (factor_one - factor_zero),
        atol=1e-6,
        rtol=1e-6,
    )
    assert model.last_two_hop_base_residual_mean > 0.0
    assert model.last_two_hop_scale_factor_mean == 0.25
    assert model.last_two_hop_effective_scale_mean == (
        0.25 * model.two_hop_structure_scale
    )
    assert model.last_two_hop_effective_residual_mean == (
        model.last_two_hop_residual_mean
    )
