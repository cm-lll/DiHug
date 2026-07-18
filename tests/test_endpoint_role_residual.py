import torch

from sparse_diffusion import utils
from sparse_diffusion.models.pyhgt_denoiser import PyHGTDenoiser


def _model():
    model = PyHGTDenoiser(
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
        use_two_hop_structure=False,
        use_endpoint_role_residual=True,
        endpoint_role_hidden_dim=16,
        endpoint_role_family_dim=4,
        edge_only_model=True,
    )
    model.eval()
    return model


def _context():
    # Undirected visible G_t with one same-type family and one cross-type family.
    edge_index = torch.tensor(
        [[0, 1, 0, 2, 1, 2, 2, 3], [1, 0, 2, 0, 2, 1, 3, 2]],
        dtype=torch.long,
    )
    family = torch.tensor([0, 0, 1, 1, 1, 1, 1, 1], dtype=torch.long)
    node_type = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    batch = torch.zeros(4, dtype=torch.long)
    return edge_index, family, node_type, batch


def _encoded(model, scale=1.0):
    edge_index, family, node_type, batch = _context()
    lookup = model.build_two_hop_structure_lookup(
        edge_index,
        num_nodes=4,
        node_type_ids=node_type,
        edge_family_ids=family,
        batch=batch,
    )
    return {
        "h": torch.zeros(4, 16),
        "y_h": torch.zeros(1, 8),
        "two_hop_structure_lookup": lookup,
        "endpoint_role_scale_factor": torch.tensor([[scale]], dtype=torch.float),
    }, batch


def test_endpoint_role_lookup_is_current_gt_and_detached():
    model = _model()
    edge_index, family, node_type, batch = _context()
    lookup = model.build_two_hop_structure_lookup(
        edge_index,
        num_nodes=4,
        node_type_ids=node_type,
        edge_family_ids=family,
        batch=batch,
    )

    assert lookup["endpoint_role_total_z"].shape == (4,)
    assert lookup["endpoint_role_family_z"].shape == (
        model.endpoint_role_family_embedding.num_embeddings,
        4,
    )
    assert torch.isfinite(lookup["endpoint_role_total_z"]).all()
    assert torch.isfinite(lookup["endpoint_role_family_z"]).all()
    assert not lookup["endpoint_role_total_z"].requires_grad
    assert not lookup["endpoint_role_family_z"].requires_grad


def test_same_type_endpoint_role_residual_is_endpoint_swap_equivariant():
    torch.manual_seed(5)
    model = _model()
    with torch.no_grad():
        model.endpoint_role_head[-1].weight.normal_(std=0.05)
        model.endpoint_role_head[-1].bias.normal_(std=0.05)
    encoded, batch = _encoded(model, scale=1.0)

    query = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    query_family = torch.tensor([0, 0], dtype=torch.long)
    residual = model._endpoint_role_residual(
        encoded, query, query_family, batch
    )

    assert torch.allclose(residual[0], residual[1], atol=1e-6, rtol=1e-6)


def test_endpoint_role_scale_factor_is_linear():
    torch.manual_seed(7)
    model = _model()
    with torch.no_grad():
        model.endpoint_role_head[-1].weight.normal_(std=0.05)
        model.endpoint_role_head[-1].bias.normal_(std=0.05)
    query = torch.tensor([[0, 0, 2], [1, 2, 3]], dtype=torch.long)
    query_family = torch.tensor([0, 1, 1], dtype=torch.long)

    encoded_zero, batch = _encoded(model, scale=0.0)
    encoded_one, _ = _encoded(model, scale=1.0)
    encoded_quarter, _ = _encoded(model, scale=0.25)
    zero = model._endpoint_role_residual(encoded_zero, query, query_family, batch)
    one = model._endpoint_role_residual(encoded_one, query, query_family, batch)
    quarter = model._endpoint_role_residual(
        encoded_quarter, query, query_family, batch
    )

    assert torch.allclose(quarter - zero, 0.25 * (one - zero), atol=1e-6)
    assert model.last_endpoint_role_base_residual_mean > 0.0
    assert model.last_endpoint_role_effective_scale_mean == 0.25


def test_endpoint_role_forward_backpropagates_to_residual_head():
    torch.manual_seed(11)
    model = _model()
    x = torch.eye(4)
    edge_index, family, node_type, batch = _context()
    edge_attr = torch.nn.functional.one_hot(
        torch.tensor([1, 1, 3, 3, 3, 3, 3, 3]), num_classes=5
    ).float()
    lookup = model.build_two_hop_structure_lookup(
        edge_index,
        num_nodes=4,
        node_type_ids=node_type,
        edge_family_ids=family,
        batch=batch,
    )
    out = model(
        X=x,
        edge_attr=edge_attr,
        edge_index=edge_index,
        y=torch.tensor([[0.25, 0.75]]),
        batch=batch,
        node_type_ids=node_type,
        node_subtype_ids=torch.tensor([0, 1, 0, 1]),
        relation_type_ids=torch.tensor([0, 0, 1, 1, 1, 1, 3, 3]),
        edge_family_ids=family,
        two_hop_structure_lookup=lookup,
        endpoint_role_scale_factor=torch.tensor([[1.0]]),
    )
    out.edge_attr.square().mean().backward()

    assert model.endpoint_role_head[-1].weight.grad is not None
    assert model.endpoint_role_head[-1].weight.grad.abs().sum() > 0
