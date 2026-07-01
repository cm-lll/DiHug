import torch

from sparse_diffusion import utils
from sparse_diffusion.models.pyhgt_denoiser import PyHGTDenoiser


def _model():
    model = PyHGTDenoiser(
        n_layers=2,
        input_dims=utils.PlaceHolder(
            X=4, E=5, y=2, charge=0
        ),
        hidden_dims={
            "dx": 16,
            "de": 8,
            "dy": 8,
            "n_head": 4,
            "dim_ffX": 32,
            "dim_ffE": 16,
            "dim_ffy": 16,
        },
        output_dims=utils.PlaceHolder(
            X=4, E=5, y=1, charge=0
        ),
        dropout=0.0,
        sn_hidden_dim=0,
        heterogeneous=True,
        num_node_types=2,
        num_node_subtypes=4,
        num_relation_types=4,
        edge_family_offsets={"aa": 1, "ab": 3},
        type_offsets={"a": 0, "b": 2},
        use_time_film=True,
        use_edge_state_update=False,
        edge_only_model=True,
    )
    model.eval()
    return model


def _inputs():
    x = torch.eye(4)
    context_edges = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]])
    context_attr = torch.nn.functional.one_hot(
        torch.tensor([1, 1, 3, 3]), num_classes=5
    ).float()
    batch = torch.zeros(4, dtype=torch.long)
    y = torch.tensor([[0.25, 0.75]])
    node_type = torch.tensor([0, 0, 1, 1])
    node_subtype = torch.tensor([0, 1, 0, 1])
    relation = torch.tensor([0, 0, 3, 3])
    family = torch.tensor([0, 0, 1, 1])
    return (
        x,
        context_edges,
        context_attr,
        batch,
        y,
        node_type,
        node_subtype,
        relation,
        family,
    )


def test_query_edges_do_not_change_context_embeddings():
    torch.manual_seed(7)
    model = _model()
    args = _inputs()
    encoded = model.encode_context(
        X=args[0],
        edge_attr=args[2],
        edge_index=args[1],
        y=args[4],
        batch=args[3],
        node_type_ids=args[5],
        node_subtype_ids=args[6],
        relation_type_ids=args[7],
        edge_family_ids=args[8],
    )
    h_before = encoded["h"].clone()
    q1 = torch.tensor([[0, 1], [2, 3]])
    q2 = torch.tensor([[0, 2, 0, 1], [2, 3, 3, 2]])
    qattr1 = torch.nn.functional.one_hot(
        torch.zeros(q1.shape[1], dtype=torch.long), num_classes=5
    ).float()
    qattr2 = torch.nn.functional.one_hot(
        torch.zeros(q2.shape[1], dtype=torch.long), num_classes=5
    ).float()
    model.decode_queries(encoded, qattr1, q1, args[3])
    model.decode_queries(encoded, qattr2, q2, args[3])
    assert torch.equal(encoded["h"], h_before)


def test_query_current_state_changes_decoder_logits():
    torch.manual_seed(11)
    model = _model()
    args = _inputs()
    encoded = model.encode_context(
        X=args[0],
        edge_attr=args[2],
        edge_index=args[1],
        y=args[4],
        batch=args[3],
        node_type_ids=args[5],
        node_subtype_ids=args[6],
        relation_type_ids=args[7],
        edge_family_ids=args[8],
    )
    query = torch.tensor([[0], [2]])
    no_edge = torch.nn.functional.one_hot(torch.tensor([0]), 5).float()
    present = torch.nn.functional.one_hot(torch.tensor([3]), 5).float()
    logits_no_edge = model.decode_queries(encoded, no_edge, query, args[3])
    logits_present = model.decode_queries(encoded, present, query, args[3])
    assert not torch.allclose(logits_no_edge, logits_present)


def test_different_context_edges_change_embeddings():
    torch.manual_seed(13)
    model = _model()
    args = _inputs()
    encoded_a = model.encode_context(
        X=args[0],
        edge_attr=args[2],
        edge_index=args[1],
        y=args[4],
        batch=args[3],
        node_type_ids=args[5],
        node_subtype_ids=args[6],
        relation_type_ids=args[7],
        edge_family_ids=args[8],
    )
    alt_edges = torch.tensor([[0, 2, 1, 3], [2, 0, 3, 1]])
    alt_attr = torch.nn.functional.one_hot(
        torch.tensor([3, 3, 3, 3]), num_classes=5
    ).float()
    encoded_b = model.encode_context(
        X=args[0],
        edge_attr=alt_attr,
        edge_index=alt_edges,
        y=args[4],
        batch=args[3],
        node_type_ids=args[5],
        node_subtype_ids=args[6],
        relation_type_ids=torch.tensor([1, 2, 1, 2]),
        edge_family_ids=torch.tensor([1, 1, 1, 1]),
    )
    assert not torch.allclose(encoded_a["h"], encoded_b["h"])


def test_split_forward_matches_legacy_combined_forward():
    torch.manual_seed(17)
    model = _model()
    args = _inputs()
    combined = model(
        X=args[0],
        edge_attr=args[2],
        edge_index=args[1],
        y=args[4],
        batch=args[3],
        node_type_ids=args[5],
        node_subtype_ids=args[6],
        relation_type_ids=args[7],
        edge_family_ids=args[8],
    )
    encoded = model.encode_context(
        X=args[0],
        edge_attr=args[2],
        edge_index=args[1],
        y=args[4],
        batch=args[3],
        node_type_ids=args[5],
        node_subtype_ids=args[6],
        relation_type_ids=args[7],
        edge_family_ids=args[8],
    )
    split_logits = model.decode_queries(
        encoded, args[2], args[1], args[3]
    )
    assert torch.equal(combined.edge_attr, split_logits)
