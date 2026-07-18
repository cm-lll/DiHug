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
        num_relation_types=2,
        edge_family_offsets={"aa": 1, "ab": 3},
        type_offsets={"a": 0, "b": 2},
        use_iterative_motif_decoder=True,
        iterative_motif_rounds=2,
        iterative_motif_shortlist_ratio=0.5,
        iterative_motif_shortlist_min=1,
        iterative_motif_shortlist_max=8,
        edge_only_model=True,
    )
    model.eval()
    return model


def _encoded(model):
    x = torch.eye(4)
    context = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    context_attr = torch.nn.functional.one_hot(
        torch.tensor([1, 1]), num_classes=5
    ).float()
    return model.encode_context(
        X=x,
        edge_attr=context_attr,
        edge_index=context,
        y=torch.tensor([[0.25, 0.75]]),
        batch=torch.zeros(4, dtype=torch.long),
        node_type_ids=torch.tensor([0, 1, 0, 1]),
        node_subtype_ids=torch.tensor([0, 0, 1, 1]),
        relation_type_ids=torch.zeros(2, dtype=torch.long),
        edge_family_ids=torch.zeros(2, dtype=torch.long),
    )


def test_zero_initialized_refinement_preserves_base_logits():
    torch.manual_seed(3)
    model = _model()
    encoded = _encoded(model)
    query = torch.tensor([[0, 1, 0, 2], [1, 2, 2, 3]], dtype=torch.long)
    query_attr = torch.nn.functional.one_hot(
        torch.zeros(query.shape[1], dtype=torch.long), num_classes=5
    ).float()
    batch = torch.zeros(4, dtype=torch.long)
    family = torch.zeros(query.shape[1], dtype=torch.long)

    model.use_iterative_motif_decoder = False
    base = model.decode_queries(encoded, query_attr, query, batch, family)
    model.use_iterative_motif_decoder = True
    refined = model.decode_queries(encoded, query_attr, query, batch, family)

    assert torch.equal(base, refined)
    assert model.last_iterative_motif_shortlist_edges == 2


def test_predicted_shortlist_exposes_typed_closure_support():
    model = _model()
    encoded = _encoded(model)
    query = torch.tensor([[0, 1, 0, 2], [1, 2, 2, 3]], dtype=torch.long)
    family = torch.zeros(query.shape[1], dtype=torch.long)
    batch = torch.zeros(4, dtype=torch.long)
    # Edges 0-1 and 1-2 form the top-ranked predicted path supporting 0-2.
    exist_scores = torch.tensor([6.0, 5.0, -2.0, -3.0])
    logits = torch.stack(
        [torch.zeros_like(exist_scores), exist_scores], dim=-1
    )
    logits = torch.cat([logits, logits.new_zeros((4, 3))], dim=-1)

    features, support, shortlist = model._predicted_motif_features(
        logits, encoded, query, batch, family
    )

    assert shortlist == 2
    assert support[2] > 0
    assert support[2] > support[0]
    # Feature 0 is the current score, 1 total support, then middle-node types.
    assert features[2, 1] > 0
    assert features[2, 3] > 0  # middle node 1 has type 1
    assert features[2, 2] == 0


def test_family_exist_score_ignores_invalid_relation_classes():
    model = _model()
    logits = torch.tensor(
        [
            [0.0, 2.0, -1.0, -2.0, 20.0],
            [0.0, 20.0, 20.0, 3.0, -1.0],
        ]
    )
    scores = model._family_exist_scores(logits, torch.tensor([0, 1]))

    assert scores[0] < 3.0
    assert scores[1] < 4.0


def test_refinement_backpropagates_through_predicted_support():
    torch.manual_seed(5)
    model = _model()
    encoded = _encoded(model)
    query = torch.tensor([[0, 1, 0, 2], [1, 2, 2, 3]], dtype=torch.long)
    family = torch.zeros(query.shape[1], dtype=torch.long)
    batch = torch.zeros(4, dtype=torch.long)
    logits = torch.randn(4, 5, requires_grad=True)

    refined = model._iterative_motif_refinement(
        logits, encoded, query, batch, family
    )
    refined[:, 1:].sum().backward()

    for head in model.iterative_motif_heads:
        assert head[-1].weight.grad is not None
        assert head[-1].weight.grad.abs().sum() > 0
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
