import torch
from torch import nn

from sparse_diffusion.structure_energy.global_context_joint_edit import (
    GlobalContextJointEditResidual,
    decode_queries_in_chunks,
)
from sparse_diffusion.structure_energy.switch_diffusion import SwitchDenoiserMLP


class _ToyQueryDecoder(nn.Module):
    def decode_queries(
        self,
        *,
        encoded,
        query_edge_attr,
        query_edge_index,
        batch,
        query_edge_family_ids=None,
        query_edge_stage_ids=None,
        edge_input_residual_scale=None,
    ):
        del batch, query_edge_stage_ids, edge_input_residual_scale
        h = encoded["h"]
        src = h[query_edge_index[0]]
        dst = h[query_edge_index[1]]
        family = (
            query_edge_attr.new_zeros((query_edge_attr.shape[0], 1))
            if query_edge_family_ids is None
            else query_edge_family_ids[:, None].to(query_edge_attr.dtype)
        )
        return torch.cat((src + dst, query_edge_attr[:, :1], family), dim=-1)


def test_chunked_query_decode_matches_full_and_is_order_invariant():
    torch.manual_seed(3)
    model = _ToyQueryDecoder()
    encoded = {"h": torch.randn(9, 4)}
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6, 7]], dtype=torch.long
    )
    attr = torch.randn(7, 3)
    family = torch.tensor([0, 1, 0, 2, 1, 2, 0])
    batch = torch.zeros(9, dtype=torch.long)
    full = model.decode_queries(
        encoded=encoded,
        query_edge_attr=attr,
        query_edge_index=edge_index,
        batch=batch,
        query_edge_family_ids=family,
    )
    forward = decode_queries_in_chunks(
        model,
        encoded=encoded,
        query_edge_attr=attr,
        query_edge_index=edge_index,
        batch=batch,
        query_edge_family_ids=family,
        chunk_size=3,
    )
    reverse = decode_queries_in_chunks(
        model,
        encoded=encoded,
        query_edge_attr=attr,
        query_edge_index=edge_index,
        batch=batch,
        query_edge_family_ids=family,
        chunk_size=3,
        chunk_order=(2, 1, 0),
    )
    assert torch.equal(full, forward)
    assert torch.equal(full, reverse)


def test_zero_initialized_global_context_residual_exactly_recovers_legacy_mlp():
    torch.manual_seed(5)
    legacy = SwitchDenoiserMLP(input_dim=11, hidden_dim=16, layers=2, dropout=0.0)
    model = GlobalContextJointEditResidual(
        legacy,
        node_hidden_dim=8,
        candidate_feature_dim=11,
        edge_slot_dim=6,
        hidden_dim=16,
    ).eval()
    features = torch.randn(2, 5, 11)
    node_hidden = torch.randn(2, 10, 8)
    endpoints = torch.tensor(
        [
            [[0, 1, 2, 3, 4, 5, 6, 7]] * 5,
            [[1, 2, 3, 4, 5, 6, -1, -1]] * 5,
        ]
    )
    edge_hidden = torch.randn(2, 5, 4, 6)
    with torch.no_grad():
        expected = legacy(features)
        actual = model(
            features,
            node_hidden=node_hidden,
            candidate_endpoints=endpoints,
            edge_slot_hidden=edge_hidden,
        )
    assert torch.equal(expected, actual)


def test_joint_residual_can_learn_from_all_removed_and_added_slots():
    torch.manual_seed(7)
    legacy = SwitchDenoiserMLP(input_dim=9, hidden_dim=12, layers=1, dropout=0.0)
    model = GlobalContextJointEditResidual(
        legacy,
        node_hidden_dim=6,
        candidate_feature_dim=9,
        hidden_dim=12,
    )
    with torch.no_grad():
        final = model.residual[-1]
        assert isinstance(final, nn.Linear)
        final.weight.normal_(std=0.1)
    features = torch.zeros(1, 2, 9)
    node_hidden = torch.randn(1, 8, 6)
    endpoints = torch.tensor(
        [[[0, 1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 3, 4, 5, -1, -1]]]
    )
    logits = model(
        features, node_hidden=node_hidden, candidate_endpoints=endpoints
    )
    assert logits.shape == (1, 2)
    assert not torch.equal(logits[:, 0], logits[:, 1])
