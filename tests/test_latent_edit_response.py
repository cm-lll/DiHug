import numpy as np
import torch

from sparse_diffusion.structure_energy.latent_edit_response import (
    LocalEditResponseScorer,
    _proposal_endpoint_data,
)
from sparse_diffusion.structure_energy.mixed_edit_diffusion import EditProposal

from test_latent_edit_representation import _heterogeneous_graph


def test_proposal_endpoint_data_preserves_degree_delta_semantics():
    graph = _heterogeneous_graph()
    proposal = EditProposal("relocation", "A-B", ((0, 3),), ((0, 4),))
    node_ids, mask, delta = _proposal_endpoint_data(
        graph, [proposal], ("A-A", "A-B")
    )
    positions = {int(node): index for index, node in enumerate(node_ids[0][mask[0]])}
    scale = np.sqrt(graph.num_nodes)
    assert np.isclose(delta[0, positions[3], 1], -1.0 / scale)
    assert np.isclose(delta[0, positions[4], 1], 1.0 / scale)
    assert np.isclose(delta[0, positions[0], 1], 0.0)


def test_local_response_scorer_shapes_and_latent_gain_are_finite():
    model = LocalEditResponseScorer(
        feature_dim=7,
        hidden_dim=8,
        role_dim=3,
        scorer_hidden_dim=16,
        scorer_layers=1,
        dropout=0.0,
    )
    features = torch.randn(2, 5, 7)
    hidden = torch.randn(2, 5, 4, 8)
    prototype = torch.randn(2, 5, 4, 8)
    role_delta = torch.randn(2, 5, 4, 3)
    mask = torch.ones(2, 5, 4, dtype=torch.bool)
    base, latent, response, gain = model(
        features, hidden, prototype, role_delta, mask
    )
    assert base.shape == latent.shape == gain.shape == (2, 5)
    assert response.shape == (2, 5, 4)
    assert torch.isfinite(gain).all()
