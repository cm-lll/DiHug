import numpy as np
import torch

from sparse_diffusion.structure_energy.edge_energy_edit_gate import (
    EdgeEnergyInferenceResidual,
    EdgeEnergyFusion,
    candidate_edge_energy_delta,
)
from sparse_diffusion.structure_energy.latent_edit_representation import (
    WarmupHGTEncoder,
)
from sparse_diffusion.structure_energy.mixed_edit_diffusion import EditProposal

from test_latent_edit_representation import _heterogeneous_graph


def test_edge_energy_fusion_zero_initialization_is_exact_baseline():
    fusion = EdgeEnergyFusion()
    base = torch.randn(2, 5)
    delta = torch.randn(2, 5)
    actions = torch.tensor([[0, 1, 2, 1, 2], [2, 1, 0, 2, 1]])
    assert torch.equal(fusion(base, delta, actions), base)


def test_edge_energy_fusion_uses_action_specific_nonnegative_weights():
    fusion = EdgeEnergyFusion()
    with torch.no_grad():
        fusion.action_weights.copy_(torch.tensor([2.0, 3.0]))
    base = torch.zeros(1, 3)
    delta = torch.tensor([[5.0, 7.0, 11.0]])
    actions = torch.tensor([[0, 1, 2]])
    scores = fusion(base, delta, actions)
    assert torch.allclose(scores, torch.tensor([[0.0, 14.0, 33.0]]))
    with torch.no_grad():
        fusion.action_weights.copy_(torch.tensor([-1.0, 2.0]))
    fusion.project_nonnegative()
    assert torch.equal(fusion.action_weights, torch.tensor([0.0, 2.0]))


def test_candidate_edge_energy_delta_matches_complete_edit_semantics():
    graph = _heterogeneous_graph()
    model = WarmupHGTEncoder(
        node_feature_dim=4,
        type_count=2,
        family_count=2,
        hidden_dim=8,
        heads=2,
        layers=1,
        dropout=0.0,
    )

    def deterministic_logits(hidden, query_edges, query_families, current, time):
        del hidden, query_families, current, time
        return query_edges[0].float() * 10.0 + query_edges[1].float()

    model.edge_logits = deterministic_logits
    hidden = torch.zeros(graph.num_nodes, 8)
    candidates = [
        EditProposal("noop", "A-A", (), ()),
        EditProposal("relocation", "A-B", ((0, 3),), ((2, 4),)),
        EditProposal(
            "switch", "A-B", ((0, 3), (1, 4)), ((0, 4), (1, 3))
        ),
    ]
    delta = candidate_edge_energy_delta(
        graph,
        candidates,
        representation=model,
        hidden=hidden,
        family_names=("A-A", "A-B"),
        time_fraction=0.5,
        device=torch.device("cpu"),
    )
    assert np.isclose(delta[0], 0.0)
    assert np.isclose(delta[1], 24.0 - 3.0)
    expected_switch = (4.0 + 13.0) / 2.0 - (3.0 + 14.0) / 2.0
    assert np.isclose(delta[2], expected_switch)


def test_zero_weight_inference_residual_is_exactly_zero():
    graph = _heterogeneous_graph()
    model = WarmupHGTEncoder(
        node_feature_dim=5,
        type_count=2,
        family_count=2,
        hidden_dim=8,
        heads=2,
        layers=1,
        dropout=0.0,
        role_condition_dim=2,
    )
    residual = EdgeEnergyInferenceResidual(
        model,
        family_names=("A-A", "A-B"),
        role_means=np.zeros((2, 2), dtype=np.float32),
        role_scales=np.ones((2, 2), dtype=np.float32),
        action_scales=(1.0, 1.0),
        action_weights=(0.0, 0.0),
    )
    hidden = torch.zeros(graph.num_nodes, 8)
    candidates = [
        EditProposal("noop", "A-A", (), ()),
        EditProposal("relocation", "A-B", ((0, 3),), ((2, 4),)),
    ]
    scores = residual.score_graph_candidates(
        graph,
        candidates,
        0.5,
        torch.device("cpu"),
        encoded=hidden,
    )
    assert torch.equal(scores, torch.zeros_like(scores))
