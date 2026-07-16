import numpy as np
import torch

from sparse_diffusion.structure_energy.cardinality_conditioned_edit import family_edge_counts
from sparse_diffusion.structure_energy.graph import FamilySpec, TypedGraph
from sparse_diffusion.structure_energy.macro_edit_diffusion import (
    MacroRSEditDenoiser,
    apply_commuting_edit_batch,
    proposal_nodes,
    sample_conflict_free_rs_batch,
)
from sparse_diffusion.structure_energy.mixed_edit_diffusion import inverse_edit
from sparse_diffusion.structure_energy.switch_diffusion import SwitchDenoiserMLP


def _graph() -> TypedGraph:
    types = np.asarray([0] * 8 + [1] * 6)
    specs = {"AB": FamilySpec("AB", 0, 1, tuple(range(8)), tuple(range(8, 14)))}
    edges = [(i, 8 + (i % 6)) for i in range(8)] + [(i, 8 + ((i + 2) % 6)) for i in range(8)]
    return TypedGraph(types, ("A", "B"), specs, {"AB": edges})


def test_macro_rs_batch_commutes_preserves_counts_and_has_valid_inverse():
    graph = _graph()
    before_edges = {
        family: set(graph.family_edges[family]) for family in graph.family_specs
    }
    before_counts = family_edge_counts(graph)
    batch = sample_conflict_free_rs_batch(graph, np.random.default_rng(7), budget=2)
    assert batch
    node_sets = [proposal_nodes(item) for item in batch]
    assert all(not (left & right) for i, left in enumerate(node_sets) for right in node_sets[i + 1 :])
    apply_commuting_edit_batch(graph, batch)
    assert family_edge_counts(graph) == before_counts
    apply_commuting_edit_batch(graph, tuple(inverse_edit(item) for item in batch))
    assert all(
        set(graph.family_edges[family]) == before_edges[family]
        for family in graph.family_specs
    )


def test_macro_model_zero_residual_recovers_legacy_candidate_scores():
    torch.manual_seed(13)
    legacy = SwitchDenoiserMLP(input_dim=12, hidden_dim=16, layers=2, dropout=0.0)
    model = MacroRSEditDenoiser(
        legacy,
        node_feature_dim=4,
        candidate_feature_dim=12,
        type_count=2,
        family_count=1,
        hidden_dim=16,
        heads=4,
        layers=1,
        max_budget=3,
        dropout=0.0,
    ).eval()
    assert all(not parameter.requires_grad for parameter in legacy.parameters())
    node_features = torch.randn(1, 6, 4)
    node_types = torch.tensor([[0, 0, 0, 1, 1, 1]])
    edge_index = torch.tensor([[0, 1, 2, 3], [3, 4, 5, 0]])
    edge_types = torch.zeros(4, dtype=torch.long)
    features = torch.randn(1, 3, 12)
    endpoints = torch.tensor([[[0, 3, -1, -1, 1, 4, -1, -1]] * 3])
    family = torch.zeros(1, 3, dtype=torch.long)
    time = torch.tensor([0.5])
    with torch.no_grad():
        expected = legacy(features)
        actual, _, budget = model(
            node_features,
            node_types,
            edge_index,
            edge_types,
            features,
            endpoints,
            family,
            time,
        )
    assert torch.equal(actual, expected)
    assert int(budget.argmax()) == 0
