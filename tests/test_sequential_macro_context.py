import numpy as np

from sparse_diffusion.structure_energy.graph import FamilySpec, TypedGraph
from sparse_diffusion.structure_energy.mixed_edit_diffusion import apply_edit
from sparse_diffusion.structure_energy.sequential_macro_context import (
    build_sequential_macro_dataset,
)
from sparse_diffusion.structure_energy.switch_diffusion import SwitchDenoiserMLP


def _graph() -> TypedGraph:
    types = np.asarray([0] * 8 + [1] * 6)
    specs = {
        "AB": FamilySpec("AB", 0, 1, tuple(range(8)), tuple(range(8, 14)))
    }
    edges = [(i, 8 + (i % 6)) for i in range(8)] + [
        (i, 8 + ((i + 2) % 6)) for i in range(8)
    ]
    return TypedGraph(types, ("A", "B"), specs, {"AB": edges})


def test_sequential_macro_dataset_records_ordered_exact_inverse():
    graph = _graph()
    model = SwitchDenoiserMLP(input_dim=30, hidden_dim=16, layers=2, dropout=0.0)
    dataset = build_sequential_macro_dataset(
        graph,
        model,
        trajectories=2,
        macro_steps=3,
        micro_steps=4,
        record_every=1,
        seed=7,
        feature_profile="legacy_local_v1",
        actions=("switch",),
    )
    assert len(dataset.examples) == 6
    assert dataset.macro_steps == 3
    assert dataset.micro_steps == 4
    assert all(len(item.inverse_edits) == 4 for item in dataset.examples)
    for item in dataset.examples:
        restored = item.graph.clone()
        for proposal in item.inverse_edits:
            apply_edit(restored, proposal)
        restored.validate()


def test_sequential_macro_uses_no_persistent_edge_state():
    graph = _graph()
    model = SwitchDenoiserMLP(input_dim=30, hidden_dim=16, layers=2, dropout=0.0)
    dataset = build_sequential_macro_dataset(
        graph,
        model,
        trajectories=2,
        macro_steps=1,
        micro_steps=2,
        seed=11,
        feature_profile="legacy_local_v1",
    )
    assert dataset.actions == ("switch",)
    assert dataset.node_feature_dim > 0
    assert dataset.candidate_feature_dim == model.input_dim
