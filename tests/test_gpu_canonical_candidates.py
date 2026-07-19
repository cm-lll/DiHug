import numpy as np
import pytest
import torch

from sparse_diffusion.structure_energy.gpu_role_guided_edit import (
    GpuTypedGraphState,
    NOOP_ID,
)
from sparse_diffusion.structure_energy.graph import FamilySpec, TypedGraph


def _toy_graph() -> TypedGraph:
    node_types = np.asarray([0] * 6 + [1] * 4, dtype=np.int64)
    return TypedGraph(
        node_types,
        ("A", "B"),
        {
            "AA": FamilySpec("AA", 0, 0, tuple(range(6)), tuple(range(6))),
            "AB": FamilySpec("AB", 0, 1, tuple(range(6)), tuple(range(6, 10))),
        },
        {
            "AA": [(0, 1), (2, 3), (4, 5)],
            "AB": [
                (0, 6),
                (0, 7),
                (1, 6),
                (1, 8),
                (2, 7),
                (2, 9),
                (3, 8),
                (3, 9),
                (4, 6),
                (4, 9),
                (5, 7),
                (5, 8),
            ],
        },
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_ordered_gpu_candidates_are_unique_and_preserve_state_invariants():
    graph = _toy_graph()
    family_names = tuple(sorted(graph.family_specs))
    target_rows = np.stack(
        [graph.family_degrees(family) for family in family_names], axis=1
    ).astype(np.float32)
    ab = family_names.index("AB")
    target_rows[0, ab] += 1
    target_rows[1, ab] -= 1
    state = GpuTypedGraphState(graph, target_rows, "cuda", use_triton=False)
    generator = torch.Generator(device="cuda")
    generator.manual_seed(23)

    for role_phase in (True, False):
        batch = state.canonical_ordered_candidates(
            8,
            generator,
            role_phase=role_phase,
            targeted_fraction=0.5,
            closure_fraction=1.0,
        )
        assert batch.action.shape == (8,)
        # Tiny graphs can exhaust the legal edit space. The backend then pads
        # with noops, while every executable edit must remain unique.
        active = batch.action != NOOP_ID
        assert int((~active).sum()) >= 1
        active_keys = state._edit_keys(batch)[active]
        assert torch.unique(active_keys, dim=0).shape[0] == active_keys.shape[0]

        selected = torch.nonzero(active)[0, 0]
        family_counts = state.family_edge_counts.clone()
        state.commit(batch, selected)
        assert torch.equal(state.family_edge_counts, family_counts)
        assert torch.equal(state.adjacency, state.adjacency.T)
        assert torch.equal(state.edge_slots >= 0, state.adjacency)
