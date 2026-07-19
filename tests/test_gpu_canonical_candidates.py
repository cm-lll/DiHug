import numpy as np
import pytest
import torch

from sparse_diffusion.structure_energy.gpu_role_guided_edit import (
    GpuTypedGraphState,
    NOOP_ID,
    sample_gpu_role_guided_edit_diffusion,
    sample_gpu_role_guided_edit_diffusion_batched,
)
from sparse_diffusion.structure_energy.graph import FamilySpec, TypedGraph
from sparse_diffusion.structure_energy.mixed_edit_diffusion import (
    mixed_edit_feature_width,
)
from sparse_diffusion.structure_energy.switch_diffusion import SwitchDenoiserMLP
from sparse_diffusion.structure_energy.rooted_context_edit import (
    RootedContextEditResidual,
    _teacher_distribution,
)


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_batched_gpu_chains_match_independent_transition_semantics():
    graph = _toy_graph()
    family_names = tuple(sorted(graph.family_specs))
    target_rows = np.stack(
        [graph.family_degrees(family) for family in family_names], axis=1
    ).astype(np.float32)
    input_dim = mixed_edit_feature_width(
        "typed_closure_compact_v2",
        family_count=len(family_names),
        type_count=len(graph.type_names),
    )
    torch.manual_seed(5)
    model = SwitchDenoiserMLP(
        input_dim=input_dim, hidden_dim=16, layers=1, dropout=0.0
    ).cuda().eval()
    seeds = (31, 47)
    kwargs = dict(
        steps=12,
        candidates_per_step=8,
        device="cuda",
        metrics_every=12,
        role_weight=0.05,
        gain_scale=10.0,
        schedule_power=1.0,
        schedule_floor=1.0,
        role_error_per_node_threshold=0.0,
        model_time_horizon_steps=24,
        model_time_start_step=0,
        closure_candidate_fraction=1.0,
        targeted_role_candidate_fraction=0.5,
        auto_role_balance=0.0,
        candidate_mode="canonical_ordered",
        use_triton=False,
    )
    independent = [
        sample_gpu_role_guided_edit_diffusion(
            graph.clone(), model, target_rows.copy(), seed=seed, **kwargs
        )
        for seed in seeds
    ]
    batched = sample_gpu_role_guided_edit_diffusion_batched(
        [graph.clone(), graph.clone()],
        model,
        [target_rows.copy(), target_rows.copy()],
        seeds=seeds,
        **kwargs,
    )
    for expected, actual in zip(independent, batched):
        assert expected.graph.family_edge_sets == actual.graph.family_edge_sets
        assert expected.action_counts == actual.action_counts


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_batched_context_has_no_cross_chain_messages_or_transition_drift():
    graph = _toy_graph()
    family_names = tuple(sorted(graph.family_specs))
    target_rows = np.stack(
        [graph.family_degrees(family) for family in family_names], axis=1
    ).astype(np.float32)
    input_dim = mixed_edit_feature_width(
        "typed_closure_compact_v2",
        family_count=len(family_names),
        type_count=len(graph.type_names),
    )
    torch.manual_seed(11)
    model = SwitchDenoiserMLP(
        input_dim=input_dim, hidden_dim=16, layers=1, dropout=0.0
    ).cuda().eval()
    context = RootedContextEditResidual(
        model,
        node_feature_dim=len(family_names) + 3,
        candidate_feature_dim=input_dim,
        type_count=len(graph.type_names),
        family_count=len(family_names),
        hidden_dim=8,
        heads=2,
        layers=1,
        dropout=0.0,
    ).cuda().eval()
    seeds = (13, 29)
    kwargs = dict(
        steps=8,
        candidates_per_step=8,
        device="cuda",
        metrics_every=8,
        role_weight=0.05,
        gain_scale=10.0,
        schedule_power=1.0,
        schedule_floor=1.0,
        role_error_per_node_threshold=1.0e9,
        model_time_horizon_steps=16,
        model_time_start_step=0,
        closure_candidate_fraction=1.0,
        targeted_role_candidate_fraction=0.5,
        context_model=context,
        context_refresh_steps=2,
        candidate_mode="canonical_ordered",
        use_triton=False,
    )
    independent = [
        sample_gpu_role_guided_edit_diffusion(
            graph.clone(), model, target_rows.copy(), seed=seed, **kwargs
        )
        for seed in seeds
    ]
    batched = sample_gpu_role_guided_edit_diffusion_batched(
        [graph.clone(), graph.clone()],
        model,
        [target_rows.copy(), target_rows.copy()],
        seeds=seeds,
        **kwargs,
    )
    for expected, actual in zip(independent, batched):
        assert expected.graph.family_edge_sets == actual.graph.family_edge_sets
        assert expected.action_counts == actual.action_counts


def test_rooted_context_teacher_is_scale_invariant_for_large_graph_utilities():
    utilities = torch.tensor(
        [[0.0, 1.0, -0.5, 0.25], [2.0, 2.0, 2.0, 2.0]],
        dtype=torch.float32,
    )
    reference = _teacher_distribution(utilities, temperature=0.5)
    tiny = _teacher_distribution(utilities * 1.0e-8, temperature=0.5)
    assert torch.allclose(reference[0], tiny[0], atol=1.0e-6, rtol=1.0e-6)
    assert torch.allclose(tiny[1], torch.full((4,), 0.25))
