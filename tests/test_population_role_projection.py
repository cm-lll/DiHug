from types import SimpleNamespace

import torch

from sparse_diffusion.diffusion_model_sparse import DiscreteDenoisingDiffusion


class _Harness:
    def __init__(self, mode="soft_quota", epoch=0):
        self.current_epoch = epoch
        self.cfg = SimpleNamespace(
            model=SimpleNamespace(
                population_role_projection_mode=mode,
                population_role_projection_temperature=0.5,
                population_role_projection_min_temperature=0.1,
                population_role_projection_anneal_epochs=100,
                population_role_projection_bisection_iters=50,
                population_role_degree_weight=1.0,
                population_role_tail_wedge_weight=1.0,
                population_role_multiset_wedge_weight=0.5,
                population_role_multiset_closure_weight=1.0,
                population_role_multiset_min_wedge=0.1,
                population_role_log_degree_temperature=0.2,
                population_role_closure_rate_huber_beta=0.1,
                gqer_score_margin=0.1,
                gqer_v2_enabled=False,
                gqer_v2_pareto_safe=True,
                gqer_v2_component_tolerance=0.02,
                gqer_v2_component_floor=0.01,
                gqer_v2_min_relative_improvement=1e-6,
                gqer_v2_pair_anchor_loss=True,
            )
        )

    def _population_role_projection_temperature(self):
        return DiscreteDenoisingDiffusion._population_role_projection_temperature(
            self
        )

    def _soft_fixed_cardinality_projection(
        self, scores, target_mass, temperature, iterations
    ):
        return DiscreteDenoisingDiffusion._soft_fixed_cardinality_projection(
            self, scores, target_mass, temperature, iterations
        )

    def _population_role_log_degree_membership(
        self, degree, max_degree, temperature=0.2
    ):
        return DiscreteDenoisingDiffusion._population_role_log_degree_membership(
            degree, max_degree, temperature
        )

    def _population_role_typed_wedge_closure(
        self, matrix, roots, first, second, same_neighbor_type=False
    ):
        return DiscreteDenoisingDiffusion._population_role_typed_wedge_closure(
            matrix,
            roots,
            first,
            second,
            same_neighbor_type=same_neighbor_type,
        )

    def _js_divergence_from_masses(self, pred_mass, target_mass, eps=1e-8):
        return DiscreteDenoisingDiffusion._js_divergence_from_masses(
            pred_mass, target_mass, eps=eps
        )

    def _compute_adaptive_multiset_population_role_loss(self, *args, **kwargs):
        return DiscreteDenoisingDiffusion._compute_adaptive_multiset_population_role_loss(
            self, *args, **kwargs
        )

    def _auxiliary_low_noise_factor(self, sparse_noisy_data, power):
        return DiscreteDenoisingDiffusion._auxiliary_low_noise_factor(
            self, sparse_noisy_data, power
        )

    def _clean_node_type_ids(self, clean_x, device):
        return clean_x.reshape(-1).long().to(device)

    def _gqer_structural_objective(self, matrix, target_matrix, clean_data):
        return DiscreteDenoisingDiffusion._gqer_structural_objective(
            self, matrix, target_matrix, clean_data
        )

    def _gqer_exist_scores(self, logits):
        return DiscreteDenoisingDiffusion._gqer_exist_scores(logits)

    def _gqer_v2_accepts_proposal(self, *args, **kwargs):
        return DiscreteDenoisingDiffusion._gqer_v2_accepts_proposal(
            self, *args, **kwargs
        )


def test_soft_projection_matches_mass_and_has_translation_invariant_gradient():
    harness = _Harness()
    scores = torch.tensor(
        [-2.0, -1.0, -0.5, 0.1, 0.8, 1.3, 2.0],
        requires_grad=True,
    )
    projected = harness._soft_fixed_cardinality_projection(
        scores,
        torch.tensor(3.0),
        temperature=0.5,
        iterations=50,
    )

    assert torch.allclose(projected.sum(), torch.tensor(3.0), atol=1e-5)
    weighted = (projected * torch.arange(1, 8, dtype=projected.dtype)).sum()
    weighted.backward()
    assert torch.isfinite(scores.grad).all()
    assert abs(float(scores.grad.sum())) < 1e-5


def test_grouped_soft_projection_matches_each_graph_family_quota():
    harness = _Harness()
    scores = torch.linspace(-3.0, 3.0, 12, requires_grad=True)
    raw_prob = torch.sigmoid(scores)
    target = torch.tensor(
        [1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0],
        dtype=torch.float32,
    )
    family = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1])
    graph = torch.zeros(12, dtype=torch.long)

    projected, stats = (
        DiscreteDenoisingDiffusion._project_population_role_probabilities(
            harness,
            scores,
            raw_prob,
            target,
            family,
            graph,
        )
    )

    assert torch.allclose(projected[:5].sum(), torch.tensor(2.0), atol=1e-5)
    assert torch.allclose(projected[5:].sum(), torch.tensor(3.0), atol=1e-5)
    assert stats["projection_groups"] == 2.0
    assert stats["projection_mass_mae"] < 1e-5
    projected.square().sum().backward()
    assert torch.isfinite(scores.grad).all()


def test_straight_through_mode_is_hard_in_forward_and_soft_in_backward():
    harness = _Harness(mode="straight_through")
    scores = torch.linspace(-2.0, 2.0, 8, requires_grad=True)
    raw_prob = torch.sigmoid(scores)
    target = torch.tensor([1, 1, 1, 0, 0, 0, 0, 0], dtype=torch.float32)
    family = torch.zeros(8, dtype=torch.long)
    graph = torch.zeros(8, dtype=torch.long)

    projected, _ = (
        DiscreteDenoisingDiffusion._project_population_role_probabilities(
            harness,
            scores,
            raw_prob,
            target,
            family,
            graph,
        )
    )

    assert torch.equal(projected.detach(), (projected.detach() > 0.5).float())
    assert float(projected.detach().sum()) == 3.0
    (projected * torch.arange(8, dtype=projected.dtype)).sum().backward()
    assert torch.isfinite(scores.grad).all()
    assert float(scores.grad.abs().sum()) > 0.0


def test_projection_mode_override_compares_soft_and_hard_without_mutating_config():
    harness = _Harness(mode="raw")
    scores = torch.linspace(-2.0, 2.0, 8)
    raw_prob = torch.sigmoid(scores)
    target = torch.tensor([1, 1, 1, 0, 0, 0, 0, 0], dtype=torch.float32)
    family = torch.zeros(8, dtype=torch.long)
    graph = torch.zeros(8, dtype=torch.long)

    soft, _ = DiscreteDenoisingDiffusion._project_population_role_probabilities(
        harness,
        scores,
        raw_prob,
        target,
        family,
        graph,
        mode_override="soft_quota",
    )
    hard, _ = DiscreteDenoisingDiffusion._project_population_role_probabilities(
        harness,
        scores,
        raw_prob,
        target,
        family,
        graph,
        mode_override="straight_through",
    )

    assert harness.cfg.model.population_role_projection_mode == "raw"
    assert torch.allclose(soft.sum(), torch.tensor(3.0), atol=1e-5)
    assert float(hard.sum()) == 3.0
    assert torch.equal(hard, (hard > 0.5).float())
    assert not torch.equal(soft, hard)


def test_population_role_probe_gap_reports_soft_hard_metric_difference():
    gap = DiscreteDenoisingDiffusion._population_role_probe_gap(
        {"tail_wedge_log_mae": 0.2, "closure_rate_mae": 0.05},
        {"tail_wedge_log_mae": 0.7, "closure_rate_mae": 0.2},
    )
    assert abs(gap["wedge_log_mae"] - 0.5) < 1e-8
    assert abs(gap["closure_rate_mae"] - 0.15) < 1e-8


def test_projection_temperature_anneals_geometrically():
    start = _Harness(epoch=0)._population_role_projection_temperature()
    middle = _Harness(epoch=50)._population_role_projection_temperature()
    end = _Harness(epoch=99)._population_role_projection_temperature()

    assert start == 0.5
    assert 0.1 < middle < 0.5
    assert abs(end - 0.1) < 1e-8


def _triangle_matrix(num_nodes=3):
    matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    for u, v in ((0, 1), (0, 2), (1, 2)):
        matrix[u, v] = 1.0
        matrix[v, u] = 1.0
    return matrix


def test_typed_multiset_wedge_closure_covers_all_type_multiplicities():
    matrix = _triangle_matrix()
    roots = torch.tensor([0])

    # AAA: the root and both neighbors share one type.
    wedge, closed = DiscreteDenoisingDiffusion._population_role_typed_wedge_closure(
        matrix,
        roots,
        torch.tensor([0, 1, 2]),
        torch.tensor([0, 1, 2]),
        same_neighbor_type=True,
    )
    assert torch.allclose(wedge, torch.ones(1))
    assert torch.allclose(closed, torch.ones(1))

    # AAB/ABB: one root sees two same-type neighbors; another root sees
    # two distinct neighbor types. No dataset-specific type name is involved.
    wedge_same, closed_same = (
        DiscreteDenoisingDiffusion._population_role_typed_wedge_closure(
            matrix,
            torch.tensor([2]),
            torch.tensor([0, 1]),
            torch.tensor([0, 1]),
            same_neighbor_type=True,
        )
    )
    wedge_distinct, closed_distinct = (
        DiscreteDenoisingDiffusion._population_role_typed_wedge_closure(
            matrix,
            roots,
            torch.tensor([1]),
            torch.tensor([2]),
            same_neighbor_type=False,
        )
    )
    assert torch.allclose(wedge_same, torch.ones(1))
    assert torch.allclose(closed_same, torch.ones(1))
    assert torch.allclose(wedge_distinct, torch.ones(1))
    assert torch.allclose(closed_distinct, torch.ones(1))

    # ABC uses the same distinct-neighbor formula as above.
    assert torch.allclose(wedge_distinct, closed_distinct)


def test_typed_multiset_statistics_are_permutation_invariant_and_differentiable():
    matrix = _triangle_matrix().requires_grad_(True)
    roots = torch.tensor([0])
    wedge, closed = DiscreteDenoisingDiffusion._population_role_typed_wedge_closure(
        matrix,
        roots,
        torch.tensor([1]),
        torch.tensor([2]),
        same_neighbor_type=False,
    )

    permutation = torch.tensor([2, 0, 1])
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(3)
    permuted = matrix.detach().index_select(0, permutation).index_select(1, permutation)
    perm_wedge, perm_closed = (
        DiscreteDenoisingDiffusion._population_role_typed_wedge_closure(
            permuted,
            inverse[roots],
            inverse[torch.tensor([1])],
            inverse[torch.tensor([2])],
            same_neighbor_type=False,
        )
    )
    assert torch.allclose(wedge.detach(), perm_wedge)
    assert torch.allclose(closed.detach(), perm_closed)

    (wedge + closed).sum().backward()
    assert matrix.grad is not None
    assert torch.isfinite(matrix.grad).all()


def test_log_degree_bins_cover_long_tails_without_a_fixed_cap():
    degree = torch.tensor([0.0, 1.0, 2.0, 3.0, 6.0, 12.0, 64.0, 674.0])
    membership, centers = (
        DiscreteDenoisingDiffusion._population_role_log_degree_membership(
            degree,
            max_degree=1024,
            temperature=0.2,
        )
    )
    assert torch.allclose(membership.sum(dim=-1), torch.ones_like(degree))
    assert torch.isfinite(membership).all()
    assert float(centers.max()) > 500.0
    assert any(abs(float(value) - 2.0) < 1e-6 for value in centers)


def test_adaptive_multiset_loss_is_zero_on_target_and_has_finite_gradient():
    harness = _Harness()
    target = _triangle_matrix()
    pred = target.clone().requires_grad_(True)
    degree = pred.sum(dim=-1)
    target_degree = target.sum(dim=-1)
    loss, stats = harness._compute_adaptive_multiset_population_role_loss(
        torch.zeros((1, 2)),
        pred,
        target,
        degree,
        target_degree,
        torch.ones(3, dtype=torch.bool),
        torch.tensor([0, 0, 1]),
        2,
        {"projection_groups": 1.0},
    )
    assert float(loss.detach()) < 1e-7
    assert stats["active_multiset_groups"] > 0
    loss.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()


def test_low_noise_factor_suppresses_high_noise_structure_supervision():
    harness = _Harness()
    low_noise = harness._auxiliary_low_noise_factor(
        {"t_float": torch.tensor([0.1])}, 2.0
    )
    high_noise = harness._auxiliary_low_noise_factor(
        {"t_float": torch.tensor([0.9])}, 2.0
    )
    assert abs(low_noise - 0.81) < 1e-6
    assert abs(high_noise - 0.01) < 1e-6
    assert low_noise > high_noise


def test_gqer_structural_utility_prefers_missing_triangle_closure_edge():
    harness = _Harness()
    harness.cfg.model.population_role_degree_weight = 0.0
    harness.cfg.model.population_role_tail_wedge_weight = 0.0
    harness.cfg.model.population_role_multiset_wedge_weight = 0.0
    harness.cfg.model.population_role_multiset_closure_weight = 1.0

    target = torch.zeros((4, 4), dtype=torch.float32)
    for u, v in ((0, 1), (0, 2), (1, 2)):
        target[u, v] = target[v, u] = 1.0
    pred = target.clone()
    pred[1, 2] = pred[2, 1] = 0.0
    pred.requires_grad_(True)
    clean = SimpleNamespace(x=torch.tensor([0, 1, 2, 2]))

    objective, stats = harness._gqer_structural_objective(pred, target, clean)
    grad = torch.autograd.grad(objective, pred)[0]
    utility = -(grad + grad.T)

    assert stats["active_multiset_groups"] > 0
    assert float(utility[1, 2]) > 0.0
    assert float(utility[1, 2]) > float(utility[1, 3])


def test_gqer_block_rank_loss_pushes_exchange_across_global_threshold():
    harness = _Harness()
    edges = torch.tensor([[0, 0, 1, 2], [1, 2, 2, 3]])
    logits = torch.tensor(
        [[0.0, 1.0], [0.0, -1.0], [0.0, 0.5], [0.0, -0.5]],
        requires_grad=True,
    )
    pred = SimpleNamespace(edge_attr=logits, edge_index=edges)
    keys = torch.tensor([1, 2, 6, 11])
    state = {
        "keys": keys,
        "owners": torch.zeros(4, dtype=torch.long),
        "selected": torch.tensor([True, False, True, False]),
        "proposed": torch.tensor([False, True, True, False]),
        "thresholds": torch.full((4,), 0.25),
        "swap_weight": torch.tensor([1.0, 1.0, 0.0, 0.0]),
    }
    clean = SimpleNamespace(x=torch.zeros(4))

    loss, stats = DiscreteDenoisingDiffusion._compute_gqer_block_rank_loss(
        harness, pred, clean, state, block_id=0
    )
    loss.backward()

    assert stats["pairs"] == 1.0
    assert logits.grad[0, 1] > 0.0
    assert logits.grad[1, 1] < 0.0


def test_gqer_v2_pareto_acceptance_rejects_closure_tradeoff():
    harness = _Harness()
    harness.cfg.model.gqer_v2_enabled = True
    accepted, reason = harness._gqer_v2_accepts_proposal(
        0.30,
        0.20,
        {
            "degree": 0.20,
            "tail_wedge": 0.30,
            "motif_wedge": 0.25,
            "closure": 0.10,
        },
        {
            "degree": 0.15,
            "tail_wedge": 0.20,
            "motif_wedge": 0.20,
            "closure": 0.20,
        },
    )
    assert not accepted
    assert reason == "component:closure"


def test_gqer_v2_pareto_acceptance_requires_largest_residual_to_improve():
    harness = _Harness()
    harness.cfg.model.gqer_v2_enabled = True
    accepted, reason = harness._gqer_v2_accepts_proposal(
        0.30,
        0.25,
        {
            "degree": 0.10,
            "tail_wedge": 0.30,
            "motif_wedge": 0.20,
            "closure": 0.10,
        },
        {
            "degree": 0.08,
            "tail_wedge": 0.30,
            "motif_wedge": 0.15,
            "closure": 0.09,
        },
    )
    assert not accepted
    assert reason == "primary:tail_wedge"


def test_gqer_v2_pair_anchor_pushes_actual_add_above_drop():
    harness = _Harness()
    harness.cfg.model.gqer_v2_enabled = True
    edges = torch.tensor([[0, 0], [1, 2]])
    logits = torch.tensor(
        [[0.0, 0.8], [0.0, 0.2]],
        requires_grad=True,
    )
    pred = SimpleNamespace(edge_attr=logits, edge_index=edges)
    state = {
        "keys": torch.tensor([1, 2]),
        "owners": torch.zeros(2, dtype=torch.long),
        "selected": torch.tensor([True, False]),
        "proposed": torch.tensor([False, True]),
        # Deliberately uninformative family threshold. V2 must use the paired
        # first-pass anchors below instead.
        "thresholds": torch.zeros(2),
        "pair_anchors": torch.tensor([0.2, 0.8]),
        "swap_weight": torch.ones(2),
    }
    clean = SimpleNamespace(x=torch.zeros(3))

    loss, stats = DiscreteDenoisingDiffusion._compute_gqer_block_rank_loss(
        harness, pred, clean, state, block_id=0
    )
    loss.backward()

    assert stats["pairs"] == 1.0
    assert logits.grad[0, 1] > 0.0
    assert logits.grad[1, 1] < 0.0


def test_exact_sequential_terminal_projection_preserves_quota_and_closes_triangle():
    harness = _Harness()
    harness.local_rank = 1
    harness.cfg.model = SimpleNamespace(
        sampling_structure_swap_limiter=True,
        sampling_structure_swap_limiter_weight=1.0,
        sampling_structure_swap_limiter_max_swaps=1,
        sampling_structure_swap_limiter_min_step_frac=0.0,
        sampling_structure_swap_limiter_max_step_frac=1.0,
        sampling_structure_swap_limiter_exact_sequential=True,
        sampling_structure_limiter_target_triangles=1.0,
        sampling_structure_limiter_triangle_weight=1.0,
        sampling_structure_limiter_clustering_weight=0.0,
        sampling_structure_limiter_lcc_weight=0.0,
        sampling_structure_limiter_assort_weight=0.0,
        sampling_structure_limiter_degree_weight=0.0,
        sampling_structure_limiter_triangle_tolerance=0.0,
        sampling_structure_swap_limiter_model_score_weight=0.0,
        sampling_structure_swap_limiter_gain_scale=5.0,
        sampling_structure_swap_limiter_score_margin=2.0,
        sampling_structure_swap_limiter_pool_multiplier=4,
        sampling_structure_swap_limiter_pair_top_adds=8,
        sampling_structure_swap_limiter_pair_top_drops=8,
        diffusion_steps=100,
    )
    harness.dataset_info = SimpleNamespace()
    harness._sampling_structure_target_value = (
        lambda metric, names: 1.0 if metric == "triangles" else None
    )
    harness._ensure_sampling_degree_pair_reference = lambda: None
    harness._sampling_lcc_floor_target = lambda: (None, None)

    # Selected path: 0-1-2-3. Swapping 2-3 for 0-2 closes one triangle while
    # retaining the same graph/family cardinality.
    edge_index = torch.tensor(
        [[0, 1, 2, 0, 0, 1], [1, 2, 3, 2, 3, 3]], dtype=torch.long
    )
    selected = torch.tensor([0, 1, 2], dtype=torch.long)
    edge_keys = edge_index[0] * 4 + edge_index[1]
    scores = torch.tensor([1.0, 1.0, 0.1, 0.09, 0.0, 0.0])
    data = SimpleNamespace(
        node=torch.zeros((4, 1)),
        batch=torch.zeros(4, dtype=torch.long),
    )

    projected = DiscreteDenoisingDiffusion._apply_global_exact_k_structure_swap_limiter(
        harness,
        data=data,
        edge_index=edge_index,
        edge_keys=edge_keys,
        raw_scores=scores,
        selection_scores=scores,
        families=torch.zeros(6, dtype=torch.long),
        edge_batch=torch.zeros(6, dtype=torch.long),
        selected=selected,
        sampling_step=1,
        log_selection=False,
    )

    assert projected.numel() == selected.numel()
    assert 3 in projected.tolist()
    assert 2 not in projected.tolist()


def test_family_inference_prefers_supported_canonical_over_zero_reverse():
    harness = _Harness()
    harness.dataset_info = SimpleNamespace(
        type_offsets={"A": 0, "B": 1},
        fam_endpoints={
            "link_A_B": {"src_type": "A", "dst_type": "B"},
            "link_B_A": {"src_type": "B", "dst_type": "A"},
        },
        edge_family_offsets={"link_A_B": 1, "link_B_A": 2},
        edge_family_avg_edge_counts={"link_A_B": 5.0, "link_B_A": 0.0},
    )
    edges = torch.tensor([[0, 1], [1, 0]])
    _, family = DiscreteDenoisingDiffusion._infer_query_family_from_clean_node_types(
        harness, edges, torch.tensor([0, 1])
    )

    assert torch.equal(family, torch.zeros(2, dtype=torch.long))
