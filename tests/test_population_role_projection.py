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
