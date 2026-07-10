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


def test_projection_temperature_anneals_geometrically():
    start = _Harness(epoch=0)._population_role_projection_temperature()
    middle = _Harness(epoch=50)._population_role_projection_temperature()
    end = _Harness(epoch=99)._population_role_projection_temperature()

    assert start == 0.5
    assert 0.1 < middle < 0.5
    assert abs(end - 0.1) < 1e-8
