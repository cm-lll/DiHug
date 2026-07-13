from types import SimpleNamespace

import torch
from torch import nn

from sparse_diffusion.diffusion_model_sparse import DiscreteDenoisingDiffusion
from sparse_diffusion.models.pyhgt_denoiser import PyHGTDenoiser


def test_absorbing_reverse_add_probability_uses_conditional_x0_probability():
    clean_p = torch.tensor([0.2, 0.8])
    alpha_s = torch.tensor([0.6, 0.6])
    alpha_t = torch.tensor([0.3, 0.3])

    actual = DiscreteDenoisingDiffusion._absorbing_reverse_add_probability(
        clean_p, alpha_s, alpha_t
    )

    expected = clean_p * (alpha_s - alpha_t) / (1.0 - alpha_t)
    torch.testing.assert_close(actual, expected)


def test_absorbing_reverse_final_step_recovers_clean_probability():
    clean_p = torch.tensor([0.01, 0.5, 0.99])
    alpha_t = torch.tensor([0.4, 0.4, 0.4])
    alpha_s = torch.ones_like(alpha_t)

    actual = DiscreteDenoisingDiffusion._absorbing_reverse_add_probability(
        clean_p, alpha_s, alpha_t
    )

    torch.testing.assert_close(actual, clean_p)


def test_binary_logit_shift_matches_expected_count():
    logits = torch.tensor([-4.0, -1.0, 0.0, 2.0, 5.0])
    target = 2.25

    shift = DiscreteDenoisingDiffusion._shift_binary_logits_to_expected_count(
        logits, target
    )

    torch.testing.assert_close(
        torch.sigmoid(logits + shift).sum(),
        torch.tensor(target),
        atol=1.0e-5,
        rtol=1.0e-5,
    )


def test_expected_count_calibration_reserves_locked_absorbing_edges():
    missing_logits = torch.tensor([-3.0, -0.5, 0.2, 1.5, 3.0])
    locked_count = 2.0
    clean_family_target = 4.5

    shift = DiscreteDenoisingDiffusion._shift_binary_logits_to_expected_count(
        missing_logits,
        clean_family_target - locked_count,
    )

    actual = locked_count + torch.sigmoid(missing_logits + shift).sum()
    torch.testing.assert_close(
        actual,
        torch.tensor(clean_family_target),
        atol=1.0e-5,
        rtol=1.0e-5,
    )


def test_absorbing_reverse_probability_is_zero_without_reverse_interval():
    clean_p = torch.tensor([0.3, 0.9])
    alpha = torch.tensor([0.4, 0.4])

    actual = DiscreteDenoisingDiffusion._absorbing_reverse_add_probability(
        clean_p, alpha, alpha
    )

    torch.testing.assert_close(actual, torch.zeros_like(actual))


def test_test_only_absorbing_posterior_rejects_uncalibrated_checkpoint():
    harness = object.__new__(DiscreteDenoisingDiffusion)
    object.__setattr__(
        harness,
        "cfg",
        SimpleNamespace(
            general=SimpleNamespace(test_only="old.ckpt"),
            model=SimpleNamespace(
                sampling_edge_selection="absorbing_posterior_bernoulli",
                absorbing_probability_calibration=True,
            ),
        ),
    )

    try:
        harness.load_state_dict({}, strict=False)
    except RuntimeError as exc:
        assert "probability calibration" in str(exc)
    else:
        raise AssertionError("uncalibrated checkpoint was accepted")


def test_absorbing_micro_intervals_are_contiguous_and_reach_endpoint():
    t = torch.tensor([[0.8]])
    s = torch.tensor([[0.6]])

    intervals = DiscreteDenoisingDiffusion._absorbing_micro_intervals(
        s, t, rounds=4
    )

    assert len(intervals) == 4
    torch.testing.assert_close(intervals[0][1], t)
    torch.testing.assert_close(intervals[-1][0], s)
    for previous, current in zip(intervals, intervals[1:]):
        torch.testing.assert_close(previous[0], current[1])


def _joint_bundle_head_harness():
    model = object.__new__(PyHGTDenoiser)
    nn.Module.__init__(model)
    model.use_joint_edge_bundle = True
    model.query_edge_repr_dim = 5  # dx=2, de=1
    model.lin_in_E = nn.Linear(1, 1)
    model.joint_edge_bundle_family_embedding = nn.Embedding(3, 2)
    model.joint_edge_bundle_head = nn.Sequential(
        nn.LayerNorm(20),
        nn.Linear(20, 8),
        nn.GELU(),
        nn.Linear(8, 1),
    )
    return model


def test_joint_edge_bundle_score_is_pair_permutation_invariant():
    torch.manual_seed(7)
    model = _joint_bundle_head_harness().eval()
    query_repr = torch.randn(2, 5, requires_grad=True)
    node_hidden = torch.randn(3, 2)
    edge_index = torch.tensor([[0, 0], [1, 2]])
    family = torch.tensor([0, 1])

    forward = model.score_joint_edge_bundles(
        query_repr,
        node_hidden,
        edge_index,
        torch.tensor([[0], [1]]),
        family,
    )
    reversed_pair = model.score_joint_edge_bundles(
        query_repr,
        node_hidden,
        edge_index,
        torch.tensor([[1], [0]]),
        family,
    )

    torch.testing.assert_close(forward, reversed_pair)
    forward.sum().backward()
    assert query_repr.grad is not None
    assert torch.isfinite(query_repr.grad).all()


def test_joint_edge_bundle_score_is_edge_orientation_invariant():
    torch.manual_seed(11)
    model = _joint_bundle_head_harness().eval()
    query_repr = torch.randn(2, 5)
    node_hidden = torch.randn(3, 2)
    edge_index = torch.tensor([[0, 0], [1, 2]])
    family = torch.tensor([0, 1])
    pair = torch.tensor([[0], [1]])

    forward = model.score_joint_edge_bundles(
        query_repr, node_hidden, edge_index, pair, family
    )
    reversed_repr = query_repr.clone()
    reversed_repr[:, :2] = query_repr[:, 2:4]
    reversed_repr[:, 2:4] = query_repr[:, :2]
    reversed_edges = edge_index.flip(0)
    reversed_score = model.score_joint_edge_bundles(
        reversed_repr, node_hidden, reversed_edges, pair, family
    )

    torch.testing.assert_close(forward, reversed_score)
