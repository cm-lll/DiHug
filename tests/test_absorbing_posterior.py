from types import SimpleNamespace

import torch

from sparse_diffusion.diffusion_model_sparse import DiscreteDenoisingDiffusion


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
                sampling_edge_selection="absorbing_posterior_bernoulli"
            ),
        ),
    )

    try:
        harness.load_state_dict({}, strict=False)
    except RuntimeError as exc:
        assert "probability calibration" in str(exc)
    else:
        raise AssertionError("uncalibrated checkpoint was accepted")
