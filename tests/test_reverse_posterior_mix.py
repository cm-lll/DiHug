from types import SimpleNamespace

import torch

from sparse_diffusion.diffusion_model_sparse import (
    DiscreteDenoisingDiffusion,
)
from sparse_diffusion.diffusion.noise_schedule import (
    PredefinedNoiseScheduleDiscrete,
)


class _Harness:
    T = 100

    def __init__(self, weights, mode="full", scale=1.0):
        self.cfg = SimpleNamespace(
            model=SimpleNamespace(
                sampling_reverse_posterior_mix_weights=weights,
                sampling_reverse_posterior_mix_mode=mode,
                sampling_reverse_posterior_mix_scale=scale,
            )
        )
        self.noise_schedule = PredefinedNoiseScheduleDiscrete(
            noise_schedule="cosine",
            timesteps=100,
            skip=1,
        )

    def _sampling_time_transitions(self):
        return [(100, 75), (75, 50), (50, 25), (25, 0)]


def _weight(harness, t_value, s_value):
    return DiscreteDenoisingDiffusion._reverse_posterior_mix_weight(
        harness,
        s_float=torch.tensor([[s_value]], dtype=torch.float32),
        t_float=torch.tensor([[t_value]], dtype=torch.float32),
    )


def test_null_weights_retain_full_reverse_posterior():
    assert _weight(_Harness(None), 1.0, 0.75) == 1.0


def test_late_posterior_weights_follow_reverse_transitions():
    harness = _Harness([0.0, 0.0, 0.25, 0.75])
    assert _weight(harness, 1.0, 0.75) == 0.0
    assert _weight(harness, 0.75, 0.5) == 0.0
    assert _weight(harness, 0.5, 0.25) == 0.25
    assert _weight(harness, 0.25, 0.0) == 0.75


def test_alpha_bar_s_continuous_weight_uses_target_time():
    harness = _Harness(None, mode="alpha_bar_s", scale=0.5)
    actual = _weight(harness, 0.75, 0.5)
    expected = 0.5 * float(
        harness.noise_schedule.get_alpha_bar(
            t_normalized=torch.tensor([[0.5]])
        ).item()
    )
    assert abs(actual - expected) < 1e-7


def test_alpha_bar_s_squared_weight():
    harness = _Harness(
        None, mode="alpha_bar_s_squared", scale=0.5
    )
    actual = _weight(harness, 0.75, 0.5)
    alpha = float(
        harness.noise_schedule.get_alpha_bar(
            t_normalized=torch.tensor([[0.5]])
        ).item()
    )
    assert abs(actual - 0.5 * alpha * alpha) < 1e-7
