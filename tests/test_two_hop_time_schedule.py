from types import SimpleNamespace

import torch

from sparse_diffusion.diffusion.noise_schedule import (
    PredefinedNoiseScheduleDiscrete,
)
from sparse_diffusion.diffusion_model_sparse import (
    DiscreteDenoisingDiffusion,
)


class _ScheduleHarness:
    device = torch.device("cpu")

    def __init__(self, schedule):
        self.cfg = SimpleNamespace(
            model=SimpleNamespace(
                use_two_hop_structure=True,
                two_hop_structure_schedule=schedule,
            )
        )
        self.noise_schedule = PredefinedNoiseScheduleDiscrete(
            noise_schedule="cosine",
            timesteps=100,
            skip=1,
        )


def _factor(schedule, t):
    harness = _ScheduleHarness(schedule)
    return DiscreteDenoisingDiffusion._two_hop_reliability_factor(
        harness, torch.tensor(t, dtype=torch.float32).reshape(-1, 1)
    )


def test_fixed_linear_and_quadratic_time_schedules():
    t = [0.0, 0.25, 0.5, 1.0]
    assert torch.equal(_factor("fixed", t), torch.ones(4, 1))
    assert torch.allclose(
        _factor("linear_t", t),
        torch.tensor([[1.0], [0.75], [0.5], [0.0]]),
    )
    assert torch.allclose(
        _factor("quadratic_t", t),
        torch.tensor([[1.0], [0.5625], [0.25], [0.0]]),
    )


def test_alpha_bar_squared_matches_noise_schedule_and_decreases():
    t = torch.tensor([[0.0], [0.25], [0.5], [0.75], [1.0]])
    harness = _ScheduleHarness("alpha_bar_squared")
    actual = DiscreteDenoisingDiffusion._two_hop_reliability_factor(
        harness, t
    )
    expected = harness.noise_schedule.get_alpha_bar(
        t_normalized=t
    ).square()

    assert torch.equal(actual, expected)
    assert torch.all(actual[:-1] >= actual[1:])


def test_explicit_sampling_seeds_are_validated_and_override_train_seed():
    harness = SimpleNamespace(
        cfg=SimpleNamespace(
            train=SimpleNamespace(seed=17),
            general=SimpleNamespace(
                test_sampling_seeds=[0, 1, 2],
                test_variance=3,
                final_model_samples_to_generate=1,
            ),
        ),
        _trainer=SimpleNamespace(num_devices=1),
    )

    assert (
        DiscreteDenoisingDiffusion._explicit_test_sampling_seeds(harness)
        == [0, 1, 2]
    )
    assert DiscreteDenoisingDiffusion._current_sampling_seed(harness) == 17
    harness._active_test_sampling_seed = 2
    assert DiscreteDenoisingDiffusion._current_sampling_seed(harness) == 2
