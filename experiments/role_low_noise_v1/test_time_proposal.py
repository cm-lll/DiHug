import copy
from pathlib import Path
import sys
import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
PREV = HERE.parent / "role_condition_diagnosis_v1"
sys.path[:0] = [str(HERE), str(PREV), str(PREV / "baseline"),
               str(HERE.parent / "role_specialization_v1")]
from time_proposal import proposal_from_atom, paired_input, weighted_input
from test_late_fusion import fixture, SMALL
from capacity_readout import CapacityDenoiser
from role_condition.model import extract_target
from hsgd.bw_preflight import common as c


def test_exact_proposal_mass_and_objective_gradient():
    info = [proposal_from_atom(i) for i in range(128)]
    counts = np.bincount([x["t"] for x in info], minlength=65)[1:]
    np.testing.assert_array_equal(counts, [9] * 8 + [1] * 56)
    theta = torch.tensor([.3, -.7], dtype=torch.float64, requires_grad=True)
    t = torch.arange(1, 65, dtype=torch.float64)
    losses = torch.nn.functional.softplus(theta[0] * t / 64 + theta[1])
    baseline = losses.mean()
    candidate = sum(x["importance_weight"] * losses[x["t"] - 1] for x in info) / 128
    torch.testing.assert_close(candidate, baseline, atol=2e-12, rtol=0)
    left = torch.autograd.grad(baseline, theta, retain_graph=True)[0]
    right = torch.autograd.grad(candidate, theta)[0]
    torch.testing.assert_close(left, right, atol=2e-12, rtol=0)


def test_pairing_preserves_original_stream_query_and_labels():
    a, _, _ = fixture()
    left = np.random.default_rng(11); right = np.random.default_rng(11)
    for u in range(12):
        new, original, info = paired_input(a, 11, u, left)
        old = c.train_input(a, 64, 4, 11, u, right)
        assert left.bit_generator.state == right.bit_generator.state
        for key in ("edges", "edge_families", "query", "ids", "labels"):
            np.testing.assert_array_equal(original[key], old[key])
        np.testing.assert_array_equal(new["query"], old["query"])
        assert new["t"] == info["t"]
        assert old["ce_weight"] == new["ce_weight"]
        replay_rng = np.random.default_rng()
        replay_rng.bit_generator.state = copy.deepcopy(right.bit_generator.state)
        x, _, _ = paired_input(a, 11, u + 1, replay_rng)
        y, _, _ = paired_input(a, 11, u + 1, right)
        np.testing.assert_array_equal(x["edges"], y["edges"])
        left.bit_generator.state = copy.deepcopy(right.bit_generator.state)


@pytest.mark.parametrize("weight", [2 / 9, 2])
def test_real_model_ce_gradient_scaling_and_input_immutability(weight):
    a, _, _ = fixture()
    torch.manual_seed(19)
    m = CapacityDenoiser(2, 1, family_pairs=a["space"].pairs,
                        prior=a["prior"], T=64, **SMALL)
    m.set_target(extract_target(a))
    op = torch.optim.AdamW(m.parameters(), lr=2e-4, weight_decay=1e-12)
    inp = c.train_input(a, 64, 4, 11, 0, np.random.default_rng(11), force_t=8)
    original_weight = inp["ce_weight"]
    first = {}; second = {}
    state = torch.get_rng_state().clone()
    r = c.update(m, op, a, inp, 64, "cpu", do_step=False, diagnostics=first)
    s = c.update(m, op, a, weighted_input(inp, weight), 64, "cpu",
                 do_step=False, diagnostics=second)
    assert torch.equal(state, torch.get_rng_state()) and not op.state
    assert inp["ce_weight"] == original_weight
    assert s["loss"] == pytest.approx(weight * r["loss"], rel=1e-6)
    for k, g in first["gradients"].items():
        if g is None: assert second["gradients"][k] is None
        else: torch.testing.assert_close(second["gradients"][k], g * weight,
                                         atol=2e-6, rtol=1e-5)


def test_invalid_proposal_and_unfrozen_weight_rejected():
    for atom in (-1, 128, 1.5):
        with pytest.raises(ValueError): proposal_from_atom(atom)
    with pytest.raises(ValueError): weighted_input({"ce_weight": 1.}, .5)
