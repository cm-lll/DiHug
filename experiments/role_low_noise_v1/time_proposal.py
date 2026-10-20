"""Change only the SGD time proposal, with exact objective correction.

The original RNG remains the driver of augmentation/topology selection. We
reconstruct its original input, then use a clone to form the new input. Sparse
Binomial noising consumes time-dependent random counts: these are coupled
streams, NOT equal per-slot uniforms or equal Xt between training arms.
"""
import copy
import numpy as np
from hsgd.bw_preflight import common as c

T = 64
LOW = 8
SEED_TAG = 2026102091


def proposal_from_atom(atom):
    if not isinstance(atom, (int, np.integer)) or not 0 <= atom < 128:
        raise ValueError("uniform integer atom in [0,128) required")
    # 72 low atoms: nine copies of each t=1..8; 56 high atoms once each.
    t = atom // 9 + 1 if atom < 72 else atom - 63
    probability = (9 if t <= LOW else 1) / 128
    return dict(atom=int(atom), t=int(t), proposal_probability=probability,
                importance_weight=(1 / T) / probability)


def paired_input(case, seed, position, rng):
    clone = np.random.default_rng()
    clone.bit_generator.state = copy.deepcopy(rng.bit_generator.state)
    original = c.train_input(case, T, 4, seed, position, rng)
    # Advance clone through precisely the baseline time draw. Forced time in
    # train_input then starts from the same query/noise RNG prefix.
    baseline_t = int(clone.integers(1, T + 1))
    assert baseline_t == original["t"]
    proposal_rng = np.random.default_rng([seed, position, SEED_TAG])
    info = proposal_from_atom(int(proposal_rng.integers(128)))
    inp = c.train_input(case, T, 4, seed, position, clone, force_t=info["t"])
    for field in ("ids", "f", "query", "labels"):
        assert np.array_equal(inp[field], original[field])
    assert inp["round_id"] == original["round_id"]
    info["baseline_t"] = baseline_t
    return inp, original, info


def weighted_input(inp, weight):
    if weight not in (2 / 9, 2):
        raise ValueError("weight outside frozen proposal")
    out = dict(inp)
    out["ce_weight"] = inp["ce_weight"] * weight
    return out
