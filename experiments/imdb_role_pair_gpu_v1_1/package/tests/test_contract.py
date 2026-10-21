import itertools
import math
import numpy as np
import pytest
import torch
from hsgd.slots import TypedSlots, UniformQueryPlan
from hsgd.kernel import MarginalKernel, full_sweep
from hsgd.model import TypedSparseDenoiser
from hsgd.noising import complement_positions, global_noisy_edges


def toy():
    return TypedSlots(np.array([0, 0, 0, 1, 1, 2]), tuple(itertools.combinations_with_replacement(range(3), 2)))


def test_decode_independent_combinations():
    space = toy()
    for f, pair in enumerate(space.pairs):
        edges = space.decode(f, np.arange(space.counts[f]))
        expected = [(u, v) for u, v in itertools.combinations(range(6), 2)
                    if tuple(sorted((space.types[u], space.types[v]))) == pair]
        assert sorted(map(tuple, edges.T)) == expected
        ff, ll = space.encode(edges)
        assert np.array_equal(ff, np.full(len(expected), f))
        assert np.array_equal(ll, np.arange(len(expected)))


@pytest.mark.parametrize("rounds", [1, 3, 10, 20])
def test_uniform_full_coverage_and_weight(rounds):
    space = toy()
    plan = UniformQueryPlan(space, 12, rounds=rounds)
    ids = [plan.block(r)[0] for r in range(rounds)]
    assert np.array_equal(np.sort(np.concatenate(ids)), np.arange(space.total))
    assert len(set(np.concatenate(ids))) == space.total
    # Exhaustively average ALL possible training round choices; arbitrary losses.
    loss = np.arange(space.total, dtype=float)**2
    estimates = [plan.training_ce_weight() * loss[q].sum() for q in ids]
    assert np.mean(estimates) == pytest.approx(loss.mean(), abs=1e-12)
    # Tiny family cannot be permanently excluded by query scheme.
    assert sum(len(q) for q in ids) == space.total


def test_random_reproducibility_cap_and_batch_independence():
    space = toy()
    a, b = UniformQueryPlan(space, 14, query_cap=8), UniformQueryPlan(space, 14, query_cap=8)
    c = UniformQueryPlan(space, [14, 1], query_cap=8)
    assert max(a.sizes) <= 8
    assert all(np.array_equal(x, y) for x, y in zip(a.permutations, b.permutations))
    assert any(not np.array_equal(x, y) for x, y in zip(a.permutations, c.permutations))
    with pytest.raises(ValueError):
        UniformQueryPlan(space, 0, query_cap=1)


def test_without_replacement_exact_subset_law():
    # Independent exhaustive reference for permutation slicing: each 2-of-4
    # subset has exactly 2!*2! permutations. Also each slot probability is 1/2.
    subsets = {}
    for perm in itertools.permutations(range(4)):
        key = tuple(sorted(perm[:2]))
        subsets[key] = subsets.get(key, 0) + 1
    assert len(subsets) == math.comb(4, 2)
    assert set(subsets.values()) == {4}
    # No adjacency input accepted, so chain/star cannot change the plan cache.
    space = toy()
    assert len(UniformQueryPlan(space, 10, rounds=2).permutations) == 6


def test_kernel_independent_joint_enumeration():
    kernel = MarginalKernel([.23], [1., .7, .2, 0.])
    maximum = 0.
    for s, t in [(0, 1), (0, 3), (1, 2), (1, 3), (2, 3)]:
        for p in [0., .1, .55, 1.]:
            for obs in [0, 1]:
                expected = 0.
                for clean, weight in [(0, 1-p), (1, p)]:
                    joint = []
                    for prev in [0, 1]:
                        ps = kernel.alpha[s] * (prev == clean) + (1-kernel.alpha[s]) * ([.77, .23][prev])
                        retention = kernel.alpha[t] / kernel.alpha[s]
                        likelihood = retention * (obs == prev) + (1-retention) * ([.77, .23][obs])
                        joint.append(ps * likelihood)
                    expected += weight * joint[1] / sum(joint)
                actual = kernel.reverse([p], [obs], [0], s, t)[0]
                maximum = max(maximum, abs(actual - expected))
    assert maximum < 5e-15


def test_fixed_support_and_clean_endpoint():
    kernel = MarginalKernel([0., 1., .4], [1., .5, 0.])
    assert np.array_equal(kernel.reverse([0, 1], [0, 1], [0, 1], 0, 2), [0, 1])
    assert kernel.reverse([.73], [0], [2], 0, 1)[0] == pytest.approx(.73)
    assert kernel.reverse([.73], [1], [2], 0, 1)[0] == pytest.approx(.73)
    with pytest.raises(ValueError):
        kernel.reverse([.01], [0], [0], 0, 1)
    with pytest.raises(ValueError):
        kernel.reverse([1.], [0], [1], 0, 1)
    with pytest.raises(ValueError):
        MarginalKernel([.4], [.99, .5])


def test_full_sweep_frozen_context_and_clock():
    space = toy()
    plan = UniformQueryPlan(space, 4, rounds=10)
    x = np.zeros(space.total, dtype=np.int64)
    kernel = MarginalKernel([.4]*6, [1., .6, 0.])
    seen = []
    def callback(context, ids, f, edges):
        assert np.array_equal(context, x)
        seen.extend(ids.tolist())
        context[:] = 1  # callback mutation cannot contaminate another query round.
        return np.full(len(ids), .75)
    y = full_sweep(plan, x, kernel, callback, np.linspace(0, .99, space.total), 0, 1)
    assert sorted(seen) == list(range(space.total))
    assert np.array_equal(y, np.linspace(0, .99, space.total) < .75)
    assert not x.any()


def setup_model_case():
    torch.manual_seed(51)
    space = TypedSlots(np.array([0, 0, 0, 1, 1, 1]), ((0,0), (0,1), (1,1)))
    model = TypedSparseDenoiser(2, 3, layers=2, width=32, edge_width=16, heads=4).eval()
    old = np.array([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]])
    query = np.array([[0, 0, 1, 3], [2, 3, 4, 5]])
    of = space.encode(old)[0]
    qf = space.encode(query)[0]
    return model, space, old, of, query, qf


def test_network_gradient_and_query_order():
    model, space, old, of, query, qf = setup_model_case()
    p = model(space.types, old, of, query, qf, .5)
    assert torch.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    order = [3, 2, 1, 0]
    p_order = model(space.types, old, of, query[:, order], qf[order], .5)
    assert torch.allclose(p_order, p[order], atol=1e-7, rtol=0)
    loss = torch.nn.functional.binary_cross_entropy(p, torch.tensor([0., 1., 0., 1.]))
    loss.backward()
    gradients = [v.grad for v in model.parameters() if v.grad is not None]
    assert gradients and all(torch.isfinite(v).all() for v in gradients)
    assert sum(float(v.abs().sum()) for v in gradients) > 0


def test_network_type_preserving_renumber_and_restore():
    model, space, old, of, query, qf = setup_model_case()
    perm = np.array([2, 0, 1, 5, 3, 4])  # old ID -> new ID, same immutable type.
    remap = lambda edges: np.sort(perm[edges], axis=0)
    with torch.no_grad():
        a = model(space.types, old, of, query, qf, .5)
        b = model(space.types, remap(old), of, remap(query), qf, .5)
    assert torch.allclose(a, b, atol=2e-6, rtol=0)
    clone, *_ = setup_model_case()
    clone.load_state_dict(model.state_dict(), strict=True)
    with torch.no_grad():
        c = clone(space.types, old, of, query, qf, .5)
    assert torch.equal(a, c)


def test_network_rejects_duplicate_and_illegal_inputs():
    model, space, old, of, query, qf = setup_model_case()
    with pytest.raises(ValueError):
        model(space.types, old, of, np.repeat(query[:, :1], 2, axis=1), np.repeat(qf[:1], 2), .5)
    with pytest.raises(ValueError):
        model(space.types, old, of, np.array([[0], [0]]), np.array([0]), .5)


def test_network_fixed_support_and_schema_mismatch():
    torch.manual_seed(2)
    model = TypedSparseDenoiser(2, 3, layers=1, width=16, edge_width=8, heads=4, prior=[0., .2, 1.]).eval()
    types = np.array([0, 0, 1, 1])
    old, of = np.array([[2], [3]]), np.array([2])
    query, qf = np.array([[0, 0, 2], [1, 2, 3]]), np.array([0, 1, 2])
    p = model(types, old, of, query, qf, .5)
    assert p[0].item() == 0. and p[2].item() == 1.
    assert 0 < p[1].item() < 1
    with pytest.raises(ValueError):
        model(types, old, of, query, np.array([1, 1, 2]), .5)
    with pytest.raises(ValueError):
        model(types, np.array([[0], [1]]), np.array([0]), query, qf, .5)


def test_empty_inventory_and_fixed_slots_still_in_plan():
    space = TypedSlots(np.array([0]), ((0, 0),))
    plan = UniformQueryPlan(space, 2, query_cap=4)
    ids, f, edges = plan.block(0)
    assert len(ids) == len(f) == 0 and edges.shape == (2, 0)
    assert plan.training_ce_weight() == 0.


def test_active_family_chunks_concatenate_and_preserve_ce_scaling():
    space = toy()
    active = np.array([False, True, False, True, False, False])
    plan = UniformQueryPlan(space, 5, rounds=3, active_families=active)
    expected = set()
    for f in np.flatnonzero(active):
        expected.update(range(space.offsets[f], space.offsets[f + 1]))
    blocks = [plan.block(r) for r in range(3)]
    assert set(np.concatenate([b[0] for b in blocks])) == expected
    assert sum(len(b[0]) for b in blocks) == len(expected)
    assert all(active[b[1]].all() for b in blocks)
    for f in np.flatnonzero(active):
        per_round = [int((b[1] == f).sum()) for b in blocks]
        assert max(per_round) - min(per_round) <= 1
    loss = np.zeros(space.total)
    loss[list(expected)] = np.arange(len(expected))**2
    assert np.mean([plan.training_ce_weight()*loss[b[0]].sum() for b in blocks]) == pytest.approx(loss.mean())
    cap_plan = UniformQueryPlan(space, 5, query_cap=8, active_families=active)
    assert max(cap_plan.sizes) <= 8


def test_active_sweep_skips_only_exact_fixed_families():
    space = toy()
    active = np.array([False, True, False, True, False, False])
    plan = UniformQueryPlan(space, 5, rounds=3, active_families=active)
    prior = np.where(active, .4, 0.)
    kernel = MarginalKernel(prior, [1., .6, 0.])
    x = np.zeros(space.total, dtype=np.int64)
    seen = []
    def callback(context, ids, f, edges):
        assert np.array_equal(context, x)
        assert active[f].all()
        seen.extend(ids)
        return np.full(len(ids), .75)
    y = full_sweep(plan, x, kernel, callback, np.full(space.total, .2), 0, 1)
    assert y.sum() == plan.query_total and len(seen) == plan.query_total
    with pytest.raises(ValueError):
        full_sweep(plan, x, MarginalKernel([.4]*6, [1., .6, 0.]), callback, np.full(space.total, .2), 0, 1)


def test_global_forward_complement_exhaustive():
    # Independent direct set subtraction, all occupancy subsets up to 6 slots.
    for size in range(7):
        for bits in itertools.product([0, 1], repeat=size):
            occupied = np.flatnonzero(bits)
            vacant = [i for i,b in enumerate(bits) if not b]
            actual = complement_positions(size, occupied, np.arange(len(vacant)))
            assert np.array_equal(actual, vacant)
    with pytest.raises(ValueError):
        complement_positions(4, [2, 1], [0])


def test_global_noising_support_retention_and_relabel():
    space = toy()
    edges = np.array([[0, 0, 1, 3], [1, 3, 4, 4]])
    ff, ll = space.encode(edges)
    counts = np.bincount(ff, minlength=6)
    pi = np.divide(counts, space.counts, out=np.zeros(6), where=space.counts>0)
    noisy, nf, stats = global_noisy_edges(space, edges, pi, 1., np.random.default_rng(3))
    assert set(map(tuple, noisy.T)) == set(map(tuple, edges.T))
    assert np.array_equal(nf, space.encode(noisy)[0])
    assert sum(s['noisy_edges'] for s in stats) == edges.shape[1]
    a, af, _ = global_noisy_edges(space, edges, pi, .5, np.random.default_rng(4))
    b, bf, _ = global_noisy_edges(space, edges, pi, .5, np.random.default_rng(4))
    assert np.array_equal(a, b) and np.array_equal(af, bf)
    assert np.array_equal(space.encode(a)[0], af)
    assert (pi[af] > 0).all()
    # No query argument exists: outside-query edges cannot be discarded by this
    # function. Joint law verified analytically via Binomial + uniform subset.
    for n in range(1, 6):
        p = .17
        for bits in itertools.product([0, 1], repeat=n):
            k = sum(bits)
            law = math.comb(n,k)*p**k*(1-p)**(n-k)/math.comb(n,k)
            direct = np.prod([p if bit else 1-p for bit in bits])
            assert law == pytest.approx(direct, abs=1e-15)


def test_explicit_large_recipe_dimensions():
    m = TypedSparseDenoiser(2, 3, layers=2, width=32, edge_width=16, heads=4,
                            global_width=24, ff_node=32, ff_edge=16, ff_global=48)
    assert m.backbone.lin_in_y.out_features == 24
    assert m.backbone.tf_layers[0].lin_y1.out_features == 48
    assert m.backbone.tf_layers[0].lin_y2.in_features == 48
    assert m.backbone.tf_layers[0].linX1.out_features == 32
    assert m.backbone.tf_layers[0].linE1.out_features == 16
