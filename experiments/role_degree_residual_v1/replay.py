"""Frozen six-chain diagnostic replay. No training, reseeding or new samples."""
import os
os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="")
import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys
import time
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "role_specialization_v1"
sys.path.insert(0, str(BASE))
from panel import CapacityDenoiser, load_panel, c, save, fingerprint, sha
from hsgd.runtime import reverse_step
from hsgd.kernel import MarginalKernel
from role_condition.model import capacities

TIMES = (64, 32, 16, 8, 4, 2, 1)

def slot_arrays(case):
    space = case["space"]
    # Use the production canonical index mapping, not length-based assumptions.
    plan = c.plan_for(case, [2026102031, 1, 61], 4)
    endpoints = np.zeros((2, space.total), dtype=np.int64)
    families = np.repeat(np.arange(case["n_families"]), space.counts)
    coverage = np.zeros(space.total, dtype=np.int64)
    for b in range(4):
        ids, fam, query = plan.block(b)
        assert np.array_equal(families[ids], fam)
        endpoints[:, ids] = query
        coverage[ids] += 1
    expected = np.repeat(case["active"].astype(int), space.counts)
    assert np.array_equal(coverage, expected)
    return endpoints, families, coverage.astype(bool)

def degree_expectation(values, endpoints, families, active, n, R):
    result = np.zeros((n, R))
    for side in (0, 1):
        np.add.at(result, (endpoints[side, active], families[active]), values[active])
    return result

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--panel", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    start = time.monotonic(); cpu = time.process_time()
    args.output.mkdir(parents=True, exist_ok=False)
    status = "failed"; reason = None; rows = []; checked = 0
    torch.set_num_threads(1); torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cfg = json.loads((args.run/"resolved_config.json").read_text())
    records, targets = load_panel(args.panel); case = records[0]["case"]
    weights = json.loads((args.run/"weight_hashes.json").read_text())
    generation = json.loads((args.run/"generation.json").read_text())
    sources = {}
    for p in (args.run/"resolved_config.json", args.run/"weight_hashes.json",
              args.run/"generation.json", args.panel/"panel.json"):
        sources[str(p)] = fingerprint(p)
    types = case["space"].types; n = len(types); R = case["n_families"]
    edges, families, active = slot_arrays(case)
    cap = capacities(types, case["space"].pairs); alpha = c.schedule(64)
    kernel = MarginalKernel(case["prior"], alpha)
    sample_seed = cfg["sampling_seeds"][0]
    save(args.output/"protocol.json", dict(times=TIMES, training_seeds=cfg["train_seeds"],
        sampling_seed=sample_seed, conditions=["balanced", "extreme"], max_seconds=300,
        scope="replay archived chains; all 64 states and final graph must match",
        no_quality_sample_addition=True))
    def check():
        if time.monotonic()-start > 295 or (args.output/"STOP").exists():
            raise TimeoutError("diagnostic bound/STOP")
    try:
        for seed in cfg["train_seeds"]:
            path = args.run/f"seed{seed}/capacity_C_final_1000.pt"
            assert fingerprint(path) == weights[str(path.relative_to(args.run))]
            sources[str(path)] = fingerprint(path)
            model = CapacityDenoiser(case["n_types"], R, family_pairs=case["space"].pairs,
                                    prior=case["prior"], T=64, **cfg["model"])
            model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
            model.eval()
            for condition in ("balanced", "extreme"):
                old = next(r for r in generation if r["seed"] == seed and
                    r["sampling_seed"] == sample_seed and r["condition"] == condition)
                graphfile = args.run/old["graph_file"]
                assert fingerprint(graphfile) == old["graph_file_sha256"]
                stepsfile = graphfile.with_name(graphfile.stem+"_steps.json")
                sources[str(graphfile)] = fingerprint(graphfile)
                sources[str(stepsfile)] = fingerprint(stepsfile)
                oldsteps = json.loads(stepsfile.read_text())
                model.set_target(targets[condition]); c.seed_all(sample_seed)
                rng = np.random.default_rng(sample_seed)
                state = (rng.random(case["space"].total) <
                         np.repeat(case["prior"], case["space"].counts)).astype(np.uint8)
                for t in range(64, 0, -1):
                    check(); before = state.copy(); captured = []
                    hook = None
                    if t in TIMES:
                        hook = model.branch.register_forward_hook(lambda m, a, out: captured.append(out[1]))
                    try:
                        with torch.no_grad():
                            state, info = reverse_step(model, case, state, kernel, t, 4,
                                sample_seed, rng, "cpu", check_time=check, trace=t in TIMES)
                    finally:
                        if hook: hook.remove()
                    assert sha(state.tolist()) == oldsteps[64-t]["state_sha256"], (seed, condition, t)
                    assert sha(rng.bit_generator.state) == oldsteps[64-t]["rng_after"]
                    checked += 1
                    if t not in TIMES: continue
                    assert len(captured) == 1
                    diag = captured[0]; tr = info["trace"]
                    mu = degree_expectation(tr["p0"], edges, families, active, n, R)
                    nxt = degree_expectation(tr["q"], edges, families, active, n, R)
                    birth = degree_expectation((1-before)*tr["q"], edges, families, active, n, R)
                    death = degree_expectation(before*(1-tr["q"]), edges, families, active, n, R)
                    soft = np.zeros((n, R)); entropy = np.zeros(n); allocation = {}
                    for d in diag["allocation"]:
                        typ = d["type"]; ix = np.flatnonzero(types == typ)
                        vec = np.array([v for v, count in targets[condition].roles[typ]])
                        a = d["attention"]
                        soft[ix] = a @ vec
                        entropy[ix] = -(np.where(a > 0, a*np.log(np.maximum(a, 1e-300)), 0)).sum(1)
                        allocation[f"attention_type{typ}"] = a
                    noisy = alpha[t]*soft+(1-alpha[t])*cap[types]*case["prior"]
                    np.testing.assert_allclose(nxt-diag["degree"], birth-death, atol=1e-12, rtol=0)
                    stem = f"seed{seed}_{condition}_t{t:02d}"
                    np.savez_compressed(args.output/(stem+".npz"), Xt=before, next_state=state,
                        types=types, p0=tr["p0"], q=tr["q"], uniform=tr["uniform"],
                        endpoints=edges, families=families, active=active,
                        current_degree=diag["degree"], clean_expected_degree=mu,
                        next_expected_degree=nxt, soft_target_degree=soft,
                        noisy_soft_target_degree=noisy, entropy=entropy,
                        expected_births=birth, expected_deaths=death, **allocation)
                    mg = case["space"].pairs.index((2, 3)); ix = np.flatnonzero(types == 3)
                    rows.append(dict(seed=seed, condition=condition, t=t,
                        G_current=diag["degree"][ix, mg].tolist(),
                        G_soft=soft[ix, mg].tolist(), G_p0=mu[ix, mg].tolist(),
                        G_q=nxt[ix, mg].tolist(), G_entropy=entropy[ix].tolist(),
                        G_births=birth[ix, mg].tolist(), G_deaths=death[ix, mg].tolist(),
                        expected_edges_by_family=np.bincount(families[active], weights=tr["p0"][active], minlength=R).tolist(),
                        current_edges_by_family=np.bincount(families[active], weights=before[active], minlength=R).tolist()))
                saved = np.load(graphfile)
                ee, ff = c.state_edges(case, state)
                assert np.array_equal(ee.T, saved["edges"]) and np.array_equal(ff+1, saved["relations"])
                print("verified archived replay", seed, condition, flush=True)
        assert checked == 384 and len(rows) == 42
        assert all(fingerprint(p) == h for p, h in sources.items())
        save(args.output/"records.json", rows); save(args.output/"sources.json", sources)
        status = "completed"
    except BaseException as exc:
        reason = repr(exc); raise
    finally:
        save(args.output/"receipt.json", dict(status=status, reason=reason, checked_steps=checked,
            recorded_states=len(rows), training_updates=0, replay_chains=checked//64,
            new_quality_samples=0, GPU=0, wall_seconds=time.monotonic()-start,
            CPU_seconds=time.process_time()-cpu,
            peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024))

if __name__ == "__main__": main()
