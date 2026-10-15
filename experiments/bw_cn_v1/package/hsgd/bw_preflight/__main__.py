"""Manual stage launcher. Separate commands, bounded own-child watchdogs."""
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

from .common import (PACKAGE, checkpoint, check_prob, code_hashes, config,
                     dimensions, environment, full_sweep, graph_stats, imdb_case,
                     json_save, make_model, memory, plan_for, query_inventory,
                     restore, rng_restore, rng_save, runtime_binding, schedule, seed_all, sha,
                     state_edges, sync, toy_case, train_input, tree_error,
                     trusted_load, torch_save, update, verify_manifest, MarginalKernel)
from hsgd.runtime import predict, StepPredictor, reverse_step, backend_name


class Deadline(Exception):
    pass


class Run:
    def __init__(self, args):
        self.args = args
        self.out = args.output.resolve()
        existed = self.out.exists()
        if existed and not args.resume:
            raise RuntimeError("output already exists; use a new directory or explicit --resume")
        if args.resume and (not existed or args.stage not in ("B", "C-imdb")):
            raise RuntimeError("resume requires an existing B/C-imdb run")
        self.out.mkdir(parents=True, exist_ok=True)
        self.lock = (self.out / ".run.lock").open("a+")
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.resume and (self.out / "summary.json").exists():
            if json.loads((self.out / "summary.json").read_text()).get("status") == "passed":
                raise RuntimeError("completed run must not be duplicated")
            archive = self.out / "attempts"
            archive.mkdir(exist_ok=True)
            (self.out / "summary.json").rename(archive / ("summary_" + str(time.time_ns()) + ".json"))
        self.started, self.cpu = time.monotonic(), time.process_time()
        self.cfg = config()
        self.graph_backend = backend_name(getattr(args,"backend","upstream"))
        self.events = (self.out / "events.jsonl").open("a", buffering=1)
        self.phase = "environment"
        self.result = dict(stage=args.stage, status="running", formal_training=False,
                           new_scientific_graphs=0, graph_backend=self.graph_backend, config=self.cfg, attempts=[])
        torch.set_num_threads(1)
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        verify_manifest()
        self.env = environment(args.device)
        self.binding = dict(code=code_hashes(), config=self.cfg,
                            backend=runtime_binding(self.env), graph_backend=self.graph_backend)
        self.result["environment"] = self.env
        if args.resume:
            old = json.loads((self.out / "run.json").read_text())
            if old["initial_binding"] != self.binding:
                raise RuntimeError("resume environment/source changed")
        else:
            json_save(self.out / "run.json", dict(initial_binding=self.binding, environment=self.env,
                      stage=args.stage, command=sys.argv, wall_seconds=args.wall_seconds))
        self.event("environment_recorded", environment=self.env)
        if args.device == "cuda":
            torch.cuda.reset_peak_memory_stats()

    def event(self, phase, **values):
        self.phase = phase
        item = dict(phase=phase, elapsed_seconds=time.monotonic() - self.started, **values)
        self.events.write(json.dumps(item, allow_nan=False) + "\n")
        print(json.dumps(item, allow_nan=False), flush=True)
        json_save(self.out / "progress.json", item)

    def check_time(self):
        if time.monotonic() - self.started >= self.args.wall_seconds:
            raise Deadline("explicit stage wall limit reached")

    def finish(self, status, **extra):
        self.result.update(status=status, wall_seconds=time.monotonic() - self.started,
                           cpu_seconds=time.process_time() - self.cpu, last_phase=self.phase,
                           **memory(self.args.device), **extra)
        self.result["code_unchanged"] = code_hashes() == self.binding["code"]
        verify_manifest()
        if not self.result["code_unchanged"]:
            raise RuntimeError("code changed during run")
        json_save(self.out / "summary.json", self.result)
        self.event("stage_finished", status=status, wall_seconds=self.result["wall_seconds"])
        print_summary(self.result)


def print_summary(r):
    print("\n=== DiHug bounded engineering summary ===", flush=True)
    for key in ["stage", "status", "completed_updates", "completed_reverse_steps", "remaining_time",
                "wall_seconds", "cpu_seconds", "peak_allocated_mib", "peak_reserved_mib",
                "process_peak_rss_mib", "stable_batch_median_seconds", "stable_batch_p90_seconds",
                "measured_sweep_seconds", "estimated_T1000_seconds", "failure", "next_step"]:
        if key in r:
            print(f"{key}: {r[key]}", flush=True)
    e = r.get("environment", {})
    print("environment:", json.dumps(e, ensure_ascii=False), flush=True)
    print("configuration:", json.dumps(r.get("config", {}), ensure_ascii=False), flush=True)
    print("No quality/convergence certification; no M/K/critic/completion.\n", flush=True)


def prerequisite(path, stage, binding):
    p = Path(path)
    r = json.loads((p / "summary.json").read_text())
    if r.get("stage") != stage or r.get("status") != "passed":
        raise RuntimeError("prerequisite stage did not pass: " + str(p))
    run = json.loads((p / "run.json").read_text())
    if run["initial_binding"] != binding:
        raise RuntimeError("prerequisite has different backend/config/source")
    return p


def symmetry_device_hooks(model, device):
    checks = {"tensor_inputs_checked": 0, "symmetric_backbone_outputs_checked": 0}
    def check_input(module, args):
        for arg in args:
            if torch.is_tensor(arg):
                assert arg.device.type == device
                checks["tensor_inputs_checked"] += 1
    def symmetric(module, args, output):
        edge = output.edge_index.detach().cpu().numpy()
        lookup = {tuple(e): i for i, e in enumerate(edge.T)}
        inverse = [lookup[(int(v), int(u))] for u, v in edge.T]
        error = float((output.edge_attr - output.edge_attr[inverse]).abs().max())
        if error > 5e-6:
            raise RuntimeError("mirrored backbone outputs differ")
        checks["symmetric_backbone_outputs_checked"] += 1
    handles = [m.register_forward_pre_hook(check_input) for m in model.modules()]
    handles.append(model.backbone.register_forward_hook(symmetric))
    return checks, handles


def stage_a(run):
    cfg, device = run.cfg, run.args.device
    seed_all(cfg["train_seed"])
    rng = np.random.default_rng(cfg["train_seed"])
    case = toy_case()
    T, rounds = 8, 3
    init_start = time.perf_counter()
    model = make_model(case, cfg, device)
    optimizer = torch.optim.AdamW(model.parameters(), **cfg["optimizer"])
    sync(device)
    run.result["model_initialization_seconds"] = time.perf_counter() - init_start
    run.result["actual_dimensions"] = dimensions(model)
    run.result["toy_schedule"] = dict(T=T, rounds=rounds, alpha=schedule(T).tolist())
    run.result["query_inventory"] = query_inventory(case, plan_for(case, 41, rounds))
    run.binding = dict(run.binding, data="deterministic six-node engineering fixture", T=T, rounds=rounds)
    checks, handles = symmetry_device_hooks(model, device)
    inp = train_input(case, T, rounds, cfg["train_seed"], 1, rng, force_t=4)
    record = update(model, optimizer, case, inp, T, device, backend=getattr(run,"graph_backend","upstream"))
    run.event("smoke_update_1", record=record)
    model.eval()
    with torch.no_grad():
        expected_output = predict(model,case["space"].types, inp["edges"], inp["edge_families"], inp["query"], inp["f"], .5,backend=getattr(run,"graph_backend","upstream")).detach().cpu()
    ck = checkpoint(model, optimizer, rng, device, 1, run.binding, [record])
    torch_save(run.out / "smoke_step1.pt", ck)
    next_input = train_input(case, T, rounds, cfg["train_seed"], 2, rng)
    expected_step = update(model, optimizer, case, next_input, T, device,backend=getattr(run,"graph_backend","upstream"))
    expected_model, expected_optimizer = checkpoint(model, optimizer, rng, device, 2, run.binding, []) ["model"], copy.deepcopy(optimizer.state_dict())
    expected_rng = rng_save(rng, device)
    restored_count = restore(trusted_load(run.out / "smoke_step1.pt"), model, optimizer, rng, device, run.binding)
    assert restored_count == 1
    model.eval()
    with torch.no_grad():
        restored_output = predict(model,case["space"].types, inp["edges"], inp["edge_families"], inp["query"], inp["f"], .5,backend=getattr(run,"graph_backend","upstream")).detach().cpu()
    output_error = float((expected_output - restored_output).abs().max())
    replay_input = train_input(case, T, rounds, cfg["train_seed"], 2, rng)
    for key in ("t", "ids", "f", "query", "labels", "edges", "edge_families", "round_id"):
        assert np.array_equal(next_input[key], replay_input[key])
    replay = update(model, optimizer, case, replay_input, T, device,backend=getattr(run,"graph_backend","upstream"))
    model_error = tree_error(expected_model, checkpoint(model, optimizer, rng, device, 2, run.binding, []) ["model"])
    optimizer_error = tree_error(expected_optimizer, optimizer.state_dict())
    actual_rng = rng_save(rng, device)
    # Compare generator/global states by serializing exact state containers, not random draws.
    import pickle
    for key in ("generator", "python", "numpy"):
        assert pickle.dumps(actual_rng[key]) == pickle.dumps(expected_rng[key])
    assert torch.equal(actual_rng["torch"], expected_rng["torch"])
    if device == "cuda":
        assert all(torch.equal(a, b) for a, b in zip(actual_rng["accelerator"], expected_rng["accelerator"]))
    for error in (output_error, model_error, optimizer_error):
        if error > cfg["recovery_atol"]:
            raise RuntimeError("recovery error exceeded predeclared absolute tolerance")
    for handle in handles:
        handle.remove()
    torch_save(run.out / "smoke_final.pt", checkpoint(model, optimizer, rng, device, 2, run.binding, [record, replay]))
    run.result.update(device_checks=checks, recovery=dict(atol=cfg["recovery_atol"], rtol=0., output_error=output_error,
                      parameter_error=model_error, optimizer_error=optimizer_error, rng_exact=True),
                      optimizer_calls=3, unique_updates=2, completed_updates=2,
                      gradient_scope="only parameters participating in edge CE; absent branches are listed, not failed")
    run.finish("passed", next_step="manually run B and C-small; full BW IMDB capacity still unknown")


def validation(model, case, cfg, device, backend="upstream"):
    rng = np.random.default_rng(cfg["validation_seed"])
    model.eval()
    records = []
    with torch.no_grad():
        for i, t in enumerate(cfg["validation_times"]):
            inp = train_input(case, cfg["T"], cfg["rounds"], cfg["validation_seed"], i, rng, force_t=t)
            p = predict(model,case["space"].types, inp["edges"], inp["edge_families"], inp["query"], inp["f"], t / cfg["T"],backend=backend)
            check_prob(p, inp["f"], case)
            label = torch.as_tensor(inp["labels"], device=device)
            ce = torch.nn.functional.binary_cross_entropy(p, label, reduction="none")
            records.append(dict(backend=backend,time=t, slots=len(label), positive_labels=int(label.sum()),
                                weighted_ce=float(ce.sum() * inp["ce_weight"]),
                                query_ids_sha256=__import__("hashlib").sha256(inp["ids"].tobytes()).hexdigest()))
    sync(device)
    return records


def stage_b(run):
    cfg, device = run.cfg, run.args.device
    prerequisite(run.args.a_dir, "A", run.binding)
    case = imdb_case(run.args.data, cfg)
    seed_all(cfg["train_seed"])
    rng = np.random.default_rng(cfg["train_seed"])
    run.result["query_inventory"] = query_inventory(case, plan_for(case, cfg["train_seed"], cfg["rounds"]))
    run.binding = dict(run.binding, data=sha(run.args.data), T=cfg["T"], rounds=cfg["rounds"])
    init_start = time.perf_counter()
    model = make_model(case, cfg, device)
    optimizer = torch.optim.AdamW(model.parameters(), **cfg["optimizer"])
    sync(device)
    run.result["model_initialization_seconds"] = time.perf_counter() - init_start
    run.result["actual_dimensions"] = dimensions(model)
    run.result["alpha_sha256"] = __import__("hashlib").sha256(schedule(cfg["T"]).tobytes()).hexdigest()
    records, completed = [], 0
    if run.args.resume:
        ck = trusted_load(run.out / "preflight_last.pt")
        completed = restore(ck, model, optimizer, rng, device, run.binding)
        records = ck["history"]
        run.event("restored", committed_updates=completed)
    else:
        run.event("low_mid_high_device_checks")
        checks = []
        for i, t in enumerate(cfg["device_check_times"]):
            run.check_time()
            inp = train_input(case, cfg["T"], cfg["rounds"], cfg["validation_seed"], 100 + i,
                              np.random.default_rng([cfg["validation_seed"], t]), force_t=t)
            run.event("device_check_start", time=t, query_slots=len(inp["ids"]), noisy_edges=inp["edges"].shape[1])
            rec = update(model, optimizer, case, inp, cfg["T"], device, do_step=False,backend=getattr(run,"graph_backend","upstream"))
            checks.append(rec)
            run.event("device_check_passed", record=rec)
        run.result["device_checks"] = checks
        run.result["validation_before"] = validation(model, case, cfg, device,backend=getattr(run,"graph_backend","upstream"))
        json_save(run.out / "baseline_checks.json", dict(device_checks=checks, validation_before=run.result["validation_before"]))
        torch_save(run.out / "preflight_last.pt", checkpoint(model, optimizer, rng, device, 0, run.binding, []))
    while completed < cfg["updates"]:
        run.check_time()
        position = completed + 1
        run.event("training_input_start", position=position)
        inp = train_input(case, cfg["T"], cfg["rounds"], cfg["train_seed"], position, rng)
        run.event("training_update_start", position=position, time=inp["t"], query_slots=len(inp["ids"]), noisy_edges=inp["edges"].shape[1])
        rec = update(model, optimizer, case, inp, cfg["T"], device,backend=getattr(run,"graph_backend","upstream"))
        rec.update(position=position, timing_phase="warmup" if position <= cfg["warmup_updates"] else "stable")
        rec["batch_compute_seconds"] = rec["preparation_seconds"] + rec["noise_seconds"] + rec["synchronized_update_seconds"]
        records.append(rec)
        completed = position
        run.result["completed_updates"] = completed
        start = time.perf_counter()
        torch_save(run.out / "preflight_last.pt", checkpoint(model, optimizer, rng, device, completed, run.binding, records))
        run.event("committed_training_update", record=rec, checkpoint_seconds=time.perf_counter() - start)
    run.result.update(records=records, completed_updates=completed, validation_after=validation(model, case, cfg, device,backend=getattr(run,"graph_backend","upstream")))
    baseline = json.loads((run.out / "baseline_checks.json").read_text())
    run.result.update(baseline)
    stable = [r["batch_compute_seconds"] for r in records if r["timing_phase"] == "stable"]
    run.result.update(stable_batch_median_seconds=float(np.median(stable)),
                      stable_batch_p90_seconds=float(np.quantile(stable, .9)),
                      timing_definition="CPU preparation/noising plus synchronized device update; excludes checkpoint IO and validation",
                      lr_schedule="constant; first five updates are timing warmup only", data_hash_end=sha(run.args.data))
    assert run.result["data_hash_end"] == cfg["imdb_sha256"]
    run.finish("passed", next_step="manually run C-imdb with fixed preflight weight and explicit wall cap; no formal training")


def callback_for(model, case, state, kernel, t, run, measure=False):
    return StepPredictor(model,case,state,kernel,t,getattr(run,"graph_backend","upstream"),
                         event=run.event,check_time=run.check_time,profile=measure)


def chain_checkpoint(run, state, rng, device, t, completed, binding):
    torch_save(run.out / "chain_last.pt", dict(state=state.copy(), rng=rng_save(rng, device),
               remaining_time=t, completed_reverse_steps=completed, binding=binding,
               contract="frozen Xt in every block; commit only after full_sweep returns"))


def ensure_sweep_cost(run, model, case, state, kernel, t, rounds, rng):
    """Measure missing receipt before continuing. No graph/RNG/step advancement."""
    path = run.out / "sweep_cost.json"
    if path.exists():
        receipt = json.loads(path.read_text())
        for key in ("measured_sweep_seconds", "estimated_T1000_seconds"):
            value = receipt[key]
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not np.isfinite(value) or value <= 0:
                raise RuntimeError("invalid sweep cost receipt: " + key)
        run.result.update(receipt)
        return False
    if not 0 < t < len(kernel.alpha):
        raise RuntimeError("missing sweep cost but no unfinished reverse time remains; cannot fabricate timing")
    device, cfg = run.args.device, run.cfg
    saved_rng = rng_save(rng, device)
    frozen = state.copy()
    run.event("measure_full_probability_sweep", time=t, resumed=run.args.resume)
    try:
        plan = plan_for(case, [cfg["sample_seed"], t, 61], rounds)
        predict = callback_for(model, case, state, kernel, t, run, measure=True)
        sync(device)
        start = time.perf_counter()
        seen = np.zeros(case["space"].total, dtype=np.uint8)
        for r in range(rounds):
            ids, fam, query = plan.block(r)
            assert not seen[ids].any()
            p0 = predict(state.copy(), ids, fam, query)
            q = kernel.reverse(p0, state[ids], fam, t-1, t)
            assert np.isfinite(q).all() and ((q >= 0) & (q <= 1)).all()
            seen[ids] += 1
        assert seen.sum() == plan.query_total and np.array_equal(state, frozen)
        sync(device)
        cost = time.perf_counter() - start
        T = len(kernel.alpha) - 1
        receipt = dict(measured_sweep_seconds=cost, estimated_T1000_seconds=T * cost,
                       measured_time=t, measured_context_edges=int(state.sum()),
                       estimate_boundary="one frozen saved state only; resumed timing may be at t<T; density can change cost/OOM; no ETA guarantee")
        json_save(path, receipt)
        run.result.update(receipt)
        run.event("cost_estimate", sweep_seconds=cost, estimated_chain_seconds=T * cost, wall_cap=run.args.wall_seconds)
        return True
    finally:
        if "predict" in locals() and hasattr(predict,"close"): predict.close()
        rng_restore(saved_rng, rng, device)


def stage_c(run, large):
    cfg, device = run.cfg, run.args.device
    a = prerequisite(run.args.a_dir, "A", run.binding)
    if large:
        b = prerequisite(run.args.b_dir, "B", run.binding)
        prerequisite(run.args.small_dir, "C-small", run.binding)
        case, T, rounds = imdb_case(run.args.data, cfg), cfg["T"], cfg["rounds"]
        weight = b / "preflight_last.pt"
    else:
        case, T, rounds, weight = toy_case(), 8, 3, a / "smoke_final.pt"
    seed_all(cfg["sample_seed"])
    rng = np.random.default_rng(cfg["sample_seed"])
    model = make_model(case, cfg, device)
    ck = trusted_load(weight)
    expected_data = cfg["imdb_sha256"] if large else "deterministic six-node engineering fixture"
    expected_binding = dict(run.binding, data=expected_data, T=T, rounds=rounds)
    if ck["binding"] != expected_binding or ck["completed_updates"] != (100 if large else 2):
        raise RuntimeError("weight binding/update count mismatch")
    model.load_state_dict(ck["model"], strict=True)
    model.eval()
    run.result["actual_dimensions"] = dimensions(model)
    kernel = MarginalKernel(case["prior"], schedule(T))
    binding = dict(run.binding, data=expected_data, T=T, rounds=rounds, weight_sha256=sha(weight))
    run.result["sampling_binding"] = binding
    run.result["query_inventory"] = query_inventory(case, plan_for(case, [cfg["sample_seed"], T, 61], rounds))
    if run.args.resume:
        saved = trusted_load(run.out / "chain_last.pt")
        if saved["binding"] != binding:
            raise RuntimeError("chain checkpoint inputs differ")
        state, t, completed = saved["state"], saved["remaining_time"], saved["completed_reverse_steps"]
        rng_restore(saved["rng"], rng, device)
        run.event("chain_restored", remaining_time=t, completed_steps=completed)
    else:
        pi = np.repeat(case["prior"], case["space"].counts)
        state = (rng.random(case["space"].total) < pi).astype(np.uint8)
        t, completed = T, 0
        chain_checkpoint(run, state, rng, device, t, completed, binding)
        run.result["prior_state_metrics"] = graph_stats(case, state)
    if large:
        ensure_sweep_cost(run, model, case, state, kernel, t, rounds, rng)
    while t > 0:
        run.check_time()
        before_rng = rng_save(rng, device)
        sync(device)
        start = time.perf_counter()
        try:
            next_state, receipt = reverse_step(model,case,state,kernel,t,rounds,cfg["sample_seed"],rng,device,
                          backend=getattr(run,"graph_backend","upstream"),event=run.event,check_time=run.check_time)
        except BaseException:
            # Discard partial block outputs. Restore RNG before this whole step.
            rng_restore(before_rng, rng, device)
            chain_checkpoint(run, state, rng, device, t, completed, binding)
            raise
        sync(device)
        state, t, completed = next_state.astype(np.uint8), t-1, completed+1
        chain_checkpoint(run, state, rng, device, t, completed, binding)
        run.result.update(remaining_time=t, completed_reverse_steps=completed)
        run.event("reverse_time_committed", remaining_time=t, completed_steps=completed,
                  step_seconds=time.perf_counter()-start, edges=int(state.sum()), receipt=receipt, **memory(device))
    stats = graph_stats(case, state)
    edges, fam = state_edges(case, state)
    np.savez_compressed(run.out / "engineering_final_graph.npz", types=case["space"].types, edges=edges.T,
                        families=fam, state=state, T=np.array(T))
    with np.load(run.out / "engineering_final_graph.npz", allow_pickle=False) as z:
        assert np.array_equal(z["state"], state) and np.array_equal(z["edges"], edges.T)
        assert np.array_equal(z["types"], case["space"].types)
    run.result.update(final_graph_metrics=stats, completed_reverse_steps=T, remaining_time=0,
                      engineering_chain_outputs=1, graph_T=T, graph_file_sha256=sha(run.out / "engineering_final_graph.npz"))
    run.finish("passed", next_step="return measured engineering results; choose formal workload separately, no automatic continuation")


def watchdog(args):
    # Only this child is managed. Parent has never allocated device tensors.
    command = [sys.executable, "-m", "hsgd.bw_preflight", *sys.argv[1:], "--worker"]
    start = time.monotonic()
    child = subprocess.Popen(command)
    try:
        code = child.wait(timeout=args.wall_seconds)
        if args.output.exists():
            json_save(args.output / "launcher.json", dict(child_exit_code=code,
                      launch_wall_seconds=time.monotonic()-start,
                      wall_cap_seconds=args.wall_seconds, pid=child.pid))
        return code
    except subprocess.TimeoutExpired:
        child.terminate()
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait(timeout=10)
        if not args.output.exists():
            args.output.mkdir(parents=True, exist_ok=False)
        out = args.output
        if out.exists():
            summary = dict(stage=args.stage, status="incomplete", reason="parent wall watchdog",
                           wall_cap_seconds=args.wall_seconds, managed_pid=child.pid,
                           max_termination_grace_seconds=20, formal_training=False)
            checkpoint_name = "chain_last.pt" if args.stage.startswith("C") else ("smoke_step1.pt" if args.stage == "A" else "preflight_last.pt")
            path = out / checkpoint_name
            if path.exists():
                saved = trusted_load(path)
                for key in ("remaining_time", "completed_reverse_steps", "completed_updates"):
                    if key in saved:
                        summary[key] = saved[key]
                summary["checkpoint_sha256"] = sha(path)
            run_path = out / "run.json"
            if run_path.exists():
                saved_run = json.loads(run_path.read_text())
                summary.update(environment=saved_run["environment"], config=saved_run["initial_binding"]["config"])
            if (out / "sweep_cost.json").exists():
                summary.update(json.loads((out / "sweep_cost.json").read_text()))
            json_save(out / "watchdog_stop.json", summary)
            if not (out / "summary.json").exists():
                json_save(out / "summary.json", summary)
            print_summary(summary)
        return 75
    except KeyboardInterrupt:
        child.terminate()
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait(timeout=10)
        print("User interrupted own child; last atomic checkpoint preserved, no automatic restart.", flush=True)
        return 130


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["A", "B", "C-small", "C-imdb", "summary"])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--backend", choices=["upstream","legacy"], default="upstream")
    parser.add_argument("--data", type=Path, default=PACKAGE / "data/IMDB_movie_reference.npz")
    parser.add_argument("--a-dir", type=Path)
    parser.add_argument("--b-dir", type=Path)
    parser.add_argument("--small-dir", type=Path)
    parser.add_argument("--wall-seconds", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.output = args.output or Path("/root/private_data/gnn_projects/bw_sparse_runs_v6") / args.stage
    args.a_dir = args.a_dir or args.output.parent / "A"
    args.b_dir = args.b_dir or args.output.parent / "B"
    args.small_dir = args.small_dir or args.output.parent / "C-small"
    args.wall_seconds = args.wall_seconds or (3600 if args.stage == "B" else 1800)
    if args.wall_seconds <= 0:
        parser.error("positive wall limit required")
    if args.stage == "summary":
        for stage in ["A", "B", "C-small", "C-imdb"]:
            path = args.output / stage / "summary.json"
            if path.exists():
                print_summary(json.loads(path.read_text()))
            else:
                print(stage + ": not run")
        return 0
    if args.device == "cpu" and args.stage in ("B", "C-imdb"):
        parser.error("full IMDB stages require accelerator; CPU allowed only for engineering fixtures")
    if not args.worker:
        return watchdog(args)
    run = None
    try:
        run = Run(args)
        if args.stage == "A":
            stage_a(run)
        elif args.stage == "B":
            stage_b(run)
        else:
            stage_c(run, args.stage == "C-imdb")
        return 0
    except BaseException as exc:
        traceback.print_exc()
        if run is not None:
            if isinstance(exc, Deadline):
                path = run.out / ("chain_last.pt" if args.stage.startswith("C") else "preflight_last.pt")
                extra = {}
                if path.exists():
                    saved = trusted_load(path)
                    extra = {k: saved[k] for k in ("remaining_time", "completed_reverse_steps", "completed_updates") if k in saved}
                run.finish("incomplete", failure=str(exc), **extra)
                return 75
            run.finish("failed", failure=repr(exc), next_step="inspect exact failing phase; no automatic fallback/retry")
        return 1


if __name__ == "__main__":
    sys.exit(main())
