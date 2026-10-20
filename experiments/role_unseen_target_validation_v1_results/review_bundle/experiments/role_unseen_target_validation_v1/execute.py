"""Read-only fixed-weight experiment. One parent clock, three CPU seed jobs.

The launcher is standard-library only. Original reverse_step is called verbatim.
No optimizer, fitting, proof graph or old generated graph is reachable.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
PLAN = EXP / "role_target_scalability_v1"


def save(path, value):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def cfg():
    return load(HERE / "config.json")


def source_files():
    # Bind the complete reused scientific source plus this adapter and frozen plan.
    sys.path.insert(0, str(EXP / "role_robust_training_v1"))
    from core import file_manifest
    result = dict(file_manifest())
    for directory in (HERE, PLAN):
        for p in directory.rglob("*"):
            if p.is_file() and p.suffix in (".py", ".json", ".md") and "proof_only" not in p.parts:
                result[str(p.relative_to(EXP))] = digest(p)
    return result


def check_sources(out):
    for name, expected in load(out / "source_manifest.json").items():
        if digest(EXP / name) != expected:
            raise ValueError("frozen source changed: " + name)


def new_output(out, protected):
    out = out.resolve()
    for p in protected:
        p = p.resolve()
        if out == p or out in p.parents or p in out.parents:
            raise ValueError("output overlaps protected path: " + str(p))
    if out.exists():
        raise FileExistsError("existing output forbidden; cumulative budget cannot restart")
    out.mkdir(parents=True)
    return out


def budget_check(out):
    ledger = load(out / "activation.json")
    def check():
        if (out / "STOP").exists():
            raise InterruptedError("STOP")
        if time.time() >= ledger["deadline_epoch"] - cfg()["exit_reserve_seconds"]:
            raise TimeoutError("unique cumulative45min deadline; no retries")
    return check


def lazy():
    os.environ.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    sys.path.insert(0, str(EXP / "role_robust_training_v1"))
    import core
    core.setup()
    return core


def target_objects(c):
    result = {}
    for rec in load(PLAN / "evidence/targets.json")["targets"]:
        x = rec["anonymous_target"]
        target = c.RoleTarget(tuple(tuple((tuple(v), int(n)) for v, n in rows) for rows in x["roles"]),
                              tuple(x["type_counts"]), tuple(map(tuple, x["schema"])))
        assert target.fingerprint == rec["fingerprint"]
        result[rec["name"]] = target
    assert list(result) == cfg()["target_order"]
    return result


def metadata_case(c, panel, targets):
    """NO clean edge identity in inference case: only inventory/schema and stubs."""
    import numpy as np
    x = load(panel / "panel.json")
    target = targets["balanced"]
    types = np.asarray(x["graphs"][0]["types"], dtype=np.int64)
    space = c.c.TypedSlots(types, tuple(map(tuple, x["schema"])))
    assert tuple(np.bincount(types)) == target.type_counts and len(types) == 34
    counts = []
    for r, (a, b) in enumerate(space.pairs):
        stubs = sum(v[r] * n for v, n in target.roles[a])
        counts.append(stubs // 2 if a == b else stubs)
    counts = np.asarray(counts, dtype=np.int64)
    assert counts.tolist() == [0, 10, 18, 0, 0, 36, 0, 0, 24, 0]
    prior = np.divide(counts, space.counts, out=np.zeros(len(counts), dtype=np.float64), where=space.counts > 0)
    return dict(space=space, prior=prior, counts=counts, active=counts > 0,
                n_types=4, n_families=10)


def verify_inputs(a, c):
    import importlib.metadata
    expected = load(PLAN / "evidence/weights.json")
    pinned = EXP / "role_robust_training_6000_v1_results/evidence/learning_run01"
    assert c.file_manifest() == load(pinned / "source_manifest.json")
    assert digest(a.panel / "panel.json") == expected["history"]["panel_sha256"]
    assert digest(pinned / "resolved_config.json") == expected["config_sha256"]
    assert load(a.run / "resolved_config.json") == expected["full_resolved_config"]
    assert expected["full_resolved_config"]["T"] == 64
    assert expected["full_resolved_config"]["rounds"] == 4
    actual = c.c.environment("cpu")
    for w in expected["weights"]:
        binding = w["checkpoint_binding"]["environment"]
        for key in ("python_version", "packages", "torch_runtime_version", "float32_matmul_precision",
                    "matmul_allow_tf32", "deterministic_algorithms", "torch_threads"):
            assert actual[key] == binding[key], (key, actual[key], binding[key])
        path = a.run / f"seedjobs/seed{w['seed']}/seed{w['seed']}/normal2_final_6000.pt"
        assert digest(path) == w["sha256"]
    assert not c.torch.cuda.is_initialized()
    return expected, actual


def chain(model, case, target, sample_seed, c, check, progress):
    """Same prior, plan seed, full_sweep and RNG as original worker.sample_final."""
    import numpy as np
    model.eval()
    model.set_target(target)
    c.c.seed_all(sample_seed)
    rng = np.random.default_rng(sample_seed)
    state = (rng.random(case["space"].total) < np.repeat(case["prior"], case["space"].counts)).astype(np.uint8)
    initial = c.sha(state.tolist())
    kernel = c.MarginalKernel(case["prior"], c.c.schedule(64))
    steps = []
    for t in range(64, 0, -1):
        check()
        uh, after = c.uniform_contract(rng, len(state))
        plan = c.c.plan_for(case, [sample_seed, t, 61], 4)
        h = hashlib.sha256()
        for b in range(4):
            for ar in plan.block(b):
                h.update(np.ascontiguousarray(ar).tobytes())
        with c.torch.no_grad():
            next_state, info = c.reverse_step(model, case, state, kernel, t, 4, sample_seed, rng, "cpu", check_time=check)
        assert c.sha(rng.bit_generator.state) == after and info["exact_coverage"] and info["blocks"] == 4
        state = next_state  # commit only complete step
        steps.append(dict(t=t, uniform_sha256=uh, query_sha256=h.hexdigest(), rng_after=after,
                          state_sha256=c.sha(state.tolist()), edge_count=int(state.sum()), exact_coverage=True,
                          blocks=info["blocks"], covered_slots=info["covered_slots"], step_seconds=info["step_seconds"]))
        progress(state, steps, rng.bit_generator.state)
    return state, steps, initial


def seed_worker(a):
    start = time.monotonic()
    import resource
    cpu = time.process_time()
    folder = a.output / f"seed{a.seed}"
    status = "failed"; reason = None; completed = 0; forwards = 0
    def on_signal(sig, frame):
        (a.output / "STOP").touch()
    for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(s, on_signal)
    check = budget_check(a.output)
    try:
        c = lazy()
        import numpy as np
        from role_condition.metrics import graph
        from analyse import target_errors
        check(); check_sources(a.output)
        expected, environment = verify_inputs(a, c)
        targets = target_objects(c); case = metadata_case(c, a.panel, targets)
        config = expected["full_resolved_config"]
        w = next(x for x in expected["weights"] if x["seed"] == a.seed)
        path = a.run / f"seedjobs/seed{a.seed}/seed{a.seed}/normal2_final_6000.pt"
        tick = time.monotonic()
        model = c.CapacityDenoiser(case["n_types"], case["n_families"], family_pairs=case["space"].pairs,
                                  prior=case["prior"], T=64, **config["model"])
        state_dict = c.torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict, strict=True)
        assert all(c.torch.equal(v, model.state_dict()[k]) for k, v in state_dict.items())
        assert np.array_equal(model.prior.numpy(), case["prior"])
        save(folder / "binding.json", dict(weight=w, environment=environment, model_config=config,
             parameters=sum(p.numel() for p in model.parameters()), model_load_seconds=time.monotonic()-tick,
             inference_case_keys=sorted(case), proof_graphs_read=False, optimizer_created=False,
             source_manifest_sha256=digest(a.output / "source_manifest.json")))
        for sample_seed in config["sampling_seeds"]:
            for name in cfg()["target_order"]:
                check(); tick = time.monotonic(); stem = f"{name}_{sample_seed}"
                def progress(state, steps, rng):
                    # Cheap observable boundary; full state/RNG committed every8 steps and at chain end.
                    if len(steps) % 8 == 0:
                        tmp = folder / "partial.npz"
                        np.savez_compressed(tmp, state=state)
                        save(folder / "partial.json", dict(condition=name, sampling_seed=sample_seed,
                             completed_steps=len(steps), rng=rng, state_sha256=c.sha(state.tolist()), steps=steps))
                        save(folder / "progress.json", dict(completed_chains=completed, condition=name,
                             sampling_seed=sample_seed, completed_steps=len(steps), elapsed_seconds=time.monotonic()-start))
                state, steps, initial = chain(model, case, targets[name], sample_seed, c, check, progress)
                edges, families = c.c.state_edges(case, state)
                path_out = folder / (stem + ".npz")
                np.savez_compressed(path_out, types=case["space"].types, edges=edges.T,
                                    relations=families+1, schema=np.asarray(case["space"].pairs))
                save(folder / (stem + "_steps.json"), steps)
                check(); metrics = graph(case, edges, c.NAMES, check)
                row = dict(seed=a.seed, condition=name, sampling_seed=sample_seed, checkpoint_update=6000,
                           weight_sha256=w["sha256"], target_fingerprint=targets[name].fingerprint,
                           graph_file=str(path_out.relative_to(a.output)), graph_sha256=digest(path_out),
                           steps_file=str((folder/(stem+"_steps.json")).relative_to(a.output)),
                           steps_sha256=digest(folder/(stem+"_steps.json")), initial_state_sha256=initial,
                           metrics=metrics, target_errors=target_errors(metrics, targets[name].as_dict()),
                           seconds=time.monotonic()-tick, newly_executed_chain=True)
                save(folder/(stem+"_result.json"), row)
                completed += 1; forwards += 256
                print(json.dumps(dict(seed=a.seed, completed=completed, condition=name, sampling_seed=sample_seed,
                      seconds=row["seconds"], G=row["target_errors"]["Genre"])), flush=True)
        check(); check_sources(a.output)
        assert digest(path) == w["sha256"] and completed == 40 and not c.torch.cuda.is_initialized()
        status = "completed"
    except BaseException as exc:
        reason = repr(exc)
        status = "stopped" if isinstance(exc, (InterruptedError, TimeoutError, KeyboardInterrupt)) else "failed"
        traceback.print_exc()
    finally:
        save(folder / "receipt.json", dict(status=status, reason=reason, completed_chains=completed,
             completed_chain_forwards=forwards, wall_seconds=time.monotonic()-start, cpu_seconds=time.process_time()-cpu,
             peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
             training_updates=0, device="cpu", gpu_used=False))
    return 0 if status == "completed" else 1


def manager(a):
    import resource
    started=time.monotonic(); cpu=time.process_time()
    jobs=[]; status="failed"; reason=None
    def stop(sig, frame):
        (a.output/"STOP").touch()
    for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(s,stop)
    check=budget_check(a.output)
    try:
        c=lazy(); check()
        save(a.output/"source_manifest.json", source_files())
        expected,environment=verify_inputs(a,c)
        save(a.output/"environment.json",environment)
        save(a.output/"resolved_config.json",dict(execution=cfg(), frozen_plan=load(PLAN/"config.json"),
             original_model_training_config=expected["full_resolved_config"],
             authorization="latest user approval: new120 CPU chains under one2700s window; no training"))
        save(a.output/"targets.json",load(PLAN/"evidence/targets.json"))
        for seed in expected["full_resolved_config"]["train_seeds"]:
            folder=a.output/f"seed{seed}";folder.mkdir()
            log=(folder/"worker.log").open("w")
            cmd=[sys.executable,"-u",str(HERE/"execute.py"),"seed","--run",str(a.run),"--panel",str(a.panel),
                 "--output",str(a.output),"--seed",str(seed)]
            # Same process group as manager: parent can clean ALL its descendants.
            child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
            jobs.append((child,folder,log))
        last=0
        while any(p.poll() is None for p,_,_ in jobs):
            check()
            for p,folder,_ in jobs:
                if p.poll() not in (None,0):
                    raise RuntimeError(f"seed child failed {folder}: exit{p.returncode}")
            if time.monotonic()-last>20:
                rows=[dict(seed=folder.name,exit_code=p.poll(),progress=load(folder/"progress.json") if (folder/"progress.json").exists() else None) for p,folder,_ in jobs]
                save(a.output/"parallel_progress.json",rows);print(json.dumps(rows),flush=True);last=time.monotonic()
            time.sleep(.3)
        assert all(p.wait()==0 and load(folder/"receipt.json")["status"]=="completed" for p,folder,_ in jobs)
        check()
        from analyse import audit
        audit(a.output,check)
        check();check_sources(a.output)
        for w in expected["weights"]:
            path=a.run/f"seedjobs/seed{w['seed']}/seed{w['seed']}/normal2_final_6000.pt"
            assert digest(path)==w["sha256"]
        check();status="completed"
    except BaseException as exc:
        reason=repr(exc);traceback.print_exc()
        status="stopped" if isinstance(exc,(InterruptedError,TimeoutError,KeyboardInterrupt)) else "failed"
        (a.output/"STOP").touch()
    finally:
        for p,_,_ in jobs:
            if p.poll() is None:p.terminate()
        for p,_,log in jobs:
            try:p.wait(timeout=2)
            except subprocess.TimeoutExpired:p.kill();p.wait()
            log.close()
        save(a.output/"worker_receipt.json",dict(status=status,reason=reason,
             completed_graph_records=len(list(a.output.glob("seed*/*_result.json"))),
             wall_seconds=time.monotonic()-started,cpu_seconds=time.process_time()-cpu,
             peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
             no_training=True,no_gpu=True,no_big_graph=True))
    return 0 if status=="completed" else 1


def main():
    started=time.time();cpu=time.process_time();p=argparse.ArgumentParser()
    p.add_argument("mode",choices=("launch","manager","seed"))
    for key in ("run","panel","output"):p.add_argument("--"+key,type=Path,required=True)
    p.add_argument("--seed",type=int);a=p.parse_args()
    if a.mode=="seed":return seed_worker(a)
    if a.mode=="manager":return manager(a)
    out=new_output(a.output,[a.run,a.panel,EXP])
    # No automatic resume or second activation. Initial clock precedes heavy imports.
    revision=subprocess.check_output(["git","rev-parse","HEAD"],cwd=HERE,text=True).strip()
    dirty=subprocess.check_output(["git","status","--porcelain"],cwd=HERE,text=True)
    if dirty:raise RuntimeError("commit/freeze source before activation")
    deadline=started+cfg()["budget_seconds"]
    save(out/"activation.json",dict(activation_epoch=started,deadline_epoch=deadline,budget_seconds=2700,
         no_reset=True,source_commit=revision,protocol_commit=cfg()["protocol_commit"],command=sys.argv,
         CPU_only=True,expected_new_chains=120,training_authorized=False))
    sys.path.insert(0,str(EXP/"role_robust_training_v1"))
    from run import supervise
    command=[sys.executable,"-u",str(HERE/"execute.py"),"manager","--run",str(a.run.resolve()),
             "--panel",str(a.panel.resolve()),"--output",str(out)]
    code=supervise(command,out,deadline,graceful_stop=True)
    result=load(out/"controller.json");result.update(wall_seconds=time.time()-started,
        parent_cpu_seconds=time.process_time()-cpu,
        budget_seconds=2700,remaining_seconds=deadline-time.time(),formal_activation_not_reset=True)
    save(out/"controller.json",result);print(json.dumps(result),flush=True)
    return code


if __name__=="__main__":sys.exit(main())
