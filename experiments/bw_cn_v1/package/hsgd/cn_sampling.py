"""Unmodified transaction/graph-stat routines from reviewed bw_single_graph_v1.
New native weight import is in cn_entry; no v9/preflight weight compatibility.
"""
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parents[1]
VERSION="bw_cn_v1_sampling"

def adapter_hashes():
    manifest=read(ROOT/"manifest_cn_v1.json")
    entries={n:v["sha256"] for n,v in manifest["files"].items()}
    for n,s in entries.items():
        if digest(required(ROOT,n))!=s:raise RuntimeError("package changed: "+n)
    return dict(manifest_sha256=digest(ROOT/"manifest_cn_v1.json"),files=entries)

def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def required(root, name):
    if not isinstance(name, str) or not name or Path(name).is_absolute():
        raise RuntimeError("expected nonempty relative evidence path")
    root = Path(root).resolve()
    p = (root / name).resolve()
    if root not in p.parents or not p.is_file() or p.stat().st_size == 0:
        raise RuntimeError("missing, empty or escaping evidence: " + name)
    return p

def write(path, value):
    p = Path(path)
    tmp = p.with_name(p.name + ".tmp")
    with tmp.open("w") as f:
        json.dump(value, f, indent=2, allow_nan=False, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(p)

def atomic_torch(path, value):
    import torch
    p = Path(path)
    tmp = p.with_name(p.name + ".tmp")
    with tmp.open("wb") as f:
        torch.save(value, f)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(p)

def graph_stats(case, state):
    import numpy as np
    from hsgd.bw_preflight.common import state_edges
    state = np.asarray(state)
    if state.shape != (case["space"].total,) or not np.isin(state, [0, 1]).all():
        raise RuntimeError("invalid full-slot state")
    for f, pi in enumerate(case["prior"]):
        if pi in (0., 1.) and not (state[case["space"].offsets[f]:case["space"].offsets[f+1]] == int(pi)).all():
            raise RuntimeError("state violates fixed support")
    edges, families = state_edges(case, state)
    types = case["space"].types
    n, k = len(types), case["n_families"]
    degree = np.zeros((n, k), dtype=np.int64)
    parent = np.arange(n)
    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = int(parent[u])
        return u
    for r in range(k):
        degree[:, r] = np.bincount(edges[:, families == r].ravel(), minlength=n)
    for a, b in edges.T:
        x, y = find(int(a)), find(int(b))
        parent[x] = y
    sizes = sorted(np.bincount([find(i) for i in range(n)]).tolist(), reverse=True)
    sizes = [s for s in sizes if s]
    total = degree.sum(1)
    def distribution(x):
        return dict(nodes=len(x), histogram=np.bincount(x).tolist(), mean=float(x.mean()) if len(x) else None,
                    median=float(np.median(x)) if len(x) else None, max=int(x.max()) if len(x) else None)
    encoded, local = case["space"].encode(edges)
    ids = case["space"].offsets[encoded] + local
    simple = bool((edges[0] < edges[1]).all() and len(np.unique(ids)) == len(ids))
    stats = dict(nodes=n, physical_edges=edges.shape[1], type_counts=np.bincount(types).tolist(),
        family_pairs=[list(x) for x in case["space"].pairs], family_edges=np.bincount(families, minlength=k).tolist(),
        isolated_nodes=int((total == 0).sum()), components=len(sizes), component_sizes=sizes,
        connected=len(sizes) == 1, largest_component_nodes=max(sizes, default=0),
        degree=distribution(total), degree_by_type={str(t): distribution(total[types == t]) for t in np.unique(types)},
        degree_by_relation={str(r): distribution(degree[:, r]) for r in range(k)},
        degree_by_type_relation={str(t): {str(r): distribution(degree[types == t, r]) for r in range(k)} for t in np.unique(types)},
        schema_legal=bool(np.array_equal(encoded, families)), simple_undirected=simple,
        edge_storage="canonical u<v once; each edge contributes degree at both endpoints",
        relation_histogram_population="all nodes, including zeros; type-relation populations also reported")
    return stats, edges, families, degree

def check_state(ck, case, binding, T):
    import numpy as np
    from hsgd.learning_budget import signature
    if ck["binding"] != binding or signature(ck["binding"]) != ck["binding_sha256"]:
        raise RuntimeError("sampling checkpoint binding changed")
    t, steps = ck["remaining_time"], ck["completed_reverse_steps"]
    if type(t) is not int or type(steps) is not int or not 0 <= t <= T or steps + t != T:
        raise RuntimeError("sampling time/step count mismatch")
    if ck["state_slots"] != case["space"].total or ck["packed_state"].dtype != np.uint8:
        raise RuntimeError("sampling state shape/type mismatch")
    if ck["packed_state"].shape != ((case["space"].total + 7)//8,):
        raise RuntimeError("packed state length mismatch")
    if set(ck["rng"]) != {"generator", "python", "numpy", "torch", "accelerator"}:
        raise RuntimeError("sampling RNG incomplete")
    state = np.unpackbits(ck["packed_state"])[:ck["state_slots"]]
    graph_stats(case, state)
    return state

def chain(model, case, cfg, receipt, out, seed, device, resume=False, stop_after=None, interrupt_block=None):
    """stop_after/interrupt_block are internal finite engineering-test hooks only."""
    import numpy as np
    import torch
    from hsgd.bw_preflight.common import seed_all, rng_save, rng_restore, trusted_load, sync, memory, schedule
    from hsgd.kernel import MarginalKernel
    from hsgd.learning_budget import signature
    from hsgd.learning_mode import require, state as mode_state
    from hsgd.runtime import reverse_step
    out = Path(out)
    binding = dict(adapter=adapter_hashes(), input=receipt, seed=seed,
        T=cfg["T"], rounds=cfg["rounds"], backend="upstream", execution_mode=mode_state(),
        contract="frozen Xt/full sweep/atomic complete-step commit; ordinary resume not bitwise certified")
    require(mode_state(), False)
    model.eval()
    seed_all(seed)
    rng = np.random.default_rng(seed)
    T, t, completed = cfg["T"], cfg["T"], 0
    kernel = MarginalKernel(case["prior"], schedule(T))
    stopped = [None]
    old_handlers = {}
    for sig in (signal.SIGINT, signal.SIGTERM):
        old_handlers[sig] = signal.signal(sig, lambda s, f: stopped.__setitem__(0, "signal_" + str(s)))
    started, cpu = time.perf_counter(), time.process_time()
    cumulative, cumulative_cpu, step_seconds, checkpoint_seconds = 0., 0., 0., 0.
    failure = None
    def event(phase, **kw):
        row = dict(phase=phase, utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **kw)
        with (out/"events.jsonl").open("a") as f:
            f.write(json.dumps(row, allow_nan=False) + "\n")
        print(json.dumps(row, ensure_ascii=False), flush=True)
    def stop_check():
        if stopped[0] or (out / "STOP_REQUESTED").exists():
            raise InterruptedError("safe stop requested; incomplete time step discarded")
        require(mode_state(), False)
    def save(last_step):
        nonlocal checkpoint_seconds
        tick = time.perf_counter()
        filename = "states/step_%04d_%d.pt" % (completed, time.time_ns())
        ck = dict(binding=binding, binding_sha256=signature(binding), packed_state=np.packbits(state),
            state_slots=len(state), remaining_time=t, completed_reverse_steps=completed,
            rng=rng_save(rng, device), cumulative_wall_seconds=cumulative+time.perf_counter()-started,
            cumulative_cpu_seconds=cumulative_cpu+time.process_time()-cpu,
            reverse_compute_seconds=step_seconds, checkpoint_seconds=checkpoint_seconds, last_step=last_step)
        atomic_torch(out/filename, ck)
        write(out/"chain_latest.json", dict(path=filename, sha256=digest(out/filename),
            remaining_time=t, completed_reverse_steps=completed, binding_sha256=signature(binding)))
        checkpoint_seconds += time.perf_counter()-tick
    try:
        if resume:
            pointer = read(required(out, "chain_latest.json"))
            p = required(out, pointer["path"])
            if digest(p) != pointer["sha256"]:
                raise RuntimeError("sampling checkpoint hash mismatch")
            ck = trusted_load(p)
            # Observational environment timestamp changes; stable binding was frozen at first launch.
            old_binding = read(required(out, "sampling_binding.json"))
            comparable = json.loads(json.dumps(binding))
            comparable["input"]["environment"]["utc"] = old_binding["input"]["environment"]["utc"]
            if comparable != old_binding:
                raise RuntimeError("sampling source/input/config/runtime/seed changed")
            binding = old_binding
            state = check_state(ck, case, binding, T)
            if (pointer["remaining_time"] != ck["remaining_time"] or pointer["completed_reverse_steps"] != ck["completed_reverse_steps"] or
                    pointer["binding_sha256"] != signature(binding)):
                raise RuntimeError("pointer/time/binding mismatch")
            t, completed = ck["remaining_time"], ck["completed_reverse_steps"]
            cumulative, cumulative_cpu = ck["cumulative_wall_seconds"], ck["cumulative_cpu_seconds"]
            step_seconds, checkpoint_seconds = ck["reverse_compute_seconds"], ck["checkpoint_seconds"]
            rng_restore(ck["rng"], rng, device)
            # Exact loaded RNG, without promising future non-deterministic trajectory identity.
            from hsgd.perf_checks import compare_rng
            compare_rng(ck["rng"], rng_save(rng, device))
            event("restored", remaining_time=t, completed_reverse_steps=completed, loaded_RNG_exact=True,
                  ordinary_future_trajectory_identity=False)
            flag = out/"STOP_REQUESTED"
            if flag.exists():
                flag.rename(out/("STOP_REQUESTED_served_%d" % time.time_ns()))
        else:
            pi = np.repeat(case["prior"], case["space"].counts)
            state = (rng.random(case["space"].total) < pi).astype(np.uint8)
            write(out/"sampling_binding.json", binding)
            (out/"states").mkdir()
            save(None)
            event("prior_initialized", T=T, rounds=cfg["rounds"], slots=len(state), seed=seed,
                  state_edges=int(state.sum()), no_time_limit=True, scientific_requests=int(receipt.get("formal_training") is True))
        # FrozenTypedContext checks Tensor._version; inference_mode removes that
        # counter. no_grad preserves the original guard without changing v9.
        with torch.no_grad():
            while t:
                stop_check()
                # reverse_step restores RNG and closes the frozen context on any partial failure.
                next_state, rec = reverse_step(model,case,state,kernel,t,cfg["rounds"],seed,rng,device,
                    backend="upstream",check_time=stop_check,
                    event=lambda phase, **kw: event(phase, **kw) if phase=="reverse_block" and kw["block_completed"]%5==0 else None,
                    interrupt_block=interrupt_block)
                if not rec["exact_coverage"] or rec["blocks"] != cfg["rounds"]:
                    raise RuntimeError("incomplete sweep cannot be committed")
                state, t, completed = next_state, t-1, completed+1
                step_seconds += rec["step_seconds"]
                save(rec)
                write(out/"progress.json", dict(status="running", completed_reverse_steps=completed,
                    remaining_time=t, last_step_seconds=rec["step_seconds"], physical_edges=int(state.sum()),
                    session_wall_seconds=time.perf_counter()-started, **memory(device)))
                event("time_committed", completed_reverse_steps=completed, remaining_time=t,
                      seconds=rec["step_seconds"], physical_edges=int(state.sum()),
                      estimated_remaining_seconds=rec["step_seconds"]*t,
                      ETA_note="last-step extrapolation only; graph density and time may change cost")
                if stop_after is not None and completed >= stop_after:
                    raise InterruptedError("finite engineering test stop")
        stats, edges, families, degree = graph_stats(case, state)
        reference_state = np.isin(np.arange(case["space"].total),case["clean_ids"]).astype(np.uint8)
        ref, _, _, _ = graph_stats(case, reference_state)
        target = out/"graph.npz"
        if target.exists():
            raise RuntimeError("final graph already exists; refusing overwrite")
        tmp = out/"graph.npz.tmp"
        with tmp.open("wb") as f:
            np.savez_compressed(f, types=case["space"].types, edges=edges.T,
                relations=families+1, family_ids=families, family_pairs=np.asarray(case["space"].pairs),
                degree=degree.sum(1), relation_degree=degree, source_model_sha256=receipt["model_sha256"], seed=seed)
            f.flush(); os.fsync(f.fileno())
        tmp.replace(target)
        with np.load(target, allow_pickle=False) as z:
            if not np.array_equal(z["edges"],edges.T) or not np.array_equal(z["types"],case["space"].types) or not np.array_equal(z["family_ids"],families):
                raise RuntimeError("final graph reload mismatch")
        result = dict(status="completed", completed_reverse_steps=completed, remaining_time=t,
            graph_sha256=digest(target), generated=stats, reference=ref,
            final_graph="graph.npz", no_completion_or_filter=True, reloaded_graph_exact=True)
    except InterruptedError as exc:
        result = dict(status="stopped", reason=str(exc), completed_reverse_steps=completed,
            remaining_time=t, final_graph=None, incomplete_step_discarded=True)
    except BaseException as exc:
        failure = traceback.format_exc()
        # Pointer is the authoritative durable commit, including a crash during save.
        if (out/"chain_latest.json").exists():
            committed = read(out/"chain_latest.json")
            t, completed = committed["remaining_time"], committed["completed_reverse_steps"]
        result = dict(status="failed", reason=type(exc).__name__+": "+str(exc),
            traceback=failure, completed_reverse_steps=completed, remaining_time=t, final_graph=None)
    finally:
        sync(device)
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
    result.update(version=VERSION, seed=seed, model_sha256=receipt["model_sha256"],
        source_final_checkpoint=receipt["final_checkpoint"], source_completed_updates=receipt["completed_updates"],
        session_wall_seconds=time.perf_counter()-started, cumulative_wall_seconds=cumulative+time.perf_counter()-started,
        session_cpu_seconds=time.process_time()-cpu, cumulative_cpu_seconds=cumulative_cpu+time.process_time()-cpu,
        reverse_compute_seconds=step_seconds, checkpoint_seconds=checkpoint_seconds,
        inference_mode="eval + no_grad, deterministic_algorithms=False", precision=cfg["precision"],
        scientific_output_count=int(result["status"]=="completed" and receipt.get("formal_training") is True), training_updates=0,
        ordinary_resume_bitwise_trajectory_certified=False, **memory(device))
    write(out/"summary.json",result)
    event("sampling_terminal", **result)
    return 1 if failure else 0

def unchanged(receipt):
    for root, files in ((receipt["source_package"]["path"], receipt["source_package"]["files"]),
                        (receipt["run"], receipt["run_evidence"])):
        for name, sha in files.items():
            if digest(required(root, name)) != sha:
                raise RuntimeError("read-only source changed during execution: " + name)
