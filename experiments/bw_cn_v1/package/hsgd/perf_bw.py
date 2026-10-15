"""Bounded old-vs-upstream diagnostics. No full chain, formal training or resume.

BW invocation imports B explicitly; only disposable model copies get one update.
One frozen full step computes probabilities/threshold comparisons, not a new
scientific output. Intermediate threshold arrays are not saved as graph samples.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import numpy as np
import torch
from hsgd.bw_preflight.common import (PACKAGE, config, environment, memory, sync, seed_all, make_model,
    toy_case, imdb_case, state_edges, plan_for, schedule, rng_save, check_prob, check_prob_device,
    json_save, sha, code_hashes, torch_save)
from hsgd.computational import FrozenTypedContext, legacy_arrays, legacy_transfer
from hsgd.kernel import MarginalKernel
from hsgd.perf_checks import (TOLERANCES, compare_forward, compare_update, compare_float, exact_graph,
                             threshold_differences, compare_rng)
from hsgd.perf_import import import_fixed_inputs


class Cap(Exception):
    pass


class Meter:
    def __init__(self,device,cap,out):
        self.device,self.cap,self.out=device,cap,out
        self.start=time.monotonic();self.cpu=time.process_time();self.records=[]
    def check(self):
        if time.monotonic()-self.start>=self.cap:raise Cap("bounded performance window exhausted")
    def measure(self,name,fn):
        self.check();sync(self.device)
        a=b=None
        hardware_phase=any(tag in name for tag in ("network","transfer","coalesce_mapping","backward","optimizer","checks_device_only","D2H"))
        if self.device=="cuda" and hardware_phase:
            a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record()
        wall=time.perf_counter();cpu=time.process_time();value=fn()
        if b is not None:b.record()
        sync(self.device)
        row=dict(phase=name,wall_seconds=time.perf_counter()-wall,cpu_seconds=time.process_time()-cpu,
                 device_event_seconds=(a.elapsed_time(b)/1000 if a is not None else None),
                 device_event_definition="device-stream interval may include host dispatch gaps; not pure kernel busy time" if a is not None else "CPU phase or CPU device; no GPU duration asserted")
        self.records.append(row)
        return value,row
    def event(self,phase,**fields):
        row=dict(phase=phase,elapsed_seconds=time.monotonic()-self.start,**fields)
        with (self.out/"events.jsonl").open("a") as f:f.write(json.dumps(row,allow_nan=False)+"\n")
        print(json.dumps(row,allow_nan=False),flush=True)


def stats(values):
    return dict(n=len(values),median=float(np.median(values)),p90=float(np.quantile(values,.9))) if values else dict(n=0)


def legacy_check_prob(p,family,case):
    # Exact v4 implementation for the baseline, no extra GPU support check.
    if not torch.isfinite(p).all() or not ((p>=0)&(p<=1)).all():
        raise RuntimeError("nonfinite/out-of-range probability")
    q=p.detach().cpu().numpy().astype(np.float64)
    pi=case["prior"][family]
    assert (q[pi==0]==0).all() and (q[pi==1]==1).all()
    return q


def graph_only(args,meter):
    cfg=config();case=imdb_case(args.data,cfg)
    # Lightweight parameter/prior provider, no neural network instantiated.
    from types import SimpleNamespace
    model=SimpleNamespace(n_types=case["n_types"],n_families=case["n_families"],
                          family_pairs=case["space"].pairs,prior=torch.tensor(case["prior"],dtype=torch.float64),
                          parameters=lambda: iter([torch.zeros(1)]))
    state=np.isin(np.arange(case["space"].total),case["clean_ids"]).astype(np.uint8)
    (plan,edges,fam),initial=meter.measure("plan_and_existing_extraction",lambda:(plan_for(case,[cfg["sample_seed"],500,61],20),*state_edges(case,state)))
    ctx,prep=meter.measure("frozen_context_setup",lambda:FrozenTypedContext(model,case["space"].types,edges,fam,500))
    blocks=[plan.block(i) for i in range(20)]
    timings={"legacy":[],"upstream":[]};detailed=[]
    for repeat in range(args.repeats+1):
        for backend in ("legacy","upstream"):
            meter.check();start=time.perf_counter();cpu=time.process_time()
            for ids,qf,qe in blocks:
                if backend=="legacy": graph=legacy_transfer(ctx,legacy_arrays(ctx,qe,qf,500),include_query_mask=False)
                else: graph=ctx.assemble(ctx.transfer_query(ctx.prepare_query(qe,qf,500),500),500)
            elapsed=time.perf_counter()-start
            if repeat:timings[backend].append(elapsed)
            detailed.append(dict(backend=backend,warmup=repeat==0,twenty_blocks_seconds=elapsed,cpu_seconds=time.process_time()-cpu))
        meter.event("graph_repeat",repeat=repeat)
    for ids,qf,qe in blocks:
        exact_graph(legacy_transfer(ctx,legacy_arrays(ctx,qe,qf,500)),ctx.assemble(ctx.transfer_query(ctx.prepare_query(qe,qf,500),500),500))
    ctx.close()
    return dict(scope="CPU graph construction only; no network",graph_exact_all_twenty_blocks=True,
                timings={k:stats(v) for k,v in timings.items()},low_interference_records=detailed,
                query_slots=plan.query_total,rounds=20,context_physical_edges=len(fam),
                initialization=[initial,prep],trained_model_calls=0,engineering_optimizer_calls=0)


def one_step(model,case,state,t,cfg,backend,meter,detailed=False):
    start=time.perf_counter();cpu=time.process_time()
    (plan,edges,ef),prep=meter.measure(backend+"/plan_existing",lambda:(plan_for(case,[cfg["sample_seed"],t,61],20),*state_edges(case,state)))
    rng=np.random.default_rng([cfg["sample_seed"],t,71])
    uniform=rng.random(case["space"].total);before=rng_save(rng,meter.device)
    kernel=MarginalKernel(case["prior"],schedule(cfg["T"]))
    ctx=None
    if backend=="upstream" or detailed:
        ctx,setup=meter.measure(backend+"/context_setup",lambda:FrozenTypedContext(model,case["space"].types,edges,ef,t))
    q_all=np.zeros(case["space"].total,dtype=np.float64);seen=np.zeros(len(state),dtype=np.uint8)
    threshold=np.asarray(state).copy();block_times=[]
    model.eval()
    with torch.no_grad():
        for r in range(20):
            meter.check();block_start=time.perf_counter()
            if detailed:(ids,qf,qe),_=meter.measure(backend+"/query_block",lambda:plan.block(r))
            else:ids,qf,qe=plan.block(r)
            if not len(ids):
                # Exactly as full_sweep: no network call for an empty block.
                block_times.append(time.perf_counter()-block_start)
                continue
            # Exact existing copy-check policy, not optimized or relaxed here.
            frozen=state.copy();assert np.array_equal(frozen,state)
            assert not seen[ids].any()
            if detailed:
                if backend=="legacy":
                    arrays,_=meter.measure(backend+"/host_construct",lambda:legacy_arrays(ctx,qe,qf,t))
                    graph,_=meter.measure(backend+"/transfer",lambda:legacy_transfer(ctx,arrays,include_query_mask=False))
                else:
                    query,_=meter.measure(backend+"/query_validation",lambda:ctx.prepare_query(qe,qf,t))
                    query,_=meter.measure(backend+"/transfer",lambda:ctx.transfer_query(query,t))
                    graph,_=meter.measure(backend+"/coalesce_mapping",lambda:ctx.assemble(query,t))
                # Use same network formula; actual old forward equivalence checked separately.
                (_,p),_=meter.measure(backend+"/network",lambda:ctx.run_network(graph,t/cfg["T"],t))
                if backend=="upstream":
                    # Diagnostic only, deliberately excluded from stable timing.
                    # Check every block, not just the first block's probabilities.
                    exact_graph(legacy_transfer(ctx,legacy_arrays(ctx,qe,qf,t)),graph)
                p0,_=meter.measure(backend+"/probability_D2H_checks",lambda:legacy_check_prob(p,qf,case))
                q,_=meter.measure(backend+"/reverse_probability",lambda:kernel.reverse(p0,state[ids],qf,t-1,t))
                output,_=meter.measure(backend+"/threshold",lambda:uniform[ids]<q)
            else:
                if backend=="legacy":p=model(case["space"].types,edges,ef,qe,qf,t/cfg["T"])
                else:p=ctx.predict(qe,qf,t/cfg["T"],t)
                p0=legacy_check_prob(p,qf,case)
                q=kernel.reverse(p0,state[ids],qf,t-1,t);output=uniform[ids]<q
            q_all[ids]=q;threshold[ids]=output;seen[ids]+=1
            sync(meter.device);block_times.append(time.perf_counter()-block_start)
            if detailed:meter.event("diagnostic_block",backend=backend,block=r+1,query=len(ids))
    # Frozen state never committed. Both branches make zero generation advances.
    assert seen.sum()==plan.query_total and (seen<=1).all()
    for f,active in enumerate(case["active"]):
        section=seen[case["space"].offsets[f]:case["space"].offsets[f+1]]
        assert (section==int(active)).all()
    compare_rng(before,rng_save(rng,meter.device))
    if ctx is not None:ctx.close()
    return dict(q=q_all,uniform=uniform,threshold=threshold,covered=seen,
                seconds=time.perf_counter()-start,cpu_seconds=time.process_time()-cpu,
                block_timings=block_times,reverse_state_advances=0)


def training_timing(model,case,inp,cfg,backend,meter,detailed=False,optimizer_state=None):
    start=time.perf_counter()
    # Exclude model/optimizer clone construction, report separately as initialization.
    clone=copy.deepcopy(model);clone.train();opt=torch.optim.AdamW(clone.parameters(),**cfg["optimizer"])
    if optimizer_state is not None:opt.load_state_dict(copy.deepcopy(optimizer_state))
    sync(meter.device);initial=time.perf_counter()-start
    sync(meter.device);start=time.perf_counter();cpu=time.process_time();records=[]
    def phase(name,fn):
        if detailed:
            value,row=meter.measure(backend+"/train/"+name,fn);records.append(row);return value
        return fn()
    opt.zero_grad(set_to_none=True)
    label=phase("label_transfer",lambda:torch.as_tensor(inp["labels"],device=meter.device))
    token=("training-input",0)
    if backend=="legacy":
        if detailed:
            ctx=phase("context_setup",lambda:FrozenTypedContext(clone,case["space"].types,inp["edges"],inp["edge_families"],token))
            arrays=phase("host_construct",lambda:legacy_arrays(ctx,inp["query"],inp["f"],token))
            graph=phase("graph_transfer",lambda:legacy_transfer(ctx,arrays,include_query_mask=False))
            _,p=phase("network",lambda:ctx.run_network(graph,inp["t"]/cfg["T"],token))
        else:
            p=phase("forward_with_construction",lambda:clone(case["space"].types,inp["edges"],inp["edge_families"],inp["query"],inp["f"],inp["t"]/cfg["T"]))
        phase("checks_and_unused_D2H",lambda:legacy_check_prob(p,inp["f"],case))
    else:
        ctx=phase("context_setup",lambda:FrozenTypedContext(clone,case["space"].types,inp["edges"],inp["edge_families"],token))
        query=phase("query_validation",lambda:ctx.prepare_query(inp["query"],inp["f"],token))
        query=phase("query_transfer",lambda:ctx.transfer_query(query,token))
        graph=phase("coalesce_mapping",lambda:ctx.assemble(query,token))
        _,p=phase("network",lambda:ctx.run_network(graph,inp["t"]/cfg["T"],token))
        phase("checks_device_only",lambda:check_prob_device(p,graph.query_families,case,device_prior=clone.prior))
    loss=phase("loss",lambda:torch.nn.functional.binary_cross_entropy(p,label,reduction="sum")*inp["ce_weight"])
    if not torch.isfinite(loss):raise RuntimeError("nonfinite loss")
    phase("backward",lambda:loss.backward())
    grads=[v.grad for v in clone.parameters() if v.grad is not None]
    if not grads or not all(torch.isfinite(g).all() for g in grads):raise RuntimeError("nonfinite/absent participating gradients")
    norm=torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g) for g in grads]))
    if not torch.isfinite(norm):raise RuntimeError("nonfinite aggregate gradient norm")
    before=clone.backbone.lin_out_E.weight.detach().clone()
    phase("optimizer",lambda:opt.step());sync(meter.device)
    if backend=="upstream" or detailed:ctx.close()
    delta=float((clone.backbone.lin_out_E.weight.detach()-before).abs().max())
    if not delta>0:raise RuntimeError("expected disposable trainable edge head did not change")
    result=dict(seconds=time.perf_counter()-start,cpu_seconds=time.process_time()-cpu,
                initialization_seconds=initial,loss=float(loss.detach()),phases=records,
                disposable_optimizer_calls=1,original_weight_updated=False,gradient_norm=float(norm),expected_head_delta=delta,
                loaded_read_only_optimizer_state=optimizer_state is not None,
                decomposition_only=detailed,
                timing_boundary="detailed uses equivalent decomposed graph path; stable legacy timing calls the actual untouched v4 forward" if detailed else "low-interference fixed-input update")
    return result


def device_run(args,meter):
    cfg=config()
    ck,case,state,t,import_record=import_fixed_inputs(args.b_dir,args.source_package,args.chain_dir)
    json_save(meter.out/"import_binding.json",import_record)
    env=environment(args.device)
    from hsgd.bw_preflight.common import runtime_binding
    if runtime_binding(env)!=ck["binding"]["backend"]:
        raise RuntimeError("actual runtime/device differs from imported B; this is not transparent cross-backend recovery")
    free,total=torch.cuda.mem_get_info()
    meter.event("device_memory_before",free_bytes=free,total_bytes=total)
    if free<24*2**30:
        raise RuntimeError("less than 24GiB free for disposable paired gradient checks; stop without claiming or terminating other tasks")
    model,init=meter.measure("model_instantiation",lambda:make_model(case,cfg,args.device))
    expected=model.state_dict()
    if expected.keys()!=ck["model"].keys():raise RuntimeError("checkpoint architecture keys differ")
    for name,value in ck["model"].items():
        if value.shape!=expected[name].shape or value.dtype!=expected[name].dtype or not torch.isfinite(value).all():
            raise RuntimeError("checkpoint shape/dtype/finite mismatch: "+name)
    model.load_state_dict(ck["model"],strict=True);model.eval()
    edges,ef=state_edges(case,state);plan=plan_for(case,[cfg["sample_seed"],t,61],20);ids,qf,qe=plan.block(0)
    inp=dict(edges=edges,edge_families=ef,query=qe,f=qf,t=t,
             labels=np.isin(ids,case["clean_ids"]).astype(np.float32),ce_weight=plan.training_ce_weight())
    meter.event("correctness_start",remaining_time=t,context_edges=len(ef),query=len(ids))
    with torch.no_grad():forward=compare_forward(model,case["space"].types,edges,ef,qe,qf,t/cfg["T"])
    update_check=compare_update(model,case,inp,cfg["T"],cfg["optimizer"],optimizer_state=ck["optimizer"])
    json_save(meter.out/"correctness.json",dict(forward=forward,update=update_check,tolerances=TOLERANCES))
    meter.event("correctness_passed",forward=forward,gradient_error=update_check["gradient_error"])
    correctness_memory=memory(args.device)
    if args.device=="cuda":torch.cuda.reset_peak_memory_stats()
    measured={"training":{},"step":{}};diffs=[];warmups={}
    # Warmup is detailed and covers all 20 blocks; low-interference runs afterwards.
    old=one_step(model,case,state,t,cfg,"legacy",meter,True)
    new=one_step(model,case,state,t,cfg,"upstream",meter,True)
    active=np.flatnonzero(old["covered"]);f=np.searchsorted(case["space"].offsets[1:],active,side="right")
    error=compare_float(old["q"][active],new["q"][active],"reverse")
    diffs=threshold_differences(active,f,old["q"][active],new["q"][active],old["uniform"][active])
    assert len(diffs)==np.count_nonzero(old["threshold"]!=new["threshold"])
    json_save(meter.out/"warmup_checks.json",dict(reverse_error=error,threshold_differences=diffs,
        frozen_integer_coverage_exact=True,RNG_unchanged=True,old_seconds=old["seconds"],new_seconds=new["seconds"],
        note="detailed diagnostics includes extra checks; not stable timing"))
    warmups["step"]={"legacy":old["seconds"],"upstream":new["seconds"]}
    warmups["training"]={k:training_timing(model,case,inp,cfg,k,meter,True,optimizer_state=ck["optimizer"]) for k in ("legacy","upstream")}
    planned_cost=old["seconds"]+new["seconds"]+sum(v["seconds"]+v["initialization_seconds"] for v in warmups["training"].values())
    remaining=meter.cap-(time.monotonic()-meter.start)
    repeats=min(args.repeats,max(0,int((remaining-20)/max(planned_cost*1.5,1))))
    meter.event("bounded_repeats_frozen",requested=args.repeats,planned=repeats,
                reason="warmup cost and time only; not speed or scientific quality")
    if repeats<1:raise Cap("warmup complete but insufficient time for one reliable end-to-end pair")
    for k in ("legacy","upstream"):measured["training"][k]=[];measured["step"][k]=[]
    for repeat in range(repeats):
        # Alternate order to reduce fixed order bias, use the same model/state/plan.
        for k in (("legacy","upstream") if repeat%2==0 else ("upstream","legacy")):
            meter.check();measured["training"][k].append(training_timing(model,case,inp,cfg,k,meter,optimizer_state=ck["optimizer"]))
            row=one_step(model,case,state,t,cfg,k,meter)
            measured["step"][k].append({key:row[key] for key in ("seconds","cpu_seconds","reverse_state_advances")})
        meter.event("stable_pair_completed",pair=repeat+1)
    # No actual model checkpoint/graph output: isolated fixed payload I/O probe.
    payload=meter.out/"checkpoint_io_probe.pt"
    _,io=meter.measure("checkpoint_io_probe",lambda:torch_save(payload,dict(state=state.copy(),
        rng=rng_save(np.random.default_rng(91),meter.device),remaining_time=t,
        completed_reverse_steps=cfg["T"]-t,binding=import_record["original_binding"],
        purpose="isolated IO diagnostic, NOT a resumable chain checkpoint")))
    _,log=meter.measure("log_io_probe",lambda:meter.event("isolated_io_probe_done"))
    assert sha(args.b_dir/"preflight_last.pt")==import_record["weight_sha256"]
    if args.chain_dir:assert sha(args.chain_dir/"chain_last.pt")==import_record["state_sha256"]
    return dict(scope="fixed-state engineering comparison; no chain resume or scientific graph output",import_binding=import_record,
        forward=forward,update_equivalence=update_check,reverse_error=error,threshold_differences=diffs,
        device_memory_before=dict(free_bytes=free,total_bytes=total),
        integer_coverage_exact=True,random_state_unchanged=True,warmup=warmups,measured=measured,
        stable_summary={kind:{k:stats([row["seconds"] for row in values]) for k,values in arm.items()} for kind,arm in measured.items()},
        repeats=repeats,initialization=init,correctness_memory=correctness_memory,checkpoint_io=io,log_io=log,
        io_probe_limit="same state/RNG/counter tensor structure as chain checkpoint, diagnostic binding is not chain-certified; no model checkpoint IO measured; per-block logs identical in detailed mode and absent in both low-interference arms",
        trained_weight_updated=False,reverse_state_advances=0,engineering_optimizer_calls=2+2*(1+repeats))


def worker(args):
    if args.output.exists():raise RuntimeError("use a fresh output directory; no resume")
    args.output.mkdir(parents=True)
    torch.set_num_threads(1);torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    # Check archive inputs without installation or environment mutation.
    from hsgd.bw_preflight.common import verify_manifest
    verify_manifest()
    seed_all(2026101419)
    env=environment(args.device)
    meter=Meter(args.device,args.wall_seconds-5,args.output);binding=code_hashes()
    json_save(args.output/"run.json",dict(command=sys.argv,code=binding,config=config(),
             environment=env,wall_cap_seconds=args.wall_seconds))
    meter.event("environment",environment=env,tolerances=TOLERANCES)
    result=dict(status="running",environment=env,tolerances=TOLERANCES,config=config(),
                formal_training=False,new_scientific_graphs=0,full_chain_runs=0,batching=False)
    try:
        result.update(graph_only(args,meter) if args.mode=="graph-cpu" else device_run(args,meter))
        result["status"]="passed";code=0
    except Cap as exc:result.update(status="incomplete",error=str(exc));code=75
    except BaseException as exc:traceback.print_exc();result.update(status="failed",error=repr(exc));code=1
    result.update(wall_seconds=time.monotonic()-meter.start,cpu_seconds=time.process_time()-meter.cpu,
                  phases=meter.records,**memory(args.device),code_unchanged=code_hashes()==binding)
    if not result["code_unchanged"]:result.update(status="failed",error="source changed");code=1
    json_save(args.output/"summary.json",result)
    terminal={k:result[k] for k in ("status","error","scope","stable_summary","timings",
        "forward","reverse_error","threshold_differences","repeats","wall_seconds","cpu_seconds",
        "peak_allocated_mib","peak_reserved_mib","process_peak_rss_mib","code_unchanged") if k in result}
    terminal.update(summary_path=str(args.output/"summary.json"),formal_training=False,full_chain_runs=0,new_scientific_graphs=0)
    print(json.dumps(terminal,indent=2,allow_nan=False),flush=True)
    return code


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=["graph-cpu","bw"])
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--device",choices=["cpu","cuda"],default="cuda")
    p.add_argument("--data",type=Path,default=PACKAGE/"data/IMDB_movie_reference.npz")
    p.add_argument("--b-dir",type=Path);p.add_argument("--source-package",type=Path);p.add_argument("--chain-dir",type=Path)
    p.add_argument("--repeats",type=int,default=3);p.add_argument("--wall-seconds",type=int,default=540)
    p.add_argument("--worker",action="store_true",help=argparse.SUPPRESS);args=p.parse_args()
    if not 30<=args.wall_seconds<=570 or not 1<=args.repeats<=10:p.error("window 30..570 seconds, repetitions 1..10 required")
    if args.mode=="graph-cpu":args.device="cpu"
    if args.mode=="bw" and (args.device!="cuda" or args.b_dir is None or args.source_package is None):p.error("bw requires accelerator plus explicit B/source-package")
    if args.worker:return worker(args)
    child=subprocess.Popen([sys.executable,"-m","hsgd.perf_bw",*sys.argv[1:],"--worker"])
    try:return child.wait(timeout=args.wall_seconds)
    except (subprocess.TimeoutExpired,KeyboardInterrupt) as exc:
        child.terminate()
        try:child.wait(timeout=5)
        except subprocess.TimeoutExpired:child.kill();child.wait(timeout=5)
        if args.output.exists():json_save(args.output/"watchdog_stop.json",dict(status="incomplete",reason=type(exc).__name__,wall_cap_seconds=args.wall_seconds,own_child_pid=child.pid))
        return 75 if isinstance(exc,subprocess.TimeoutExpired) else 130


if __name__=="__main__":sys.exit(main())
