"""Manual bounded v6 engineering acceptance. Never a formal training launcher.

Worker processes use the same update/train_input/reverse_step as A/B/C entries.
Each parent-created worker has a hard timeout within one <=570s suite window.
Restores occur in a fresh Python process with exact native bindings.
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
from hsgd.bw_preflight.common import (PACKAGE,config,code_hashes,environment,runtime_binding,
    sha,json_save,torch_save,trusted_load,make_case,toy_case,make_model,memory,sync,
    seed_all,rng_save,rng_restore,train_input,update,cpu_tree,restore,checkpoint,
    tree_error,schedule,graph_stats,state_edges,verify_manifest)
from hsgd.kernel import MarginalKernel
from hsgd.runtime import reverse_step
from hsgd.perf_checks import compare_float,compare_rng,threshold_differences,TOLERANCES
from hsgd.perf_import import import_fixed_inputs
from hsgd.learning_mode import configure

SEED = 2026101506
JOBS = ("initialize","small","training_pair","training_continuous","training_resume",
        "chain_pair","chain_resume","interrupt","interrupt_resume","performance")
CHECKS = ("numerical_tolerance","sampling_decisions_exact","short_chain_states_exact",
          "save_restore","performance_complete","next_stage_conditions")


def fixture_case():
    # Full declared architecture; smaller inventory ONLY for local CPU engineering.
    return make_case(np.repeat([0,1],10),np.stack((np.arange(19),np.arange(1,20))),((0,0),(0,1),(1,1)))


def load_native(path,binding):
    ck=trusted_load(path)
    if ck["binding"] != binding: raise RuntimeError("v6 native binding mismatch")
    return ck


def model_opt(initial,case,cfg,device):
    model=make_model(case,cfg,device)
    model.load_state_dict(initial["model"],strict=True)
    opt=torch.optim.AdamW(model.parameters(),**cfg["optimizer"])
    # CPU .to() may return the same tensor storage. Disposable comparisons
    # must not share B100 Adam moments with each other or the read-only input.
    opt.load_state_dict(copy.deepcopy(initial["optimizer"]))
    return model,opt


def step_count(opt):
    return sorted(set(int(v["step"]) for v in opt["state"].values()))


def save_chain(out,name,initial,model,state,rng,t,count,history,device):
    ck=dict(binding=initial["binding"],model=cpu_tree(model.state_dict()),state=state.copy(),
            rng=cpu_tree(rng_save(rng,device)),remaining_time=t,completed_reverse_steps=count,
            test_start_time=initial["start_time"],history=history,purpose="engineering fixed-input short chain, NOT scientific output")
    if count+t != ck["test_start_time"]: raise AssertionError("short-chain count mismatch")
    torch_save(out/name,ck)
    return ck


def write_trace(out,name,rec):
    arrays=rec.pop("trace")
    np.savez_compressed(out/name,**arrays)
    rec["trace_file"]=name
    rec["trace_sha256"]=sha(out/name)
    return arrays


def record_input(out,name,inp):
    torch_save(out/name,{k:v for k,v in inp.items() if not k.endswith("seconds")})
    return sha(out/name)


def worker(args):
    out=args.output; start=time.monotonic();cpu=time.process_time()
    torch.set_num_threads(1);torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    configure(True)  # Isolated engineering worker; formal learning is a different process.
    env=environment(args.device);verify_manifest();cfg=config();seed_all(SEED)
    if args.device=="cuda": torch.cuda.reset_peak_memory_stats()
    def check_time():
        if time.monotonic()-start > args.worker_cap: raise TimeoutError("worker cooperative cap")
    def event(phase,**kw):
        value=dict(job=args.job,phase=phase,elapsed=time.monotonic()-start,**kw)
        with (out/(args.job+".events.jsonl")).open("a") as f:f.write(json.dumps(value)+"\n")
        print(json.dumps(value),flush=True)
    record=dict(job=args.job,status="running",environment=env,formal_training=False,new_scientific_graphs=0)
    event("worker_started",device=args.device,environment=env,code=code_hashes())
    try:
        if args.job=="initialize":
            origin=None
            if args.fixture:
                case=fixture_case();model=make_model(case,cfg,args.device)
                opt=torch.optim.AdamW(model.parameters(),**cfg["optimizer"])
                state=np.isin(np.arange(case["space"].total),case["clean_ids"]).astype(np.uint8)
                t=500; origin_steps=0;data="deterministic 20-node CPU engineering inventory"
            else:
                if not args.b_dir or not args.source_package: raise RuntimeError("explicit v4 B/source paths required")
                old,case,state,t,origin=import_fixed_inputs(args.b_dir,args.source_package,args.chain_dir)
                model=make_model(case,cfg,args.device);model.load_state_dict(old["model"],strict=True)
                opt=torch.optim.AdamW(model.parameters(),**cfg["optimizer"]);opt.load_state_dict(old["optimizer"])
                origin_steps=100;data=cfg["imdb_sha256"]
            if t<4: raise RuntimeError("fixed test state must support four remaining reverse steps")
            seed_all(SEED);rng=np.random.default_rng(SEED)
            binding=dict(code=code_hashes(),config=cfg,backend=runtime_binding(env),graph_backend="upstream",
                         data=data,T=cfg["T"],rounds=cfg["rounds"],origin=origin,
                         scope="v6 engineering local updates/counters only; not old chain continuation")
            initial=checkpoint(model,opt,rng,args.device,0,binding,[])
            initial.update(state=state,start_time=t,origin_optimizer_updates=origin_steps,origin=origin)
            torch_save(out/"initial.pt",initial)
            json_save(out/"run.json",dict(stage="v6-runtime-gate",initial_binding=binding,environment=env,
                command=sys.argv,bounded_launch=args.bounded_launch,
                scope="20-node local fixture" if args.fixture else "complete IMDB engineering only"))
            record.update(binding=binding,initial_sha256=sha(out/"initial.pt"),origin_optimizer_updates=origin_steps,
                          nodes=len(case["space"].types),slots=case["space"].total,start_time=t,
                          initialization_seconds=time.monotonic()-start)
        else:
            initial=trusted_load(out/"initial.pt");binding=initial["binding"]
            if binding["code"]!=code_hashes() or binding["config"]!=cfg or binding["backend"]!=runtime_binding(env):
                raise RuntimeError("source/config/runtime changed between processes")
            if binding["graph_backend"]!="upstream":raise RuntimeError("wrong native backend")
            from hsgd.bw_preflight.common import imdb_case
            case=fixture_case() if args.fixture else imdb_case(args.data,cfg)
            if not args.fixture and sha(args.data)!=binding["data"]:raise RuntimeError("worker data binding mismatch")
            model,opt=model_opt(initial,case,cfg,args.device)
            record["backend_comparison_scope"] = (
                "CN legacy API aliases canonical upstream; NOT independent implementations"
                if model.common_neighbor is not None else "baseline legacy vs upstream implementations")
            rng=np.random.default_rng(SEED);rng_restore(initial["rng"],rng,args.device)
            T,R=cfg["T"],cfg["rounds"]
            if args.job=="small":
                small=toy_case();seed_all(SEED+1);sm=make_model(small,cfg,args.device).eval()
                small_start=time.perf_counter();ks=MarginalKernel(small["prior"],schedule(8))
                r0=np.random.default_rng(SEED+1);pi=np.repeat(small["prior"],small["space"].counts)
                state=(r0.random(len(pi))<pi).astype(np.uint8)
                a,b=state.copy(),state.copy();ra=np.random.default_rng(SEED+2);rb=np.random.default_rng(SEED+2)
                diffs=[];errors=[]
                for t in range(8,0,-1):
                    a,x=reverse_step(sm,small,a,ks,t,3,SEED+3,ra,args.device,"legacy",check_time=check_time,trace=True)
                    b,y=reverse_step(sm,small,b,ks,t,3,SEED+3,rb,args.device,"upstream",check_time=check_time,trace=True)
                    errors.append(compare_float(x["trace"]["q"],y["trace"]["q"],"reverse"))
                    compare_rng(rng_save(ra,args.device),rng_save(rb,args.device))
                    diffs.append(int(np.count_nonzero(a!=b)))
                    if diffs[-1]:raise AssertionError("small trajectory threshold difference")
                    event("small_complete_step",time=t,remaining=t-1)
                ei,ef=state_edges(small,b)
                np.savez_compressed(out/"small_final_graph.npz",state=b,types=small["space"].types,edges=ei.T,families=ef,T=np.array(8))
                with np.load(out/"small_final_graph.npz") as z:assert np.array_equal(z["state"],b)
                record.update(completed_reverse_steps=8,remaining_time=0,max_reverse_error=max(errors),
                    state_difference_counts=diffs,graph=graph_stats(small,b),small_seconds=time.perf_counter()-small_start)
            elif args.job=="training_pair":
                inp=train_input(case,T,R,SEED,1,rng)
                input_sha=record_input(out,"training_pair_input.pt",inp)
                left,lo=model_opt(initial,case,cfg,args.device)
                right,ro=model_opt(initial,case,cfg,args.device)
                old,new={},{}
                saved=rng_save(rng,args.device)
                a=update(left,lo,case,inp,T,args.device,backend="legacy",diagnostics=old)
                rng_restore(saved,rng,args.device)
                b=update(right,ro,case,inp,T,args.device,backend="upstream",diagnostics=new)
                pair_evidence=dict(binding=binding,errors={},verification="pending",input_sha256=input_sha,
                    legacy=cpu_tree(left.state_dict()),upstream=cpu_tree(right.state_dict()),
                    legacy_optimizer=cpu_tree(lo.state_dict()),upstream_optimizer=cpu_tree(ro.state_dict()),
                    diagnostics={"legacy":old,"upstream":new},local_updates=1)
                torch_save(out/"training_pair.pt",pair_evidence)
                errors={k:compare_float(old[k],new[k],k) for k in ("logits","p0","loss")}
                if old["gradients"].keys()!=new["gradients"].keys():raise AssertionError("gradient keys")
                grad=[]
                for name,v in old["gradients"].items():
                    w=new["gradients"][name]
                    if v is None:
                        if w is not None:raise AssertionError("gradient participation")
                    else:grad.append(compare_float(v,w,"gradient"))
                errors["gradient"]=max(grad)
                errors["updated_parameter"]=max(compare_float(x,y,"updated_parameter") for x,y in zip(left.parameters(),right.parameters()))
                pair_evidence.update(errors=errors,verification="completed")
                torch_save(out/"training_pair.pt",pair_evidence)
                record.update(errors=errors,legacy_record=a,upstream_record=b,input_sha256=input_sha,
                              participating_parameters=len(grad),local_updates=1)
            elif args.job=="training_continuous":
                history=[]
                for i in (1,2):
                    check_time();inp=train_input(case,T,R,SEED,i,rng)
                    ih=record_input(out,"training_input_%d.pt"%i,inp)
                    rec=update(model,opt,case,inp,T,args.device,backend="upstream")
                    rec.update(position=i,input_sha256=ih,timing_phase="warmup" if i==1 else "stable")
                    history.append(rec)
                    io_start=time.perf_counter()
                    ck=checkpoint(model,opt,rng,args.device,i,binding,history)
                    ck["origin_optimizer_updates"]=initial["origin_optimizer_updates"]
                    torch_save(out/("training_split.pt" if i==1 else "training_continuous.pt"),ck)
                    record.setdefault("checkpoint_seconds",[]).append(time.perf_counter()-io_start)
                    event("committed_training_update",position=i,backend=rec["actual_dispatch"],loss=rec["loss"])
                record.update(completed_updates=2,optimizer_steps=step_count(opt.state_dict()),history=history)
            elif args.job=="training_resume":
                ck=load_native(out/"training_split.pt",binding)
                if restore(ck,model,opt,rng,args.device,binding)!=1:raise AssertionError("training split count")
                inp=train_input(case,T,R,SEED,2,rng)
                replay={k:v for k,v in inp.items() if not k.endswith("seconds")}
                saved=trusted_load(out/"training_input_2.pt")
                for k,v in replay.items():
                    if not np.array_equal(v,saved[k]):raise AssertionError("resumed input: "+k)
                rec=update(model,opt,case,inp,T,args.device,backend="upstream")
                rec.update(position=2,input_sha256=sha(out/"training_input_2.pt"),timing_phase="stable")
                history=ck["history"]+[rec]
                resumed=checkpoint(model,opt,rng,args.device,2,binding,history)
                resumed["origin_optimizer_updates"]=initial["origin_optimizer_updates"]
                torch_save(out/"training_recovered.pt",resumed)
                expected=load_native(out/"training_continuous.pt",binding)
                pe=max(compare_float(v,resumed["model"][k],"updated_parameter") for k,v in expected["model"].items())
                oe=tree_error(expected["optimizer"],resumed["optimizer"])
                if oe>cfg["recovery_atol"]:raise AssertionError("optimizer recovery tolerance")
                compare_rng(expected["rng"],resumed["rng"])
                for a,b in zip(expected["history"],history):
                    for k in ("position","time","query_slots","query_round","noisy_edges","input_sha256","backend","actual_dispatch"):
                        if a[k]!=b[k]:raise AssertionError("semantic log mismatch: "+k)
                    compare_float(a["loss"],b["loss"],"loss")
                record.update(completed_updates=2,parameter_error=pe,optimizer_error=oe,rng_exact=True,
                    input_exact=True,semantic_log_exact=True,new_process_restore=True,optimizer_steps=step_count(opt.state_dict()))
            elif args.job=="chain_pair":
                model.eval();k=MarginalKernel(case["prior"],schedule(T));a=initial["state"].copy();b=a.copy()
                ra=np.random.default_rng(SEED+4);rb=np.random.default_rng(SEED+4);history=[];errors=[];diffs=[]
                for i in range(1,5):
                    t=initial["start_time"]-i+1
                    a,x=reverse_step(model,case,a,k,t,R,config()["sample_seed"],ra,args.device,"legacy",event,check_time,True)
                    b,y=reverse_step(model,case,b,k,t,R,config()["sample_seed"],rb,args.device,"upstream",event,check_time,True)
                    xx=write_trace(out,"legacy_step_%d.npz"%i,x);yy=write_trace(out,"upstream_step_%d.npz"%i,y)
                    errors.append(dict(p0=compare_float(xx["p0"],yy["p0"],"p0"),reverse=compare_float(xx["q"],yy["q"],"reverse")))
                    assert np.array_equal(xx["uniform"],yy["uniform"]) and np.array_equal(xx["coverage"],yy["coverage"])
                    families=np.repeat(np.arange(case["n_families"]),case["space"].counts)
                    ds=threshold_differences(np.arange(len(a)),families,xx["q"],yy["q"],xx["uniform"])
                    diffs.extend([dict(time=t,**d) for d in ds])
                    compare_rng(rng_save(ra,args.device),rng_save(rb,args.device))
                    if not np.array_equal(a,b):
                        json_save(out/"threshold_differences.json",diffs)
                        record.update(first_divergence_time=t,threshold_differences=diffs,numerical_tolerance="passed",
                                      sampling_decisions_exact="failed",short_chain_states_exact="incomplete")
                        raise AssertionError("threshold divergence; stop paired trajectory comparison")
                    history.append(dict(position=i,legacy=x,upstream=y))
                    save_chain(out,"chain_continuous.pt",initial,model,b,rb,t-1,i,history,args.device)
                    if i==2:save_chain(out,"chain_split.pt",initial,model,b,rb,t-1,i,history,args.device)
                    event("paired_time_committed",position=i,remaining_time=t-1)
                json_save(out/"threshold_differences.json",diffs)
                record.update(completed_reverse_steps=4,remaining_time=initial["start_time"]-4,errors=errors,
                              state_exact=True,rng_exact=True,threshold_differences=diffs,history=history)
            elif args.job in ("chain_resume","interrupt_resume"):
                name="chain_split.pt" if args.job=="chain_resume" else "chain_interrupted.pt"
                ck=load_native(out/name,binding)
                if ck["completed_reverse_steps"]!=2 or ck["remaining_time"]!=initial["start_time"]-2:raise AssertionError("chain split count")
                model.load_state_dict(ck["model"],strict=True);model.eval();rng_restore(ck["rng"],rng,args.device)
                state,t,history=ck["state"],ck["remaining_time"],ck["history"]
                k=MarginalKernel(case["prior"],schedule(T));max_error=0.
                for i in (3,4):
                    state,rec=reverse_step(model,case,state,k,t,R,config()["sample_seed"],rng,args.device,"upstream",event,check_time,True)
                    arrays=write_trace(out,args.job+"_step_%d.npz"%i,rec)
                    with np.load(out/("upstream_step_%d.npz"%i)) as expected:
                        for key in ("state","uniform","coverage"):
                            if not np.array_equal(arrays[key],expected[key]):raise AssertionError("restore exact array: "+key)
                        max_error=max(max_error,compare_float(arrays["q"],expected["q"],"reverse"))
                    history=history+[dict(position=i,upstream=rec)];t-=1
                    final=save_chain(out,args.job+".pt",initial,model,state,rng,t,i,history,args.device)
                expected=load_native(out/"chain_continuous.pt",binding);compare_rng(expected["rng"],final["rng"])
                record.update(completed_reverse_steps=4,remaining_time=t,probability_error=max_error,state_exact=True,
                              rng_exact=True,new_process_restore=True,source_checkpoint=name)
            elif args.job=="interrupt":
                ck=load_native(out/"chain_split.pt",binding);model.load_state_dict(ck["model"],strict=True);model.eval()
                rng_restore(ck["rng"],rng,args.device);state=ck["state"].copy();t=ck["remaining_time"]
                k=MarginalKernel(case["prior"],schedule(T))
                try:
                    reverse_step(model,case,state,k,t,R,config()["sample_seed"],rng,args.device,"upstream",event,check_time,True,3)
                    raise AssertionError("interruption did not occur")
                except InterruptedError:
                    assert np.array_equal(state,ck["state"])
                    compare_rng(ck["rng"],rng_save(rng,args.device))
                    save_chain(out,"chain_interrupted.pt",initial,model,state,rng,t,2,ck["history"],args.device)
                record.update(injected_after_block=3,completed_reverse_steps=2,remaining_time=t,state_rollback_exact=True,rng_rollback_exact=True)
            elif args.job=="performance":
                model.eval();k=MarginalKernel(case["prior"],schedule(T));timings=[]
                warmups=[]
                # Warm up IN THIS fresh process. Trace disabled: identical logs/checks;
                # no extra trace posterior/export. One full 20-block step per side.
                for backend in ("legacy","upstream"):
                    r=np.random.default_rng(SEED+4)
                    _,rec=reverse_step(model,case,initial["state"],k,initial["start_time"],R,config()["sample_seed"],r,args.device,
                                       backend,event,check_time,False)
                    warmups.append(rec)
                for backend in ("legacy","upstream"):
                    r=np.random.default_rng(SEED+4)
                    _,rec=reverse_step(model,case,initial["state"],k,initial["start_time"],R,config()["sample_seed"],r,args.device,
                                       backend,event,check_time,False)
                    timings.append(rec)
                record.update(full_step_timings=timings,warmup_step_timings=warmups,rounds=R,stable_repeats=1,
                              limitation="one low-interference complete sweep per side; insufficient tail precision; no full-chain throughput claim")
            else:raise RuntimeError("unknown job")
        record["status"]="completed"
        event("worker_completed",backend="upstream; legacy explicit comparison only")
        record.update(wall_seconds=time.monotonic()-start,cpu_seconds=time.process_time()-cpu,**memory(args.device))
        json_save(out/(args.job+".json"),record)
        return 0
    except BaseException as exc:
        record.update(status="incomplete" if isinstance(exc,(TimeoutError,KeyboardInterrupt)) else "failed",
                      failure=type(exc).__name__+": "+str(exc),traceback=traceback.format_exc(),
                      wall_seconds=time.monotonic()-start,cpu_seconds=time.process_time()-cpu,**memory(args.device))
        json_save(out/(args.job+".json"),record);print(record["traceback"],flush=True)
        return 75 if record["status"]=="incomplete" else 1


def summarize(out):
    records={}
    for job in JOBS:
        p=out/(job+".json")
        records[job]=json.loads(p.read_text()) if p.exists() else dict(status="incomplete",reason="not executed / evidence missing")
    complete=lambda *names:all(records[n]["status"]=="completed" for n in names)
    result=dict(stage="v6-runtime-gate",purpose="engineering only",formal_training=False,new_scientific_graphs=0,
                statuses={k:"incomplete" for k in CHECKS},jobs=records,tolerances=TOLERANCES)
    result["ready_for_formal_training"]=False
    result["validation_status"]="not executed; unchanged validation protocol, outside this finite equivalence gate"
    result["BW_IMDB_tested"] = records["initialize"].get("environment",{}).get("device_name")=="BW"
    groups={"numerical_tolerance":("small","training_pair","chain_pair"),
            "sampling_decisions_exact":("small","chain_pair"),"short_chain_states_exact":("small","chain_pair"),
            "save_restore":("training_resume","chain_resume","interrupt","interrupt_resume"),
            "performance_complete":("training_continuous","performance")}
    for key,names in groups.items():
        if key=="numerical_tolerance" and all(records[n]["status"]=="completed" or records[n].get("numerical_tolerance")=="passed" for n in names):
            result["statuses"][key]="passed"
        elif key=="short_chain_states_exact" and any(records[n].get("short_chain_states_exact")=="incomplete" for n in names):
            result["statuses"][key]="incomplete"
        elif complete(*names):result["statuses"][key]="passed"
        elif any(records[n]["status"]=="failed" for n in names):result["statuses"][key]="failed"
    if any(v=="failed" for v in result["statuses"].values()): result["statuses"]["next_stage_conditions"]="failed"
    elif all(result["statuses"][k]=="passed" for k in CHECKS[:-1]):result["statuses"]["next_stage_conditions"]="passed"
    result["next_stage_meaning"]="only engineering gate; formal workload/configuration and full IMDB chain remain separately unauthorized"
    result["files"]={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob("*"))
                     if p.is_file() and p.name not in ("summary.json","launcher.json","verification.json")}
    json_save(out/"summary.json",result)
    print(json.dumps({"statuses":result["statuses"],"output":str(out)},indent=2),flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--device",choices=("cpu","cuda"),default="cuda")
    p.add_argument("--fixture",action="store_true",help="smaller inventory local engineering only; full architecture")
    p.add_argument("--arm",choices=("baseline","cn"),required=True)
    p.add_argument("--source-package",type=Path);p.add_argument("--b-dir",type=Path);p.add_argument("--chain-dir",type=Path)
    p.add_argument("--data",type=Path,default=PACKAGE/"data/IMDB_movie_reference.npz")
    p.add_argument("--wall-seconds",type=int,default=570)
    p.add_argument("--job",choices=JOBS,help=argparse.SUPPRESS)
    p.add_argument("--bounded-launch",action="store_true",help=argparse.SUPPRESS)
    p.add_argument("--worker-cap",type=float,default=540,help=argparse.SUPPRESS)
    args=p.parse_args();args.output=args.output.resolve()
    from hsgd.bw_preflight.common import set_arm
    set_arm(args.arm)
    if not 1<=args.wall_seconds<=570:p.error("suite cap must be 1..570s; startup and cleanup share this budget")
    if args.device=="cpu" and not args.fixture:p.error("full IMDB manual gate requires accelerator")
    if args.job:return worker(args)
    if args.output.exists():raise RuntimeError("new independent output directory required; no duplicate gate")
    args.output.mkdir(parents=True);start=time.monotonic();launch=[]
    for job in JOBS:
        left=args.wall_seconds-(time.monotonic()-start)-10
        if left<=0:break
        command=[sys.executable,"-m","hsgd.runtime_gate",*sys.argv[1:],"--job",job,"--worker-cap",str(max(1,left-5))]
        log=(args.output/(job+".stdout.log")).open("w")
        print("Starting",job,"remaining suite seconds",round(left,1),flush=True)
        child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
        try:code=child.wait(timeout=left)
        except (subprocess.TimeoutExpired,KeyboardInterrupt):
            child.terminate()
            try:child.wait(timeout=3)
            except subprocess.TimeoutExpired:child.kill();child.wait(timeout=3)
            code=75
        finally:log.close()
        launch.append(dict(job=job,exit_code=code,pid=child.pid))
        print(job,"exit",code,flush=True)
        if code:break
    result=summarize(args.output)
    json_save(args.output/"launcher.json",dict(jobs=launch,suite_wall_seconds=time.monotonic()-start,
        wall_cap_seconds=args.wall_seconds,max_child_cleanup_seconds=6,
        overhead="each fresh Python import/startup and worker initialization counted in suite cap; verification startup additional CPU-only"))
    from hsgd.runtime_verify import audit
    verified=audit(args.output,require_outer=False)
    if verified["exit_code"]==1:
        result["statuses"]["next_stage_conditions"]="failed"
        result["evidence_validation_error"]=verified
        json_save(args.output/"summary.json",result)
    json_save(args.output/"verification.json",verified)
    print(json.dumps(verified,indent=2),flush=True)
    return verified["exit_code"]


if __name__=="__main__":sys.exit(main())
