"""IMDB B: strict preflight, separate ordinary-mode learning, explicit stop.

Manual BW entry is learning_launch. This module is its owned worker only.
No sampling chain, checkpoint quality selection or automatic parameter changes.
"""
import argparse
import copy
import fcntl
import json
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
import numpy as np
import torch
from hsgd.bw_preflight.common import (PACKAGE,config,code_hashes,environment,runtime_binding,dimensions,
    make_model,imdb_case,seed_all,rng_save,rng_restore,train_input,update,cpu_tree,torch_save,trusted_load,
    json_save,sha,memory,sync,verify_manifest,query_inventory,plan_for)
from hsgd.runtime_gate import fixture_case,SEED
from hsgd.perf_checks import compare_float,compare_rng,TOLERANCES
from hsgd.learning_budget import signature,model_fingerprint
from hsgd.learning_control import elapsed,validate as validate_control,stop_reason
from hsgd.learning_mode import configure,require as require_mode,state as mode_state,same_platform
from hsgd.learning_restore import verify_loaded
from hsgd.learning_validation import freeze_inputs,validate


class StopRequested(Exception):pass


PROTOCOL=dict(version="bw_cn_v2",graph_backend="upstream",training_seed=2026101407,
    validation_seed=2026101408,validation_times=[1,250,500,750,1000],validation_blocks=20,
    validation_interval_seconds=1800,checkpoint_interval_seconds=1800,
    eight_hour_limit=False,default_stop="maximum10000 updates, explicit override before start",default_evaluation_weight="final",
    cached_prediction_stages=["initial","first_periodic","final"],
    preflight_strict=True,training_strict=False,validation_strict=False,post_resume_trajectory_identity_required=False,
    fresh_initialization_after_gate=True,sampling_gate_required=False,automatic_generation=False,
    loss="(R/S)*sum FP32 BCE on one uniformly selected active query block; fixed-none CE0 included in S",
    time_sampling="uniform integer1..T",batch=1,microbatch=1,gradient_accumulation=1,
    learning_rate_schedule="constant2e-4; first5 successful updates per worker invocation timing warmup only, no LR warmup",gradient_clipping=False,
    parameter_zero_change="finite completed optimizer step counts even if FP32 edge-head delta rounds to0; not skipped",
    label_statistics="observations plus exact unique-slot coverage bitset, not independent samples",
    features="baseline v9 or single common-neighbor readout; Xt-only shared first-layer node context, no IDs; dropout.1 inactive",
    convergence_claim=False)


def setup(args):
    torch.set_num_threads(1);torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    configure(args.phase=="gate_initialize")
    verify_manifest();cfg=config();env=environment(args.device)
    require_mode(runtime_binding(env),args.phase=="gate_initialize")
    print(json.dumps(dict(event="process_started",phase=args.phase,execution_mode=mode_state(),environment=env)),flush=True)
    case=fixture_case() if args.fixture else imdb_case(args.data,cfg)
    if not args.fixture:
        if env["device_name"]!="BW" or not env["hip"]:raise RuntimeError("full training entry requires actual BW HIP device")
        for name,v in (("torch-geometric","2.7.0"),("numpy","1.26.4"),("scipy","1.14.1")):
            if env["packages"][name]!=v:raise RuntimeError("dependency differs: "+name+"; no automatic installation")
    if args.device=="cuda":torch.cuda.reset_peak_memory_stats()
    return cfg,env,case


def source_commit():
    m=PACKAGE/"manifest_cn_v2.json"
    if m.exists():return json.loads(m.read_text())["source_commit"]
    return subprocess.check_output(["git","-C",str(PACKAGE),"rev-parse","HEAD"],text=True).strip()+" (local development; actual files hashed)"


def gate_initialize(args):
    cfg,env,case=setup(args);out=args.output;out.mkdir(parents=True,exist_ok=False)
    seed_all(SEED);rng=np.random.default_rng(SEED)
    model=make_model(case,cfg,args.device);opt=torch.optim.AdamW(model.parameters(),**cfg["optimizer"])
    binding=dict(code=code_hashes(),config=cfg,backend=runtime_binding(env),graph_backend="upstream",
        data="deterministic 20-node CPU engineering fixture" if args.fixture else cfg["imdb_sha256"],
        T=cfg["T"],rounds=cfg["rounds"],origin=None,scope="fresh engineering training-only gate; no scientific weights")
    initial=dict(model=cpu_tree(model.state_dict()),optimizer=cpu_tree(opt.state_dict()),rng=cpu_tree(rng_save(rng,args.device)),
        completed_updates=0,binding=binding,history=[],origin_optimizer_updates=0,origin=None,start_time=500,
        state=np.isin(np.arange(case["space"].total),case["clean_ids"]).astype(np.uint8))
    torch_save(out/"initial.pt",initial)
    json_save(out/"run.json",dict(initial_binding=binding,environment=env,stage="cn-v2-strict-training-only-gate"))
    print("Fresh training-only gate initialized; no B100/C-chain import",flush=True)


def verify_training_gate(out,write=False):
    out=Path(out);init=trusted_load(out/"initial.pt");binding=init["binding"]
    require_mode(binding["backend"],True)
    if binding["code"]!=code_hashes() or binding["config"]!=config():raise RuntimeError("gate source/config changed")
    reports={}
    for name in ("training_pair","training_continuous","training_resume"):
        r=json.loads((out/(name+".json")).read_text())
        if r["status"]!="completed" or runtime_binding(r["environment"])!=binding["backend"]:raise RuntimeError("training gate failed: "+name)
        reports[name]=r
    pair=trusted_load(out/"training_pair.pt")
    if pair.get("verification")!="completed":raise RuntimeError("pair comparison incomplete")
    if pair["binding"]!=binding or pair["local_updates"]!=1 or pair["input_sha256"]!=sha(out/"training_pair_input.pt"):raise RuntimeError("pair binding/count/input mismatch")
    a,b=pair["diagnostics"]["legacy"],pair["diagnostics"]["upstream"]
    errors={k:compare_float(a[k],b[k],k) for k in ("logits","p0","loss")}
    grads=[]
    for k,v in a["gradients"].items():
        if v is None:
            if b["gradients"][k] is not None:raise RuntimeError("gradient participation changed")
        else:grads.append(compare_float(v,b["gradients"][k],"gradient"))
    if not grads:raise RuntimeError("empty participating gradients")
    errors["gradient"]=max(grads)
    errors["updated_parameter"]=max(compare_float(v,pair["upstream"][k],"updated_parameter") for k,v in pair["legacy"].items())
    checkpoints=[]
    for name,n in (("training_split.pt",1),("training_continuous.pt",2),("training_recovered.pt",2)):
        ck=trusted_load(out/name)
        if ck["binding"]!=binding or ck["completed_updates"]!=n or len(ck["history"])!=n:raise RuntimeError("gate checkpoint count/binding")
        from hsgd.cn_optimizer import verify_steps
        verify_steps(ck)
        for i,r in enumerate(ck["history"],1):
            if not (r["backend"]==r["actual_dispatch"]=="upstream"):raise RuntimeError("gate dispatch")
            if r["input_sha256"]!=sha(out/("training_input_%d.pt"%i)):raise RuntimeError("gate actual input")
        checkpoints.append(ck)
    continuous,recovered=checkpoints[1:]
    for name,v in continuous["model"].items():compare_float(v,recovered["model"][name],"updated_parameter")
    from hsgd.bw_preflight.common import tree_error
    if tree_error(continuous["optimizer"],recovered["optimizer"])>config()["recovery_atol"]:raise RuntimeError("recovered Adam differs")
    compare_rng(continuous["rng"],recovered["rng"])
    if reports["training_resume"].get("new_process_restore") is not True or not reports["training_resume"].get("semantic_log_exact"):raise RuntimeError("no cross-process restore evidence")
    evidence=dict(status="passed",errors=errors,tolerances=TOLERANCES,normal_training_input=True,
        cross_process_restore=True,sampling_short_chain="not executed; not a training prerequisite",
        formal_initialization="will be fresh; gate weights not reused",binding=binding,
        files={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!="training_gate.json"})
    if write:json_save(out/"training_gate.json",evidence)
    elif json.loads((out/"training_gate.json").read_text())!=evidence:raise RuntimeError("preflight certificate changed")
    return evidence


def checkpoint_steps(opt):return sorted(set(int(v["step"]) for v in opt.state.values()))


def train(args):
    out=args.output;control=json.loads((out/"run_control.json").read_text());validate_control(control)
    cfg,env,case=setup(args)
    require_mode(runtime_binding(env),False)
    lock=(out/".training.lock").open("a+");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    gate=verify_training_gate(out/"preflight",write=not args.resume)
    same_platform(gate["binding"]["backend"],runtime_binding(env))
    seed_all(cfg["train_seed"]);rng=np.random.default_rng(cfg["train_seed"])
    model=make_model(case,cfg,args.device);opt=torch.optim.AdamW(model.parameters(),**cfg["optimizer"])
    protocol=copy.deepcopy(PROTOCOL)
    if args.fixture:protocol.update(validation_interval_seconds=999999,checkpoint_interval_seconds=999999,engineering_fixture=True)
    actual=dict(model=dimensions(model),optimizer={k:v for k,v in opt.param_groups[0].items() if k!="params"})
    if actual["optimizer"]["amsgrad"] is not False:raise RuntimeError("unexpected AMSGrad change")
    data="deterministic CPU fixture" if args.fixture else sha(args.data)
    binding=dict(code=code_hashes(),config=cfg,execution_protocol=protocol,data=data,
        backend=runtime_binding(env),graph_backend="upstream",run_control=control,
        device_inventory=dict(visible_devices=env.get("visible_devices"),device_name=env["device_name"]),
        data_provenance_sha256=None if args.fixture else sha(PACKAGE/"data/IMDB_preprocessing_provenance.json"))
    stop=[None]
    def request(signum,frame):stop[0]="manual_interrupt" if signum==signal.SIGINT else "termination_requested"
    signal.signal(signal.SIGINT,request);signal.signal(signal.SIGTERM,request)
    events=(out/"events.jsonl").open("a",buffering=1)
    logs=(out/"updates.jsonl").open("a",buffering=1)
    def event(phase,**kw):
        r=dict(phase=phase,utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),run_elapsed_seconds=elapsed(control))
        r.update(kw)
        events.write(json.dumps(r,allow_nan=False)+"\n");print(json.dumps(r),flush=True)
    def valid_source():
        if code_hashes()!=binding["code"] or config()!=binding["config"]:raise RuntimeError("code/config changed during training")
        if not args.fixture and sha(args.data)!=binding["data"]:raise RuntimeError("reference source changed")
        if signature(json.loads((out/"run_control.json").read_text()))!=signature(control):raise RuntimeError("run control mutated")
        require_mode(mode_state(),False)
        verify_manifest()
    def check_validation_time():
        require_mode(mode_state(),False)  # Stop requests are served at safe update boundaries.
    history=[];validations=[];save_times=[];logging_seconds=0.;session_updates=0;completed=0;coverage=np.zeros(case["space"].total,dtype=np.uint8)
    observations=np.zeros((case["n_families"],2),dtype=np.int64);save_cost=[1.];validation_cost=[0.]
    periodic_saved=False;last_ck=None;milestone_due=elapsed(control)+protocol["checkpoint_interval_seconds"]
    started_cpu=time.process_time();reason=None;failure=None;final_path=None
    def save(name):
        nonlocal last_ck
        valid_source();sync(args.device);start=time.perf_counter()
        ck=dict(model=cpu_tree(model.state_dict()),optimizer=cpu_tree(opt.state_dict()),rng=cpu_tree(rng_save(rng,args.device)),
            optimizer_parameter_names=[n for n,p in model.named_parameters()],
            completed_updates=completed,binding=binding,history=history,validations=validations,
            cumulative_run_elapsed_seconds=elapsed(control),next_milestone_elapsed_seconds=milestone_due,
            coverage_bits=np.packbits(coverage),observations=observations.copy(),periodic_prediction_saved=periodic_saved,
            purpose="engineering fixture only" if args.fixture else "IMDB B explicit-stop training; final is default evaluation weight")
        ck["model_sha256"]=model_fingerprint(ck["model"])
        path=out/"checkpoints"/name;torch_save(path,ck);last_ck=path
        json_save(out/"latest.json",dict(path=str(path.relative_to(out)),sha256=sha(path),completed_updates=completed,
                                        run_elapsed_seconds=elapsed(control),checkpoint_seconds=time.perf_counter()-start))
        event("checkpoint",file=name,completed_updates=completed,seconds=time.perf_counter()-start)
        save_duration=time.perf_counter()-start;save_cost[0]=max(save_cost[0],save_duration);save_times.append(save_duration)
        return path
    def validation(stage,snapshot=False):
        valid_source();event("validation_start",stage=stage,completed_updates=completed)
        r=validate(model,case,cfg,out/"validation_inputs",out/"validation"/(stage+"_u%07d_a%d"%(completed,time.time_ns())),
                   rng,args.device,stage,completed,check_validation_time,event,snapshot)
        validations.append(r);validation_cost[0]=max(validation_cost[0],r["wall_seconds"])
        event("validation_complete",stage=stage,completed_updates=completed,pooled_ce=r["pooled_ce"],
              prior_ce=r["baseline"]["pooled_ce"],positive_ce=r["positive_ce"],negative_ce=r["negative_ce"],seconds=r["wall_seconds"])
    try:
        (out/"checkpoints").mkdir(exist_ok=True)
        if args.resume:
            freeze=json.loads((out/"freeze.json").read_text())
            if freeze["binding"]!=binding:raise RuntimeError("native source/data/config/backend/Python/Torch/run-control binding changed")
            latest=json.loads((out/"latest.json").read_text());path=out/latest["path"]
            if sha(path)!=latest["sha256"]:raise RuntimeError("latest checkpoint digest mismatch")
            ck=trusted_load(path)
            if ck["binding"]!=binding:raise RuntimeError("checkpoint source/config/control binding invalid")
            if ck["model_sha256"]!=model_fingerprint(ck["model"]):raise RuntimeError("checkpoint model fingerprint mismatch")
            model.load_state_dict(ck["model"],strict=True);opt.load_state_dict(copy.deepcopy(ck["optimizer"]));rng_restore(ck["rng"],rng,args.device)
            loaded=verify_loaded(ck,model,opt,rng,args.device)
            loaded.update(checkpoint_sha256=sha(path),execution_mode=mode_state(),data_binding=binding["data"],config_sha256=signature(cfg))
            json_save(out/("resume_loaded_%d.json"%time.time_ns()),loaded)
            if signature({k:v for k,v in opt.param_groups[0].items() if k!="params"})!=signature(actual["optimizer"]):
                raise RuntimeError("restored optimizer defaults/settings differ")
            completed=ck["completed_updates"];history=ck["history"];validations=ck["validations"]
            from hsgd.cn_optimizer import verify_steps
            verify_steps(ck,model)
            coverage=np.unpackbits(ck["coverage_bits"])[:len(coverage)];observations=ck["observations"]
            periodic_saved=ck["periodic_prediction_saved"];milestone_due=ck["next_milestone_elapsed_seconds"];last_ck=path
            validation_cost[0]=max([r["wall_seconds"] for r in validations] or [0.])
            # Archive orphan updates beyond the last saved state; never silently reuse them.
            if (out/"summary.json").exists():(out/"summary.json").rename(out/("summary_attempt_%d.json"%time.time_ns()))
            logs.close();old_lines=(out/"updates.jsonl").read_text().splitlines()
            committed=[json.loads(x) for x in old_lines if x.strip()]
            if len(committed)<completed:raise RuntimeError("committed update ledger missing rows")
            for expected,actual in zip(history,committed[:completed]):
                if expected!=actual:raise RuntimeError("committed update ledger/checkpoint mismatch")
            if len(committed)>completed:
                (out/"updates.jsonl").rename(out/("updates_attempt_%d.jsonl"%time.time_ns()))
                with (out/"updates.jsonl").open("w") as f:
                    for row in history:f.write(json.dumps(row)+"\n")
            logs=(out/"updates.jsonl").open("a",buffering=1)
            if not any(r["stage"]=="initial" for r in validations):
                validation("initial",True);save("initial_validated_resumed_%d.pt"%time.time_ns())
            event("restored",completed_updates=completed,loaded_state=loaded,stop_policy=control["stop_mode"])
        else:
            inputs=freeze_inputs(case,cfg,out/"validation_inputs")
            freeze=dict(binding=binding,source_commit=source_commit(),actual_dimensions=actual,
                expanded_training_config=dict(method=cfg,execution=protocol,actual_optimizer=actual["optimizer"],
                    actual_linear_dimensions=actual["model"]["linear_layers"],update_limit=control["stop_after_updates"],
                    legacy_preflight_updates_field_ignored=cfg["updates"]),
                environment=env,command=sys.argv,seed=cfg["train_seed"],reference=dict(nodes=len(case["space"].types),
                edges=len(case["clean_ids"]),types=np.bincount(case["space"].types).tolist(),relations=case["counts"].tolist(),
                family_pairs=[list(x) for x in case["space"].pairs],slots=case["space"].counts.tolist(),prior=case["prior"].tolist()),
                validation_input_manifest_sha256=sha(out/"validation_inputs/manifest.json"),disk_estimate=inputs["disk_estimate"],
                checkpoint_bytes_estimate=actual["model"]["parameters"]*12,
                retained_checkpoint_count="depends on explicit stop; all milestones retained",
                declared_scope="same-reference denoising validation; no automatic generation")
            freeze["query_inventory"]=query_inventory(case,plan_for(case,[cfg["train_seed"],1,21],cfg["rounds"]))
            json_save(out/"freeze.json",freeze);event("fresh_formal_initialization",actual_dimensions=actual,seed=cfg["train_seed"])
            save("initial.pt");validation("initial",True)
            save("initial_validated.pt")
        while True:
            require_mode(mode_state(),False)
            reason=stop[0] or stop_reason(out,control,completed)
            if reason:break
            position=completed+1;before=rng_save(rng,args.device);tick=time.perf_counter()
            inp=train_input(case,cfg["T"],cfg["rounds"],cfg["train_seed"],position,rng)
            rec=update(model,opt,case,inp,cfg["T"],args.device,backend="upstream",require_parameter_change=False)
            if not all(torch.isfinite(p).all() for p in model.parameters()):raise RuntimeError("nonfinite updated parameter")
            completed=position
            session_updates+=1
            counts=np.zeros((case["n_families"],2),dtype=np.int64)
            for label in (0,1):counts[:,label]=np.bincount(inp["f"][inp["labels"]==label],minlength=case["n_families"])
            observations+=counts;coverage[inp["ids"]]=1
            rec.pop("no_gradient_parameters",None)
            rec.update(update=completed,utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
                execution_mode=mode_state(),
                run_elapsed_seconds=elapsed(control),lr=opt.param_groups[0]["lr"],skipped=False,exception=None,
                timing_phase="warmup" if session_updates<=cfg["warmup_updates"] else "stable",
                batch_seconds=time.perf_counter()-tick,family_label_counts=counts.tolist())
            log_start=time.perf_counter()
            history.append(rec);logs.write(json.dumps(rec,allow_nan=False)+"\n")
            json_save(out/"progress.json",dict(phase="training",completed_updates=completed,last_loss=rec["loss"],
                run_elapsed_seconds=elapsed(control),stop_mode=control["stop_mode"],backend=rec["actual_dispatch"]))
            logging_seconds+=time.perf_counter()-log_start
            if completed%10==0:event("training_progress",completed_updates=completed,loss=rec["loss"],batch_seconds=rec["batch_seconds"],**memory(args.device))
            due=elapsed(control)>=milestone_due
            if due:
                if due:
                    while milestone_due<=elapsed(control):milestone_due+=protocol["checkpoint_interval_seconds"]
                stage="first_periodic" if not periodic_saved else "periodic"
                validation(stage,not periodic_saved)
                periodic_saved=True
                unique=[int(coverage[case["space"].offsets[f]:case["space"].offsets[f+1]].sum()) for f in range(case["n_families"])]
                event("family_coverage",observation_counts=observations.tolist(),unique_slots=unique,
                      denominator_slots=case["space"].counts.tolist(),definition="unique query IDs plus repeated observation counts, not independent repeats")
                save("milestone_%07d.pt"%completed)
        validation("final",True)
        final_path=save("final_u%07d_%d.pt"%(completed,time.time_ns()))
    except BaseException as exc:
        failure=type(exc).__name__+": "+str(exc);reason="error_stop"
        attempt=dict(update=locals().get("position"),time=locals().get("inp",{}).get("t"),
            query_round=locals().get("inp",{}).get("round_id"),backend="upstream",exception=failure,
            completed_updates=completed,run_elapsed_seconds=elapsed(control),traceback=traceback.format_exc())
        with (out/"exceptions.jsonl").open("a") as f:f.write(json.dumps(attempt)+"\n")
        event("failed",**attempt)
        # Never label potentially half-updated weights as valid. Last atomic milestone remains.
    finally:
        stable=[r["batch_seconds"] for r in history if r["timing_phase"]=="stable"]
        result=dict(stage="cn-v2-IMDB-B-learning",status="failed" if failure else ("stopped_limit" if reason=="configured_update_limit" else "stopped_manual"),
            reason=reason,failure=failure,completed_updates=completed,run_control=control,run_elapsed_seconds=elapsed(control),
            execution_mode=mode_state(),post_resume_trajectory_identity_required=False,
            effective_training_compute_seconds=sum(r["batch_seconds"] for r in history),
            retained_milestone_updates=[r["completed_updates"] for r in validations],
            cpu_seconds=time.process_time()-started_cpu,formal_training=not args.fixture,new_scientific_graphs=0,
            final_checkpoint=str(final_path.relative_to(out)) if final_path else None,
            latest_valid_checkpoint=str(last_ck.relative_to(out)) if last_ck else None,
            validation_records=validations,stable_updates=len(stable),stable_median_seconds=float(np.median(stable)) if stable else None,
            stable_p90_seconds=float(np.quantile(stable,.9)) if stable else None,converged="not certified by stopping or update count",
            periodic_prediction_saved=periodic_saved,validation_wall_seconds=sum(r["wall_seconds"] for r in validations),
            peak_checkpoint_seconds=save_cost[0],**memory(args.device))
        result.update(checkpoint_wall_seconds=sum(save_times),checkpoint_seconds=save_times,
                      progress_logging_seconds=logging_seconds,
                      timing_note="batch excludes weight saves/validation/progress IO; CPU here is worker phase, not summed imports/gate CPU")
        if validations:
            first,last=validations[0],validations[-1]
            result["validation_trend"]=dict(initial_pooled_ce=first["pooled_ce"],last_pooled_ce=last["pooled_ce"],
                prior_ce=first["baseline"]["pooled_ce"],initial_positive_ce=first["positive_ce"],last_positive_ce=last["positive_ce"],
                initial_negative_ce=first["negative_ce"],last_negative_ce=last["negative_ce"],
                last_interval_pooled_delta=last["pooled_ce"]-validations[-2]["pooled_ce"] if len(validations)>1 else None,
                interpretation="descriptive same-reference denoising trend; no convergence or generation certification")
        json_save(out/"summary.json",result);event("training_stopped",summary=result)
        logs.close();events.close()
    return 1 if failure else 0


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--phase",choices=("gate_initialize","train"),required=True)
    p.add_argument("--output",type=Path,required=True);p.add_argument("--data",type=Path,default=PACKAGE/"data/IMDB_movie_reference.npz")
    p.add_argument("--device",choices=("cuda","cpu"),default="cuda");p.add_argument("--fixture",action="store_true")
    p.add_argument("--resume",action="store_true")
    p.add_argument("--arm",choices=("baseline","cn"),required=True)
    args=p.parse_args();args.output=args.output.resolve()
    from hsgd.bw_preflight.common import set_arm
    set_arm(args.arm)
    if args.device=="cpu" and not args.fixture:p.error("CPU only for explicit engineering fixture")
    if args.phase=="gate_initialize":gate_initialize(args);return 0
    return train(args)


if __name__=="__main__":sys.exit(main())
