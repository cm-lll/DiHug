"""One frozen candidate, paired with already archived late-C reference.

CPU-only, one nonrenewable 2700s window, no model/seed/checkpoint selection.
"""
import os
os.environ.update(OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",CUDA_VISIBLE_DEVICES="")
import argparse
import gc
import hashlib
import json
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent
PREV=ROOT.parent/"role_condition_diagnosis_v1"
PAIRED=ROOT.parent/"role_late_paired_v1"
sys.path[:0]=[str(ROOT),str(PAIRED),str(PREV),str(PREV/"baseline")]
from capacity_readout import CapacityDenoiser
from pairing import input_digest,uniform_contract,atomic_weights,sha
from worker import response
from diagnose import load_panel
from hsgd.bw_preflight import common as c
from hsgd.runtime import reverse_step
from hsgd.kernel import MarginalKernel
from role_condition.experiment import save,emit,opt,fixed_validation,validate
from role_condition.panel import augment,NAMES
from role_condition.metrics import graph,compare

def fingerprint(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def files():
    paths=list(ROOT.glob("*.py"))+list(ROOT.glob("*.md"))
    paths+=list(PAIRED.glob("*.py"))+[PREV/"late_fusion.py",PREV/"diagnose.py"]
    paths+=[x for x in (PREV/"baseline").rglob("*") if x.is_file() and x.suffix in [".py",".json"]]
    return {str(p.relative_to(ROOT.parent)):fingerprint(p) for p in sorted(paths)}

def new_model(case,cfg,source,seed,hashes):
    m=CapacityDenoiser(case["n_types"],case["n_families"],family_pairs=case["space"].pairs,
        prior=case["prior"],T=64,**cfg["model"])
    path=source/f"seed{seed}/late_C_initial.pt"
    assert fingerprint(path)==hashes[str(path.relative_to(source))]
    old=torch.load(path,map_location="cpu",weights_only=True)
    state=m.state_dict()
    assert set(old)-set(state)=={"branch.query.weight","branch.query.bias","branch.key.weight","branch.key.bias"}
    assert not set(state)-set(old)
    for name in state:state[name]=old[name].clone()
    m.load_state_dict(state,strict=True)
    assert all(torch.equal(m.state_dict()[k],old[k]) for k in state)
    return m,dict(source=str(path),sha256=fingerprint(path),all_shared_tensors_exact=True,
                  shared_tensors=len(state),removed_tensors=sorted(set(old)-set(state)),
                  parameters=sum(p.numel() for p in m.parameters()))

def worker(source,out,deadline_epoch):
    torch.set_num_threads(1);torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(False)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    started=time.monotonic();cpu=time.process_time();status="failed";reason=None
    completed=0;generated=[];m=None;optimizer=None
    def check():
        if (out/"STOP").exists():raise InterruptedError("STOP")
        if time.time()>deadline_epoch-30:raise TimeoutError("original budget reserve")
    def stopped(sig,frame):raise InterruptedError(f"signal {sig}")
    for sig in [signal.SIGTERM,signal.SIGINT,signal.SIGHUP]:signal.signal(sig,stopped)
    cfg=json.loads((source/"resolved_config.json").read_text())
    cfg.update(version="role_specialization_v1",arms=["capacity_C"],baseline_arm="late_C",
        budget_seconds=2700,updates=1000,expected_updates=3000,expected_chains=48,
        method="capacity-coupled forward-degree likelihood soft role readout")
    save(out/"resolved_config.json",cfg);save(out/"environment.json",c.environment("cpu"))
    old_environment=json.loads((source/"environment.json").read_text())
    current_environment=c.environment("cpu")
    for key in ["python_version","packages","torch_runtime_version","float32_matmul_precision",
                "matmul_allow_tf32","deterministic_algorithms","torch_threads"]:
        assert old_environment[key]==current_environment[key],f"numerical environment mismatch: {key}"
    binding=files();save(out/"source_manifest.json",binding)
    save(out/"revision.json",dict(commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()))
    prior_hashes=json.loads((source/"weight_hashes.json").read_text())
    receipt=json.loads((source/"budget_receipt.json").read_text());assert receipt["status"]=="completed"
    records,targets=load_panel(source);case=records[0]["case"]
    source_hashes={str(p):fingerprint(p) for p in [source/"panel.json",source/"resolved_config.json",source/"weight_hashes.json",
            source/"updates.jsonl",source/"generation.json",source/"audit.json"]}
    old_updates={(r["seed"],r["update"]):r for r in map(json.loads,(source/"updates.jsonl").read_text().splitlines()) if r["arm"]=="late_C"}
    old_graphs=[r for r in json.loads((source/"generation.json").read_text()) if r["arm"]=="late_C"]
    for g in old_graphs:
        p=source/g["graph_file"];assert fingerprint(p)==g["graph_file_sha256"];source_hashes[str(p)]=fingerprint(p)
    save(out/"baseline_sources.json",source_hashes)
    references=json.loads((source/"reference_metrics.json").read_text())
    save(out/"reference_metrics.json",references)
    values=fixed_validation(records,cfg);train=[r for r in records if r["split"]=="train"]
    weight_hashes={}
    try:
        # Full-size engineering cost gate, no changes based on generated quality.
        m,_=new_model(case,cfg,source,cfg["train_seeds"][0],prior_hashes)
        m.set_target(targets["extreme"]);optimizer=opt(m,cfg)
        current=next(r["case"] for r in train if r["condition"]=="extreme")
        durations=[]
        for i in range(5):
            check();inp=c.train_input(current,64,4,2026102088,i,np.random.default_rng([2026102088,i]),force_t=32)
            st=time.monotonic();c.update(m,optimizer,current,inp,64,"cpu");durations.append(time.monotonic()-st)
        m.eval();times=[];dense=np.repeat(case["active"].astype(np.uint8),case["space"].counts)
        for i in range(3):
            check();st=time.monotonic()
            with torch.no_grad():reverse_step(m,case,dense,MarginalKernel(case["prior"],c.schedule(64)),32,4,12,np.random.default_rng(12),"cpu",check_time=check)
            times.append(time.monotonic()-st)
        val=validate(m,values,cfg,targets,check)["seconds"]
        forecast=float(1.2*(3000*np.quantile(durations[2:],.9)+48*64*np.quantile(times[1:],.9)+12*val)+120)
        gate=dict(updates_seconds=durations,dense_step_seconds=times,validation_seconds=val,
            forecast_seconds=forecast,remaining_seconds=deadline_epoch-time.time(),passed=bool(forecast<deadline_epoch-time.time()-30))
        save(out/"cost_gate.json",gate)
        if not gate["passed"]:raise RuntimeError("complete candidate panel exceeds bound; no science starts")
        del m,optimizer;gc.collect();m=None;optimizer=None
        for seed in cfg["train_seeds"]:
            check();m,init=new_model(case,cfg,source,seed,prior_hashes);optimizer=opt(m,cfg);assert not optimizer.state
            sd=out/f"seed{seed}";sd.mkdir();save(sd/"initialization.json",init)
            p=sd/"capacity_C_initial.pt";weight_hashes[str(p.relative_to(out))]=atomic_weights(p,m.state_dict())
            save(out/"weight_hashes.json",weight_hashes)
            rng=np.random.default_rng(seed)
            for u in range(1001):
                if u in [0,250,500,1000]:
                    check();v=validate(m,values,cfg,targets,check)
                    save(sd/f"capacity_C_validation_{u:04d}.json",dict(seed=seed,update=u,**v))
                    emit(out/"progress.jsonl",dict(phase="validation",seed=seed,update=u,mean_ce=v["mean_ce"]))
                if u==1000:break
                check();rec=train[(u%2)*4+int(rng.integers(4))];current=augment(rec["case"],rng)
                inp=c.train_input(current,64,4,seed,u,rng);digest=input_digest(current,inp)
                old=old_updates[seed,u+1];assert digest==old["input_sha256"]
                # Original network has no stochastic forward operators. Explicitly
                # check both runs' unchanged RNG consumption, rather than just seed.
                before=torch.get_rng_state().clone();m.set_target(targets[rec["condition"]])
                st=time.monotonic();r=c.update(m,optimizer,current,inp,64,"cpu");dt=time.monotonic()-st
                assert torch.equal(before,torch.get_rng_state())
                assert old["torch_rng_before"]==old["torch_rng_after"]
                assert input_digest(current,inp)==digest
                step=int(optimizer.state[m.backbone.lin_out_E.weight]["step"]);assert step==u+1
                completed+=1
                with (out/"updates.jsonl").open("a") as f:f.write(json.dumps(dict(seed=seed,arm="capacity_C",update=u+1,
                    input_sha256=digest,paired_baseline_input_verified=True,t=inp["t"],query_block=inp["round_id"],
                    query_count=len(inp["ids"]),loss=r["loss"],gradient_norm=r["gradient_norm"],optimizer_step=step,
                    seconds=dt,no_torch_forward_randomness=True))+"\n")
                if (u+1)%100==0:emit(out/"progress.jsonl",dict(phase="training",seed=seed,update=u+1,total=completed,remaining=deadline_epoch-time.time()))
            p=sd/"capacity_C_final_1000.pt";weight_hashes[str(p.relative_to(out))]=atomic_weights(p,m.state_dict())
            save(out/"weight_hashes.json",weight_hashes)
            rr,aa=response({"capacity_C":m},values,targets,cfg,check);save(sd/"condition_response.json",rr)
            np.savez_compressed(sd/"condition_response.npz",**aa)
            m.eval()
            for sample_seed in cfg["sampling_seeds"]:
                for condition in ["balanced","extreme"]:
                    check();m.set_target(targets[condition]);c.seed_all(sample_seed);rng=np.random.default_rng(sample_seed)
                    state=(rng.random(case["space"].total)<np.repeat(case["prior"],case["space"].counts)).astype(np.uint8)
                    original=next(r for r in old_graphs if r["seed"]==seed and r["condition"]==condition and r["sampling_seed"]==sample_seed)
                    assert sha(state.tolist())==original["initial_state_sha256"]
                    steps_path=(source/original["graph_file"]).with_name(Path(original["graph_file"]).stem+"_steps.json")
                    source_hashes[str(steps_path)]=fingerprint(steps_path)
                    old_steps=json.loads(steps_path.read_text());steps=[];st=time.monotonic()
                    for t in range(64,0,-1):
                        check();uh,expected=uniform_contract(rng,case["space"].total)
                        plan=c.plan_for(case,[sample_seed,t,61],4);qh=hashlib.sha256()
                        for j in range(4):
                            for arr in plan.block(j):qh.update(np.ascontiguousarray(arr).tobytes())
                        prior_step=old_steps[64-t]
                        assert prior_step["t"]==t and prior_step["uniform_sha256"]==uh and prior_step["query_sha256"]==qh.hexdigest()
                        with torch.no_grad():state,info=reverse_step(m,case,state,MarginalKernel(case["prior"],c.schedule(64)),t,4,sample_seed,rng,"cpu",check_time=check)
                        assert sha(rng.bit_generator.state)==expected==prior_step["rng_after"]
                        steps.append(dict(t=t,uniform_sha256=uh,query_sha256=qh.hexdigest(),rng_after=expected,
                            state_sha256=sha(state.tolist()),edge_count=int(state.sum()),exact_coverage=info["exact_coverage"],
                            blocks=info["blocks"],seconds=info["step_seconds"]))
                    edges,f=c.state_edges(case,state);stem=f"capacity_C_{condition}_{sample_seed}"
                    np.savez_compressed(sd/(stem+".npz"),types=case["space"].types,edges=edges.T,relations=f+1,schema=np.array(case["space"].pairs))
                    save(sd/(stem+"_steps.json"),steps);metrics=graph(case,edges,NAMES,check);comparison=compare(metrics,references[condition])
                    mg=case["space"].pairs.index((2,3));g=metrics["typed_degree"]["Genre"]["by_relation"][mg]
                    high=int(np.array(g["histogram"])[12:].sum())
                    row=dict(seed=seed,arm="capacity_C",condition=condition,sampling_seed=sample_seed,
                        graph_file=str((sd/(stem+".npz")).relative_to(out)),graph_file_sha256=fingerprint(sd/(stem+".npz")),
                        metrics=metrics,comparison=comparison,G_W1=comparison["typed_relation_degree_W1"]["Genre"][mg],
                        G_joint_TV=comparison["joint_relation_degree_TV"]["Genre"],G_max=g["max"],G_high_ge12=high,
                        G_multiple_hubs=high>1,G_zero=g["zero"],G_low_le1=g["low_le1"],seconds=time.monotonic()-st)
                    generated.append(row)
                    with (out/"generation.jsonl").open("a") as stream:stream.write(json.dumps(row)+"\n")
                    emit(out/"progress.jsonl",dict(phase="sample",seed=seed,condition=condition,sampling_seed=sample_seed,
                        chains=len(generated),edges=metrics["edges"],G_max=g["max"],high=high,remaining=deadline_epoch-time.time()))
            del m,optimizer;gc.collect();m=None;optimizer=None
        assert completed==3000 and len(generated)==48
        save(out/"generation.json",generated);save(out/"baseline_sources.json",source_hashes)
        assert all(fingerprint(p)==h for p,h in source_hashes.items())
        assert files()==binding
        status="completed";return 0
    except BaseException as exc:
        reason=repr(exc)
        if m is not None:atomic_weights(out/"interrupted.pt",dict(model=m.state_dict(),optimizer=optimizer.state_dict() if optimizer else None,completed=completed))
        raise
    finally:
        save(out/"worker_receipt.json",dict(status=status,reason=reason,completed_updates=completed,chains=len(generated),
            wall_seconds=time.monotonic()-started,CPU_seconds=time.process_time()-cpu,
            peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,GPU=0,baseline_training_reexecuted=False))

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--source",type=Path,required=True);ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--worker",action="store_true");ap.add_argument("--deadline",type=float)
    ap.add_argument("--continue-preflight-fix",action="store_true")
    a=ap.parse_args()
    if a.worker:return worker(a.source,a.output,a.deadline)
    if a.continue_preflight_fix:
        ledger=json.loads((a.output/"activation.json").read_text())
        old=json.loads((a.output/"worker_receipt.json").read_text())
        assert old["status"]=="failed" and old["completed_updates"]==0 and old["chains"]==0
        assert not (a.output/"updates.jsonl").exists() and not list(a.output.glob("seed*"))
        activated=ledger["activation_epoch"];deadline=ledger["deadline_epoch"]
        assert time.time()<deadline and deadline-activated==2700
        archive=a.output/"preflight_failure_01";archive.mkdir(exist_ok=False)
        for p in list(a.output.iterdir()):
            if p.is_file() and p.name!="activation.json":p.rename(archive/p.name)
        save(a.output/"preflight_fix.json",dict(reason="NumPy bool serialization in cost receipt; zero science updates",
             original_activation_epoch=activated,original_deadline_epoch=deadline,no_reset=True))
    else:
        a.output.mkdir(parents=True,exist_ok=False);activated=time.time();deadline=activated+2700
        save(a.output/"activation.json",dict(activation_epoch=activated,deadline_epoch=deadline,budget_seconds=2700,
            no_reset=True,new_task="user delegates one bounded minimal correction; previous 60min panel remains closed"))
    child=None;status="failed";exit_code=1;reason=None
    def interrupt(sig,frame):raise InterruptedError(f"parent signal {sig}")
    for sig in [signal.SIGINT,signal.SIGTERM,signal.SIGHUP]:signal.signal(sig,interrupt)
    try:
        with (a.output/"worker.log").open("w") as log:
            child=subprocess.Popen([sys.executable,"-u",str(Path(__file__).resolve()),"--worker","--source",str(a.source),
                "--output",str(a.output),"--deadline",str(deadline)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            while child.poll() is None:
                if time.time()>deadline-10:raise TimeoutError("single panel deadline")
                if (a.output/"STOP").exists():raise InterruptedError("STOP")
                time.sleep(.5)
            exit_code=child.returncode;status="completed" if exit_code==0 else "failed"
    except BaseException as exc:reason=repr(exc)
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=5)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        save(a.output/"budget_receipt.json",dict(status=status,exit_code=exit_code,reason=reason,
            activation_epoch=activated,deadline_epoch=deadline,wall_seconds=time.time()-activated,budget_seconds=2700,no_reset=True))
    return exit_code
if __name__=="__main__":sys.exit(main())
