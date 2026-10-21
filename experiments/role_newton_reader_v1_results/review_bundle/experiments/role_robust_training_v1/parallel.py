"""Three independent seed jobs, one isolated GPU each, ONE shared deadline."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
def save(p,x):
    tmp=p.with_name(p.name+".tmp");tmp.write_text(json.dumps(x,indent=2,allow_nan=False)+"\n");tmp.replace(p)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def merge(out,jobs,complete):
    updates=[];graphs=[];weights={};index=[];resources=[]
    for job in jobs:
        folder=job["folder"];prefix=str(folder.relative_to(out))+"/"
        for name in ("resolved_config.json","source_manifest.json","input_bindings.json","frozen_binding.json"):
            if (folder/name).exists() and not (out/name).exists():(out/name).write_bytes((folder/name).read_bytes())
        if (folder/"updates.jsonl").exists():updates.extend(json.loads(x) for x in (folder/"updates.jsonl").read_text().splitlines())
        if (folder/"generation.jsonl").exists():
            for line in (folder/"generation.jsonl").read_text().splitlines():
                r=json.loads(line);r["graph_file"]=prefix+r["graph_file"];graphs.append(r)
        if (folder/"weight_hashes.json").exists():
            weights.update({prefix+k:v for k,v in json.loads((folder/"weight_hashes.json").read_text()).items()})
        if (folder/"checkpoint_index.json").exists():
            for r in json.loads((folder/"checkpoint_index.json").read_text()):r["path"]=prefix+r["path"];index.append(r)
        if (folder/"worker_receipt.json").exists():resources.append(dict(seed=job["seed"],gpu=job["gpu"],**json.loads((folder/"worker_receipt.json").read_text())))
        pressure=folder/f"pressure_seed{job['seed']}.json"
        if pressure.exists():
            rows=json.loads(pressure.read_text())
            for r in rows:r["probability_file"]=prefix+r["probability_file"]
            save(out/pressure.name,rows)
    with (out/"updates.jsonl").open("w") as f:
        for r in updates:f.write(json.dumps(r,allow_nan=False)+"\n")
    with (out/"generation.jsonl").open("w") as f:
        for r in graphs:f.write(json.dumps(r,allow_nan=False)+"\n")
    save(out/"generation.json",graphs);save(out/"weight_hashes.json",weights);save(out/"checkpoint_index.json",index)
    save(out/"seed_resources.json",resources)
    return len(updates),len(graphs)

def main():
    p=argparse.ArgumentParser()
    for key in ("source","panel","frozen","output"):p.add_argument("--"+key,type=Path,required=True)
    p.add_argument("--deadline",type=float,required=True);p.add_argument("--gpus",nargs=3)
    p.add_argument("--device",choices=("cpu","cuda"),default="cpu");a=p.parse_args()
    cfg=json.loads((HERE/"config.json").read_text());activation=json.loads((a.output/"activation.json").read_text())
    slots=a.gpus if a.device=="cuda" else ["CPU slot0","CPU slot1","CPU slot2"]
    assert activation["budget_seconds"]==28800 and len(set(slots))==3
    started=time.monotonic();children=[];jobs=[];probes=[];status="failed";reason=None;counts=(0,0);phase="resource_check"
    requested=[False]
    def stop(sig,frame):requested[0]=True;(a.output/"STOP").touch()
    for s in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP):signal.signal(s,stop)
    def check():
        if time.time()>a.deadline-15:raise TimeoutError("shared eight-hour deadline")
        if requested[0] or (a.output/"STOP").exists():
            for child in children:(child["folder"]/"STOP").touch()
            requested[0]=True
            if phase=="independent_global_audit":raise InterruptedError("STOP during audit")
    def launch(mode,seed,gpu,folder,probe=None):
        folder.mkdir(parents=True,exist_ok=False)
        ledger=dict(activation,substage=mode,seed=seed,gpu=gpu,inherited_global_deadline=True)
        save(folder/"activation.json",ledger)
        deadline=min(a.deadline,time.time()+300) if mode=="probe" else a.deadline
        cmd=[sys.executable,"-u",str(HERE/"worker.py"),mode,"--source",str(a.source),"--panel",str(a.panel),
             "--frozen",str(a.frozen),"--output",str(folder),"--deadline",str(deadline),"--device",a.device,"--parallel-workers","3"]
        if mode=="seed-panel":cmd += ["--seed",str(seed),"--probe-receipt",str(probe)]
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu if a.device=="cuda" else "",OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1")
        # Ordinary training stays ordinary; strict gate is a separate process.
        env.pop("CUBLAS_WORKSPACE_CONFIG",None)
        log=(folder/"worker.log").open("w")
        process=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
        child=dict(process=process,folder=folder,gpu=gpu,seed=seed,log=log,deadline=deadline);children.append(child)
        return child
    def wait_all(active):
        last=0
        while any(c["process"].poll() is None for c in active):
            check()
            for child in active:
                code=child["process"].poll()
                if code is None and time.time()>child["deadline"]:
                    raise TimeoutError(f"bounded child exceeded deadline: {child['folder']}")
                if code is not None and code!=0:
                    receipt=child["folder"]/"worker_receipt.json"
                    r=json.loads(receipt.read_text()) if receipt.exists() else {}
                    if r.get("status")=="stopped":raise InterruptedError(f"child stopped: {r.get('reason')}")
                    raise RuntimeError(f"child failed {child['folder']} exit{code}: {r.get('reason')}")
            if time.monotonic()-last>=30:
                report=[]
                for child in active:
                    path=child["folder"]/"progress.jsonl";r={}
                    if path.exists():
                        lines=path.read_text().splitlines()
                        if lines:r=json.loads(lines[-1])
                    report.append(dict(seed=child["seed"],gpu=child["gpu"],exit_code=child["process"].poll(),progress=r))
                save(a.output/"parallel_progress.json",dict(phase=phase,children=report,remaining_seconds=a.deadline-time.time()))
                print(json.dumps(dict(phase=phase,children=report,remaining_seconds=a.deadline-time.time())),flush=True);last=time.monotonic()
            time.sleep(.5)
        for child in active:
            code=child["process"].wait();child["log"].close()
            r=json.loads((child["folder"]/"worker_receipt.json").read_text())
            if r["status"]=="stopped":raise InterruptedError(f"child stopped: {r.get('reason')}")
            if code!=0 or r["status"]!="completed":raise RuntimeError(f"noncompleted child {child['folder']}: {r}")
        if requested[0]:raise InterruptedError("STOP")
    try:
        if a.device=="cuda":
            snapshot=subprocess.check_output(["nvidia-smi","--query-gpu=uuid,name,memory.total,memory.used,utilization.gpu","--format=csv,noheader"],text=True,timeout=30)
            for gpu in slots:
                line=next(l for l in snapshot.splitlines() if l.startswith(gpu+","))
                if int(line.split(",")[3].strip().split()[0])>256:raise RuntimeError(f"assigned device occupied: {gpu}")
        else:
            if len(os.sched_getaffinity(0))<3:raise RuntimeError("fewer than three available CPU affinity slots")
            snapshot="CPU fallback: GPU gradient gate exceeded frozen tolerance; no GPU science"
        save(a.output/"resource_allocation.json",dict(snapshot=snapshot,assigned=slots,one_seed_pair_per_worker=True,device=a.device))
        phase="parallel_cost_gate"
        for seed,gpu in zip(cfg["train_seeds"],slots):probes.append(launch("probe",seed,gpu,a.output/"cost_probes"/f"seed{seed}"))
        wait_all(probes)
        reports=[json.loads((x["folder"]/"probe.json").read_text()) for x in probes]
        worst=max(r["parallel_wall_forecast_seconds"] for r in reports)
        passed=worst<a.deadline-time.time()-30
        save(a.output/"parallel_cost_gate.json",dict(passed=passed,worst_parallel_forecast_seconds=worst,remaining_seconds=a.deadline-time.time(),reports=reports))
        if not passed:raise RuntimeError("all fixed work cannot fit remaining eight-hour window; no learning starts")
        phase="parallel_learning_and_evaluation"
        for probe in probes:
            jobs.append(launch("seed-panel",probe["seed"],probe["gpu"],a.output/"seedjobs"/f"seed{probe['seed']}",probe["folder"]/"probe.json"))
        wait_all(jobs);counts=merge(a.output,jobs,True)
        assert counts==(36000,288)
        phase="independent_global_audit"
        sys.path.insert(0,str(HERE))
        from evaluate import audit
        audit(a.output,cfg,check)
        status="completed";return 0
    except BaseException as exc:
        reason=repr(exc);status="stopped" if isinstance(exc,(InterruptedError,TimeoutError)) else "failed";raise
    finally:
        for child in children:
            if child["process"].poll() is None:(child["folder"]/"STOP").touch()
        end=time.monotonic()+5
        while any(c["process"].poll() is None for c in children) and time.monotonic()<end:time.sleep(.1)
        for child in children:
            if child["process"].poll() is None:
                try:os.killpg(child["process"].pid,signal.SIGKILL)
                except ProcessLookupError:pass
            child["process"].wait();child["log"].close()
        if jobs and status!="completed":counts=merge(a.output,jobs,False)
        save(a.output/"worker_receipt.json",dict(status=status,reason=reason,phase=phase,scientific_updates=counts[0],complete_chains=counts[1],
            wall_seconds=time.monotonic()-started,learning_panel_activated=True,GPU=len(slots) if a.device=="cuda" else 0,
            budget_seconds=28800,activation_epoch=activation["activation_epoch"],deadline_epoch=activation["deadline_epoch"],no_reset=True))
if __name__=="__main__":sys.exit(main())
