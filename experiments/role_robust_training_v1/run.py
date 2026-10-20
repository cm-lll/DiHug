"""Lightweight parent: budgets cover child imports and initialization."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent

def new_output(path,protected):
    path=path.resolve()
    for p in protected:
        p=p.resolve()
        if path==p or path in p.parents or p in path.parents:
            raise ValueError(f"output overlaps protected input/source: {p}")
    if path.exists():raise FileExistsError("new output required; no budget restart")
    path.mkdir(parents=True)
    return path

def supervise(command,out,deadline):
    child=None;status="failed";code=1;reason=None;clean_start=None
    def interrupt(sig,frame):raise InterruptedError(f"parent signal {sig}")
    previous={s:signal.signal(s,interrupt) for s in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP)}
    try:
        env=dict(os.environ,OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",CUDA_VISIBLE_DEVICES="")
        with (out/"worker.log").open("w") as f:
            child=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT,env=env,start_new_session=True)
            print(json.dumps(dict(pid=child.pid,output=str(out),deadline_epoch=deadline)),flush=True)
            while child.poll() is None:
                if time.time()>=deadline:raise TimeoutError("single cumulative deadline")
                if (out/"STOP").exists():raise InterruptedError("STOP")
                time.sleep(.2)
            code=int(child.returncode)
            receipt=json.loads((out/"worker_receipt.json").read_text())
            status="completed" if code==0 and receipt["status"]=="completed" else "failed"
            if status!="completed" and code==0:code=1
    except BaseException as exc:
        reason=repr(exc);code=124 if isinstance(exc,TimeoutError) else 1
        status="stopped" if isinstance(exc,(InterruptedError,TimeoutError,KeyboardInterrupt)) else "failed"
    finally:
        clean_start=time.time()
        # Ignore repeated termination while reaping only our own child group.
        for s in previous:signal.signal(s,signal.SIG_IGN)
        if child is not None:
            try:os.killpg(child.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            try:child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGKILL);child.wait()
        for s,handler in previous.items():signal.signal(s,handler)
        record=dict(status=status,exit_code=code,reason=reason,cleanup_seconds=time.time()-clean_start)
        (out/"controller.json").write_text(json.dumps(record,indent=2)+"\n")
    return code

def main():
    started=time.time();p=argparse.ArgumentParser()
    p.add_argument("mode",choices=("freeze","probe","panel"))
    p.add_argument("--source",type=Path,required=True);p.add_argument("--panel",type=Path,required=True)
    p.add_argument("--frozen",type=Path);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--execute-learning-panel",action="store_true")
    a=p.parse_args()
    if a.mode=="panel" and not a.execute_learning_panel:
        p.error("panel requires explicit --execute-learning-panel; preparation does not start learning")
    if a.mode!="freeze" and a.frozen is None:p.error("--frozen required")
    config=json.loads((HERE/"config.json").read_text())
    cap=config["learning_budget_seconds"] if a.mode=="panel" else config["probe_budget_seconds"]
    protected=[a.source,a.panel,HERE,HERE.parent/"role_specialization_v1",HERE.parent/"role_late_paired_v1",HERE.parent/"role_condition_diagnosis_v1"]
    if a.frozen:protected.append(a.frozen)
    out=new_output(a.output,protected);deadline=started+cap
    (out/"activation.json").write_text(json.dumps(dict(mode=a.mode,activation_epoch=started,deadline_epoch=deadline,
        budget_seconds=cap,learning_panel_activated=a.mode=="panel",no_reset=True,command=sys.argv),indent=2)+"\n")
    args=[sys.executable,"-u",str(HERE/"worker.py"),a.mode,"--source",str(a.source.resolve()),
          "--panel",str(a.panel.resolve()),"--output",str(out),"--deadline",str(deadline)]
    if a.frozen:args += ["--frozen",str(a.frozen.resolve())]
    code=supervise(args,out,deadline)
    r=json.loads((out/"controller.json").read_text());r.update(wall_seconds=time.time()-started,
         budget_seconds=cap,within_budget_plus_cleanup=time.time()-started<=cap+5,learning_panel_activated=a.mode=="panel")
    (out/"controller.json").write_text(json.dumps(r,indent=2)+"\n");print(json.dumps(r),flush=True)
    return code
if __name__=="__main__":sys.exit(main())
