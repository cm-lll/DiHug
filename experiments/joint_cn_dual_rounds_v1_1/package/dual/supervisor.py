"""Standard-library-only wall supervisor. No torch/numpy/data/CUDA/lock imports."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

def safe_output(output):
    out=Path(output).resolve();package=Path(__file__).resolve().parents[1]
    if out==package or package in out.parents or out in package.parents:raise ValueError('output overlaps package/source')
    if out.exists():raise ValueError('bounded output exists')
    return out

def supervise(command,seconds,output,started=None,cleanup_seconds=3,completion_check=None):
    started=time.monotonic() if started is None else started
    out=safe_output(output);out.parent.mkdir(parents=True,exist_ok=True)
    deadline=started+seconds;received=[];handlers={};proc=None;reason=None;rc=1;cleanup_start=None
    for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP):handlers[sig]=signal.signal(sig,lambda s,f:received.append(s))
    try:
        if time.monotonic()>=deadline:reason='timeout';rc=124
        else:
            proc=subprocess.Popen(command,start_new_session=True)
            while True:
                if received:reason='signal';rc=130;break
                if (out/'STOP').exists():reason='STOP';rc=130;break
                if time.monotonic()>=deadline:reason='timeout';rc=124;break
                if proc.poll() is not None:rc=proc.returncode;break
                time.sleep(.02)
        if rc==0 and completion_check:
            try:completion_check(out)
            except Exception:reason='missing/invalid complete evidence';rc=1
        if received:reason='signal';rc=130
        elif (out/'STOP').exists():reason='STOP';rc=130
        elif time.monotonic()>=deadline:reason='timeout';rc=124
        return rc
    finally:
        compute_end=time.monotonic();cleanup_start=compute_end
        if proc is not None:
            # Reap the group even if the direct worker exited but left children.
            try:os.killpg(proc.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            try:proc.wait(timeout=cleanup_seconds)
            except subprocess.TimeoutExpired:
                try:os.killpg(proc.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                proc.wait()
            # Direct worker may exit promptly while descendants ignore TERM.
            try:os.killpg(proc.pid,signal.SIGKILL)
            except ProcessLookupError:pass
        cleanup_end=time.monotonic()
        for sig,h in handlers.items():signal.signal(sig,h)
        out.mkdir(exist_ok=True)
        receipt=dict(exit_code=rc,child_exit_code=proc.returncode if proc else None,reason=reason,
            completed=rc==0 and reason is None,cap_seconds=seconds,
            started_monotonic=started,computation_window_seconds=compute_end-started,
            cleanup_seconds=cleanup_end-cleanup_start,cleanup_budget_seconds=cleanup_seconds,
            wall_seconds=cleanup_end-started,scope='includes interpreter/module startup, data, CUDA, device lock, compute and save; cleanup recorded separately')
        tmp=out/'supervisor.json.tmp';tmp.write_text(json.dumps(receipt,indent=2));tmp.replace(out/'supervisor.json')

def short_main(argv,started):
    p=argparse.ArgumentParser(description='570s TOTAL startup+device short test; no formal training')
    p.add_argument('--dataset',required=True,choices=['IMDB','PubMed']);p.add_argument('--output',required=True)
    p.add_argument('--device-index',type=int,default=0);p.add_argument('--max-seconds',type=int,default=570)
    p.add_argument('--device',choices=['cuda'],default='cuda')
    args=p.parse_args(argv)
    if not 30<=args.max_seconds<=570:raise ValueError('cap must be 30..570 seconds')
    command=[sys.executable,'-u','-m','dual.cli','_short-worker','--dataset',args.dataset,
        '--output',args.output,'--device-index',str(args.device_index),'--max-seconds',str(args.max_seconds),
        '--deadline-monotonic',str(started+args.max_seconds)]
    def completed(out):
        summary=json.loads((out/'summary.json').read_text());costs=json.loads((out/'costs.json').read_text())
        if summary['status']!='completed' or len(costs['updates'])!=6 or costs['sweep']['blocks']!=20 or not costs['sweep']['exact_coverage'] or len(costs['validation']['times'])!=5:
            raise ValueError('short test incomplete')
    return supervise(command,args.max_seconds,args.output,started,completion_check=completed)
