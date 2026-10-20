"""CPU-only bounded launcher, no remote/model/data access in the parent."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent

def main():
    started=time.monotonic();p=argparse.ArgumentParser()
    p.add_argument('--output',required=True,type=Path);args=p.parse_args();out=args.output.resolve()
    if out.exists() or out==ROOT or ROOT in out.parents or out in ROOT.parents:raise ValueError('new output separate from source required')
    manifest=json.loads((ROOT/'manifest.json').read_text())
    import hashlib
    for name,sha in manifest.items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=sha:raise ValueError('release changed: '+name)
    cfg=json.loads((ROOT/'role_condition/config.json').read_text())
    cap=cfg['budget_seconds']-cfg['correctness_reserve_seconds']-cfg['parent_cleanup_seconds'];deadline=started+cap
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    # Local CPU job only; child inherits no accelerator access and installs no packages.
    args.output.parent.mkdir(parents=True,exist_ok=True)
    proc=subprocess.Popen([sys.executable,'-u','-m','role_condition.experiment',str(out),str(deadline)],cwd=ROOT,env=env,start_new_session=True)
    received=[];old={};rc=1;reason=None
    for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP):old[sig]=signal.signal(sig,lambda s,f:received.append(s))
    try:
        while proc.poll() is None:
            if received or time.monotonic()>=deadline or (out/'STOP').exists():
                reason='signal' if received else 'budget' if time.monotonic()>=deadline else 'STOP'
                rc=124 if reason=='budget' else 130;break
            time.sleep(.1)
        else:rc=proc.returncode
        return rc
    finally:
        cleanup=time.monotonic()
        if proc.poll() is None:
            os.killpg(proc.pid,signal.SIGTERM)
            try:proc.wait(timeout=cfg['parent_cleanup_seconds'])
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
        for sig,handler in old.items():signal.signal(sig,handler)
        out.mkdir(exist_ok=True)
        (out/'controller.json').write_text(json.dumps(dict(exit_code=rc,child_exit_code=proc.returncode,reason=reason,
            wall_seconds=time.monotonic()-started,cleanup_seconds=time.monotonic()-cleanup,computation_cap=cap,
            total_envelope_seconds=cfg['budget_seconds'],correctness_reserve_seconds=cfg['correctness_reserve_seconds'],
            full_panel_completed=rc==0,no_automatic_retry=True),indent=2))

if __name__=='__main__':raise SystemExit(main())
