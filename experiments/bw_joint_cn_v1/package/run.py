"""Stdlib bounded launcher. No train/generate command and no automatic installs."""
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
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['test','plan','toy-check','device-check'])
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--confirm-bounded-BW-check',action='store_true');p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    a=p.parse_args()
    if a.command=='device-check' and (a.device!='cuda' or not a.confirm_bounded_BW_check):
        p.error('BW check needs --device cuda --confirm-bounded-BW-check; not authorized for automatic execution')
    if a.command!='device-check' and a.device!='cpu':p.error('only explicit BW device-check may use accelerator')
    if a.worker:
        from hsgd.fusion_preflight import run
        try:run(a.command,a.output,a.device)
        except BaseException as exc:
            a.output.mkdir(parents=True,exist_ok=True)
            (a.output/'failure.json').write_text(json.dumps(dict(status='failed',error=repr(exc),automatic_retry=False)))
            raise
        return 0
    if a.output.exists():p.error('new independent output directory required')
    cap=480 if a.command=='device-check' else 180
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    if a.command=='test':
        a.output.mkdir(parents=True)
        cmd=[sys.executable,'-m','pytest',str(ROOT/'tests'),'-q','--junitxml='+str(a.output.resolve()/'tests.xml')]
    else:
        cmd=[sys.executable,str(Path(__file__).resolve()),a.command,'--output',str(a.output.resolve()),'--device',a.device,'--worker']
        if a.confirm_bounded_BW_check:cmd+=['--confirm-bounded-BW-check']
    started=time.perf_counter();log=a.output.parent/(a.output.name+'.log');log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('x') as f:
        child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        print(json.dumps(dict(pid=child.pid,log=str(log),timeout_seconds=cap)),flush=True)
        try:code=child.wait(timeout=cap)
        except (subprocess.TimeoutExpired,KeyboardInterrupt) as exc:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=3)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
            code=75
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/'launcher.json').write_text(json.dumps(dict(exit_code=code,wall_seconds=time.perf_counter()-started,
        cap_seconds=cap,cleanup_reserve_seconds=3,log=str(log),status='worker_exited' if code==0 else 'failed_or_incomplete')))
    print(log.read_text());return code


if __name__=='__main__':raise SystemExit(main())
