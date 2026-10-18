"""Bounded manual controller; no BW connection, training, or chain launcher."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def main():
    started=time.monotonic()
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=['test','short','worker'])
    p.add_argument('--output',required=True)
    p.add_argument('--seconds',type=int,default=900)
    p.add_argument('--manifest');p.add_argument('--manifest-sha256')
    p.add_argument('--evaluation');p.add_argument('--evaluation-sha256')
    p.add_argument('--deadline',type=float)
    a=p.parse_args()
    if not 10<=a.seconds<=900:p.error('single invocation must be 10..900 seconds, including cleanup')
    if a.action=='worker':
        from multiscale_role.short import run
        return run(a.manifest,a.manifest_sha256,a.evaluation,a.evaluation_sha256,a.output,a.deadline)
    if Path(a.output).exists():p.error('new output directory required; never overwrite')
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1')
    deadline=started+a.seconds
    if a.action=='test':
        Path(a.output).mkdir(parents=True,exist_ok=False)
        command=[sys.executable,'-m','pytest','-q','tests','--disable-warnings','--junitxml='+str(Path(a.output)/'tests.xml')]
    else:
        if not all((a.manifest,a.manifest_sha256,a.evaluation,a.evaluation_sha256)):p.error('pinned six-state manifest and original evaluation specification required')
        command=[sys.executable,'-u',__file__,'worker','--output',a.output,'--seconds',str(a.seconds),
                 '--deadline',str(deadline-8),'--manifest',a.manifest,'--manifest-sha256',a.manifest_sha256,
                 '--evaluation',a.evaluation,'--evaluation-sha256',a.evaluation_sha256]
    child=subprocess.Popen(command,env=env,start_new_session=True)
    result=None
    try:result=child.wait(timeout=max(.01,deadline-time.monotonic()-4))
    except (subprocess.TimeoutExpired,KeyboardInterrupt):
        os.killpg(child.pid,signal.SIGINT)
        try:child.wait(timeout=max(.01,deadline-time.monotonic()-.1))
        except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        result=124
    finally:
        out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
        (out/'controller.json').write_text(json.dumps(dict(exit_code=result,controller_wall_seconds=time.monotonic()-started,
             status='completed' if result==0 else 'incomplete' if result==124 else 'failed',
             single_run_limit_seconds=a.seconds,no_absolute_old_deadline=True,scientific_chains=0,training=0),indent=2))
    return result


if __name__=='__main__':raise SystemExit(main())
