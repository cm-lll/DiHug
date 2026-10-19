#!/usr/bin/env python3
"""Bounded CPU experiments; no training, network, BW or diffusion-chain runner."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent


def atomic(path,value):
    tmp=path.with_name(path.name+'.tmp')
    with tmp.open('w') as f:json.dump(value,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


def release_hash():
    path=ROOT/'SHA256SUMS'
    if not path.exists():raise ValueError('release checklist missing; run the documented build prepare first')
    for line in path.read_text().splitlines():
        h,n=line.split('  ',1);p=(ROOT/n).resolve()
        if ROOT not in p.parents or hashlib.sha256(p.read_bytes()).hexdigest()!=h:raise ValueError('release payload hash mismatch: '+n)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    activated=time.monotonic();p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['small','imdb']);p.add_argument('--output',required=True)
    p.add_argument('--max-seconds',type=float);p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    p.add_argument('--deadline',type=float,help=argparse.SUPPRESS)
    args=p.parse_args();cap=(300 if args.action=='imdb' else 120) if args.max_seconds is None else args.max_seconds
    if not 10<=cap<=(300 if args.action=='imdb' else 120):raise ValueError('invalid bounded action budget')
    out=Path(args.output).resolve()
    if not args.worker:
        if out==ROOT or ROOT in out.parents or out in ROOT.parents or out.exists():raise ValueError('fresh output disjoint from package required')
        package_sha=release_hash();out.mkdir(parents=True)
        atomic(out/'activation.json',dict(action=args.action,inclusive_wall_seconds=cap,engineering_only=True,release_checklist_sha256=package_sha))
        caught=[];old={};child=None;status='failed';reason='parent_error';code=1
        for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP):
            old[sig]=signal.signal(sig,lambda s,f:caught.append(s))
        try:
            env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
            command=[sys.executable,str(Path(__file__).resolve()),*sys.argv[1:],'--worker','--deadline',str(activated+cap-8)]
            child=subprocess.Popen(command,env=env,start_new_session=True)
            atomic(out/'child.json',dict(pid=child.pid,pgid=child.pid))
            while True:
                if caught or (out/'STOP').exists():status='stopped';reason=signal.Signals(caught[0]).name if caught else 'STOP';code=128+caught[0] if caught else 75;break
                if time.monotonic()>=activated+cap-3:status='incomplete';reason='wall_timeout';code=124;break
                ret=child.poll()
                if ret is not None:
                    summary=json.loads((out/'summary.json').read_text()) if (out/'summary.json').exists() else {}
                    if ret==0 and summary.get('status')=='complete':
                        if release_hash()!=package_sha:raise ValueError('release changed during experiment')
                        if args.action=='small':
                            if len(summary.get('cases',[]))!=4:raise ValueError('partial small panel')
                            for row in summary['cases']:
                                for name in ('result.json','exact_tables.npz'):
                                    if not (out/row['name']/name).is_file():raise ValueError('missing exact evidence')
                        elif not (out/'engineering_graph.npz').exists():raise ValueError('missing engineering graph')
                        status='complete';reason='completed';code=0
                    else:
                        status=summary.get('status','failed');reason=summary.get('reason','worker_noncomplete');code=124 if status=='incomplete' else 75 if status=='stopped' else (ret or 1)
                    break
                time.sleep(.025)
        except BaseException as exc:status='failed';reason=type(exc).__name__+': '+str(exc);code=1
        finally:
            reaped=child is None;group_gone=child is None
            if child is not None:
                try:os.killpg(child.pid,signal.SIGTERM)
                except ProcessLookupError:pass
                try:child.wait(timeout=min(1.,max(.05,activated+cap-time.monotonic()-1)))
                except subprocess.TimeoutExpired:
                    try:os.killpg(child.pid,signal.SIGKILL)
                    except ProcessLookupError:pass
                    child.wait(timeout=1.)
                reaped=True
                try:os.killpg(child.pid,0)
                except ProcessLookupError:group_gone=True
            if not reaped or not group_gone:status='incomplete';reason='cleanup_incomplete';code=124
            record=dict(status=status,reason=reason,exit_code=code,child_reaped=reaped,owned_group_gone=group_gone,
                inclusive_wall_seconds=time.monotonic()-activated,cap_seconds=cap,SIGKILL_catchable=False)
            atomic(out/'controller.json',record)
            for sig,handler in old.items():signal.signal(sig,handler)
        print(json.dumps(record));return code
    if args.deadline is None:raise ValueError('worker requires owned parent deadline')
    for _ in range(20):
        if (out/'child.json').exists():break
        time.sleep(.025)
    if not (out/'child.json').exists() or json.loads((out/'child.json').read_text())['pid']!=os.getpid():raise ValueError('owned child required')
    stopped=[]
    for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP):signal.signal(sig,lambda s,f:stopped.append(s))
    def check():
        if stopped or (out/'STOP').exists():raise InterruptedError('safe stop requested')
        if time.monotonic()>=args.deadline:raise TimeoutError('inclusive work deadline; cleanup reserve remains')
    try:
        if args.action=='small':
            from joint_decode.experiments import small
            result=small(out,check)
        else:
            from joint_decode.imdb import run
            result=run(out,args.deadline,check)
        check();atomic(out/'summary.json',result)
        return 0 if result['status']=='complete' else 124 if result['status']=='incomplete' else 1
    except BaseException as exc:
        result=dict(status='incomplete' if isinstance(exc,TimeoutError) else 'stopped' if isinstance(exc,(InterruptedError,KeyboardInterrupt)) else 'failed',
                    reason=type(exc).__name__+': '+str(exc),scope='engineering only',model_calls=0,GPU_use=0,scientific_chains=0)
        atomic(out/'summary.json',result);print(json.dumps(result),flush=True)
        return 124 if result['status']=='incomplete' else 75 if result['status']=='stopped' else 1


if __name__=='__main__':sys.exit(main())
