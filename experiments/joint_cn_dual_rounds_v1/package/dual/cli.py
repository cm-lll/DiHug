import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import fcntl
import numpy as np
import torch
from hsgd.bw_preflight import common as c
from hsgd.kernel import MarginalKernel
from hsgd.runtime import reverse_step
from .data import ROOT,load,inventory,risk,digest
from .io import (Control,Stopped,atomic_json,configure,optimizer,protect,verify_release,device_lock)
from .engine import run_train,sample,preflight,verify_completed_sample
from . import validation
from .metrics import graph,compare

def gate_command(args,output):
    command=[sys.executable,'-m','dual.cli','gate','--dataset',args.dataset,'--output',str(output),
             '--device',args.device,'--device-index',str(args.device_index)]
    return command

def bounded_child(command,seconds,output):
    """Entire disposable device process, with bounded cleanup and explicit receipt."""
    import signal
    out=protect(output)
    if out.exists():raise ValueError('bounded output exists')
    out.parent.mkdir(parents=True,exist_ok=True)
    proc=subprocess.Popen(command,start_new_session=True)
    handlers={};received=[];start=time.monotonic();reason=None
    for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP):
        handlers[sig]=signal.signal(sig,lambda s,f:received.append(s))
    try:
        while proc.poll() is None:
            if received or time.monotonic()-start>=seconds or (out/'STOP').exists():
                reason='signal' if received else 'STOP' if (out/'STOP').exists() else 'timeout'
                os.killpg(proc.pid,signal.SIGTERM)
                try:proc.wait(timeout=3)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                break
            time.sleep(.1)
        rc=proc.wait()
        if reason:rc=124 if reason=='timeout' else 130
        return rc
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid,signal.SIGKILL);proc.wait()
        for sig,h in handlers.items():signal.signal(sig,h)
        # No writes to protected package tree; out is a dedicated new test directory.
        out.mkdir(exist_ok=True)
        atomic_json(out/'supervisor.json',dict(exit_code=proc.returncode,reason=reason,
                    wall_seconds=time.monotonic()-start,cap_seconds=seconds,completed=proc.returncode==0 and reason is None))

def short_test(case,cfg,args):
    out=protect(args.output)
    if out.exists():raise ValueError('short test output exists')
    out.mkdir(parents=True);env=configure(args.device,False);c.seed_all(cfg['train_seed'])
    rng=np.random.default_rng(cfg['train_seed']);started=time.monotonic()
    model=c.make_model(case,cfg,args.device);opt=optimizer(model,cfg);init=time.monotonic()-started
    records=[];status='incomplete'
    try:
        with Control(out,args.max_seconds-10) as ctl:
            for j,t in enumerate([1,500,1000,20,500,1000]):
                ctl.check();s=time.monotonic()
                inp=c.train_input(case,cfg['T'],cfg['rounds'],cfg['train_seed'],j,rng,force_t=t)
                r=c.update(model,opt,case,inp,cfg['T'],args.device,backend=cfg['backend'])
                r.update(total_seconds=time.monotonic()-s,warmup=j<3);records.append(r)
                atomic_json(out/'partial.json',dict(phase='training_updates',initialization_seconds=init,updates=records))
                print(json.dumps(dict(short_update=j+1,seconds=r['total_seconds'],loss=r['loss'])),flush=True)
            model.eval();pi=np.repeat(case['prior'],case['space'].counts)
            state=(rng.random(case['space'].total)<pi).astype(np.uint8)
            with torch.no_grad():
                _,step=reverse_step(model,case,state,MarginalKernel(case['prior'],c.schedule(cfg['T'])),cfg['T'],
                    cfg['rounds'],cfg['sample_seed'],rng,args.device,check_time=ctl.check,trace=True)
            step.pop('trace')
            atomic_json(out/'partial.json',dict(phase='full_sweep_complete',initialization_seconds=init,updates=records,sweep=step))
            # Full five-time-panel cost, not a single-block extrapolation.
            val=validation.run(model,case,cfg,args.device,validation.panel(case,cfg),ctl.check)
            stable=[r['total_seconds'] for r in records[3:]]
            estimate=dict(training_12000_seconds=float(np.median(stable)*12000),
                validation_13_panels_seconds=val['seconds']*13,
                five_T1000_chains_seconds=step['step_seconds']*1000*5,
                caveat='one prior-state sweep only; trace overhead present; cost changes with Xt/CN density; IO extra')
            atomic_json(out/'costs.json',dict(environment=env,initialization_seconds=init,updates=records,
                sweep=step,validation=val,estimates=estimate,**c.memory(args.device)))
            status='completed'
    finally:
        atomic_json(out/'summary.json',dict(status=status,wall_seconds=time.monotonic()-started,
            scientific_samples=0,formal_training_updates=0,**c.memory(args.device)))
    return 0

def parser():
    p=argparse.ArgumentParser(description='Joint CN IMDB/PubMed fresh 12000-update rounds; raw outputs only')
    p.add_argument('command',choices=['inventory','train','resume','pipeline','gate','short-test','_short-worker',
            'progress','stop','sample-checkpoint','evaluate'])
    p.add_argument('--dataset',choices=['IMDB','PubMed'],required=True)
    p.add_argument('--output',type=Path);p.add_argument('--run',type=Path)
    p.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    p.add_argument('--device-index',type=int,default=0)
    p.add_argument('--fresh-start',action='store_true');p.add_argument('--resume',action='store_true')
    p.add_argument('--accept-config-provenance-gap',action='store_true')
    p.add_argument('--update',type=int);p.add_argument('--graph',type=Path)
    p.add_argument('--max-seconds',type=int,default=570)
    return p

def main(argv=None):
    args=parser().parse_args(argv);verify_release();case,cfg=load(args.dataset)
    if args.command=='inventory':
        print(json.dumps(dict(resolved_config=cfg,inventory=inventory(case),risk=risk(case)),indent=2));return 0
    if args.output is None:raise ValueError('--output required')
    args.output=protect(args.output)
    if args.command in ('progress','stop'):
        if not args.output.is_dir():raise ValueError('run does not exist')
        if args.command=='stop':(args.output/'STOP').touch();print('Safe stop requested after committed update/time step.');return 0
        for name in ('progress.json','latest.json','training_summary.json','summary.json'):
            if (args.output/name).is_file():print(name,(args.output/name).read_text())
        return 0
    if args.command=='evaluate':
        if args.output.exists():raise ValueError('evaluation output exists')
        if args.graph is None:raise ValueError('--graph required')
        with np.load(args.graph,allow_pickle=False) as z:
            if not np.array_equal(z['types'],case['space'].types) or not np.array_equal(z['schema'],case['space'].pairs):raise ValueError('graph inventory/schema mismatch')
            edges=z['edges'].T
            if not np.array_equal(z['relations']-1,case['space'].encode(edges)[0]):raise ValueError('graph labels mismatch')
        result=graph(case,edges,cfg['data']['type_names']);ref=graph(case,case['clean'],cfg['data']['type_names'])
        args.output.mkdir(parents=True);atomic_json(args.output/'evaluation.json',dict(graph=result,reference=ref,comparison=compare(result,ref),graph_sha256=c.sha(args.graph)))
        return 0
    if args.device=='cpu':raise ValueError('public real-data execution requires allocated accelerator; CPU is only used by explicit small tests')
    torch.cuda.set_device(args.device_index)
    print(json.dumps(dict(device_index=args.device_index,device=torch.cuda.get_device_name(args.device_index),
         visible_devices=torch.cuda.device_count(),visibility={k:os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES')})),flush=True)
    if args.command=='short-test':
        if not 30<=args.max_seconds<=570:raise ValueError('short test cap must be 30..570 seconds including startup')
        command=[sys.executable,'-m','dual.cli','_short-worker','--dataset',args.dataset,'--output',str(args.output),
            '--device-index',str(args.device_index),'--max-seconds',str(args.max_seconds)]
        with device_lock(args.output.parent):
            return bounded_child(command,args.max_seconds,args.output)
    if args.command=='_short-worker':return short_test(case,cfg,args)
    if args.command=='gate':preflight(case,cfg,args.output,args.device);return 0
    if args.command=='sample-checkpoint':
        if args.run is None or args.update is None:raise ValueError('--run and --update required')
        with device_lock(args.run.parent):
            return 0 if sample(case,cfg,args.run,args.output,args.update,args.device,args.resume) else 2
    # One process per allocated device for the entire train-then-sample pipeline.
    with device_lock(args.output.parent):
        resume=args.resume or args.command=='resume'
        if not resume:
            if not args.fresh_start:raise ValueError('fresh launch requires --fresh-start')
            if args.output.exists():raise ValueError('fresh run exists; refused before gate')
            gate=args.output.with_name(args.output.name+'.gate')
            rc=bounded_child(gate_command(args,gate),570,gate)
            if rc:return rc
            receipt=json.loads((gate/'gate.json').read_text())
            if (receipt['status']!='passed' or receipt['engine_updates']!=2 or receipt['parameter_error']!=0
                or receipt['optimizer_error']!=0 or not receipt['binding']['runtime']['deterministic_algorithms']
                or c.sha(gate/'gate.pt')!=receipt['checkpoint_sha256']):raise ValueError('strict gate evidence invalid')
        done=run_train(case,cfg,args.output,args.device,fresh=args.fresh_start,resume=resume,
                       accept_gap=args.accept_config_provenance_gap)
        if not done:return 2
        if args.command=='pipeline':
            for update in cfg['evaluation_updates']:
                target=args.output/'samples'/('u%06d'%update)
                if target.exists() and (target/'summary.json').exists():
                    receipt=json.loads((target/'summary.json').read_text())
                    if receipt['status']=='completed':
                        verify_completed_sample(case,cfg,args.output,target,update)
                        continue
                if not sample(case,cfg,args.output,target,update,args.device,resume=target.exists()):return 2
        return 0

if __name__=='__main__':
    try:sys.exit(main())
    except Exception as exc:
        print('FAILED: '+repr(exc),file=sys.stderr,flush=True);raise
