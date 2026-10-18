"""Bounded engineering tests / saved-probability cost. Never a scientific chain."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['test','inventory','cost'])
    p.add_argument('--output',required=True)
    p.add_argument('--max-seconds',type=int,default=480)
    p.add_argument('--input',help='trusted saved NPZ: state,p0,t; no synthetic or regenerated scientific input')
    p.add_argument('--expected-sha256',help='user verified saved NPZ hash')
    args=p.parse_args()
    if not 1<=args.max_seconds<=570: p.error('bounded check requires 1..570 seconds')
    out=Path(args.output)
    out.mkdir(parents=True,exist_ok=False)
    def timeout(*_):raise TimeoutError('bounded engineering wall limit reached')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(args.max_seconds)
    start,cpu=time.perf_counter(),time.process_time()
    from hsgd.bw_preflight import common as c
    import torch
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(False,warn_only=False)
    report=dict(status='incomplete',action=args.action,environment=c.environment('cpu'),
                GPU_use=0,model_calls=0,optimizer_updates=0,scientific_graphs=0)
    def resources():
        child=resource.getrusage(resource.RUSAGE_CHILDREN)
        own=time.process_time()-cpu
        return dict(cpu_seconds_self=own,cpu_seconds_children=child.ru_utime+child.ru_stime,
                    cpu_seconds_total=own+child.ru_utime+child.ru_stime,
                    peak_rss_mib_self=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.,
                    peak_rss_mib_child_max=child.ru_maxrss/1024.,
                    memory_scope='per-process peaks, not simultaneous process-tree memory')
    try:
        if args.action=='test':
            command=[sys.executable,'-m','pytest','-q','tests','--disable-warnings','--junitxml='+str(out/'tests.xml')]
            with (out/'tests.log').open('w') as f:
                child=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
                try:code=child.wait(timeout=max(1,args.max_seconds-(time.perf_counter()-start)))
                except BaseException:
                    os.killpg(child.pid,signal.SIGTERM)
                    try:child.wait(timeout=3)
                    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                    raise
            report.update(status='passed' if code==0 else 'failed',exit_code=code,
                          model_calls='small CPU engineering fixtures only; see test scopes',
                          scientific_graphs=0,optimizer_updates=0)
            if code:raise RuntimeError('CPU regression failure; see tests.log')
        else:
            from role_degree.role_statistics import joint_profile
            from stepwise_role.core import Layout,freeze,StepwiseGuide
            from hsgd.kernel import MarginalKernel
            case=c.imdb_case(c.PACKAGE/'data/IMDB_movie_reference.npz',c.config())
            l=Layout.canonical(case['space'].types,case['space'].pairs,case['prior'])
            f=case['space'].encode(case['clean'])[0]
            profile=joint_profile(l.types,case['clean'],f,l.pairs)
            tick=time.perf_counter();frozen=freeze(profile,l,c.schedule(1000));g=StepwiseGuide(l,frozen)
            report.update(target_initialization_seconds=time.perf_counter()-tick,nodes=len(l.types),
                          reference_edges=case['clean'].shape[1],all_slots=len(l.families),
                          active_slots=int(case['space'].counts[case['active']].sum()),
                          target_shape={key:list(value.shape) for key,value in g.targets.items()},
                          guidance_binding=g.binding(),target_CF_bytes=sum(a.nbytes for a in g.targets.values()))
            c.json_save(out/'frozen_guidance.json',frozen)
            if args.action=='cost':
                if not args.input or not args.expected_sha256:raise ValueError('cost requires a saved complete input AND verified hash')
                path=Path(args.input)
                if hashlib.sha256(path.read_bytes()).hexdigest()!=args.expected_sha256:raise ValueError('saved input hash mismatch')
                with np.load(path,allow_pickle=False) as z:
                    if not {'state','p0','t','data_sha256','alpha','source_record'}.issubset(z.files):
                        raise ValueError('saved input lacks required provenance/arrays; do not regenerate to fill')
                    state=z['state'].copy();p0=z['p0'].copy();t=int(z['t'])
                    if str(z['data_sha256'].item())!=c.config()['imdb_sha256'] or not np.array_equal(z['alpha'],g.alpha):
                        raise ValueError('saved input data/T1000 schedule differs')
                    source=str(z['source_record'].item())
                if not source.strip():raise ValueError('empty saved input source record')
                if state.shape!=p0.shape or not np.isin(state,[0,1]).all():raise ValueError('saved frozen state invalid')
                print(json.dumps(dict(event='statistic_cost_start',t=t,slots=len(p0))),flush=True)
                _,_,result=g.apply(p0,state,MarginalKernel(case['prior'],g.alpha),t)
                report.update(status='measured_probability_only',guidance=result,input_sha256=args.expected_sha256,
                              input_source_record=source,provenance_boundary='source text recorded, not independently reconstructed',
                              probability_collection_seconds=None,collection_reason='not included: pre-saved p0 only',
                              GPU_inference_not_measured=True)
                if result['status']=='numerical_failure':raise FloatingPointError(result['reason'])
            else:report['status']='inventory_only_not_model_cost'
        report.update(wall_seconds=time.perf_counter()-start,**resources())
        c.json_save(out/'summary.json',report)
        print(json.dumps(report,ensure_ascii=False),flush=True)
    except BaseException as exc:
        report.update(status='incomplete' if isinstance(exc,(TimeoutError,KeyboardInterrupt)) else 'failed',
                      exception=type(exc).__name__,reason=str(exc),wall_seconds=time.perf_counter()-start,**resources())
        c.json_save(out/'summary.json',report)
        raise
    finally:signal.alarm(0)


if __name__=='__main__':main()
