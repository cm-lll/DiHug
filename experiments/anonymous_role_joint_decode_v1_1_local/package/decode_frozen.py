#!/usr/bin/env python3
"""One explicit local unguided A1/B1 fixed-matching MAP. CPU <=300s, no model."""
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


def paths(args):
    # ALL checks precede mkdir, logs and activation. Resolve symlinks as well.
    manifest=Path(args.input).resolve()
    if manifest.is_dir():manifest=manifest/'manifest.json'
    out=Path(args.output).resolve();protected=[manifest.parent,ROOT]
    # Recognized repository layout only; an unrelated /tmp/.git must not make
    # every /tmp output forbidden after stand-alone extraction.
    if ROOT.parent.parent.name=='experiments' and (ROOT.parents[2]/'.git').exists():protected.append(ROOT.parents[2])
    for name in ('original_graph','original_receipt'):
        if getattr(args,name):protected.append(Path(getattr(args,name)).resolve())
    if out.exists() or any(out==p or p in out.parents or out in p.parents for p in protected):
        raise ValueError('fresh output must be disjoint from input and source roots')
    return manifest,out


def worker(args,out,deadline):
    import resource
    import platform
    import numpy as np
    import scipy
    from joint_decode.frozen_input import Frozen,array_sha,POSTERIOR_ATOL
    from joint_decode.role_assignment import match
    from joint_decode.decoder import decode
    from joint_decode.structure import evaluate,differences
    stopped=[]
    for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP):signal.signal(sig,lambda s,f:stopped.append(s))
    start=time.monotonic();cpu=time.process_time();times={}
    def check():
        if stopped or (out/'STOP').exists():raise InterruptedError('STOP/signal')
        if time.monotonic()>=deadline:raise TimeoutError('inclusive local CPU work deadline')
    def timed(name,fn):
        check();tick=time.monotonic();value=fn();times[name]=time.monotonic()-tick;check();return value
    try:
        frozen=timed('input_verification',lambda:Frozen(args.input,args.manifest_sha256,args.arm,check))
        original=None
        if args.original_graph:
            original=timed('original_verification',lambda:frozen.original(args.original_graph,args.original_sha256,args.original_receipt,args.original_receipt_sha256))
        assigned,matching=timed('matching',lambda:match(frozen.schema,frozen.target,frozen.p0,check))
        bits,solver=timed('decoding',lambda:decode(frozen.schema,frozen.target,assigned,frozen.law,seconds=max(.001,deadline-time.monotonic()),check=check,
              event=lambda row:print(json.dumps(dict(phase='relation',family=row['family'],status=row['status'])),flush=True)))
        if bits is None or solver['status']!='optimal_fixed_matching':
            atomic(out/'summary.json',dict(status='failed' if bits is None else 'incomplete',reason='fixed matching decoder not certified optimal',solver=solver))
            return 1
        decoded=timed('decoded_evaluation',lambda:evaluate(frozen.schema,bits,frozen.semantic_types,frozen.law,check))
        baseline=timed('original_evaluation',lambda:evaluate(frozen.schema,original,frozen.semantic_types,frozen.law,check)) if original is not None else None
        comparison=differences(frozen.schema,original,bits) if original is not None else None
        check();tick=time.monotonic();graph=out/'decoded_graph.npz'
        with graph.with_suffix('.npz.tmp').open('xb') as f:
            np.savez_compressed(f,types=frozen.schema.types,edges=frozen.schema.slots[:,bits.astype(bool)],
                families=frozen.schema.families[bits.astype(bool)],assigned_joint_degrees=assigned,
                actual_joint_degrees=frozen.schema.degree(bits));f.flush();os.fsync(f.fileno())
        check();os.replace(graph.with_suffix('.npz.tmp'),graph);times['graph_save']=time.monotonic()-tick
        timed('final_input_rehash',frozen.rehash);check()
        from run import release_hash
        release=release_hash();check()
        report=dict(status='complete',arm=args.arm,t=1,seed=frozen.meta['seed'],source_kind=frozen.receipt['source_kind'],
            source_limit=frozen.source_limit,scope='ONE fixed JOINT role matching MAP, not anonymous global MAP, not a sampler',
            matching=matching,solver=solver,checks=dict(schema=True,support=solver['exact_support'],
                assigned_degrees=solver['exact_assigned_degrees'],anonymous_joint_frequencies=solver['exact_joint_role_multiset']),
            schema=dict(nodes=frozen.schema.n,type_counts=frozen.schema.sizes,pairs=frozen.schema.pairs.tolist(),
                slots=frozen.schema.s,canonical_slots_sha256=array_sha(frozen.schema.slots),families_sha256=array_sha(frozen.schema.families),
                input_to_canonical_sha256=array_sha(frozen.mapping)),semantic_types=frozen.semantic_types,
            source_receipt={k:frozen.receipt[k] for k in ('weight_sha256','source_manifest_sha256','config_sha256','reference_sha256','alpha_sha256')},
            posterior_absolute_tolerance=POSTERIOR_ATOL,
            structural_fixed0=int((frozen.support==0).sum()),structural_fixed1=int((frozen.support==1).sum()),
            query_sha256=frozen.meta['query_sha256'],q_p0_max_error=frozen.q_error,numerical_saturated_free_slots=frozen.saturated,
            score_semantics=frozen.meta['score_semantics'],decoded=decoded,original=baseline,changes_by_family=comparison,
            missing_comparisons=[] if original is not None else ['no original graph: original quality, edge edits and paired scores unavailable'],
            input_hashes={str(p):h for p,h in frozen.hashes.items()},output_sha256=hashlib.sha256(graph.read_bytes()).hexdigest(),
            release_checklist_sha256=release,times=times,worker_wall_seconds=time.monotonic()-start,worker_cpu_seconds=time.process_time()-cpu,
            peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
            environment=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,CPU_threads=1),
            model_calls=0,training_updates=0,GPU_use=0,scientific_chains=0,no_automatic_followup=True)
        check();atomic(out/'summary.json',report);check();return 0
    except BaseException as exc:
        status='incomplete' if isinstance(exc,TimeoutError) else 'stopped' if isinstance(exc,(InterruptedError,KeyboardInterrupt)) else 'failed'
        atomic(out/'summary.json',dict(status=status,reason=type(exc).__name__+': '+str(exc),times=times,real_input_executed=False if 'frozen' not in locals() else frozen.receipt['source_kind']!='synthetic_fixture'))
        print(status,type(exc).__name__,str(exc),flush=True);return 124 if status=='incomplete' else 75 if status=='stopped' else 1


def main():
    start=time.monotonic();p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',required=True,help='explicit local standard manifest or its root; never searches elsewhere')
    p.add_argument('--manifest-sha256',required=True);p.add_argument('--arm',required=True,choices=['A','B'])
    p.add_argument('--output',required=True);p.add_argument('--max-seconds',type=float,default=300.)
    for name in ('original-graph','original-sha256','original-receipt','original-receipt-sha256'):p.add_argument('--'+name)
    p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS);p.add_argument('--deadline',type=float,help=argparse.SUPPRESS)
    a=p.parse_args()
    if not 1<=a.max_seconds<=300:raise ValueError('inclusive bound must be1..300 seconds')
    original=[a.original_graph,a.original_sha256,a.original_receipt,a.original_receipt_sha256]
    if any(original) and not all(original):raise ValueError('all four original graph/receipt arguments are required together')
    if a.worker:
        out=Path(a.output).resolve()
        for _ in range(20):
            if (out/'child.json').exists():break
            time.sleep(.025)
        owner=json.loads((out/'child.json').read_text())
        if owner['pid']!=os.getpid() or a.deadline is None:raise ValueError('worker requires owned parent')
        return worker(a,out,a.deadline)
    manifest,out=paths(a);a.input=str(manifest)
    from run import release_hash
    release=release_hash();out.mkdir(parents=True);atomic(out/'activation.json',dict(max_seconds=a.max_seconds,
         input=str(manifest),expected_manifest_sha256=a.manifest_sha256,arm=a.arm,release_checklist_sha256=release))
    old={};caught=[];child=None;status='failed';reason='parent failure';code=1
    reserve=min(5.,a.max_seconds*.2);deadline=start+a.max_seconds-reserve
    for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP):old[sig]=signal.signal(sig,lambda s,f:caught.append(s))
    try:
        env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
        command=[sys.executable,str(Path(__file__).resolve()),*sys.argv[1:],'--input',str(manifest),'--worker','--deadline',str(deadline)]
        with (out/'worker.log').open('w') as log:
            child=subprocess.Popen(command,env=env,start_new_session=True,stdout=log,stderr=subprocess.STDOUT)
            atomic(out/'child.json',dict(pid=child.pid,pgid=child.pid))
            while True:
                if caught or (out/'STOP').exists():status='stopped';reason='STOP' if not caught else signal.Signals(caught[0]).name;code=75;break
                if time.monotonic()>=deadline:status='incomplete';reason='wall_timeout';code=124;break
                ret=child.poll()
                if ret is not None:
                    summary=json.loads((out/'summary.json').read_text()) if (out/'summary.json').exists() else {}
                    status=summary.get('status','failed');reason=summary.get('reason','worker returned')
                    evidence=summary.get('checks',{})
                    graph=out/'decoded_graph.npz'
                    valid=bool(evidence) and all(evidence.values()) and graph.is_file()
                    valid=valid and hashlib.sha256(graph.read_bytes()).hexdigest()==summary.get('output_sha256')
                    code=0 if ret==0 and status=='complete' and valid and release_hash()==release else (ret or 1)
                    if code and status=='complete':status='failed';reason='missing/changed completion evidence'
                    break
                time.sleep(.025)
    except BaseException as exc:status='failed';reason=type(exc).__name__+': '+str(exc);code=1
    finally:
        gone=child is None
        if child is not None:
            try:os.killpg(child.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            try:child.wait(timeout=max(.02,min(1.,start+a.max_seconds-time.monotonic()-.1)))
            except subprocess.TimeoutExpired:
                try:os.killpg(child.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                child.wait(timeout=.1)
            try:os.killpg(child.pid,0)
            except ProcessLookupError:gone=True
        if not gone or time.monotonic()>start+a.max_seconds:code=124;status='incomplete';reason='cleanup/deadline'
        if code and (out/'summary.json').exists():
            summary=json.loads((out/'summary.json').read_text());summary.update(status=status,controller_reason=reason);atomic(out/'summary.json',summary)
        receipt=dict(status=status,reason=reason,exit_code=int(code),owned_group_gone=gone,wall_seconds=time.monotonic()-start,
             cap_seconds=a.max_seconds,SIGKILL_catchable=False)
        atomic(out/'controller.json',receipt)
        for sig,handler in old.items():signal.signal(sig,handler)
    print(json.dumps(receipt));return int(code)


if __name__=='__main__':sys.exit(main())
