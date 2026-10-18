#!/usr/bin/env python3
"""900-second inclusive supervised jobs; no neural calls or BW connection."""
import argparse
import json
import os
import sys
import time
from pathlib import Path


def check_release(check=lambda:None):
    from multiscale_role.runtime_guard import checked_sha
    root=Path(__file__).resolve().parent
    for line in (root/'DYNAMIC_SHA256SUMS').read_text().splitlines():
        check();h,name=line.split('  ',1);path=(root/name).resolve()
        if root not in path.parents or checked_sha(path,check)!=h:raise ValueError('release binding differs: '+name)


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['pack','check','b1','six'])
    p.add_argument('--export');p.add_argument('--short-run');p.add_argument('--portable');p.add_argument('--portable-sha256')
    p.add_argument('--b1-receipt');p.add_argument('--b1-receipt-sha256')
    p.add_argument('--output',required=True);p.add_argument('--max-seconds',type=float,default=900)
    p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    p.add_argument('--deadline',type=float,help=argparse.SUPPRESS);p.add_argument('--cleanup-deadline',type=float,help=argparse.SUPPRESS)
    return p


def main():
    started=time.monotonic();args=parser().parse_args()
    if not 20<=args.max_seconds<=900:raise ValueError('inclusive budget must be 20..900 seconds')
    if args.action=='pack':
        if not args.export or not args.short_run:raise ValueError('pack requires original export and short-run paths')
    elif not args.portable or not args.portable_sha256:raise ValueError('pinned portable manifest required')
    from multiscale_role.runtime_guard import supervise,Budget,atomic_json
    if not args.worker:
        check_release()
        # No output creation before path validation. Avoid heavy NumPy import in
        # parent so signal supervision covers numerical imports too.
        out=Path(args.output).resolve()
        inputs=[Path(__file__).resolve().parent]
        inputs += [Path(p) for p in (args.export,args.short_run) if p]
        if args.portable:inputs.append(Path(args.portable).resolve().parent)
        if args.b1_receipt:inputs.append(Path(args.b1_receipt).resolve().parent)
        if args.action=='pack':
            # Discover external referenced evidence roots BEFORE creating the
            # output/controller files. No source path is made writable.
            from dynamic_role.portable import PINS
            import hashlib
            def read_before_write(path,expected):
                if time.monotonic()>=started+args.max_seconds-12:raise TimeoutError('deadline during read-only path preflight')
                raw=Path(path).read_bytes()
                if hashlib.sha256(raw).hexdigest()!=expected:raise ValueError('sealed path-preflight input differs: '+str(path))
                return json.loads(raw)
            export=Path(args.export).resolve();short=Path(args.short_run).resolve()
            source=read_before_write(export/'six_states.json',PINS['six_states'])
            spec=read_before_write(export/'original_evaluation.json',PINS['original_evaluation'])
            receipt=read_before_write(short/'summary.json',PINS['summary'])
            def original_path(p):
                p=Path(p);return p if p.is_absolute() else export/p
            inputs += [Path(p).parent for p in receipt['input_hashes']]
            inputs.append(original_path(spec['original_artifact_source']).parent)
            for row in source['records']:
                meta=read_before_write(original_path(row['metadata']),row['metadata_sha256'])
                inputs += [original_path(row[k]).parent for k in ('checkpoint','source_manifest','config','npz','metadata')]
                inputs += [Path(meta[k]).parent for k in ('legacy_collection','legacy_report')]
        for p in inputs:
            p=p.resolve()
            if out==p or out in p.parents or p in out.parents:raise ValueError('output overlaps protected input: '+str(p))
        if out.exists():raise FileExistsError('fresh independent output required')
        out.mkdir(parents=True)
        command=[sys.executable,str(Path(__file__).resolve()),*sys.argv[1:],'--worker',
            '--deadline',str(started+args.max_seconds-12),'--cleanup-deadline',str(started+args.max_seconds-6)]
        env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
        def validate(check):
            check()
            check_release(check)
            data=json.loads((out/'summary.json').read_text())
            if data.get('status')!='complete':raise ValueError('missing/incomplete worker result')
            if args.action in ('six','check') and data.get('completed_records')!=6:raise ValueError('partial panel cannot pass')
            if args.action=='b1' and data.get('completed_records')!=1:raise ValueError('B1 evidence absent')
            return data
        result=supervise(command,out,args.max_seconds,validate,env,started=started)
        print(json.dumps(result,ensure_ascii=False));return 0 if result.get('status')=='completed' else 1
    if args.deadline is None or args.cleanup_deadline is None:raise ValueError('worker requires supervisor deadlines')
    owner=Path(args.output)/'controller_child.json'
    for _ in range(20):
        if owner.exists():break
        time.sleep(.025)
    if not owner.exists() or json.loads(owner.read_text()).get('pid')!=os.getpid():
        raise ValueError('worker requires this supervisor-owned fresh output; no direct worker execution')
    budget=Budget(args.deadline,Path(args.output)/'STOP',args.cleanup_deadline);budget.install()
    summary=dict(status='incomplete',scope=args.action,model_calls=0,training_updates=0,scientific_graphs=0,GPU_use=0)
    try:
        if args.action=='pack':
            from dynamic_role.portable import pack
            summary=pack(args.export,args.short_run,args.output,budget.check)
        else:
            from dynamic_role.runner import run
            summary=run(args,args.output,budget.check)
        budget.check();atomic_json(Path(args.output)/'summary.json',summary,budget.check)
        return 0
    except BaseException as exc:
        summary.update(status='stopped' if isinstance(exc,(KeyboardInterrupt,InterruptedError)) else 'incomplete' if isinstance(exc,TimeoutError) else 'failed',
                       error=type(exc).__name__+': '+str(exc))
        # Cleanup receipt writes are bounded separately; never promote partial
        # scientific records to complete. SIGKILL cannot run Python finally.
        try:atomic_json(Path(args.output)/'summary.json',summary,budget.cleanup_check)
        except BaseException:pass
        raise
    finally:budget.restore()


if __name__=='__main__':sys.exit(main())
