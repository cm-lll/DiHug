"""Manual single-case diagnostic; never a model, training or sampling entry."""
import time
START=time.monotonic()
import argparse
import json
import os
from pathlib import Path
import sys
sys.dont_write_bytecode=True
import uuid
from multiscale_role.runtime_guard import supervise,atomic_json,checked_sha
from multiscale_role.entry_guard import test_environment
from b1_attribution.entry import SHORT,EXPORT,admit,inspect_inputs,validate_done
ROOT=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['inspect','test','diagnose'])
    p.add_argument('--short-run',default=SHORT);p.add_argument('--export',default=EXPORT)
    p.add_argument('--summary-sha256');p.add_argument('--case-sha256');p.add_argument('--output')
    p.add_argument('--seconds',type=int,default=900)
    p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    p.add_argument('--deadline',type=float,help=argparse.SUPPRESS);p.add_argument('--cleanup-deadline',type=float,help=argparse.SUPPRESS)
    a=p.parse_args()
    if not 15<=a.seconds<=900:p.error('single invocation max900s includes imports/verification/calculation/IO/cleanup')
    def check():
        if time.monotonic()>=START+a.seconds-12:raise TimeoutError('admission/inspection budget')
    if a.action=='inspect':
        if a.output or a.worker:p.error('inspection is stdout only; no output directory')
        result=inspect_inputs(a.short_run,a.export,check);print(json.dumps(result,indent=2))
        return 0 if result['status']=='files_present_not_authenticated' else 2
    if not a.output:p.error('--output required')
    out=Path(a.output).resolve()
    if a.worker:
        marker=json.loads((out/'activation.json').read_text())
        if os.environ.get('B1_WORKER_TOKEN')!=marker['worker_token'] or a.deadline is None:p.error('internal bounded worker only')
        if a.action=='diagnose':
            from b1_attribution.runner import run
            return run(a,out,a.deadline,a.cleanup_deadline)
        from b1_attribution.testing import run_tests
        return run_tests(ROOT,out)
    if out.exists():p.error('fresh output required')
    try:
        if a.action=='diagnose':admit(out,ROOT,a.short_run,a.export,a.summary_sha256,a.case_sha256,check)
        else:
            from multiscale_role.entry_guard import validate_output_path
            validate_output_path('test',out,ROOT,check=check)
    except (ValueError,OSError,KeyError,TypeError) as exc:p.error(str(exc))
    out.mkdir(parents=True,exist_ok=False);deadline=START+a.seconds;token=uuid.uuid4().hex
    env=test_environment(dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='',
                              PYTHONUNBUFFERED='1',B1_WORKER_TOKEN=token))
    command=[sys.executable,str(ROOT/'run_b1.py'),a.action,'--worker','--output',str(out),'--seconds',str(a.seconds),
             '--deadline',str(deadline-12),'--cleanup-deadline',str(deadline-4),
             '--short-run',a.short_run,'--export',a.export]
    for key in ('summary_sha256','case_sha256'):
        if getattr(a,key):command+=['--'+key.replace('_','-'),getattr(a,key)]
    def preflight(c):
        for line in (ROOT/'B1_SHA256SUMS').read_text().splitlines():
            c();h,name=line.split('  ',1)
            if checked_sha(ROOT/name,c)!=h:raise ValueError('B1 deployment source differs: '+name)
        atomic_json(out/'activation.json',dict(worker_token=token,controller_pid=os.getpid(),action=a.action,
                    base_commit='7049e64b108f2ef434c40815457c74037c4455eb',seconds=a.seconds,
                    source_scope='no training/model calls/graph draws; B1 cached probabilities only'),c)
    def validate(c):
        if a.action=='diagnose':validate_done(out,c)
        else:
            from b1_attribution.testing import validate_tests
            validate_tests(ROOT,out);c()
    return supervise(command,out,a.seconds,validate,env,START,preflight)


if __name__=='__main__':raise SystemExit(main())
