"""Manual bounded repair entry; no model/chain jobs. Worker is internal only."""
import time
ENTRY_STARTED=time.monotonic()
import argparse
import json
import os
from pathlib import Path
import sys
import uuid
from multiscale_role.runtime_guard import supervise,checked_sha,atomic_json

ROOT=Path(__file__).resolve().parent


def validate_result(action,out,check):
    check();out=Path(out)
    if action=='test':
        import xml.etree.ElementTree as ET
        suites=list(ET.parse(out/'tests.xml').getroot().iter('testsuite'))
        if sum(int(s.get('tests',0)) for s in suites)<85 or any(int(s.get(k,0)) for s in suites for k in ('errors','failures','skipped')):
            raise ValueError('missing/incomplete regression evidence')
        return
    summary=json.loads((out/'summary.json').read_text())
    expected={'short':'complete','export-legacy':'exported','cost':'completed_synthetic_engineering_cost'}[action]
    if summary.get('status')!=expected or summary.get('input_hash_verification_complete') is not True:
        raise ValueError('worker result/evidence incomplete')
    if action=='short':
        if summary.get('completed_records')!=6 or len(summary.get('records',[]))!=6 or summary.get('input_hashes_unchanged') is not True:
            raise ValueError('six-case final integrity evidence missing')
        for i,row in enumerate(summary['records']):
            check();saved=json.loads((out/('case_%02d.json'%i)).read_text())
            if saved!=row or not row.get('legacy_baseline_alignment',{}).get('passed'):
                raise ValueError('case output/alignment missing')
        for name in ('frozen_objective.json','policy.json'):json.loads((out/name).read_text())
    elif action=='export-legacy':
        if len(summary.get('records',[]))!=6 or summary.get('input_hashes_unchanged') is not True:
            raise ValueError('six-case conversion evidence missing')
        for name,key in (('six_states.json','manifest_sha256'),('original_evaluation.json','evaluation_sha256')):
            if checked_sha(out/name,check)!=summary.get(key):raise ValueError('converted manifest/evaluation missing or changed')
        manifest=json.loads((out/'six_states.json').read_text())
        if len(manifest.get('records',[]))!=6:raise ValueError('missing converted records')
        for row in manifest['records']:
            for key in ('npz','metadata'):
                if checked_sha(out/row[key],check)!=row[key+'_sha256']:raise ValueError('converted evidence missing or changed')
    else:
        if not summary.get('cold_warm_results_exact') or len(summary.get('passes',[]))!=2 or not summary.get('reference_static_work',{}).get('channels'):
            raise ValueError('cost evidence incomplete')
        for name in ('cold.json','warm_reuse.json'):
            if not json.loads((out/name).read_text()).get('timings'):raise ValueError('missing phase costs')
    check()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=['test','short','export-legacy','cost'])
    p.add_argument('--output',required=True);p.add_argument('--seconds',type=int,default=900)
    p.add_argument('--manifest');p.add_argument('--manifest-sha256');p.add_argument('--evaluation');p.add_argument('--evaluation-sha256')
    p.add_argument('--legacy-zip');p.add_argument('--legacy-run');p.add_argument('--legacy-ledger-sha256')
    p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    p.add_argument('--deadline',type=float,help=argparse.SUPPRESS)
    p.add_argument('--cleanup-deadline',type=float,help=argparse.SUPPRESS)
    a=p.parse_args()
    if not 10<=a.seconds<=900:p.error('single invocation must be 10..900 seconds including cleanup')
    out=Path(a.output).resolve()
    if a.worker:
        marker=json.loads((out/'activation.json').read_text())
        if os.environ.get('DIHUG_WORKER_TOKEN')!=marker.get('worker_token') or a.deadline is None:
            p.error('worker may only be launched by the bounded controller')
        if a.action=='short':
            from multiscale_role.short import run
            return run(a.manifest,a.manifest_sha256,a.evaluation,a.evaluation_sha256,out,a.deadline,a.cleanup_deadline,owned_output=True)
        if a.action=='export-legacy':
            from multiscale_role.legacy import export_run
            return export_run(a.legacy_zip,a.legacy_run,a.legacy_ledger_sha256,out,a.deadline,a.cleanup_deadline)
        if a.action=='cost':
            from multiscale_role.cost import run_cost
            return run_cost(out,a.deadline,a.cleanup_deadline)
        p.error('test has no internal worker')
    if out.exists():p.error('fresh independent output required')
    if a.action=='short' and not all((a.manifest,a.manifest_sha256,a.evaluation,a.evaluation_sha256)):
        p.error('pinned converted manifest and evaluation required')
    if a.action=='export-legacy' and not all((a.legacy_zip,a.legacy_run,a.legacy_ledger_sha256)):
        p.error('original ZIP, actual results root and pinned original evidence ledger required')
    out.mkdir(parents=True,exist_ok=False)
    deadline=ENTRY_STARTED+a.seconds;token=uuid.uuid4().hex
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1',DIHUG_WORKER_TOKEN=token)
    if a.action=='test':
        command=[sys.executable,'-m','pytest','-q','tests','--disable-warnings','--junitxml='+str(out/'tests.xml')]
    else:
        command=[sys.executable,'-u',str(Path(__file__).resolve()),a.action,'--worker','--output',str(out),
                 '--seconds',str(a.seconds),'--deadline',str(deadline-12),'--cleanup-deadline',str(deadline-4)]
        for field in ('manifest','manifest_sha256','evaluation','evaluation_sha256','legacy_zip','legacy_run','legacy_ledger_sha256'):
            value=getattr(a,field)
            if value:command+=['--'+field.replace('_','-'),value]
    def preflight(check):
        checks=ROOT/'SHA256SUMS'
        if not checks.is_file():raise ValueError('bundle manifest missing')
        for line in checks.read_text().splitlines():
            check();expected,name=line.split('  ',1)
            if checked_sha(ROOT/name,check)!=expected:raise ValueError('bundle hash mismatch: '+name)
        atomic_json(out/'activation.json',dict(action=a.action,worker_token=token,controller_pid=os.getpid(),
            absolute_old_deadline=None,single_run_seconds=a.seconds),check)
    def validate(check):
        validate_result(a.action,out,check)
    return supervise(command,out,a.seconds,validate,env,ENTRY_STARTED,preflight)


if __name__=='__main__':raise SystemExit(main())
