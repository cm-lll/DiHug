"""Bounded failure-payload recovery and cost/audit, SAME original ledger."""
import hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;OUT=Path('/data2/lyh/gnn_project/DiHug/results/role_logscale_20261021_v1')
def main():
    ledger=json.loads((OUT/'activation.json').read_text());jobs=[];children=[];stop=[False]
    def sig(s,f):stop[0]=True
    for s in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP):signal.signal(s,sig)
    def kill(p):
        if p.poll() is None:
            try:os.killpg(p.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            try:p.wait(timeout=3)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL)
        p.wait()
    def batch(tasks,seconds):
        active=[]
        try:
            for label,cmd in tasks:
                log=(OUT/(label+'.log')).open('w');p=subprocess.Popen([sys.executable,'-u',*map(str,cmd)],stdout=log,stderr=subprocess.STDOUT,
                  env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'),start_new_session=True)
                active.append((p,log,label,time.monotonic()));children.append(p)
            while any(p.poll() is None for p,_,_,_ in active):
                if stop[0] or (OUT/'STOP').exists() or time.time()>ledger['deadline_epoch']-120:raise InterruptedError('original budget/STOP')
                for p,_,label,t in active:
                    if p.poll() is None and time.monotonic()-t>seconds:raise TimeoutError(label)
                time.sleep(.2)
        finally:
            for p,log,label,t in active:
                kill(p);log.close();jobs.append(dict(job=label,exit=p.returncode,seconds=time.monotonic()-t))
        assert all(p.returncode==0 for p,_,_,_ in active),jobs
    status='failed';reason=None
    try:
        assert json.loads((OUT/'controller.json').read_text())['status']=='completed'
        files=['failure_evidence.py','small_cost.py','analyse.py','supplement.py','adapter.py','test_additional.py']
        rec=dict(time=time.time(),only_logging_cost_audit_extension=True,source={n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in files},original_scientific_files_unchanged=True)
        (OUT/'supplement_freeze.json').write_text(json.dumps(rec,indent=2)+'\n')
        batch([(f'failure_{s}',[HERE/'failure_evidence.py','--seed',s]) for s in (2026102011,2026102012,2026102013)],600)
        batch([('small_cost',[HERE/'small_cost.py']),('analysis_final',[HERE/'analyse.py'])],120)
        status='completed'
    except BaseException as exc:reason=repr(exc)
    finally:
        for p in children:kill(p)
        (OUT/'supplement_receipt.json').write_text(json.dumps(dict(status=status,reason=reason,jobs=jobs,
          elapsed_seconds=time.time()-ledger['activation_epoch'],budget_reset=False,new_independent_samples=0),indent=2)+'\n')
    return int(status!='completed')
if __name__=='__main__':sys.exit(main())
