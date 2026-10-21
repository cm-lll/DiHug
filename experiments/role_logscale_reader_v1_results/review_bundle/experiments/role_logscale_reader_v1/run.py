"""Standard-library parent: original60min ledger, own process-group cleanup."""
import argparse,hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
OUT=Path('/data2/lyh/gnn_project/DiHug/results/role_logscale_20261021_v1')
def read(p):return json.loads(p.read_text())
def save(p,x):q=p.with_suffix('.tmp');q.write_text(json.dumps(x,indent=2)+'\n');q.replace(p)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--execute',action='store_true');a=ap.parse_args()
    if not a.execute:ap.error('explicit --execute required; only diagnostics')
    ledger=read(OUT/'activation.json');frozen=read(OUT/'freeze.json')
    assert all(sha(HERE/n)==h for n,h in frozen['source'].items())
    with (OUT/'launch.json').open('x') as f:json.dump(dict(time=time.time(),activation_sha256=sha(OUT/'activation.json'),freeze_sha256=sha(OUT/'freeze.json')),f)
    jobs=[];children=[];stop=[None]
    def handler(sig,frame):stop[0]=signal.Signals(sig).name
    for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP):signal.signal(sig,handler)
    def cleanup(p):
        if p.poll() is None:
            try:os.killpg(p.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            try:p.wait(timeout=3)
            except subprocess.TimeoutExpired:
                try:os.killpg(p.pid,signal.SIGKILL)
                except ProcessLookupError:pass
        p.wait()
    def batch(commands,cap):
        active=[]
        try:
            for label,args in commands:
                log=(OUT/f'{label}.log').open('w');p=subprocess.Popen([sys.executable,'-u',*map(str,args)],env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                x=dict(label=label,p=p,log=log,start=time.monotonic());active.append(x);children.append(p)
            tick=0
            while any(x['p'].poll() is None for x in active):
                if stop[0] or (OUT/'STOP').exists() or time.time()>ledger['deadline_epoch']-120:raise InterruptedError(stop[0] or 'STOP/deadline')
                for x in active:
                    if x['p'].poll() is None and time.monotonic()-x['start']>cap:cleanup(x['p']);x['timeout']=True
                if time.monotonic()-tick>20:print('progress',[(x['label'],x['p'].poll()) for x in active],flush=True);tick=time.monotonic()
                time.sleep(.2)
        finally:
            for x in active:
                cleanup(x['p']);x['log'].close();jobs.append(dict(job=x['label'],exit_code=x['p'].returncode,timeout=x.get('timeout',False),seconds=time.monotonic()-x['start']))
            save(OUT/'jobs.json',jobs)
        return all(x['p'].returncode==0 for x in active)
    status='failed';reason=None
    try:
        if not batch([(f'diag_{s}',[HERE/'diagnose.py','--seed',s]) for s in (2026102011,2026102012,2026102013)],900):raise RuntimeError('diagnostic child failure; inspect receipts')
        batch([(f'large_{n}',[HERE/'large.py',n]) for n in ('IMDB','PubMed')],150)
        status='completed'
    except BaseException as exc:reason=repr(exc)
    finally:
        for p in children:cleanup(p)
        save(OUT/'controller.json',dict(status=status,reason=reason,jobs=jobs,elapsed_seconds=time.time()-ledger['activation_epoch'],budget_seconds=3600,
            new_independent_samples=0,training_updates=0,GPU_used=False,all_children_reaped=True))
    return int(status!='completed')
if __name__=='__main__':sys.exit(main())
