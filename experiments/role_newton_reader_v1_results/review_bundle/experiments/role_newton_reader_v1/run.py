"""Light supervisor; all costs share already activated unique60min ledger."""
import argparse,json,os,signal,subprocess,sys,time,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent;OUT=Path('/data2/lyh/gnn_project/DiHug/results/role_newton_20261021_v1')
def read(p):return json.loads(p.read_text())
def save(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--execute',action='store_true');a=parser.parse_args()
    if not a.execute:parser.error('explicit --execute required; no training/generation')
    ledger=read(OUT/'activation.json');f=read(OUT/'freeze.json')
    for n,h in f['source'].items():assert hashlib.sha256((HERE/n).read_bytes()).hexdigest()==h,n
    with (OUT/'launch.json').open('x') as ff:json.dump(dict(epoch=time.time(),original_activation=ledger['activation_epoch']),ff)
    jobs=[];stop=[None];status='failed';reason=None
    def handler(s,frame):stop[0]=signal.Signals(s).name
    for s in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP):signal.signal(s,handler)
    def kill(p):
        if p.poll() is None:
            try:os.killpg(p.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            try:p.wait(timeout=3)
            except subprocess.TimeoutExpired:
                try:os.killpg(p.pid,signal.SIGKILL)
                except ProcessLookupError:pass
        p.wait()
    try:
        for scope in ('small','large'):
            log=(OUT/(scope+'.log')).open('w');p=subprocess.Popen([sys.executable,'-u',str(HERE/'panel.py'),scope],
              env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            jobs.append(dict(scope=scope,p=p,log=log,start=time.monotonic()))
        tick=0
        while any(x['p'].poll() is None for x in jobs):
            if stop[0] or (OUT/'STOP').exists() or time.time()>ledger['deadline_epoch']-120:raise InterruptedError(stop[0] or 'STOP/deadline')
            if time.monotonic()-tick>20:print([(x['scope'],x['p'].poll()) for x in jobs],flush=True);tick=time.monotonic()
            time.sleep(.2)
        if any(x['p'].returncode for x in jobs):raise RuntimeError('child failed; no implicit retry')
        status='completed'
    except BaseException as exc:reason=repr(exc)
    finally:
        records=[]
        for x in jobs:
            kill(x['p']);x['log'].close();records.append(dict(scope=x['scope'],exit_code=x['p'].returncode,observation_window_seconds=time.monotonic()-x['start']))
        save(OUT/'controller.json',dict(status=status,reason=reason,jobs=records,all_children_reaped=True,
          elapsed_seconds=time.time()-ledger['activation_epoch'],budget_seconds=3600,training=0,new_chains=0,GPU=0))
    return int(status!='completed')
if __name__=='__main__':sys.exit(main())
