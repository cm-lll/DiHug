"""Light parent: original2700s ledger includes imports, all jobs and cleanup."""
import os,sys,json,time,signal,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent;OUT=Path('/data2/lyh/gnn_project/DiHug/results/role_newton_generation_20261021_v1')
def main():
    a=json.loads((OUT/'activation.json').read_text());jobs=[];stopped=[None];status='failed';reason=None
    def stop(s,f):stopped[0]=signal.Signals(s).name;(OUT/'STOP').touch()
    for s in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP):signal.signal(s,stop)
    def kill(p):
        if p.poll() is None:
            try:os.killpg(p.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            try:p.wait(timeout=4)
            except subprocess.TimeoutExpired:
                try:os.killpg(p.pid,signal.SIGKILL)
                except ProcessLookupError:pass
        p.wait()
    try:
        with (OUT/'launch.json').open('x') as f:json.dump(dict(launch_epoch=time.time(),activation=a),f)
        for arm in ('DP','Newton'):
            for seed in (2026102011,2026102012,2026102013):
                log=(OUT/f'{arm}_{seed}.log').open('w');p=subprocess.Popen([sys.executable,'-u',str(HERE/'worker.py'),'--arm',arm,'--seed',str(seed)],env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'),stdout=log,stderr=subprocess.STDOUT,start_new_session=True);jobs.append((arm,seed,p,log))
        while any(p.poll() is None for _,_,p,_ in jobs):
            if time.time()>a['deadline_epoch']-180 or stopped[0] or (OUT/'STOP').exists():raise InterruptedError(stopped[0] or 'deadline/STOP')
            if any(p.poll() not in (None,0) for _,_,p,_ in jobs):raise RuntimeError('child failed; stop peers, no fallback/retry')
            time.sleep(.3)
        if any(p.returncode for _,_,p,_ in jobs):raise RuntimeError('child failed')
        status='completed'
    except BaseException as exc:reason=repr(exc)
    finally:
        for _,_,p,log in jobs:kill(p);log.close()
        (OUT/'controller.json').write_text(json.dumps(dict(status=status,reason=reason,jobs=[dict(arm=arm,seed=seed,exit_code=p.returncode) for arm,seed,p,_ in jobs],elapsed_seconds=time.time()-a['activation_epoch'],all_children_reaped=True),indent=2)+'\n')
    return int(status!='completed')
if __name__=='__main__':sys.exit(main())
