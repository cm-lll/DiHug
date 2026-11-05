"""Independent resource watchdog and authorized process ownership ledger; no model imports."""
import os,sys,json,time,signal,subprocess,argparse,hashlib
from pathlib import Path
OUT=Path(os.environ['STRUCTURE_RUN']);ROOT=Path(__file__).resolve().parents[2]
def read(p):return json.loads(p.read_text())
def save(p,d):
    q=p.with_suffix('.tmp');q.write_text(json.dumps(d,indent=2)+'\n');q.replace(p)
def ticks(pid):
    try:return Path(f'/proc/{pid}/stat').read_text().split()[21]
    except FileNotFoundError:return None
def same(row):return ticks(row['pid'])==row['start_ticks']
def usage():
    t=time.time();return sum((x.get('released_epoch',t)-x['acquired_epoch']) for p in (OUT/'leases').glob('*.json') for x in [read(p)])/3600
def watch(parent):
    ledger=read(OUT/'ledger.json');stopping=None
    while ticks(parent) is not None:
        rows=[read(p) for p in (OUT/'leases').glob('*.json')];live=[x for x in rows if 'released_epoch' not in x and same(x)]
        used=usage();t=time.time();n=max(1,len(live))
        # 60s grace is charged to the quota; request stop before the hard cap.
        request=t>=ledger['science_stop_epoch']-60 or used>=ledger['GPU_hours_limit']-n*60/3600
        if request and stopping is None:
            stopping=t;(OUT/'STOP_SCIENCE').touch()
            for x in live:os.killpg(x['pid'],signal.SIGTERM)
        hard=t>=ledger['science_stop_epoch'] or used>=ledger['GPU_hours_limit'] or stopping is not None and t-stopping>=55
        if hard:
            for x in live:os.killpg(x['pid'],signal.SIGKILL)
        save(OUT/'watchdog.json',dict(epoch=t,used_GPU_hours=used,live=[x['id'] for x in live],stop_requested=stopping,hard_stop=hard))
        time.sleep(2)
def spawn(module,gpu,ident,extra):
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    log=(OUT/'logs'/(ident+'.log')).open('x');start=time.time()
    p=subprocess.Popen([sys.executable,'-u','-m',module,*extra],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    path=OUT/'leases'/(ident+'.json');save(path,dict(id=ident,pid=p.pid,start_ticks=ticks(p.pid),gpu=gpu,acquired_epoch=start))
    return p,log,path
def finish(item):
    p,log,path=item;r=read(path);r.update(released_epoch=time.time(),returncode=p.returncode);save(path,r);log.close()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['engineering','formal','watch']);ap.add_argument('--parent',type=int);ap.add_argument('--attempt',type=int,default=1);a=ap.parse_args()
    if a.mode=='watch':return watch(a.parent)
    lock=OUT/(a.mode+str(a.attempt)+'_started.json')
    with lock.open('x') as f:json.dump(dict(pid=os.getpid(),epoch=time.time()),f)
    monitor=subprocess.Popen([sys.executable,'-m','experiments.structure_output_v1.supervise_formal','watch','--parent',str(os.getpid())],cwd=ROOT,start_new_session=True)
    jobs=[]
    try:
        if a.mode=='engineering':
            jobs=[spawn('experiments.structure_output_v1.formal_engineering',4,'engineering'+str(a.attempt),[])]
            try:jobs[0][0].wait(timeout=300)
            except subprocess.TimeoutExpired:os.killpg(jobs[0][0].pid,signal.SIGTERM);jobs[0][0].wait(timeout=30)
            finish(jobs[0]);save(OUT/('engineering_exit'+str(a.attempt)+'.json'),dict(returncode=jobs[0][0].returncode,GPU_hours=usage()));return jobs[0][0].returncode
        if not read(OUT/'budget.json')['fits'] or not read(OUT/'engineering/receipt.json')['passed']:raise RuntimeError('gate not passed')
        approval=read(OUT/'approval.json');approval['launcher_pid']=os.getpid();save(OUT/'approval.json',approval)
        tasks=read(OUT/'tasks.json')
        for job in tasks:jobs.append(spawn('experiments.structure_output_v1.formal_worker',job['gpu'],job['task_id'],['--task',job['task_id']]))
        active=jobs[:]
        while active:
            for item in active[:]:
                if item[0].poll() is not None:finish(item);active.remove(item)
            save(OUT/'launcher_progress.json',dict(epoch=time.time(),active=[read(x[2])['id'] for x in active],GPU_hours=usage()))
            if active:time.sleep(5)
        save(OUT/'launcher_summary.json',dict(finished_epoch=time.time(),GPU_hours=usage(),returncodes={read(x[2])['id']:x[0].returncode for x in jobs}))
    finally:
        for p,log,path in jobs:
            if p.poll() is None:
                os.killpg(p.pid,signal.SIGTERM)
                try:p.wait(timeout=30)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
                finish((p,log,path))
        monitor.terminate();monitor.wait(timeout=10)
if __name__=='__main__':sys.exit(main() or 0)
