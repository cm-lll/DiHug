"""Same-ticket control queue; combined accounting, no changes to running workers."""
import os,sys,json,time,signal,subprocess
from pathlib import Path
from .supervise_formal import read,save,ticks,same,spawn,finish
OUT=Path(os.environ['STRUCTURE_RUN']);ROOT=Path(__file__).resolve().parents[1]
PARENT=OUT.parent/'joint_output_formal_20261108T021741'
def leases():
    return [(root,p,read(p)) for root in [PARENT,OUT] for p in (root/'leases').glob('*.json')]
def usage():
    now=time.time();return sum(r.get('released_epoch',now)-r['acquired_epoch'] for _,_,r in leases())/3600
def watch(parent):
    ledger=read(OUT/'ledger.json');stopping=None
    while True:
        now=time.time();allrows=leases();live=[(root,r) for root,_,r in allrows if 'released_epoch' not in r and same(r)]
        own=[(root,r) for root,r in live if root==OUT]
        if ticks(parent) is None and not own:break
        total=usage();global_request=now>=ledger['science_stop_epoch']-60 or total>=ledger['GPU_hours_limit']-max(1,len(live))/60
        orphan=ticks(parent) is None
        affected=live if global_request else own if orphan else []
        if affected and stopping is None:
            stopping=now
            for root in {root for root,_ in affected}:(root/'STOP_SCIENCE').touch()
            for root,r in affected:os.killpg(r['pid'],signal.SIGTERM)
        hard=now>=ledger['science_stop_epoch'] or total>=ledger['GPU_hours_limit'] or stopping is not None and now-stopping>=55
        if hard:
            for root,r in affected:os.killpg(r['pid'],signal.SIGKILL)
        save(OUT/'watchdog.json',dict(epoch=now,combined_GPU_hours=total,live=[r['id'] for _,r in live],stop_requested=stopping,hard_stop=hard))
        time.sleep(2)
def original_released(gpu):
    rows=[r for root,_,r in leases() if root==PARENT and r['gpu']==gpu]
    return bool(rows) and all('released_epoch' in r for r in rows)
def main():
    if len(sys.argv)>1:return watch(int(sys.argv[1]))
    with (OUT/'control_started.json').open('x') as f:json.dump(dict(pid=os.getpid(),epoch=time.time()),f)
    monitor=subprocess.Popen([sys.executable,'-u','-m','runtime.control_supervisor',str(os.getpid())],cwd=ROOT,start_new_session=True)
    jobs=[]
    try:
        if not read(OUT/'budget.json')['fits']:raise RuntimeError('revised full-panel budget failed')
        approval=read(OUT/'approval.json');approval['launcher_pid']=os.getpid();save(OUT/'approval.json',approval)
        # CPU waiting holds no GPU; main queue owns4-7 until its task and all four chains finish.
        while not original_released(4):
            if (OUT/'STOP_SCIENCE').exists():raise TimeoutError('before engineering')
            save(OUT/'launcher_progress.json',dict(epoch=time.time(),status='waiting_for_original_GPU4_release',GPU_hours=usage()))
            time.sleep(10)
        engineer=spawn('runtime.formal_engineering',4,'engineering_controls',[]);jobs.append(engineer)
        engineer[0].wait(timeout=600);finish(engineer)
        if engineer[0].returncode or not read(OUT/'engineering/receipt.json')['passed']:raise RuntimeError('CE control minimal integration failed')
        pending=read(OUT/'tasks.json');active=[]
        # Freeze full panel before launch; no selection from results and no borrowing0-3.
        mins=max(sum(x['device_minutes'] for x in read(OUT/'budget.json')['new_task_bounds'] if x['gpu']==gpu) for gpu in [4,5,6,7])
        if time.time()+mins*60>=read(OUT/'ledger.json')['science_stop_epoch']-120:raise RuntimeError('full control panel no longer fits wall window')
        while pending or active:
            for item in active[:]:
                if item[0].poll() is not None:finish(item);active.remove(item)
            busy={read(x[2])['gpu'] for x in active}
            for job in pending[:]:
                if (OUT/'STOP_SCIENCE').exists():break
                gpu=job['gpu']
                if gpu in busy or not original_released(gpu):continue
                item=spawn('runtime.formal_worker',gpu,job['task_id'],['--task',job['task_id']])
                jobs.append(item);active.append(item);busy.add(gpu);pending.remove(job)
            save(OUT/'launcher_progress.json',dict(epoch=time.time(),active=[read(x[2])['id'] for x in active],pending=[x['task_id'] for x in pending],combined_GPU_hours=usage()))
            if (OUT/'STOP_SCIENCE').exists() and not active:break
            if pending or active:time.sleep(10)
        save(OUT/'launcher_summary.json',dict(finished_epoch=time.time(),combined_GPU_hours=usage(),returncodes={read(x[2])['id']:x[0].returncode for x in jobs}))
        # Release all cards before CPU evaluation; wait only for parent metric cache, not for extra science.
        while not (PARENT/'evaluation_exit.json').exists():
            if time.time()>read(OUT/'ledger.json')['deadline_epoch']-120:break
            time.sleep(10)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
        with (OUT/'logs/evaluation.log').open('w') as f:
            p=subprocess.run([sys.executable,'-u','-m','runtime.evaluate_formal'],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,env=env,
                timeout=max(1,read(OUT/'ledger.json')['deadline_epoch']-time.time()-45))
            save(OUT/'evaluation_exit.json',dict(returncode=p.returncode,epoch=time.time()))
    finally:
        for p,log,path in jobs:
            if p.poll() is None:
                os.killpg(p.pid,signal.SIGTERM)
                try:p.wait(timeout=30)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
                finish((p,log,path))
        monitor.terminate();monitor.wait(timeout=10)
if __name__=='__main__':main()
