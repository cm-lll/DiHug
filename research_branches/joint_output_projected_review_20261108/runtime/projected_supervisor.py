"""Conditional second development panel; shared ownership ledger and hard stops."""
import os,sys,json,time,signal,subprocess
from pathlib import Path
from .supervise_formal import read,save,ticks,same,spawn,finish
OUT=Path(os.environ['STRUCTURE_RUN']);ROOT=Path(__file__).resolve().parents[1]
PREVIOUS=[OUT.parent/'joint_output_formal_20261108T021741',OUT.parent/'joint_output_formal_controls_20261108T025000']
def leases():return [(root,p,read(p)) for root in PREVIOUS+[OUT] for p in (root/'leases').glob('*.json')]
def usage():
 now=time.time();return sum(r.get('released_epoch',now)-r['acquired_epoch'] for _,_,r in leases())/3600
def watch(parent):
 ledger=read(OUT/'ledger.json');stopping=None
 while True:
  now=time.time();live=[(root,r) for root,_,r in leases() if 'released_epoch' not in r and same(r)];own=[(root,r) for root,r in live if root==OUT]
  if ticks(parent) is None and not own:break
  used=usage();global_stop=now>=ledger['science_stop_epoch']-60 or used>=ledger['GPU_hours_limit']-max(1,len(live))/60
  eng=sum(r.get('released_epoch',now)-r['acquired_epoch'] for root,_,r in leases() if root==OUT and r['id'].startswith('engineering'))/3600
  own_stop=ticks(parent) is None or eng>=ledger['engineering_GPU_hours_limit']-1/60 and any(r['id'].startswith('engineering') for _,r in own)
  affected=live if global_stop else own if own_stop else []
  if affected and stopping is None:
   stopping=now
   for root in {root for root,_ in affected}:(root/'STOP_SCIENCE').touch()
   for _,r in affected:os.killpg(r['pid'],signal.SIGTERM)
  hard=now>=ledger['science_stop_epoch'] or used>=ledger['GPU_hours_limit'] or stopping is not None and now-stopping>=55
  if hard:
   for _,r in affected:os.killpg(r['pid'],signal.SIGKILL)
  save(OUT/'watchdog.json',dict(epoch=now,combined_GPU_hours=used,live=[r['id'] for _,r in live],stop_requested=stopping,hard_stop=hard));time.sleep(2)
def released(gpu):
 for root in PREVIOUS:
  for t in read(root/'tasks.json'):
   if t['gpu']!=gpu:continue
   p=root/'leases'/(t['task_id']+'.json')
   if not p.exists() or 'released_epoch' not in read(p):return False
 return True
def main():
 if len(sys.argv)>1:return watch(int(sys.argv[1]))
 with (OUT/'projected_started.json').open('x') as f:json.dump(dict(pid=os.getpid(),epoch=time.time()),f)
 monitor=subprocess.Popen([sys.executable,'-u','-m','runtime.projected_supervisor',str(os.getpid())],cwd=ROOT,start_new_session=True);jobs=[]
 try:
  approval=read(OUT/'approval.json');approval['launcher_pid']=os.getpid();save(OUT/'approval.json',approval)
  while not released(0):
   if (OUT/'STOP_SCIENCE').exists():raise TimeoutError('before engineering')
   time.sleep(10)
  engineer=spawn('runtime.formal_engineering',0,'engineering_projected',[]);jobs.append(engineer);engineer[0].wait(timeout=600);finish(engineer)
  if engineer[0].returncode or not read(OUT/'engineering/receipt.json')['passed']:raise RuntimeError('projected objective engineering gate failed')
  from .projected_budget import budget
  b=budget()
  if not b['fits']:
   save(OUT/'not_started.json',dict(reason='complete8model32chain panel does not fit remaining shared budget',budget=b,scientific_formula_unchanged=True));return
  pending=read(OUT/'tasks.json');active=[]
  while pending or active:
   for item in active[:]:
    if item[0].poll() is not None:finish(item);active.remove(item)
   busy={read(x[2])['gpu'] for x in active}
   for job in pending[:]:
    if (OUT/'STOP_SCIENCE').exists():break
    if job['gpu'] in busy or not released(job['gpu']):continue
    item=spawn('runtime.formal_worker',job['gpu'],job['task_id'],['--task',job['task_id']]);jobs.append(item);active.append(item);busy.add(job['gpu']);pending.remove(job)
   save(OUT/'launcher_progress.json',dict(epoch=time.time(),active=[read(x[2])['id'] for x in active],pending=[x['task_id'] for x in pending],combined_GPU_hours=usage()))
   if (OUT/'STOP_SCIENCE').exists() and not active:break
   if pending or active:time.sleep(10)
  save(OUT/'launcher_summary.json',dict(finished_epoch=time.time(),combined_GPU_hours=usage(),returncodes={read(x[2])['id']:x[0].returncode for x in jobs}))
  while not (PREVIOUS[1]/'evaluation_exit.json').exists():
   if time.time()>read(OUT/'ledger.json')['deadline_epoch']-120:break
   time.sleep(10)
  with (OUT/'logs/evaluation.log').open('w') as f:
   p=subprocess.run([sys.executable,'-u','-m','runtime.evaluate_formal'],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,
    timeout=max(1,read(OUT/'ledger.json')['deadline_epoch']-time.time()-45),env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1'))
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
