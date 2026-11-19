"""Reuse passed engineering; prospective resource amendment under user's8h grant.

This launcher does not run engineering again and rejects any prior formal run.
Only aggregate internal GPU reserve changes32->36; user wall limits do not change.
"""
from .projected_supervisor import *
def main():
 if (OUT/'run').exists() and any((OUT/'run').iterdir()):raise RuntimeError('formal state already exists; do not restart')
 if not read(OUT/'engineering/receipt.json')['passed']:raise RuntimeError('engineering not passed')
 if not read(OUT/'resource_amendment.json')['engineering_scientific_files_unchanged']:raise RuntimeError('engineering compatibility')
 with (OUT/'formal_continuation_started.json').open('x') as f:json.dump(dict(pid=os.getpid(),epoch=time.time()),f)
 monitor=subprocess.Popen([sys.executable,'-u','-m','runtime.projected_supervisor',str(os.getpid())],cwd=ROOT,start_new_session=True);jobs=[]
 try:
  approval=read(OUT/'approval.json');approval['launcher_pid']=os.getpid();save(OUT/'approval.json',approval)
  from .projected_budget import budget
  b=budget()
  if not b['fits']:raise RuntimeError('complete panel still does not fit amended aggregate budget')
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
