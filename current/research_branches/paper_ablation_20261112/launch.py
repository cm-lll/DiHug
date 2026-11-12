"""Exclusive ready queue; independent wall/GPU-hour watchdog; owns only this ticket."""
import os,sys,time,json,signal,subprocess,shutil,hashlib,traceback
from pathlib import Path
def read(p):return json.loads(Path(p).read_text())
def save(p,x):
 t=Path(str(p)+'.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2));t.replace(p)
def main():
 out=Path(sys.argv[1]).resolve();p=read(out/'protocol.json');os.close(os.open(out/'launch.lock',os.O_CREAT|os.O_EXCL|os.O_WRONLY))
 assert read(out/'cpu_check/receipt.json')['status']==read(out/'gpu_check/receipt.json')['status']=='passed'
 for path,h in p['bindings'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==h,path
 def devices():
  text=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used','--format=csv,noheader,nounits'],text=True,timeout=20)
  return [dict(index=int(a),uuid=b.strip(),memory=float(c)) for a,b,c in (l.split(',') for l in text.splitlines()) if int(a) in p['allowed_GPU_indices']]
 free=[x for x in devices() if x['memory']==0];assert len(free)>=6,'full queue budget assumes six actually idle cards'
 pending=sorted(p['jobs'],key=lambda j:-j['planning_seconds']);loads=[0.]*6
 for j in pending:loads[loads.index(min(loads))]+=j['planning_seconds']
 assert time.time()+max(loads)+120<p['science_stop_epoch'],'complete panel no longer fits remaining window'
 assert p['preflight_GPU_hours']+sum(x['planning_seconds'] for x in pending)/3600<p['max_GPU_hours']
 procs=[];reason=None;fatal=None
 def stop():
  for x in procs:
   if x['proc'].poll() is None:
    try:os.killpg(x['proc'].pid,signal.SIGTERM)
    except ProcessLookupError:pass
 signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(InterruptedError('supervisor interrupted')))
 try:
  while pending or any(x['proc'].poll() is None for x in procs):
   now=time.time()
   for x in procs:
    if x['end'] is None and x['proc'].poll() is not None:x['end']=now
   used=p['preflight_GPU_hours']+sum((x['end'] or now)-x['start'] for x in procs)/3600
   free_disk={a:shutil.disk_usage(a).free/2**30 for a in ('/','/data2')}
   if now>=p['science_stop_epoch']-40 or used>=p['max_GPU_hours']-.08 or min(free_disk.values())<20:
    reason='science wall / cumulative occupied GPU-hours / disk reserve';stop();break
   active=[x for x in procs if x['end'] is None]
   if pending and len(active)<p['max_GPUs']:
    reserved={x['device']['index'] for x in active}
    available=[x for x in devices() if x['memory']==0 and x['index'] not in reserved]
    for dev in available[:p['max_GPUs']-len(active)]:
     if not pending:break
     job=pending[0]
     # Never launch a job that the frozen upper estimate cannot finish.
     if now+job['planning_seconds']>=p['science_stop_epoch'] or used+job['planning_seconds']/3600>=p['max_GPU_hours']:
      reason='remaining budget cannot cover unstarted job';pending=[];break
     pending.pop(0);env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=dev['uuid'],TMPDIR=str(out.parents[1]/'.runtime_tmp'),TASK_ALLOCATION_EPOCH=str(time.time()))
     log=(out/'logs'/(job['id']+'.log')).open('w');start=time.time()
     proc=subprocess.Popen([sys.executable,str(Path(__file__).with_name('job.py')),str(out),job['id']],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
     procs.append(dict(proc=proc,device=dev,job=job['id'],start=start,end=None,log=log))
     save(out/'launches.json',[dict(job=x['job'],device=x['device'],pid=x['proc'].pid,start_epoch=x['start']) for x in procs])
   save(out/'watchdog.json',dict(at_epoch=now,allocated_GPU_hours=used,remaining_science_seconds=p['science_stop_epoch']-now,
     free_GiB=free_disk,pending=[j['id'] for j in pending],processes=[dict(job=x['job'],pid=x['proc'].pid,exit_code=x['proc'].poll()) for x in procs],stop_reason=reason))
   time.sleep(5)
 except BaseException:fatal=traceback.format_exc();reason='supervisor error';stop()
 finally:
  grace=time.time()+25
  if reason:
   while time.time()<grace and any(x['proc'].poll() is None for x in procs):time.sleep(.2)
   for x in procs:
    try:os.killpg(x['proc'].pid,signal.SIGKILL)
    except ProcessLookupError:pass
  for x in procs:
   x['proc'].wait();x['end']=x['end'] or time.time();x['log'].close()
  used=p['preflight_GPU_hours']+sum(x['end']-x['start'] for x in procs)/3600
  results=[dict(job=x['job'],device=x['device'],pid=x['proc'].pid,start_epoch=x['start'],end_epoch=x['end'],exit_code=x['proc'].returncode) for x in procs]
  save(out/'GPU_release.json',dict(at_epoch=time.time(),allocated_GPU_hours=used,processes=results,all_owned_groups_exited=True,stop_reason=reason,error=fatal))
 remaining=p['delivery_epoch']-time.time()-20;erc=None
 if remaining>0:
  with (out/'evaluation.log').open('w') as f:
   env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=''
   ev=subprocess.Popen([sys.executable,str(Path(__file__).with_name('evaluate.py')),str(out)],env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
   try:erc=ev.wait(timeout=remaining)
   except subprocess.TimeoutExpired:os.killpg(ev.pid,signal.SIGKILL);erc=ev.wait()
 er=read(out/'evaluation_receipt.json') if (out/'evaluation_receipt.json').exists() else {}
 save(out/'completion.json',dict(status='complete' if len(results)==12 and all(r['exit_code']==0 for r in results) and er.get('status')=='complete' else 'partial',
    at_epoch=time.time(),wall_minutes=(time.time()-p['start_epoch'])/60,allocated_GPU_hours=used,evaluation_exit=erc,results=results,stop_reason=reason,error=fatal,no_followup=True))
if __name__=='__main__':main()
