"""Owned-process watchdog, actual allocation accounting and CPU finalization."""
import os,sys,time,json,signal,subprocess,traceback
from pathlib import Path
def read(p):return json.loads(Path(p).read_text())
def save(p,x):
 p=Path(p);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2,ensure_ascii=False));tmp.replace(p)
def main():
 out=Path(sys.argv[1]).resolve();p=read(out/'protocol.json');os.close(os.open(out/'launch.lock',os.O_CREAT|os.O_EXCL|os.O_WRONLY))
 gate=read(out/'preflight_receipt.json');assert gate['status']=='passed' and len(gate['conditions'])==4
 prior=read(out/'preflight_allocation.json')['GPU_hours'];jobs=p['new_jobs']
 text=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used','--format=csv,noheader,nounits'],text=True,timeout=30)
 devices=[dict(index=int(a),uuid=b.strip()) for a,b,c in (line.split(',') for line in text.splitlines()) if float(c)==0]
 assert len(devices)>=4,'four idle GPUs required; no partial launch'
 assert time.time()+max(x['planning_seconds'] for x in jobs)<p['science_stop_epoch'],'complete panel no longer fits wall budget'
 assert prior+sum(x['planning_seconds'] for x in jobs)/3600<p['max_GPU_hours'],'complete panel no longer fits GPU hours'
 procs=[];halt=None
 def stop():
  for x in procs:
   if x['proc'].poll() is None:os.killpg(x['proc'].pid,signal.SIGTERM)
 signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(InterruptedError('watchdog interrupted')))
 try:
  for job,dev in zip(jobs,devices):
   env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=dev['uuid'];at=time.time();env['TASK_ALLOCATION_EPOCH']=str(at)
   log=(out/'logs'/(job['id']+'.log')).open('w');proc=subprocess.Popen([sys.executable,str(Path(__file__).with_name('worker.py')),str(out),job['dataset'],job['arm']],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   procs.append(dict(proc=proc,job=job,dev=dev,start=at,end=None,log=log))
  save(out/'launch.json',[dict(job=x['job']['id'],**x['dev'],pid=x['proc'].pid,start_epoch=x['start']) for x in procs])
  while True:
   now=time.time()
   for x in procs:
    if x['end'] is None and x['proc'].poll() is not None:x['end']=now
   hours=prior+sum((x['end'] or now)-x['start'] for x in procs)/3600
   save(out/'watchdog.json',dict(at_epoch=now,GPU_hours=hours,science_remaining_seconds=p['science_stop_epoch']-now,processes=[dict(job=x['job']['id'],pid=x['proc'].pid,returncode=x['proc'].poll()) for x in procs]))
   if all(x['end'] is not None for x in procs):break
   if now>=p['science_stop_epoch']-30 or hours>=p['max_GPU_hours']-.06:halt='science or GPU-hour limit';stop();break
   time.sleep(2)
 except BaseException:halt=traceback.format_exc();stop()
 finally:
  grace_end=time.time()+15
  for x in procs:
   if x['proc'].poll() is None:
    try:x['proc'].wait(timeout=max(.01,grace_end-time.time()))
    except subprocess.TimeoutExpired:os.killpg(x['proc'].pid,signal.SIGKILL);x['proc'].wait()
   if x['end'] is None:x['end']=time.time()
   x['log'].close()
  hours=prior+sum(x['end']-x['start'] for x in procs)/3600
  results=[dict(job=x['job']['id'],**x['dev'],pid=x['proc'].pid,start_epoch=x['start'],end_epoch=x['end'],exit_code=x['proc'].returncode) for x in procs]
  save(out/'GPU_release.json',dict(at_epoch=time.time(),allocated_GPU_hours=hours,all_own_children_exited=True,processes=results,stop_reason=halt))
 remaining=p['delivery_epoch']-time.time()-15;erc=None
 if remaining>0:
  env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=''
  with (out/'evaluation.log').open('w') as log:
   proc=subprocess.Popen([sys.executable,str(Path(__file__).with_name('evaluate.py')),str(out)],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   try:erc=proc.wait(timeout=remaining)
   except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);erc=proc.wait()
 save(out/'completion.json',dict(at_epoch=time.time(),GPU_hours=hours,evaluation_exit=erc,finished=results,status='complete' if erc==0 and len(results)==4 and all(x['exit_code']==0 for x in results) else 'partial',no_followup_started=True,stop_reason=halt))
if __name__=='__main__':main()
