"""Only own four jobs; enforce wall, allocated GPU hours and disk reserve."""
import os,sys,time,json,signal,subprocess,traceback,shutil
from pathlib import Path
def read(p):return json.loads(Path(p).read_text())
def save(p,x):
 p=Path(p);t=p.with_suffix('.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2));t.replace(p)
def main():
 out=Path(sys.argv[1]).resolve();p=read(out/'protocol.json');os.close(os.open(out/'launch.lock',os.O_CREAT|os.O_EXCL|os.O_WRONLY));assert read(out/'preflight/receipt.json')['status']=='passed';assert read(out/'cpu_check/receipt.json')['status']=='passed'
 output=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used','--format=csv,noheader,nounits'],text=True,timeout=30)
 devices=[dict(index=int(a),uuid=b.strip()) for a,b,c in (x.split(',') for x in output.splitlines()) if int(a) in p['allowed_GPU_indices'] and float(c)==0];assert len(devices)==4,'need all four designated idle cards'
 assert time.time()+max(j['planning_seconds'] for j in p['jobs'])<p['science_stop_epoch'];assert p['preflight_GPU_hours']+sum(j['planning_seconds'] for j in p['jobs'])/3600<p['max_GPU_hours']
 procs=[];halt=None
 def stop():
  for x in procs:
   if x['proc'].poll() is None:os.killpg(x['proc'].pid,signal.SIGTERM)
 signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(InterruptedError('owned watchdog interrupted')))
 try:
  for job,dev in zip(p['jobs'],devices):
   env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=dev['uuid'];env['TMPDIR']=str(out.parents[1]/'.runtime_tmp');at=time.time();env['TASK_ALLOCATION_EPOCH']=str(at)
   log=(out/'logs'/(job['id']+'.log')).open('w');proc=subprocess.Popen([sys.executable,str(Path(__file__).with_name('job.py')),str(out),job['id']],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);procs.append(dict(proc=proc,job=job['id'],dev=dev,start=at,end=None,log=log))
  save(out/'launch.json',[dict(job=x['job'],**x['dev'],pid=x['proc'].pid,start_epoch=x['start']) for x in procs])
  while True:
   now=time.time()
   for x in procs:
    if x['end'] is None and x['proc'].poll() is not None:x['end']=now
   hours=p['preflight_GPU_hours']+sum((x['end'] or now)-x['start'] for x in procs)/3600
   free={path:shutil.disk_usage(path).free/2**30 for path in ('/','/data2')};save(out/'watchdog.json',dict(at_epoch=now,allocated_GPU_hours=hours,remaining_science_seconds=p['science_stop_epoch']-now,free_GiB=free,processes=[dict(job=x['job'],pid=x['proc'].pid,returncode=x['proc'].poll()) for x in procs]))
   if all(x['end'] is not None for x in procs):break
   if now>=p['science_stop_epoch']-35 or hours>=p['max_GPU_hours']-.07:halt='wall or allocated GPU-hours limit';stop();break
   if min(free.values())<20:halt='disk reserve under 20GiB';stop();break
   time.sleep(5)
 except BaseException:halt=traceback.format_exc();stop()
 finally:
  grace=time.time()+20
  if halt:
   # A job driver may exit before its GPU child finishes saving; inspect groups,
   # not only driver PIDs, so no orphan calculation outlives the watchdog.
   def group_alive(x):
    try:os.killpg(x['proc'].pid,0);return True
    except ProcessLookupError:return False
   while time.time()<grace and any(group_alive(x) for x in procs):time.sleep(.2)
   for x in procs:
    if group_alive(x):os.killpg(x['proc'].pid,signal.SIGKILL)
  for x in procs:
   if x['proc'].poll() is None:
    try:x['proc'].wait(timeout=max(.01,grace-time.time()))
    except subprocess.TimeoutExpired:os.killpg(x['proc'].pid,signal.SIGKILL);x['proc'].wait()
   if x['end'] is None:x['end']=time.time()
   x['log'].close()
  hours=p['preflight_GPU_hours']+sum(x['end']-x['start'] for x in procs)/3600;results=[dict(job=x['job'],**x['dev'],pid=x['proc'].pid,start_epoch=x['start'],end_epoch=x['end'],exit_code=x['proc'].returncode) for x in procs]
  save(out/'GPU_release.json',dict(at_epoch=time.time(),allocated_GPU_hours=hours,processes=results,all_owned_process_groups_finished=True,stop_reason=halt))
 remaining=p['delivery_epoch']-time.time()-15;erc=None
 if remaining>0:
  env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=''
  with (out/'evaluation.log').open('w') as log:
   proc=subprocess.Popen([sys.executable,str(Path(__file__).with_name('evaluate.py')),str(out)],stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
   try:erc=proc.wait(timeout=remaining)
   except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);erc=proc.wait()
 ev=read(out/'evaluation_receipt.json') if (out/'evaluation_receipt.json').exists() else {}
 save(out/'completion.json',dict(at_epoch=time.time(),allocated_GPU_hours=hours,evaluation_exit=erc,status='complete' if len(results)==4 and all(x['exit_code']==0 for x in results) and ev.get('status')=='complete' else 'partial',jobs=results,stop_reason=halt,no_followup_started=True))
if __name__=='__main__':main()
