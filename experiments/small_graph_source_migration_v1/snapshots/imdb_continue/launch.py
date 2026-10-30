"""Independent four-card IMDB continuation; parent supervisor never modified."""
import sys,subprocess,os,time,shutil,traceback
from common import *
def main():
 check();pre=read(OUT/'preflight.json');assert pre['status']=='passed' and pre['source_sha256']==source_binding()[1]
 if time.time()+pre['planned_remaining_science_seconds']>SCIENCE:raise RuntimeError('queue no longer fits before08:30')
 gpu=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
 apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory','--format=csv,noheader,nounits'],text=True)
 for dev in DEVICES:
  row=next(r.split(',') for r in gpu.splitlines() if r.startswith(dev));assert float(row[1])>7000
  if any(r.startswith(dev+',') for r in apps.splitlines()):raise RuntimeError('device occupied '+dev)
 save(OUT/'resources.json',dict(gpus=gpu,apps=apps,allowed_by='explicit user authorizes released GPU4-7 for fixedIMDB continuation'))
 ledger=dict(start_epoch=START,science_stop_epoch=SCIENCE,deadline_epoch=DEADLINE,source_sha256=source_binding()[1],output=str(OUT/'run'),gpu_uuids=DEVICES,
  parent=str(PARENT),authorizing_request='IMDB4x3000->6000+8T1000, deliver before09:00; oldPubMed unchanged')
 approval=dict(user_authorized=True,source_sha256=source_binding()[1],formula_sha256=sha(HERE/'formula.json'),protocol_sha256=sha(HERE/'protocol.json'),
  updates=6000,resume_updates=3000,new_updates_total=12000,new_chains=8,next_input3001_hashes=pre['next_input3001_hashes'],
  environment=read(PARENT.parent/'approval.json')['environment'],preflight_sha256=sha(OUT/'preflight.json'),gpu_uuids=DEVICES,deadline_epoch=DEADLINE)
 save(OUT/'ledger.json',ledger);save(OUT/'approval.json',approval)
 env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',
  CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONDONTWRITEBYTECODE='1',TMPDIR=str(OUT/'tmp'),MPLCONFIGDIR=str(OUT/'tmp/matplotlib'))
 workers=[];finished=[]
 for i,j in enumerate(train_jobs()):
  f=(OUT/(j['id']+'.log')).open('x');p=subprocess.Popen([sys.executable,'-B',str(HERE/'continue_worker.py'),'--slot',str(i)],cwd=HERE,env=dict(env,CUDA_VISIBLE_DEVICES=DEVICES[i]),stdout=f,stderr=subprocess.STDOUT)
  workers.append((i,p,f))
 save(OUT/'startup.json',dict(launcher_pid=os.getpid(),actual_launch_epoch=time.time(),ticket_start_epoch=START,science_stop_epoch=SCIENCE,deadline_epoch=DEADLINE,
  planned_science_finish_epoch=time.time()+pre['planned_remaining_science_seconds'],planned_delivery_epoch=time.time()+pre['planned_remaining_science_seconds']+1800,
  workers=[dict(job=train_jobs()[i],pid=p.pid,gpu_uuid=DEVICES[i]) for i,p,f in workers],source_sha256=source_binding()[1]))
 while workers:
  for item in list(workers):
   i,p,f=item
   if p.poll() is not None:f.close();finished.append(dict(job=train_jobs()[i],exit_code=p.returncode));workers.remove(item)
  save(OUT/'launcher_progress.json',dict(epoch=time.time(),finished=finished,running=[dict(job=train_jobs()[i],pid=p.pid) for i,p,f in workers]))
  if workers:time.sleep(10)
 save(OUT/'scientific_completion.json',dict(epoch=time.time(),finished=finished))
 # Complete and incomplete runs both retain all expected cells and original failure reasons.
 for name in ('report.py','audit_outputs.py','detail_tables.py','finalize_catalog.py','write_summary.py'):
  if time.time()>DEADLINE-30:save(OUT/'delivery_failure.json',dict(reason='delivery deadline',script=name));break
  file=HERE/'report.py' if name=='report.py' else OUT/name
  try:
   p=subprocess.run([sys.executable,'-B',str(file)],env=env,cwd=HERE,capture_output=True,text=True,timeout=max(1,DEADLINE-time.time()-20))
   save(OUT/(name+'.receipt.json'),dict(exit_code=p.returncode,stdout=p.stdout,stderr=p.stderr))
   if p.returncode:save(OUT/'delivery_failure.json',dict(script=name,error=p.stderr[-3000:]));break
  except BaseException as exc:save(OUT/'delivery_failure.json',dict(script=name,error=repr(exc),traceback=traceback.format_exc()));break
if __name__=='__main__':main()
