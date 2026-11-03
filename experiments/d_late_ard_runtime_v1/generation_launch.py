"""Sole seven-worker launcher; all occupied GPU time counted, no retries."""
import os,signal,socket,subprocess,shutil,statistics
from common_run import *
from resources import register,unregister,lease_start,lease_end
PY='/opt/di-hug-env/bin/python'
def devices():
 text=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.used,memory.free','--format=csv,noheader,nounits'],text=True)
 return [dict(zip(['gpu','uuid','name','used','free'],[v.strip() for v in line.split(',')])) for line in text.strip().splitlines()]
def gpu_seconds():
 t=time.time();return sum(max(0,x.get('end_epoch',t)-x['start_epoch']) for x in read(TASKOUT/'registry.json')['gpu_leases'])
def main():
 if (OUT/'approval.json').exists():raise FileExistsError('Duplicate launcher forbidden')
 check_time();gate=read(OUT/'recovery_gate.json');assert gate['status']=='passed' and gate['temporary_optimizer_steps']==6
 jobs=[j for j in read(OUT/'tasks.json') if not j['reuse_completed_baseline']]
 jobs.sort(key=lambda j:({'A':0,'D':1,'ard':2}[j['method']],j['train_seed']))
 assert len(jobs)==7 and len({j['task_id'] for j in jobs})==7
 cards=devices();apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader,nounits'],text=True)
 busy={line.split(',')[1].strip() for line in apps.strip().splitlines() if line.strip()}
 free=[d for d in cards if '2080 Ti' in d['name'] and int(d['used'])<100 and int(d['free'])>10000 and d['uuid'] not in busy]
 if len(free)<7:raise RuntimeError('Seven allowed free devices unavailable; preserve full panel')
 if shutil.disk_usage(ROOT).free<40*1024**3:raise RuntimeError('Insufficient 40GiB disk reserve')
 planned=read(ROOT/'results/d_late_ard_plan_20261103/budget.json');ledger=read(TASKOUT/'ledger.json')
 if gpu_seconds()/3600+planned['estimated_task_GPU_hours']>22 or time.time()+186.25*60+300>ledger['science_stop_epoch']:
  raise RuntimeError('Remaining complete-panel budget does not fit; no reduction')
 for j,d in zip(jobs,free):j['gpu_slot']=int(d['gpu'])
 (OUT/'logs').mkdir();(OUT/'tickets').mkdir()
 files=list(HERE.glob('*.py'))+list(OLD.glob('*.py'))+list(SCI.glob('*.py'))+list((SCI/'package').rglob('*.py'))
 files += [ROOT/'experiments/adaptive_role_repair_v1/generation_adapter.py']
 files += [OUT/n for n in ('tasks.json','dictionaries.json','initial_banks.json','input_contracts.json','recovery_gate.json','reused_samples.json')]
 files += list((OUT/'prepared').glob('*.json'))
 records=read(OUT/'initial_banks.json');dictionaries=read(OUT/'dictionaries.json')
 bindings={str(p):sha(p) for p in files}
 for bank in records.values():
  if sha(bank['file'])!=bank['sha256']:raise ValueError('Changed untrained bank')
  bindings[bank['file']]=bank['sha256']
 for record in dictionaries.values():
  if record.get('artifact'):
   if sha(record['artifact'])!=record['artifact_sha256']:raise ValueError('Changed frozen dictionary')
   bindings[record['artifact']]=record['artifact_sha256']
 for row in read(TASKOUT/'reviews/reference_data_binding.json')['files']:
  if sha(row['file'])!=row['legacy_approval_sha256']:raise ValueError('Changed reference asset')
  bindings[row['file']]=row['legacy_approval_sha256']
 save(OUT/'source_manifest.json',dict(files=bindings,frozen=now(),parent='382c824778429533b4bcc8a5005019df6e5358b2',plan='f28fb0f'))
 save(OUT/'approval.json',dict(authorized=True,host=socket.gethostname(),launcher_pid=os.getpid(),source_files=bindings,
  dictionaries=dictionaries,initial_banks=records,authorization='User approved D3 ARD3 A-third, 260min22GPUh, recovery gate passed',
  ledger_sha256=sha(TASKOUT/'ledger.json'),jobs=jobs,devices=free[:7],new_models=7,new_graphs=28))
 save(OUT/'resource_start.json',dict(at=now(),devices=cards,compute_processes=apps,disk_free_bytes=shutil.disk_usage(ROOT).free))
 register(os.getpid(),'gpu_controller');running={};done=[];launched=set();stop=False;failure=None;last=0
 def stopping(*_):
  nonlocal stop
  stop=True
 signal.signal(signal.SIGTERM,stopping);signal.signal(signal.SIGINT,stopping)
 try:
  for job,dev in zip(jobs,free):
   check_time();ticket=OUT/'tickets'/(job['task_id']+'.json');save(ticket,dict(job=job,GPU_UUID=dev['uuid']))
   env=dict(os.environ,CUDA_VISIBLE_DEVICES=dev['gpu'],CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TMPDIR=str(TASKOUT/'temporary'))
   log=(OUT/'logs'/(job['task_id']+'.log')).open('x')
   p=subprocess.Popen([PY,'-u',str(HERE/'generation_worker.py'),'--ticket',str(ticket)],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   register(p.pid,'gpu_worker',task=job['task_id']);lease_start(p.pid,int(dev['gpu']),dev['uuid'],job['task_id'])
   running[p.pid]=(p,log,job,dev);launched.add(job['task_id'])
  while running:
   check_time()
   if stop:raise InterruptedError('Own supervisor stop requested')
   # Signal before the cap to include checkpoint/shutdown occupancy in22GPUh.
   if gpu_seconds()+90*len(running)>=22*3600:raise TimeoutError('Cumulative GPU budget save reserve')
   for pid,(p,log,job,dev) in list(running.items()):
    if p.poll() is None:continue
    lease_end(pid);unregister(pid);log.close();path=OUT/'run'/job['task_id']/'summary.json'
    s=read(path) if path.exists() else dict(status='missing_receipt')
    done.append(dict(s,task_id=job['task_id'],method=job['method'],returncode=p.returncode));del running[pid]
    if p.returncode or s['status']!='complete':save(OUT/'failure_notice.json',dict(task=job,receipt=s,at=now()))
   if time.time()-last>10:
    save(OUT/'queue_progress.json',dict(at=now(),running=[dict(task=x[2]['task_id'],GPU=x[3]['gpu'],pid=pid) for pid,x in running.items()],complete=done,GPU_hours=gpu_seconds()/3600))
    last=time.time()
   time.sleep(3)
 except BaseException as exc:failure=repr(exc);save(OUT/'controller_exception.json',dict(at=now(),error=failure))
 finally:
  for p,_,_,_ in running.values():
   if p.poll() is None:p.terminate()
  began=time.time()
  while any(p.poll() is None for p,_,_,_ in running.values()) and time.time()-began<45:time.sleep(1)
  for pid,(p,log,job,dev) in running.items():
   if p.poll() is None:p.kill();p.wait()
   log.close();lease_end(pid);unregister(pid)
  unregister(os.getpid())
  save(OUT/'queue_receipt.json',dict(finished=now(),completed=done,interrupted=[x[2]['task_id'] for x in running.values()],
       unstarted=[dict(task_id=j['task_id'],reason=failure or 'not_dispatched') for j in jobs if j['task_id'] not in launched],error=failure,GPU_hours=gpu_seconds()/3600))
 # CPU delivery is separate from scientific/GPU process registry; original deadlines persist.
 if time.time()<ledger['final_deadline_epoch']-120:
  with (TASKOUT/'evaluation.log').open('x') as log:
   proc=subprocess.run([PY,'-u',str(HERE/'generation_evaluate.py')],stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'),timeout=max(1,ledger['final_deadline_epoch']-time.time()-90))
  save(TASKOUT/'evaluation_exit.json',dict(at=now(),returncode=proc.returncode))
if __name__=='__main__':main()
