"""One owner, eight-card fixed queue, independent watchdog and CPU delivery."""
import json,os,pathlib,signal,socket,subprocess,sys,time,traceback
from common_run import ROOT,CHECKOUT,SCI,HERE,OUT,read,save,sha,now,spawn_gpu,release

def resources():
 raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu','--format=csv,noheader'],text=True,timeout=30)
 ps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory','--format=csv,noheader'],text=True,timeout=30)
 rows={int(x.split(',')[0]):[v.strip()for v in x.split(',')]for x in raw.splitlines()}
 for i in range(8):
  r=rows[i]
  if '2080 Ti'not in r[2]or int(r[3].split()[0])-int(r[4].split()[0])<9000 or any(r[1]in s for s in ps.splitlines()):raise RuntimeError('GPU unavailable '+str(i))
 return dict(devices=rows,compute_processes=ps,free_bytes=os.statvfs(OUT).f_bavail*os.statvfs(OUT).f_frsize,host=socket.gethostname(),at=now())

def alive(r):
 try:
  s=pathlib.Path('/proc')/str(r['pid'])/'stat';v=s.read_text().split();return v[2]!='Z'and v[21]==r['proc_start_ticks']
 except(FileNotFoundError,ProcessLookupError):return False

def watchdog():
 ledger=read(OUT/'ledger.json');dead={};sent={};total=0
 while True:
  t=time.time();active=[];total=0
  for p in(OUT/'leases').glob('*.json'):
   r=read(p);running=alive(r)
   if not running and str(p)not in dead:dead[str(p)]=t
   end=r.get('released_epoch')or dead.get(str(p))or t;total+=max(0,end-r['acquired_epoch'])
   if running:active.append(r)
  stop=t>=ledger['science_stop_epoch']-60 or total>=ledger['GPU_hour_limit']*3600-8*60
  if stop:
   if not(OUT/'STOP_SCIENCE').exists():save(OUT/'STOP_SCIENCE',dict(at=now(),reason='wall/GPUh ceiling',GPU_hours=total/3600))
   for r in active:
    if r['pid']not in sent:os.kill(r['pid'],signal.SIGTERM);sent[r['pid']]=t
    if t>=ledger['science_stop_epoch']or t-sent[r['pid']]>45:
     if alive(r):os.kill(r['pid'],signal.SIGKILL)
  save(OUT/'watchdog_state.json',dict(at=now(),pid=os.getpid(),active=active,charged_GPU_hours=total/3600,stop_triggered=stop))
  if(OUT/'DONE').exists()and not active:break
  if t>=ledger['deadline_epoch']:
   for r in active:
    if alive(r):os.kill(r['pid'],signal.SIGKILL)
   break
  time.sleep(5)
 save(OUT/'watchdog_exit.json',dict(at=now(),GPU_hours=total/3600))

def launch():
 ledger=read(OUT/'ledger.json')
 if(OUT/'approval.json').exists():raise FileExistsError('No duplicate activation')
 if read(OUT/'prepared/receipt.json')['status']!='passed':raise ValueError('asset gate')
 for n in ['run','logs','leases','tickets']:(OUT/n).mkdir(exist_ok=True)
 jobs=read(OUT/'tasks.json');rs=resources();save(OUT/'resources_at_start.json',rs)
 if rs['free_bytes']<20*1024**3:raise RuntimeError('insufficient output data volume reserve')
 # Extrapolate from real completed full-size updates/chains, not synthetic speed.
 costs={};per=[]
 for ds,arm,u,c in [('IMDB','O',1.4,500),('IMDB','A',1.45,510),('PubMed','O',1.8,650),('PubMed','A',1.85,670)]:costs[(ds,arm)]=(3000*u+4*c+480)*1.2
 available=[0.]*8
 for j in jobs:
  slot=min(range(8),key=lambda i:(available[i],i));secs=costs[(j['dataset'],j['arm'])];available[slot]+=secs;per.append(dict(task_id=j['task_id'],estimated_seconds=secs,planned_GPU=slot))
 predicted=sum(x['estimated_seconds']for x in per)/3600
 gate=dict(at=now(),per_job=per,critical_queue_seconds=max(available),estimated_GPU_hours=predicted,remaining_science_seconds=ledger['science_stop_epoch']-time.time(),
          training_seconds_basis={'IMDB_O':1.4,'IMDB_A':1.45,'PubMed_O':1.8,'PubMed_A':1.85},chain_seconds_basis={'IMDB_O':500,'IMDB_A':510,'PubMed_O':650,'PubMed_A':670},validation_save_allowance_seconds=480,concurrency_factor=1.2,source='Completed O/A3000 full-graph training logs and full T1000 chain timings, plus measured recovery overhead;8-card contention is planning uncertainty')
 gate['passed']=max(available)<gate['remaining_science_seconds']-60 and predicted<ledger['GPU_hour_limit']
 save(OUT/'cost_gate.json',gate)
 if not gate['passed']:raise RuntimeError('Complete fixed panel does not fit remaining wall/GPUh budget')
 # Scientific kernels exactly retain175373c; only declared execution wiring changes.
 bound=[]
 for p in CHECKOUT.rglob('*.py'):
  if '.git'not in p.parts and '__pycache__'not in p.parts:bound.append(p)
 bound +=[SCI/'package/config.json',SCI/'package/assets/IMDB.npz',SCI/'package/assets/PubMed.npz']
 bound +=[OUT/n for n in ['ledger.json','preregistration.json','tasks.json','initial_banks.json','reused_jobs.json','protected_parent_assets.json','cost_gate.json']]
 bound+=list((OUT/'prepared').glob('*.json'))
 sources={str(p):sha(p)for p in sorted(set(bound))};save(OUT/'source_manifest.json',sources)
 approval=dict(authorized=True,authorization=ledger['authorization'],host=socket.gethostname(),launcher_pid=os.getpid(),source_files=sources,
   scientific_parent='175373c83202b6c8b80aa570e0092c4652031ce6',runtime_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=CHECKOUT,text=True).strip(),initial_banks=read(OUT/'initial_banks.json'),dictionaries={},tasks=[j['task_id']for j in jobs],GPU_UUIDs={str(i):rs['devices'][i][1]for i in range(8)})
 save(OUT/'approval.json',approval)
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
 with(OUT/'logs/watchdog.log').open('x')as log:
  watch=subprocess.Popen([sys.executable,'-u',__file__,'--watchdog'],env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
 save(OUT/'watchdog_process.json',dict(pid=watch.pid,at=now()))
 active={};finished=[];pending=list(jobs);assignments=[]
 def issue(j,gpu):
  jj=dict(j,gpu_slot=gpu);tf=OUT/'tickets'/f"{j['task_id']}.json";save(tf,dict(job=jj,GPU_UUID=rs['devices'][gpu][1]))
  p,log,lease=spawn_gpu([HERE/'worker_run.py','--ticket',tf],gpu,j['task_id']);active[j['task_id']]=dict(process=p,log=log,lease=lease,job=jj)
  assignments.append(dict(task_id=j['task_id'],GPU=gpu,UUID=rs['devices'][gpu][1],pid=p.pid,at=now()));save(OUT/'device_assignments.json',assignments)
 try:
  while pending or active:
   for key,x in list(active.items()):
    code=x['process'].poll()
    if code is not None:
     release(x['lease']);x['log'].close();finished.append(dict(task_id=key,exit_code=code,at=now()));del active[key]
   stopping=(OUT/'STOP_SCIENCE').exists()or time.time()>=ledger['science_stop_epoch']-60
   if stopping:
    for x in active.values():
     if x['process'].poll()is None:x['process'].send_signal(signal.SIGTERM)
    if pending:save(OUT/'not_started.json',pending);pending=[]
   else:
    used={x['job']['gpu_slot']for x in active.values()}
    for gpu in range(8):
     if gpu not in used and pending:issue(pending.pop(0),gpu)
   save(OUT/'launcher_state.json',dict(at=now(),status='running',active={k:dict(pid=x['process'].pid,GPU=x['job']['gpu_slot'])for k,x in active.items()},pending=[j['task_id']for j in pending],finished=finished))
   if active:time.sleep(5)
 except BaseException as exc:
  save(OUT/'launcher_failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),at=now()))
  for x in active.values():
   if x['process'].poll()is None:x['process'].send_signal(signal.SIGTERM)
  for key,x in active.items():
   try:x['process'].wait(timeout=45)
   except subprocess.TimeoutExpired:x['process'].kill();x['process'].wait()
   release(x['lease']);x['log'].close();finished.append(dict(task_id=key,exit_code=x['process'].returncode))
 save(OUT/'GPU_release.json',dict(at=now(),finished=finished,leases=[read(p)for p in(OUT/'leases').glob('*.json')]))
 for script in ['evaluate_multiseed.py','finalize_multiseed.py']:
  with(OUT/'logs'/(script+'.log')).open('x')as log:
   try:p=subprocess.run([sys.executable,'-u',str(HERE/script)],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=max(1,ledger['deadline_epoch']-time.time()-60));save(OUT/(script+'.exit.json'),dict(exit_code=p.returncode,at=now()))
   except BaseException as exc:save(OUT/(script+'.failure.json'),dict(error=repr(exc),at=now()))
 save(OUT/'DONE',dict(at=now(),finished=finished))
 try:watch.wait(timeout=15)
 except subprocess.TimeoutExpired:pass

if __name__=='__main__':
 if '--watchdog'in sys.argv:watchdog()
 else:
  try:launch()
  except BaseException as exc:save(OUT/'activation_failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),at=now()));raise
