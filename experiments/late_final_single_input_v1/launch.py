"""Single formal launcher, fixed eight jobs and bounded CPU delivery."""
import os,pathlib,signal,subprocess,sys,time,traceback,json
from common_run import ROOT,SCI,HERE,OUT,read,save,sha,now,spawn_gpu,release
def resources():
 d=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu','--format=csv,noheader'],text=True,timeout=30)
 ps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory','--format=csv,noheader'],text=True,timeout=30)
 rows={int(x.split(',')[0]):[v.strip() for v in x.split(',')] for x in d.splitlines()}
 for i in range(8):
  r=rows[i]
  if '2080 Ti' not in r[2] or int(r[3].split()[0])-int(r[4].split()[0])<9000 or any(r[1] in x for x in ps.splitlines()):raise RuntimeError('authorized GPU not available '+str(i))
 return dict(devices=rows,compute_processes=ps,permission='Latest explicit user authorization for eight node1 GPUs; no conflicting compute task or notified reservation; utilization recorded, not admission threshold',free_bytes=os.statvfs(OUT).f_bavail*os.statvfs(OUT).f_frsize,at=now())
def main():
 ledger=read(OUT/'ledger.json');start=time.time()
 if (OUT/'approval.json').exists():raise FileExistsError('formal activation already exists; no duplicate launch')
 if not read(OUT/'prepared/receipt.json')['complete'] or read(OUT/'CPU_checks.json')['status']!='passed':raise ValueError('CPU preparation gate')
 costs={}
 for ds in ['IMDB','PubMed']:
  r=read(OUT/'engineering'/ds/'receipt.json')
  if r['status']!='passed' or r['temporary_optimizer_steps']!=3:raise ValueError('integration gate')
  costs[ds]=max(x['update_seconds'] for x in r['records'])
 elapsed=time.time()-ledger['start_epoch'];remaining_science=ledger['science_stop_epoch']-time.time()
 # Short single-card costs x1.25 are planning only; chains use measured historical upper11.123min, rounded12.
 seconds={ds:(3000*c+10*60+24*60)*1.25 for ds,c in costs.items()}
 actual_engineering_GPU_hours=read(OUT/'watchdog_state.json')['charged_GPU_hours']
 predicted_GPU_hours=sum(4*x for x in seconds.values())/3600+actual_engineering_GPU_hours
 gate=dict(at=now(),measured_single_input_update_seconds=costs,predicted_per_model_seconds=seconds,
   historical_chain_minutes=11.123,planned_chain_minutes=12,planning_concurrency_factor=1.25,
   remaining_science_seconds=remaining_science,estimated_total_GPU_hours=predicted_GPU_hours,
   passed=max(seconds.values())<remaining_science-60 and predicted_GPU_hours<25,
   calibration_scope='six engineering updates, single-card costs; eight-card contention remains estimated')
 save(OUT/'cost_gate.json',gate)
 if not gate['passed']:raise RuntimeError('scope cannot fit remaining wall/GPU-hour budget')
 r=resources();save(OUT/'resources_before.json',r)
 if r['free_bytes']<12*1024**3:raise RuntimeError('output volume below12GiB')
 audit=read(ROOT/'results/frozen_late_method_audit_20261102T121837/source_code_and_metadata_bindings.json')
 for p,v in audit.items():
  if str(SCI) in p and sha(p)!=v['sha256']:raise ValueError('reviewed scientific source drift '+p)
 paths=list(HERE.glob('*.py'))+list((SCI/'package').rglob('*.py'))+list(SCI.glob('*.py'))+list((SCI/'lcc_eval').rglob('*.py'))
 paths += [SCI/'package/config.json',SCI/'package/assets/IMDB.npz',SCI/'package/assets/PubMed.npz']
 paths += list((SCI/'dictionaries').glob('dictionary_*/*'))
 paths += [OUT/n for n in ['ledger.json','protocol.json','tasks.json','initial_banks.json','dictionaries.json','input_contracts.json','initialization_pairing.json']]
 paths += list((OUT/'prepared').glob('*.json'))
 sources={str(p):sha(p) for p in sorted(set(paths)) if p.is_file()};save(OUT/'source_manifest.json',sources)
 tasks=read(OUT/'tasks.json');approval=dict(authorized=True,authorization=ledger['authorization'],issued=now(),launcher_pid=os.getpid(),host=ledger['host'],ledger_sha256=sha(OUT/'ledger.json'),source_files=sources,initial_banks=read(OUT/'initial_banks.json'),dictionaries=read(OUT/'dictionaries.json'),tasks=[j['task_id'] for j in tasks],GPU_UUIDs={str(i):r['devices'][i][1] for i in range(8)},cost_gate_sha256=sha(OUT/'cost_gate.json'))
 save(OUT/'approval.json',approval);save(OUT/'controller.json',dict(pid=os.getpid(),proc_start_ticks=pathlib.Path(f'/proc/{os.getpid()}/stat').read_text().split()[21]))
 active={};finished=[];risks=[]
 try:
  for j in tasks:
   tf=OUT/'tickets'/f"{j['task_id']}.json";save(tf,dict(job=j,GPU_UUID=r['devices'][j['gpu_slot']][1]));p,log,lease=spawn_gpu([HERE/'worker_run.py','--ticket',tf],j['gpu_slot'],j['task_id']);active[j['task_id']]=dict(process=p,log=log,lease=lease,job=j,started=time.time(),stopped=False)
  last_risk=0
  while active:
   t=time.time()
   for identifier,x in list(active.items()):
    code=x['process'].poll()
    if code is not None:
     release(x['lease']);x['log'].close();finished.append(dict(task_id=identifier,exit_code=code,finished=now(),summary=str(OUT/'run'/identifier/'summary.json')));del active[identifier]
   if t-last_risk>=60:
    risks=[]
    for identifier,x in active.items():
     f=OUT/'run'/identifier/'progress.json'
     if not f.exists():continue
     pr=read(f);u=pr.get('completed_updates',0)
     if pr['status']=='training' and u>=50:
      tail=(OUT/'run'/identifier/'training.jsonl').read_text().splitlines()[-30:];mean=sum(json.loads(z)['total_update_seconds'] for z in tail)/len(tail)
      predicted=(3000-u)*mean+28*60
      if predicted>ledger['science_stop_epoch']-t-30:
       risk=dict(task_id=identifier,at=now(),reason='measured throughput no longer fits unchanged scope',predicted_remaining_seconds=predicted,available=ledger['science_stop_epoch']-t);risks.append(risk)
       if not x['stopped']:x['process'].send_signal(signal.SIGTERM);x['stopped']=True;save(OUT/'logs'/f'{identifier}.budget_stop.json',risk)
    save(OUT/'budget_risks.json',dict(at=now(),risks=risks));last_risk=t
   save(OUT/'launcher_state.json',dict(at=now(),status='running',pid=os.getpid(),active={k:dict(pid=x['process'].pid,GPU=x['job']['gpu_slot']) for k,x in active.items()},finished=finished,science_stop=ledger['science_stop'],deadline=ledger['deadline']))
   if t>=ledger['science_stop_epoch']-45 or (OUT/'STOP_SCIENCE').exists():
    for x in active.values():
     if not x['stopped']:x['process'].send_signal(signal.SIGTERM);x['stopped']=True
   time.sleep(5)
 except BaseException as exc:
  save(OUT/'launcher_failure.json',dict(at=now(),error=repr(exc),traceback=traceback.format_exc()))
  for x in active.values():
   if x['process'].poll() is None:x['process'].send_signal(signal.SIGTERM)
  for identifier,x in active.items():
   try:x['process'].wait(timeout=45)
   except subprocess.TimeoutExpired:x['process'].kill();x['process'].wait()
   release(x['lease']);x['log'].close();finished.append(dict(task_id=identifier,exit_code=x['process'].returncode))
  active.clear()
 save(OUT/'GPU_release.json',dict(at=now(),all_eight_own_workers_exited=True,leases=[read(p) for p in (OUT/'leases').glob('*.json')],finished=finished))
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
 def cpu_stage(script,logname,reserve):
  with (OUT/'logs'/logname).open('x') as log:
   try:return subprocess.run([sys.executable,'-u',str(HERE/script)],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=max(1,ledger['deadline_epoch']-time.time()-reserve)).returncode
   except BaseException as e:save(OUT/(script+'.failure.json'),dict(at=now(),error=repr(e)));return None
 evaluation=cpu_stage('evaluate_all.py','evaluation.log',180)
 save(OUT/'completion.json',dict(at=now(),finished_jobs=finished,evaluation_exit_code=evaluation,elapsed_seconds=time.time()-ledger['start_epoch'],within_deadline=time.time()<=ledger['deadline_epoch']))
 save(OUT/'launcher_state.json',dict(at=now(),status='science_finished_CPU_delivery',finished=finished,evaluation_exit_code=evaluation))
 delivery=cpu_stage('finalize.py','delivery.log',30)
 save(OUT/'launcher_state.json',dict(at=now(),status='finished',finished=finished,evaluation_exit_code=evaluation,delivery_exit_code=delivery))
 save(OUT/'DONE',dict(at=now(),status='complete' if evaluation==0 and delivery==0 else 'incomplete_delivery'))
if __name__=='__main__':
 try:main()
 except BaseException as e:
  save(OUT/'activation_failure.json',dict(at=now(),error=repr(e),traceback=traceback.format_exc(),scientific_scope_not_reduced=True))
  save(OUT/'DONE',dict(at=now(),status='activation_failed'))
  raise
