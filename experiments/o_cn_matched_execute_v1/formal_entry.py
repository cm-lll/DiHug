"""New authorized wall binding; reuse all passed checks, never re-run engineering."""
import os,sys,time,pathlib,datetime,socket,subprocess,traceback
from common_run import ROOT,OUT,HERE,read,save,sha,now
from launch import resources

OLD=ROOT/'results/o_cn_matched_normal1_20261103T163400'
PLAN=ROOT/'results/o_cn_budget_replan_20261103T2030'

def main():
 began=time.time()
 if (OUT/'ledger.json').exists()or (OUT/'approval.json').exists():raise FileExistsError('formal stage already bound; no duplicate start')
 for n in ['run','prepared','engineering','logs','leases','tickets','temporary']:(OUT/n).mkdir(parents=True,exist_ok=True)
 plan=read(PLAN/'budget.json')
 if plan['status']!='asset_and_budget_gate_passed'or plan['issues']or plan['estimated_total_GPU_hours']>=12:raise ValueError('asset/budget gate not passed')
 # Carry forward ACTUAL released card time, not the previous wall-window remainder.
 prior=0.;leases=list((OLD/'leases').glob('*.json'))
 if len(leases)!=2:raise ValueError('unexpected historical lease count; do not lose accounting')
 for p in leases:
  r=read(p)
  if not r.get('released_epoch'):raise ValueError('historical GPU lease not released')
  prior+=(r['released_epoch']-r['acquired_epoch'])/3600
 if abs(prior-plan['existing_GPU_hours'])>1e-12:raise ValueError('historical charged time changed')
 tz=datetime.timezone(datetime.timedelta(hours=8));iso=lambda t:datetime.datetime.fromtimestamp(t,tz).isoformat()
 ledger=dict(start_epoch=began,start=iso(began),science_stop_epoch=began+165*60,science_stop=iso(began+165*60),
       deadline_epoch=began+195*60,deadline=iso(began+195*60),host=socket.gethostname(),
       wall_minutes=195,science_minutes=165,delivery_reserve_minutes=30,GPU_hour_limit=12,prior_GPU_hours=prior,
       previous_window_closed=True,previous_ledger=str(OLD/'ledger.json'),previous_ledger_sha256=sha(OLD/'ledger.json'),
       budget_basis=str(PLAN/'budget.json'),budget_basis_sha256=sha(PLAN/'budget.json'),
       authorization='User explicitly says start fixed O→A test if current assets and budget pass:4 O models3000 CE normal1 EMA.99;2 paired T1000 chains each;4GPUs; carry prior charge into12GPUh. New195min wall binding within previously proposed240min; no repeated GPU short checks.')
 save(OUT/'ledger.json',ledger);save(OUT/'ENTRY_CLAIM.json',dict(at=now(),pid=os.getpid(),one_stage=True))
 # No model construction in this entry; immutable weights remain at their old paths.
 for n in ['initial_banks.json','reused_A_jobs.json','pairing_scope.json','protected_parent_assets.json','dictionaries.json','protocol.json']:
  save(OUT/n,read(OLD/n))
 tasks=read(OLD/'tasks.json')
 for j in tasks:j['folder']=str(OUT/'run'/j['task_id'])
 save(OUT/'tasks.json',tasks)
 for p in (OLD/'prepared').glob('*.json'):save(OUT/'prepared'/p.name,read(p))
 for j in tasks:
  b=read(OUT/'initial_banks.json')[j['task_id']]
  if sha(b['file'])!=b['sha256']:raise ValueError('frozen theta0 hash changed')
 reused={}
 for ds in ['IMDB','PubMed']:
  p=OLD/'engineering'/ds/'receipt.json';r=read(p)
  if r['status']!='passed'or not all(r['replayed_state_exact'].values())or not r['generation_benchmark_training_state_unchanged']:raise ValueError('historical recovery/timing gate')
  save(OUT/'engineering'/ds/'receipt.json',r);reused[ds]=dict(source=str(p),sha256=sha(p),updates_repeated=0)
 save(OUT/'reused_checks.json',dict(at=now(),sources=reused,prior_GPU_hours=prior,repeated_GPU_updates=0,repeated_model_forwards=0))
 save(OUT/'CPU_checks.json',dict(read(OLD/'CPU_checks.json'),reused_from=str(OLD/'CPU_checks.json'),repeated_checks=False))
 save(OUT/'resume_import_check.json',dict(source=str(OLD/'resume_import_check.json'),sha256=sha(OLD/'resume_import_check.json'),new_runtime=str(HERE)))
 save(OUT/'asset_budget_gate.json',plan)
 # Snapshot plan has no right to reserve GPUs: inspect again before workers start.
 rs=resources();save(OUT/'resources_at_entry.json',rs)
 if rs['free_bytes']<12*1024**3:raise RuntimeError('data disk reserve below12GiB')
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='',CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TMPDIR=str(OUT/'temporary'))
 with (OUT/'logs/watchdog.log').open('x')as log:
  watch=subprocess.Popen([sys.executable,'-u',str(HERE/'watchdog.py')],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 save(OUT/'watchdog_process.json',dict(pid=watch.pid,at=now()))
 wait_until=time.time()+10
 while not (OUT/'watchdog_state.json').exists():
  if watch.poll()is not None or time.time()>wait_until:raise RuntimeError('watchdog did not initialize')
  time.sleep(.1)
 with (OUT/'logs/launcher.log').open('x')as log:
  launcher=subprocess.Popen([sys.executable,'-u',str(HERE/'launch.py')],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  save(OUT/'launcher_process.json',dict(pid=launcher.pid,at=now()))
  code=launcher.wait()
 if not (OUT/'DONE').exists():save(OUT/'DONE',dict(at=now(),status='launcher_failed',exit_code=code))
 try:watch.wait(timeout=20)
 except subprocess.TimeoutExpired:save(OUT/'watchdog_wait_issue.json',dict(at=now(),pid=watch.pid))
 save(OUT/'formal_entry_exit.json',dict(at=now(),launcher_exit=code,engineering_not_repeated=True))

if __name__=='__main__':
 try:main()
 except BaseException as exc:
  save(OUT/'entry_failure.json',dict(at=now(),error=repr(exc),traceback=traceback.format_exc()))
  save(OUT/'DONE',dict(at=now(),status='entry_failed'))
  raise
