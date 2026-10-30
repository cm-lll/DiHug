"""One bounded eight-card queue. No retries, restarts or additional candidates."""
import os,sys,time,subprocess,traceback,datetime
from common import *
PYTHON='/data2/lyh/miniconda3/envs/hgen_env/bin/python'
def resources():
 cards=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.free,utilization.gpu','--format=csv,noheader,nounits'],universal_newlines=True)
 jobs=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],universal_newlines=True)
 parsed=[]
 for row in cards.splitlines():
  i,u,n,free,util=[x.strip() for x in row.split(',')];parsed.append(dict(index=int(i),uuid=u,name=n,free_mib=int(free),utilization=int(util)))
 assert len(parsed)==8 and not jobs.strip(),'eight available devices required; do not stop anyone'
 assert all('2080 Ti' in x['name'] and x['free_mib']>=9000 for x in parsed)
 return dict(cards=parsed,compute_processes=jobs,permission_basis='same node1 eight2080Ti resource pool explicitly allowed by user; latest user requested multi-card repeated HGEN tests; absence of processes alone is not authorization',time=time.time())
def cost():
 rows=[]
 for d in PROTOCOL['datasets']:
  p=read(OUT/'engineering'/d/'receipt.json');a=read(OUT/'data'/d/'embedding_receipt.json');assert p['status']=='passed' and a['status']=='complete'
  train=3750*p['native_D_period_p95_seconds'];sample=3000*p['eight_walk_batches_seconds']/8
  # Existing native assembler is CPU-only; allowance is a planning assumption,
  # not a throughput measurement or a substitute for the per-request watchdog.
  model=(train+sample+60)*1.5;assembly=200
  rows.append(dict(dataset=d,train_seconds_p95_extrapolation=train,walk_seconds_extrapolation=sample,save_IO_allowance_seconds=60,concurrency_planning_margin=1.5,model_phase_with_margin=model,assembly_two_graphs_planning_seconds=assembly,assembly_two_graphs_hard_watchdog_seconds=1200,critical_path_planned_seconds=model+assembly,asset_actual_seconds=a['seconds']))
 planned=max(r['critical_path_planned_seconds'] for r in rows);remaining=read(OUT/'ledger.json')['science_stop_epoch']-time.time()
 result=dict(status='passed' if planned<remaining else 'budget_insufficient',rows=rows,queue=[dict(gpu=i,dataset=d,train_seed=s,D_updates=3750,G_updates=750,sample_seeds=PROTOCOL['sample_seeds']) for i,(d,s) in enumerate((d,s) for d in PROTOCOL['datasets'] for s in PROTOCOL['train_seeds'][d])],planned_science_seconds=planned,remaining_science_seconds=remaining,all_eight_parallel=True,measured_simultaneous_eight_card_throughput=False,CPU_evaluation_planning_seconds=300,delivery_reserve_seconds=900,clock=read(OUT/'ledger.json'))
 save(OUT/'cost.json',result);return result
def main():
 if (OUT/'formal_started.json').exists():raise FileExistsError('already launched; no retry')
 assert read(OUT/'cpu_tests.json')['passed']==8
 plan=cost();assert plan['status']=='passed','remaining window cannot accommodate fixed scope'
 r=resources();save(OUT/'resources_at_launch.json',r)
 manifest=source_manifest();save(OUT/'source_manifest.json',manifest)
 queue=plan['queue'];now=time.time();save(OUT/'formal_started.json',dict(start_epoch=now,start_Asia_Shanghai=datetime.datetime.fromtimestamp(now,datetime.timezone(datetime.timedelta(hours=8))).isoformat(),queue=queue,cards=r['cards'],scope='8 native HGEN fits and16 assemblies; no diffusion training',authorization='user approved baseline tests and multi-card repeats; quantitative gates input/engineering/CPU/cost/resources passed',clock=read(OUT/'ledger.json'),source_manifest_sha256=sha(OUT/'source_manifest.json')))
 children=[]
 for j in queue:
  log=(OUT/(cell(j['dataset'],j['train_seed'])+'.log')).open('w');env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(j['gpu']),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',TMPDIR=str(OUT/'tmp'),PYTHONDONTWRITEBYTECODE='1')
  p=subprocess.Popen([PYTHON,'-B',str(ROOT/'worker.py'),j['dataset'],str(j['train_seed'])],env=env,stdout=log,stderr=subprocess.STDOUT);children.append((p,log,j))
 save(OUT/'startup.json',dict(launcher_pid=os.getpid(),workers=[dict(j,pid=p.pid) for p,_,j in children]))
 done=[]
 while children:
  for item in list(children):
   p,log,j=item
   if p.poll() is not None:log.close();done.append(dict(j,exit_code=p.returncode,finished_epoch=time.time()));children.remove(item)
  save(OUT/'launcher_progress.json',dict(running=[dict(j,pid=p.pid) for p,_,j in children],finished=done))
  # Worker checks its own absolute science stop at every update/model batch;
  # assembly retains the already audited600s per-request watchdog. No restarts.
  if children:time.sleep(5)
 save(OUT/'scientific_completion.json',dict(epoch=time.time(),finished=done,new_HGEN_D_updates=sum(3750 for x in done if x['exit_code']==0),new_HGEN_G_updates=sum(750 for x in done if x['exit_code']==0)))
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',TMPDIR=str(OUT/'tmp'))
 with (OUT/'evaluation.log').open('w') as f:
  p=subprocess.run([PYTHON,'-B',str(ROOT/'evaluate.py')],env=env,stdout=f,stderr=subprocess.STDOUT)
 save(OUT/'evaluation_exit.json',dict(exit_code=p.returncode,epoch=time.time()))
 print('HGEN science and CPU evaluation finished',time.time(),p.returncode)
if __name__=='__main__':
 try:main()
 except BaseException:save(OUT/'launcher_failure.json',dict(traceback=traceback.format_exc(),epoch=time.time()));raise
