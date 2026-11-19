"""Prospective second development candidate, same window and shared32GPUh cap."""
import json,hashlib,shutil,tarfile,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];SRC=Path(__file__).resolve().parent
PARENT=ROOT/'results/joint_output_formal_20261108T021741';CONTROL=ROOT/'results/joint_output_formal_controls_20261108T025000'
OUT=ROOT/'results/joint_output_formal_projected_20261108T043000'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def main():
 if (OUT/'source_binding.json').exists():raise RuntimeError('already frozen')
 for d in ['engineering','prepared','leases','logs','run','evaluation']:(OUT/d).mkdir(parents=True,exist_ok=True)
 assert read(ROOT/'results/projected_degree_cpu_20261108/receipt.json')['passed']
 ledger=read(PARENT/'ledger.json');ledger.update(shared_lease_roots=[str(PARENT),str(CONTROL),str(OUT)],same_ticket=True,engineering_GPU_hours_limit=.25)
 save(OUT/'ledger.json',ledger)
 save(OUT/'approval.json',dict(authorized=True,launcher_pid=None,scope='conditional8PS O/A models3000+32chains; only if real full-panel budget fits shared32GPUh/original8h',
  development=True,preregistered_before_this_candidate_generation=True,independent_confirmation=False,
  reason='JK development outcomes motivate retaining PB metric and type weights while adding symmetric within-node pair projections; no coefficient grid',
  maximum_further_candidates=0,user_instruction='8h autonomous output correction; resources authorized; preserve previous experiments'))
 old=read(CONTROL/'tasks.json');tasks=[]
 for arch in ['A','O']:
  for i,t in enumerate(x for x in old if x['architecture']==arch):
   tasks.append(dict(t,task_id=f"{t['dataset']}_{t['train_seed']}_{arch}_PS",arm=arch+'_PS',objective_kind='PS',gpu=len(tasks),queue_order=len(tasks)))
 save(OUT/'tasks.json',tasks)
 jobs=read(CONTROL/'evaluation_jobs.json')
 for j in jobs:
  j['reused']=True;j['metric_dir']=str(CONTROL/'evaluation'/j['task_id']/'u3000'/str(j['sample_seed']))
 for t in tasks:
  folder=OUT/'run'/t['task_id']
  for ss in [2026102821,2026102822,2026110231,2026110232]:jobs.append(dict(dataset=t['dataset'],train_seed=t['train_seed'],arm=t['arm'],task_id=t['task_id'],base_task=t['task_id'],updates=3000,sample_seed=ss,
   folder=str(folder/'ema_0.99'/f'sample_{ss}'),readout=str(folder/'readout_ema_0.99.pt'),configuration=t['arm']+'_late',reused=False))
 save(OUT/'evaluation_jobs.json',jobs)
 for p in (PARENT/'prepared').glob('*.json'):shutil.copy2(p,OUT/'prepared'/p.name)
 sources=read(CONTROL/'source_binding.json')
 for p,h in sources.items():assert sha(p)==h,p
 for p in SRC.rglob('*.py'):sources[str(p)]=sha(p)
 save(OUT/'source_binding.json',sources);save(OUT/'assets.json',read(CONTROL/'assets.json'))
 save(OUT/'source_version.json',dict(parent_manifest=str(CONTROL/'source_binding.json'),parent_sha256=sha(CONTROL/'source_binding.json'),manifest_sha256=sha(OUT/'source_binding.json'),identity='prospective isolated pair-rotation CDF; development candidate2'))
 save(OUT/'budget.json',dict(fits=False,reason='must measure real projected objective engineering cost before conditional training'))
 with tarfile.open(OUT/'frozen_source.tar.gz','w:gz') as f:
  for p in sources:f.add(p,arcname=str(Path(p).relative_to(ROOT)))
 print(str(OUT))
if __name__=='__main__':main()
