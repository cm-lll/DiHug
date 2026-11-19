"""Read existing bindings, hash assets, and estimate a future run; never launch."""
import json,hashlib,statistics,time,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/pb_degree_preparation_20261106';OUT.mkdir(exist_ok=True)
BASE=ROOT/'results/cn_multiseed_20261104T042546'
RECENT=ROOT/'results/structure_input_formal_triangle_20261106T131231'
SAMPLES=[2026102821,2026102822,2026110231,2026110232]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(n,x):(OUT/(n+'.json')).write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def main():
 started=time.monotonic();manifest=read(BASE/'source_manifest.json');diff=[]
 for p,h in manifest.items():
  if not Path(p).exists() or sha(p)!=h:diff.append(p)
 tasks=[];graphs=[];cost=[]
 for r in read(RECENT/'tasks.json'):
  task=r['task_id'];ds=r['dataset'];seed=r['train_seed'];folder=BASE/'run'/task;b=read(folder/'binding.json');cfg=b['config']
  assert b['arm']=='A' and b['configuration']=='A_late' and b['updates']==3000 and cfg['inputs_per_update']==1 and cfg['rounds']==1 and cfg['T']==1000
  assert sha(BASE/'source_manifest.json')==b['source_manifest_sha256']
  initial=Path(r['initial']);assert sha(initial)==r['initial_sha256']
  ro=folder/'readout_ema_0.99.pt';rh=sha(ro);assert rh==read(ro.with_suffix('.sha256.json'))['sha256']
  lines=[json.loads(x) for x in (folder/'training.jsonl').read_text().splitlines()];assert len(lines)==3000 and [x['update'] for x in lines]==list(range(1,3001)) and all(x['paired_input_exact'] for x in lines)
  entry=dict(task=task,dataset=ds,train_seed=seed,initial=str(initial),initial_sha256=r['initial_sha256'],readout=str(ro),readout_sha256=rh,source_manifest_sha256=sha(BASE/'source_manifest.json'),config=cfg,training_log=str(folder/'training.jsonl'),training_log_sha256=sha(folder/'training.jsonl'),inputs=3000,slots_per_input=lines[0]['slots'],schema_denominator=lines[0]['S'],sampling_seeds=SAMPLES)
  times=[]
  for ss in SAMPLES:
   p=folder/'ema_0.99'/f'sample_{ss}';binding=read(p/'binding.json');graph=p/'graph.npz';gh=sha(graph)
   assert binding['status']=='complete' and binding['graph_sha256']==gh and binding['model_sha256']==rh and binding['sample_seed']==ss and binding['train_seed']==seed and binding['updates']==3000 and binding['R']==1 and binding['T']==1000 and binding['readout']=='EMA0.99' and binding['canonical_all1000_plans_and_draws']
   times.append(binding['seconds']);graphs.append(dict(dataset=ds,seed=seed,sample=ss,file=str(graph),sha256=gh,binding_file=str(p/'binding.json')))
  tasks.append(entry)
  train=sum(x['total_update_seconds'] for x in lines)/60;chain=sum(times)/60
  # Additional GPU loss cost is an admission scenario, not a measurement.
  planned_train=(train+3000*.30/60)*1.15;planned_chain=chain*1.20;device=planned_train+planned_chain+5
  cost.append(dict(task=task,baseline_training_minutes=train,baseline_four_chain_minutes=chain,assumed_extra_loss_seconds_per_update=.30,extra_loss_is_measured_GPU=False,planned_training_minutes=planned_train,planned_four_chain_minutes=planned_chain,load_save_minutes=5,device_minutes=device))
 save('assets',dict(tasks=tasks,graphs=graphs,source_drift=diff,baseline_source_files=len(manifest),initials_rehashed=4,EMA_readouts_rehashed=4,graphs_rehashed=len(graphs),missing=[],seconds=time.monotonic()-started))
 gh=sum(r['device_minutes'] for r in cost)/60+.25+.25;science=20+max(r['device_minutes'] for r in cost);wall=science+30
 save('budget',dict(formal_authorized=False,tasks=cost,engineering_max_GPU_hours=.25,additional_GPU_margin_hours=.25,total_planned_GPU_hours=gh,CPU_preparation_and_engineering_window_minutes=20,planned_science_minutes=science,planned_wall_minutes=wall,proposed_wall_limit_minutes=240,science_stop_minutes=210,proposed_GPU_hours_limit=12,scenario_fits=gh<=12 and science<=210,actual_GPU_loss_cost_unknown=True,launch_requires_actual_calibration=True,current_GPU_availability_not_claimed=True,disk_free_GiB=shutil.disk_usage(ROOT).free/2**30,CPU_saved_loss_cost_file=str(OUT/'saved_probability_checks.json')))
 assert not diff,'Frozen baseline source drift; report and do not claim launch readiness'
 print(json.dumps(dict(assets=4,graphs=len(graphs),planned_GPU_hours=gh,planned_wall_minutes=wall,source_drift=diff),indent=2))
if __name__=='__main__':main()
