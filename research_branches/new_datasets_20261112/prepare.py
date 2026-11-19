"""Freeze complete pilot before any formal model update; cost-based gate."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,time,datetime,zipfile
from pathlib import Path
from support import *
def main():
 out=Path(sys.argv[1]).resolve();assert not (out/'protocol.json').exists();gate=read(out/'preflight/receipt.json');assert gate['status']=='passed';assert read(out/'cpu_check/receipt.json')['status']=='passed'
 start=datetime.datetime.fromisoformat('2026-11-12T15:00:00+08:00').timestamp();science=start+210*60;delivery=start+240*60
 # Allocation starts at process/log launch, conservatively includes import and teardown.
 preflight_start=min((out/'preflight.log').stat().st_mtime,gate['start_epoch'])
 prior=(gate['end_epoch']-preflight_start+10)/3600
 jobs=[];(out/'contracts').mkdir();(out/'logs').mkdir(exist_ok=True)
 for dsi,ds in enumerate(('DBLP_HGEN','ACM_HAN_full7')):
  g=next(c for c in gate['conditions'] if c['dataset']==ds);case=case_for(ds)
  # 3000 CE updates + 8 training chains + 4 baseline and 4 final chains.
  update_cost=float(np.quantile(g['update_seconds'][1:],.9));step_cost=float(np.quantile(g['reverse_seconds'][1:],.9));nominal=3000*update_cost+16*1000*step_cost
  for seed in (2026111211,2026111212):
   task=f'{ds}_{seed}';cfg=config(ds,seed);rng=np.random.default_rng(np.random.SeedSequence([2026111202,seed]));contracts=[]
   for pos in range(1,3001):
    inp=single.one_input(case,cfg,pos,rng);contracts.append(dict(update=pos,t=inp['t'],input_hash=s.input_hash([inp]),query_hash=s.digest(inp['query']),slots=len(inp['ids'])))
   base.write(out/'contracts'/f'{task}.json',contracts)
   aux=np.random.default_rng(np.random.SeedSequence([2026111205,dsi,seed]));terminal={}
   for u in range(1,5):terminal[str(u)]=[dict(seed=int(aux.integers(1,2**31-1)),times=sorted(map(int,aux.choice(np.arange(1,1001),8,replace=False)),reverse=True)) for _ in range(2)]
   jobs.append(dict(id=task,dataset=ds,train_seed=seed,config=cfg,data=str(PREP/ds/'small_192.npz'),type_names=read(PREP/ds/'audit.json')['type_names'],terminal_plan=terminal,nominal_seconds=nominal,planning_seconds=1.25*nominal+600,measured_update_seconds=update_cost,measured_reverse_step_seconds=step_cost,whole_chain_cost_is_extrapolated=True))
 bindings={}
 for mod in list(sys.modules.values()):
  f=getattr(mod,'__file__',None)
  if f and str(f).startswith(str(ROOT)) and str(f).endswith('.py') and Path(f).exists():bindings[str(Path(f).resolve())]=base.sha(f)
 for f in Path(__file__).parent.glob('*.py'):bindings[str(f.resolve())]=base.sha(f)
 for ds in ('DBLP_HGEN','ACM_HAN_full7'):
  for name in ('small_192.npz','full_graph.npz','audit.json'):f=PREP/ds/name;bindings[str(f)]=base.sha(f)
 for f in (out/'contracts').glob('*.json'):bindings[str(f)]=base.sha(f)
 with zipfile.ZipFile(out/'frozen_source.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
  for f in bindings:
   if f.endswith('.py'):z.write(f,arcname=str(Path(f).relative_to(ROOT)))
 p=dict(start_epoch=start,science_stop_epoch=science,delivery_epoch=delivery,max_GPU_hours=12,max_GPUs=4,allowed_GPU_indices=[4,5,6,7],preflight_GPU_hours=prior,jobs=jobs,sample_seeds=[2026111221,2026111222,2026111223,2026111224],terminal_updates=4,bindings=bindings,formal_CE_models=4,CE_updates=12000,terminal_updates_total=16,training_chains=32,evaluation_chains=32,T=1000,R=1,initialization='fresh A lateCN per dataset and seed; terminal paired from its exact CE3000 EMA, fresh AdamW and EMA',readouts='baseline: CE3000 EMA; terminal: fixed raw after 4 updates, matching established terminal protocol; save both raw/EMA; no best checkpoint',score='unchanged type-equal joint heat-degree MMD + declared-family mean count risk; reference-only frozen scaling; inactive families remain in original denominator',bias='clean-head terminal LOO surrogate, not unbiased transition policy gradient or GDPO replication',evaluation='complete raw graph primary; LCC separately; per-training-model four draws then across two training seeds; all metrics and costs retained',gate='CPU generic metrics and historical 4-type equivalence; finite temporary update; exact saved/load state and next input; EMA restore; active coverage; fits wall/GPU budget. No effect threshold.',limits=['small induced samples are type-balanced connected regions, not representative random full graphs','full source data prepared; full-graph GPU training NOT included or demonstrated','single-reference generation; no independent graph test set','fresh dataset schemas change parameter dimensions; cross-dataset model initial weights not claimed identical','CUDA recomputed update bitwise equivalence not certified','DBLP bipartite support has no triangles/CN on legal edges; not evidence of CN structural benefit','ACM directions/weights simplified for current undirected binary model; source retained','no author baseline results or multi-graph generalization claim'])
 p['budget_gate']=dict(planning_GPU_hours=prior+sum(j['planning_seconds'] for j in jobs)/3600,longest_minutes=max(j['planning_seconds'] for j in jobs)/60,remaining_science_minutes=(science-time.time())/60,margin='25% concurrent throughput margin plus 10min/model loading/checkpoint/validation; CPU evaluation after GPU release')
 assert p['budget_gate']['planning_GPU_hours']<12 and time.time()+max(j['planning_seconds'] for j in jobs)<science,'complete panel does not fit; stop before formal training'
 base.write(out/'protocol.json',p);print(json.dumps(p['budget_gate']))
if __name__=='__main__':main()
