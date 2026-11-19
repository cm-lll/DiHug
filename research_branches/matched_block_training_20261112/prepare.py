"""Bind an explicitly resumed ticket; audit existing assets without model calls."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import sys,json,time,hashlib,datetime,py_compile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'research_branches/one_step_vs_terminal_plan_20261111'))
import update_preflight as s
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def main():
 out=Path(sys.argv[1]).resolve();out.mkdir(exist_ok=False);(out/'logs').mkdir()
 old=ROOT/'results/terminal_continuation_20261112T003000';p=read(old/'protocol.json')
 panel=read(ROOT/'results/block_generation_panel_20261112T015400/protocol.json')
 bindings=p['bindings'].copy()
 for path,h in bindings.items():assert sha(path)==h,path
 rows=[];reuse=[];checks=[]
 for ds in ('IMDB','PubMed'):
  state=s.torch.load(Path(p['resume_parent'])/(ds+'_T_eager/final_complete.pt'),map_location='cpu',weights_only=False)
  assert state['updates']==state['ema']['updates']==2 and state['recovery']['exact_resume_allowed']
  assert {float(v['step']) for v in state['optimizer']['state'].values()}=={2.}
  data=s.np.load(p['parents'][ds]['data']);ee=data['edges'];ee=ee.T if ee.shape[0]!=2 else ee
  case=s.w.common.make_case(data['types'],ee,tuple(map(tuple,data['schema'])))
  for r in (10,20):
   observed=[read(Path(j['directory'])/'summary.json')['seconds'] for j in panel['all_jobs'] if j['dataset']==ds and j['rounds']==r]
   # Six complete chains + loading/checkpoint/feedback allowance per card.
   seconds=6*max(observed)+180
   rows.append(dict(id=f'{ds}_R{r}',dataset=ds,arm=f'R{r}',rounds=r,train_seed=p['parents'][ds]['train_seed'],observed_chain_seconds=observed,nominal_seconds=seconds,planning_seconds=seconds*1.10))
   for t in (1,500,1000):
    checks.append(dict(dataset=ds,rounds=r,t=t,inventory=s.w.common.query_inventory(case,s.w.common.plan_for(case,[2026111341,t,61],r))))
  for ss in p['sample_seeds']:
   d=old/(ds+'_T_eager')/'stage_4'/f'final_sample_{ss}'
   ident=read(d/'identity.json');summary=read(d/'summary.json')
   assert ident['updates']==4 and ident['R']==1 and ident['sample_seed']==ss and summary['steps']==1000
   assert sha(d/'graph.npz')==summary['graph_sha256']
   for name in ('identity.json','summary.json','steps.jsonl','graph.npz'):bindings[str(d/name)]=sha(d/name)
   reuse.append(dict(dataset=ds,arm='R1',sample_seed=ss,directory=str(d)))
  del state
 start=datetime.datetime.fromisoformat('2026-11-12T14:47:00+08:00').timestamp()
 for mod in list(sys.modules.values()):
  path=getattr(mod,'__file__',None)
  if path and path.endswith('.py') and str(ROOT) in str(Path(path).resolve()):bindings[str(Path(path).resolve())]=sha(path)
 for f in Path(__file__).parent.glob('*.py'):py_compile.compile(str(f),doraise=True);bindings[str(f)]=sha(f)
 gpu=sum(x['planning_seconds'] for x in rows)/3600
 p.update(start_epoch=start,science_stop_epoch=start+330*60,delivery_epoch=start+360*60,max_GPU_hours=18,GPU_hours=18,max_GPUs=4,
  bindings=bindings,arms=['R10','R20'],final_updates=4,evaluate_at_updates=[4],terminal_rounds=4,
  terminal_plan={d:{u:v for u,v in pp.items() if int(u)<=4} for d,pp in p['terminal_plan'].items()},
  training_chain_count=16,new_evaluation_chains=8,reused_evaluation_graphs=4,new_terminal_rounds_per_dataset_and_arm=2,
  new_jobs=rows,reused=reuse,training_full_sweep=True,forecast_GPU_hours=gpu,
  stopping='330min science /360min delivery /18 allocated GPUh, first applicable limit; new ticket authorized after disk cleanup',
  old_unstarted_window_not_reopened=True,primary_readout='update4 raw; EMA validation separate',
  limitations=['one development training seed per dataset','two evaluation draws per model','R1 parent then two matched-block adaptation updates, not from-scratch block training','clean-head terminal surrogate is biased, not exact transition REINFORCE','CUDA recomputed updates not certified bitwise'])
 p['budget_gate']=dict(planning_GPU_hours=gpu,preflight_reserve_GPU_hours=.20,wall_remaining_seconds=p['science_stop_epoch']-time.time(),longest_planning_seconds=max(x['planning_seconds'] for x in rows),margin_factor=1.10)
 assert gpu+.20<18 and time.time()+max(x['planning_seconds'] for x in rows)+300<p['science_stop_epoch'],'full-panel forecast does not fit'
 write(out/'protocol.json',p);write(out/'CPU_preflight.json',dict(status='passed',checks=checks,bindings=len(bindings),GPU=0,model_calls=0,budget=p['budget_gate']))
 print(json.dumps(dict(output=str(out),budget=p['budget_gate'],jobs=rows),indent=2))
if __name__=='__main__':main()
