import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']: os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import csv,hashlib,importlib.util,json,resource,signal,time,traceback
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]; HERE=Path(__file__).resolve().parent
OUT=ROOT/'results/multiobjective_equalstep_cpu_20261110'; OLD=ROOT/'results/one_step_to_terminal_cpu_20261110'
PREV=ROOT/'results/multiobjective_cpu_20261110'; BIND={}; START=time.monotonic(); COUNT=0
def bind(p):
 p=Path(p); BIND[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest(); return p
def read(p): return json.loads(bind(p).read_text())
def save(name,x):
 (OUT/(name+'.json')).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False))
 if isinstance(x,list) and x:
  with (OUT/(name+'.csv')).open('w') as f:
   w=csv.DictWriter(f,list(dict.fromkeys(k for r in x for k in r)));w.writeheader();w.writerows(x)
def load_module():
 p=bind(ROOT/'research_branches/multiobjective_cpu_20261110/run_cpu.py')
 s=importlib.util.spec_from_file_location('frozen_mgda',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def main():
 global COUNT
 OUT.mkdir(exist_ok=False); signal.alarm(180); resource.setrlimit(resource.RLIMIT_AS,(2*1024**3,2*1024**3))
 cfg=read(HERE/'protocol.json');save('protocol',cfg);bind(Path(__file__))
 prior_bind=read(PREV/'bindings.json')
 for p,h in prior_bind.items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
 assert read(PREV/'gate.json')['passed']; assert read(PREV/'independent_audit.json')['passed']
 m=load_module();f,C,tri=m.setup();BIND.update(m.BIND)
 a=np.load(bind(OLD/'states_and_distributions.npz'));ix=read(OLD/'distribution_index.json'); old_updates=read(OLD/'updates.json')
 curves=[];updates=[];states=[];distributions=[];index=[];failures=[];gates=[]
 names=['relation_count','anonymous_joint_degree','triangle001','triangle011','components']
 for seed in cfg['initializations']:
  oi=next(i for i,x in enumerate(ix) if x['seed']==seed and x['step']==0 and x['arm']=='exact_forward_clean_structure')
  theta=a['parameters'][oi].copy();initial=theta.copy()
  lengths={x['step']:x['displacement_norm'] for x in old_updates if x['seed']==seed and x['arm']=='exact_forward_clean_structure'}
  assert set(lengths)==set(range(1,21)) and all(np.isfinite(v) and v>0 for v in lengths.values())
  _,G=m.local(f,C,theta);avg=(G[0]+G[1]+(G[2]+G[3])/2+G[4])/4
  error=abs(.2*np.linalg.norm(avg)-lengths[1]);assert error<1e-12
  gates.append(dict(seed=seed,first_baseline_step_length_error=float(error)))
  for step in range(21):
   loss,G=m.local(f,C,theta);ev=f.evaluate(theta);dist=ev['distribution'];orbit=np.bincount(f.orbits,weights=dist)
   metric=dict(ev['metrics']);metric['generated_state_clean_risk']=metric.pop('single_step_clean_risk')
   metric.update(forward_clean_risk=float((loss[0]+loss[1]+(loss[2]+loss[3])/2+loss[4])/4),
      isomorph_entropy=float(-np.sum(orbit[orbit>0]*np.log(orbit[orbit>0]))),
      reference_graph_probability=float(dist[f.refid]),max_graph_probability=float(dist.max()),
      parameter_displacement=float(np.linalg.norm(theta-initial)))
   for j,name in enumerate(names):metric['forward_'+name]=float(loss[j]);metric['terminal_'+name]=float(dist@C[:,j])
   for j,name in enumerate(['triangle001','triangle011']):metric[name+'_mean']=float(dist@tri[:,j])
   curves.append(dict(seed=seed,arm='MGDA_equal_baseline_step',step=step,**metric));states.append(theta.copy());distributions.append(dist);index.append(dict(seed=seed,step=step))
   if step==20:break
   v,lam,res=m.minnorm(G);norm=float(np.linalg.norm(v));assert np.isfinite(norm)
   if norm<=1e-12:
    failures.append(dict(seed=seed,step=step+1,reason='Numerical zero common direction; cannot match nonzero baseline length'));break
   length=lengths[step+1];delta=-length*v/norm;nxt=theta+delta;actual=nxt-theta
   assert np.isfinite(nxt).all();length_error=abs(np.linalg.norm(actual)-length);assert length_error<1e-12
   after,_=m.local(f,C,nxt);avg=(G[0]+G[1]+(G[2]+G[3])/2+G[4])/4
   updates.append(dict(seed=seed,step=step+1,coefficients=lam.tolist(),combined_norm=norm,
       prescribed_length=length,actual_length=float(np.linalg.norm(actual)),length_error=float(length_error),
       scale_over_raw_mgda=length/(.2*norm),KKT=res,
       cosine_to_same_state_average=float(v@avg/(norm*np.linalg.norm(avg))),
       predicted_changes=(G@actual).tolist(),actual_changes=(after-loss).tolist(),
       all_actual_decreased=bool(np.all(after-loss<0)),
       **{'change_'+name:float(after[j]-loss[j]) for j,name in enumerate(names)}))
   theta=nxt;COUNT+=1
  save('curves',curves);save('updates',updates);save('failures',failures)
 save('gate',dict(passed=True,reused_gradient_solver_gate=True,checks=gates))
 save('index',index);np.savez_compressed(OUT/'states.npz',parameters=np.array(states),distributions=np.array(distributions))
 for p,h in BIND.items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
 save('bindings',BIND)
 save('receipt',dict(status='complete_stopped' if not failures else 'partial_stopped',seconds=time.monotonic()-START,
    RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,CPU_threads=1,GPU_hours=0,
    formal_model_calls=0,toy_updates=COUNT,new_sampled_graphs=0,
    all_local_costs_decreased=sum(x['all_actual_decreased'] for x in updates),
    max_length_error=max(x['length_error'] for x in updates),max_KKT=max(x['KKT'] for x in updates)))
 print(json.dumps(dict(receipt=json.loads((OUT/'receipt.json').read_text()),finals=[x for x in curves if x['step']==20])))
if __name__=='__main__':
 try: main()
 except BaseException:
  if OUT.exists():save('failure',dict(traceback=traceback.format_exc(),updates=COUNT))
  raise
