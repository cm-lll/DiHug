"""One analytic scale revision, on the same already-viewed development outputs."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']=os.environ['OPENBLAS_NUM_THREADS']='1'
import sys,json,time,csv,importlib.util,hashlib,resource
from pathlib import Path
import numpy as np
import torch
from objective import CEUnitsRank
torch.set_num_threads(1);ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/local_allocation_design_20261108';OLD=ROOT/'results/reproducible_objective_20261108T135622/attempt2'
start=time.time();checks=[];rows=[];groups=[]
rng=np.random.default_rng(2026110815)
for i in range(24):
 n=6;q=np.array(np.triu_indices(n,1));f=np.zeros(q.shape[1],int);y=rng.integers(0,2,len(f));S=len(f)
 obj=CEUnitsRank(np.zeros(n,int),[(0,0)],[.5],q,f,y,schema_slots=S)
 z=torch.tensor(rng.normal(size=S),requires_grad=True);loss,_=obj(z);grad=torch.autograd.grad(loss,z)[0]
 assert (grad[torch.as_tensor(y==1)]<=1e-12).all() and (grad[torch.as_tensor(y==0)]>=-1e-12).all()
 assert float(grad[torch.as_tensor(y==1)].abs().max())<=1/S+1e-12 if (y==1).any() else True
 assert float(grad.abs().sum())<=2*sum(y)/S+1e-12 and abs(float(grad.sum()))<1e-12
 assert abs(float(obj(z+29)[0]-loss))<1e-12
 explicit=z.sum()*0
 for g in obj.groups:
  dif=z[g['neg']][:,None]-z[g['pos']][None,:]
  explicit+=torch.log1p(torch.exp(dif).mean(0)).sum()/(2*S)
 assert abs(float(loss-explicit))<1e-12
 j=i%S;eps=1e-5;a=z.detach().clone();b=a.clone();a[j]+=eps;b[j]-=eps
 assert abs(float((obj(a)[0]-obj(b)[0])/(2*eps))-float(grad[j]))<1e-8
 checks.append(dict(test=i,passed=True))
for label in [0,1]:
 obj=CEUnitsRank([0]*3,[(0,0)],[.5],np.array([[0,0,1],[1,2,2]]),[0]*3,[label]*3,schema_slots=3)
 z=torch.zeros(3,requires_grad=True);loss,_=obj(z);assert float(loss)==0 and torch.equal(torch.autograd.grad(loss,z)[0],torch.zeros_like(z))
 checks.append(dict(empty_label=label,passed=True))
spec=importlib.util.spec_from_file_location('helpers',ROOT/'results/output_boundary_replay_20261108T081654/worker.py');w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w)
for task in w.read(ROOT/'results/output_boundary_replay_20261108T081654/tasks.json'):
 ds,cfg=task['dataset'],task['row']['config'];data=ROOT/w.g.SCI.relative_to(w.g.ROOT)/'package'/cfg['datasets'][ds]['file']
 with np.load(data) as archive:case=w.g.common.make_case(archive['types'],archive['edges'].T,tuple(map(tuple,archive['schema'])))
 inputs=torch.load(OLD/f'{ds}_fixed_inputs.pt',map_location='cpu')['inputs']
 for t in [1,500,1000]:
  inp=next(x for panel,tt,seed,x in inputs if panel=='new_diagnostic' and tt==t)
  obj=CEUnitsRank(case['space'].types,case['space'].pairs,case['prior'],inp['query'],inp['f'],inp['labels'],schema_slots=case['space'].total)
  for arm in ['CE','PS']:
   with np.load(OLD/f'{ds}_{arm}_32_raw_new_diagnostic_{t}.npz') as archive:p=archive['probability'].astype(float)
   active=obj.active;assert ((p[active]>0)&(p[active]<1)).all();z=np.zeros_like(p);z[active]=np.log(p[active])-np.log1p(-p[active]);z=torch.tensor(z,requires_grad=True)
   tick=time.perf_counter();loss,_=obj(z);g=torch.autograd.grad(loss,z)[0].numpy();elapsed=time.perf_counter()-tick
   with np.load(OUT/f'{ds}_{arm}_{t}_free_logit_gradients.npz') as archive:ce=archive['CE'];ps=archive['PS']
   y=inp['labels'];assert (g[y==1]<=1e-12).all() and (g[y==0]>=-1e-12).all() and abs(g.sum())<1e-10
   assert abs(g[y==1]).max()<=1/case['space'].total+1e-12
   shifted=z.detach().clone()
   for r in range(case['n_families']):shifted[torch.as_tensor(inp['f']==r)]+=3*(r+1)
   assert abs(float(obj(shifted)[0]-loss))<1e-10
   row=dict(dataset=ds,arm=arm,t=t,S=case['space'].total,loss=float(loss),L1=float(abs(g).sum()),rank_over_CE_L2=float(np.linalg.norm(g)/np.linalg.norm(ce)),rank_CE_cosine=float(g@ce/(np.linalg.norm(g)*np.linalg.norm(ce))),max_positive_abs_derivative=float(abs(g[y==1]).max()),seconds=elapsed,eligible_groups=len(obj.groups),excluded_groups=len(obj.excluded))
   rows.append(row)
   for r in [-1]+list(range(case['n_families'])):
    for label in [0,1]:
     m=active&(y==label)&((inp['f']==r) if r>=0 else True)
     if m.any():groups.append(dict(dataset=ds,arm=arm,t=t,relation=r,label=label,support=int(m.sum()),rank_L1=float(abs(g[m]).sum()),CE_L1=float(abs(ce[m]).sum()),rank_opposed_count=int((g[m]*ce[m]<-1e-18).sum()),rank_zero_count=int((g[m]==0).sum())))
   np.savez_compressed(OUT/f'{ds}_{arm}_{t}_ce_units_gradients.npz',rank=g,CE=ce,PS=ps)
assert not torch.cuda.is_initialized()
(OUT/'ce_units_results.json').write_text(json.dumps(rows,indent=2));(OUT/'ce_units_tests.json').write_text(json.dumps(checks,indent=2))
for name,rr in [('ce_units_results.csv',rows),('ce_units_groups.csv',groups)]:
 with (OUT/name).open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=list(rr[0]));writer.writeheader();writer.writerows(rr)
(OUT/'ce_units_receipt.json').write_text(json.dumps(dict(status='complete',panels=12,synthetic_cases=len(checks),seconds=time.time()-start,max_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,model_calls=0,GPU_hours=0),indent=2))
print(json.dumps(rows))
