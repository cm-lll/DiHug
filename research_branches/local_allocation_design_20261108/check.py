"""Only CPU free-score differentiation on immutable exported predictions."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']=os.environ['OPENBLAS_NUM_THREADS']='1'
import sys,json,time,hashlib,importlib.util,csv,resource
from pathlib import Path
import numpy as np
import torch
from objective import LocalAllocationRank
torch.set_num_threads(1)
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/local_allocation_design_20261108'
OLD=ROOT/'results/reproducible_objective_20261108T135622/attempt2'
start=time.time();rows=[];details=[];tests=[];assets={}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def dump(name,x):
 (OUT/name).write_text(json.dumps(x,indent=2,ensure_ascii=False))
def csvout(name,x):
 with (OUT/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in x for k in r)));w.writeheader();w.writerows(x)
rng=np.random.default_rng(2026110815)
for trial in range(24):
 n=6;q=np.array(np.triu_indices(n,1));y=rng.integers(0,2,q.shape[1]);z=torch.tensor(rng.normal(size=len(y)),dtype=torch.float64,requires_grad=True)
 obj=LocalAllocationRank(np.zeros(n,int),[(0,0)],[.5],q,np.zeros(len(y),int),y)
 loss,_=obj(z);grad=torch.autograd.grad(loss,z)[0]
 assert (grad[torch.as_tensor(y==1)]<=1e-12).all() and (grad[torch.as_tensor(y==0)]>=-1e-12).all()
 assert float(grad.abs().sum())<=1+1e-12 and abs(float(grad.sum()))<1e-12
 shifted,_=obj(z+17);assert abs(float(loss-shifted))<1e-12
 explicit=0.
 for g in obj.groups:
  dif=z[g['neg']][:,None]-z[g['pos']][None,:]
  exact=torch.log1p(torch.exp(dif).mean())
  assert float(exact)+1e-12>=float(torch.nn.functional.softplus(dif).mean())
  explicit=explicit+.5*g['weight']*exact
 assert abs(float(explicit-loss))<1e-12
 j=trial%len(y);eps=1e-5;zp=z.detach().clone();zm=zp.clone();zp[j]+=eps;zm[j]-=eps
 numeric=float((obj(zp)[0]-obj(zm)[0])/(2*eps));assert abs(numeric-float(grad[j]))<1e-8
 perm=rng.permutation(len(y));other=LocalAllocationRank(np.zeros(n,int),[(0,0)],[.5],q[:,perm],np.zeros(len(y),int),y[perm])
 assert abs(float(other(z[perm])[0]-loss))<1e-12
 tests.append(dict(trial=trial,passed=True))
# No eligible positive-negative groups: exactly zero, still a valid gradient path.
for label in [0,1]:
 y=np.full(3,label);obj=LocalAllocationRank([0,0,0],[(0,0)],[.5],np.array([[0,0,1],[1,2,2]]),[0,0,0],y)
 z=torch.zeros(3,dtype=torch.float64,requires_grad=True);v,_=obj(z);assert float(v)==0 and torch.equal(torch.autograd.grad(v,z)[0],torch.zeros_like(z))
 tests.append(dict(empty_label=label,passed=True))
spec=importlib.util.spec_from_file_location('asset_helpers',ROOT/'results/output_boundary_replay_20261108T081654/worker.py')
w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w)
for task in w.read(ROOT/'results/output_boundary_replay_20261108T081654/tasks.json'):
 ds,cfg=task['dataset'],task['row']['config'];data=ROOT/w.g.SCI.relative_to(w.g.ROOT)/'package'/cfg['datasets'][ds]['file']
 assets[str(data)]=sha(data);assert assets[str(data)]==cfg['datasets'][ds]['sha256']
 with np.load(data,allow_pickle=False) as archive:case=w.g.common.make_case(archive['types'],archive['edges'].T,tuple(map(tuple,archive['schema'])))
 inpfile=OLD/f'{ds}_fixed_inputs.pt';assets[str(inpfile)]=sha(inpfile);inputs=torch.load(inpfile,map_location='cpu')['inputs']
 ps=w.make_objective(case,cfg,ds,'PS')
 for t in [1,500,1000]:
  inp=next(i for panel,tt,seed,i in inputs if panel=='new_diagnostic' and tt==t)
  obj=LocalAllocationRank(case['space'].types,case['space'].pairs,case['prior'],inp['query'],inp['f'],inp['labels'])
  dump(f'{ds}_group_inventory.json',{'eligible':[dict(node=g['node'],type=g['type'],relation=g['relation'],positive=len(g['pos']),negative=len(g['neg']),weight=g['weight']) for g in obj.groups],'excluded':obj.excluded})
  for arm in ['CE','PS']:
   file=OLD/f'{ds}_{arm}_32_raw_new_diagnostic_{t}.npz';assets[str(file)]=sha(file)
   with np.load(file) as a:p=a['probability'].astype('float64')
   active=obj.active;assert ((p[active]>0)&(p[active]<1)).all(),'cannot reconstruct saturated logits'
   z0=np.zeros_like(p);z0[active]=np.log(p[active])-np.log1p(-p[active]);z=torch.tensor(z0,requires_grad=True)
   tick=time.perf_counter();loss,_=obj(z);grad=torch.autograd.grad(loss,z)[0].numpy();elapsed=time.perf_counter()-tick
   probability=torch.tensor(p,requires_grad=True);oldloss,_=ps(probability,inp['query'],inp['f'],inp,logits=None);dp=torch.autograd.grad(oldloss,probability)[0].numpy();oldgrad=dp*p*(1-p)
   ce=(p-inp['labels'])/case['space'].total;ce[~active]=0
   assert np.isfinite(grad).all() and (grad[inp['labels']==1]<=1e-12).all() and (grad[inp['labels']==0]>=-1e-12).all()
   assert abs(grad.sum())<1e-10 and abs(grad).sum()<=1+1e-10
   shifted=z.detach().clone()
   for r in range(case['n_families']):shifted[torch.as_tensor(inp['f']==r)]+=3*(r+1)
   assert abs(float(obj(shifted)[0])-float(loss))<1e-10
   row=dict(dataset=ds,parent_arm=arm,t=t,loss=float(loss),PS_loss=float(oldloss),rank_over_CE_L2=float(np.linalg.norm(grad)/np.linalg.norm(ce)),PS_over_CE_L2=float(np.linalg.norm(oldgrad)/np.linalg.norm(ce)),rank_CE_cosine=float(grad@ce/(np.linalg.norm(grad)*np.linalg.norm(ce))),L1=float(abs(grad).sum()),net_logit_gradient=float(grad.sum()),seconds=elapsed,eligible_groups=len(obj.groups),excluded_groups=len(obj.excluded))
   rows.append(row)
   for r in [-1]+list(range(case['n_families'])):
    for label in [0,1]:
     mask=active&(inp['labels']==label)&((inp['f']==r) if r>=0 else True)
     if not mask.any():continue
     details.append(dict(dataset=ds,arm=arm,t=t,relation=r,label=label,support=int(mask.sum()),CE_L1=float(abs(ce[mask]).sum()),rank_L1=float(abs(grad[mask]).sum()),PS_L1=float(abs(oldgrad[mask]).sum()),rank_opposed_count=int((grad[mask]*ce[mask]<-1e-18).sum()),PS_opposed_count=int((oldgrad[mask]*ce[mask]<-1e-18).sum()),rank_zero_count=int((grad[mask]==0).sum())))
   np.savez_compressed(OUT/f'{ds}_{arm}_{t}_free_logit_gradients.npz',CE=ce,PS=oldgrad,rank=grad)
   dump('progress.json',{'finished':len(rows),'target':12})
assert all(sha(p)==h for p,h in assets.items())
assert not torch.cuda.is_initialized()
dump('asset_binding.json',assets);dump('synthetic_checks.json',tests);dump('results.json',rows);csvout('results.csv',rows);csvout('relation_gradients.csv',details)
dump('receipt.json',dict(status='complete',saved_prediction_panels=len(rows),synthetic_cases=len(tests),seconds=time.time()-start,max_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,model_calls=0,optimizer_updates=0,new_graphs=0,GPU_hours=0,cuda_initialized=False,failures=[]))
print(json.dumps(rows,ensure_ascii=False))
