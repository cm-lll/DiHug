"""24 frozen probability tensors only; no denoiser instantiated or imported."""
import csv,hashlib,json,os,sys,time,resource,random
from pathlib import Path
import numpy as np
import torch
from pb_degree import RelationDegreeCDF
torch.set_num_threads(1);torch.set_num_interop_threads(1)
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/pb_degree_preparation_20261106';OUT.mkdir(exist_ok=True)
PROB=ROOT/'results/input_mask_screen_20261106';DATA=ROOT/'research_branches/typed_structure_early_cn_expansion_review/experiments/typed_structure_early_cn_expansion_v1/package/assets'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name,x):
 (OUT/(name+'.json')).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
 if isinstance(x,list) and x:
  with (OUT/(name+'.csv')).open('w',newline='') as f:
   w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in x for k in r)));w.writeheader();w.writerows(x)
def reference(p,q,fam,types,schema,ref):
 n=len(types);groups=[]
 for f in np.unique(fam):
  for t in sorted(set(schema[f])):groups.append((f,t))
 g=np.zeros_like(p);loss=0.
 for f,t in groups:
  nodes=np.flatnonzero(types==t);allh=[];idxs=[];degs=[]
  for i in nodes:
   ids=np.flatnonzero((fam==f)&((q==i).any(0)));v=p[ids];h=np.ones(1);hist=[]
   for x in v:
    hist.append(h);nxt=np.zeros(len(h)+1);nxt[:-1]+=(1-x)*h;nxt[1:]+=x*h;h=nxt
   allh.append((h,hist));idxs.append(ids);degs.append(int(ref[q[0,ids],q[1,ids]].sum()))
  cap=len(allh[0][0])-1;H=np.mean([h for h,_ in allh],0);R=np.bincount(degs,minlength=cap+1)/len(nodes);e=np.cumsum(H-R)[:-1]
  loss+=.5*np.mean(e*e)/len(groups);dh=np.r_[np.cumsum((e/cap)[::-1])[::-1],0]/len(nodes)/len(groups)
  for ids,(_,hist) in zip(idxs,allh):
   a=dh.copy()
   for j in range(len(ids)-1,-1,-1):
    k=ids[j];g[k]+=hist[j]@(a[1:]-a[:-1]);a=(1-p[k])*a[:-1]+p[k]*a[1:]
 return float(loss),g

def main():
 start=time.monotonic();bindings=json.loads((PROB/'bindings.json').read_text());prev=json.loads((ROOT/'results/degree_uncertainty_cpu_20261106/receipt.json').read_text())
 rows=[];objects={};sources={};numpy_state=repr(np.random.get_state());python_state=repr(random.getstate());torch_state=torch.get_rng_state().clone()
 for b in bindings:
  ds=b['task'].split('_')[0];file=PROB/f'{b["task"]}_{b["kind"]}_{b["t"]}_FULL.npz';sources[str(file)]=sha(file);assert sources[str(file)]==prev['source_hashes'][str(file)]
  z=np.load(file);p=z['probabilities'].astype(float);q=z['query'];fam=z['family']
  if ds not in objects:
   data=DATA/(ds+'.npz');sources[str(data)]=sha(data);z=np.load(data);types=z['types'];schema=z['schema'];edges=z['edges'].T;n=len(types);ref=np.zeros((n,n));ref[edges[0],edges[1]]=ref[edges[1],edges[0]]=1
   prior=[]
   for a,c in schema:
    mask=((types[:,None]==a)&(types[None,:]==c))|((types[:,None]==c)&(types[None,:]==a));np.fill_diagonal(mask,False);prior.append(float(ref[mask].mean()))
   obj=RelationDegreeCDF(types,schema,prior,edges,data_sha256=sha(data));objects[ds]=(obj,types,schema,ref);save('objective_'+ds,obj.objective_binding())
  obj,types,schema,ref=objects[ds];v,rg=reference(p,q,fam,types,schema,ref)
  for dtype in [torch.float64,torch.float32]:
   x=torch.tensor(p,dtype=dtype,requires_grad=True);ts=time.perf_counter();loss,per=obj(x,q,fam);fs=time.perf_counter()-ts;ts=time.perf_counter();g=torch.autograd.grad(loss,x)[0];bs=time.perf_counter()-ts
   ve=abs(v-loss.item());ge=float(np.max(np.abs(g.detach().numpy()-rg)))
   assert ve<(1e-13 if dtype==torch.float64 else 1e-7)
   assert ge<(1e-11 if dtype==torch.float64 else 1e-7)
   labels=ref[q[0],q[1]];cg=(p-labels)/(len(types)*(len(types)-1)/2);ag=rg*p*(1-p);ratio=float(np.linalg.norm(ag)/np.linalg.norm(cg)) if b['kind']=='forward' else None
   rows.append(dict(dataset=ds,task=b['task'],kind=b['kind'],t=b['t'],dtype=str(dtype),loss=float(loss),value_max_error=ve,gradient_max_error=ge,forward_seconds=fs,backward_seconds=bs,ratio_aux_to_CE_logit_norm_lambda1=ratio,
    groups=len(obj.groups),batch_rows=len(obj.indices),padded_capacity=obj.indices.shape[1],history_tensor_MiB=len(obj.indices)*obj.indices.shape[1]*(obj.indices.shape[1]+1)/2*x.element_size()/2**20))
 assert numpy_state==repr(np.random.get_state()) and python_state==repr(random.getstate()) and torch.equal(torch_state,torch.get_rng_state()) and not torch.cuda.is_initialized()
 save('saved_probability_checks',rows)
 save('CPU_receipt',dict(status='pass',states=24,dtype_checks=48,seconds=time.monotonic()-start,peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,source_hashes=sources,RNG_unchanged=True,CUDA_initialized=False,denoiser_calls=0,training_updates=0,new_graphs=0,limitations=['CPU loss-only derivative and cost, not network training/optimizer/recovery acceptance','Old full-input probabilities, not original A or new trained candidate','No higher-order derivatives; exact first-order custom adjoint']))
 print(json.dumps(dict(seconds=time.monotonic()-start,checks=len(rows),max64=max(r['gradient_max_error'] for r in rows if r['dtype']=='torch.float64'),max32=max(r['gradient_max_error'] for r in rows if r['dtype']=='torch.float32')),indent=2))
if __name__=='__main__':
 try:main()
 except Exception:
  import traceback
  save('failure',dict(traceback=traceback.format_exc()));raise
