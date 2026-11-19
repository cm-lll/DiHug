"""Frozen finite probability-redistribution diagnostic, no model/autograd."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import json,time,itertools,hashlib
from pathlib import Path
import numpy as np
import closure_semantics_audit as a
ROOT=a.ROOT;OUT=a.OUT;PANEL=a.PAIRED;SRC=a.SRC
def main():
 start=time.time();p=json.loads((SRC/'protocol.json').read_text());manifest=json.loads((PANEL/'input_manifest.json').read_text());sigma=json.loads((ROOT/'results/hetero_micro_macro_calibration_20261116/frozen_scales.json').read_text())['sigma_squared']
 a.dump('direction_protocol.json',dict(inputs='all16 saved free inputs, candidate128 p0 only',positions='within each active relation, Xt absent slots with maximum vs minimum CN; tied positions all included; no labels used for selection',transfer='high-to-low and low-to-high; epsilon=min(.01,.1*each of four group available capacities)',skip='no CN distinction, or epsilon<1e-12',budget_seconds=180,score='unchanged frozen joint degree+closure CF; both unscaled type equal and original residual scaled closure block',scope='Preserves expected relation edge counts, not each node degree; synthetic probability intervention only, not proposed training or realizable network update',model_calls=0,autograd=0))
 rows=[];ctx={};bindings={}
 def bind(path):bindings[str(path)]=a.sha(path)
 for ds in ['IMDB','PubMed']:
  r=next(r for r in p['parents'] if r['dataset']==ds);bind(r['data']);z=np.load(r['data']);ty=z['types'];schema=z['schema'];e=z['edges'];e=e if e.shape[0]==2 else e.T;ref=np.zeros((len(schema),len(ty),len(ty)));lu={tuple(x):i for i,x in enumerate(schema)}
  for i,j in e.T:ref[lu[tuple(sorted((ty[i],ty[j])))],i,j]=ref[lu[tuple(sorted((ty[i],ty[j])))],j,i]=1
  path=ROOT/'results/hetero_micro_macro_cpu_20261116_v2'/(ds+'_frozen.json');bind(path);f=json.loads(path.read_text());ctx[ds]=(ty,a.obj.make_plan(schema,len(np.unique(ty))),ref,f)
 def evaluate(P,ds):
  if time.time()-start>180:raise TimeoutError('CPU direction bound')
  ty,plan,ref,f=ctx[ds];d,c,o=a.counts(P,ty,plan);rd,rc,ro=a.counts(ref,ty,plan);x=np.concatenate([d,c],1);errors=[]
  for typ in np.unique(ty):
   b=f[str(typ)]['closure'];cf=np.exp(1j*((x[ty==typ]/np.array(b['scale']))@np.array(b['omega']))).mean(0);target=np.array(b['target_real'])+1j*np.array(b['target_imag']);errors.append(float(np.mean(abs(cf-target)**2)))
  scaled=float(np.mean([e/(4*sigma[ds][str(t)]['closure']) for t,e in enumerate(errors)]))
  return dict(CF=float(np.mean(errors)),scaled_closure=scaled,triangles=float(c.sum()/3),triangle_absolute_error=abs(float(c.sum()/3-rc.sum()/3)),degree_distribution_L1=float(np.mean([np.abs(np.sort(d[ty==t],axis=0)-np.sort(rd[ty==t],axis=0)).mean() for t in np.unique(ty)])))
 for en in manifest:
  if en['kind']!='free':continue
  bind(en['input']);z=np.load(en['input']);path=PANEL/(en['key']+'_calibrated_micro_macro_prediction.npz');bind(path);pp=np.load(path)['p'];ty,plan,ref,f=ctx[en['dataset']];query=z['query'];fam=z['f'];ids=z['ids'];fullq=z['fullquery'];X=np.zeros((len(ty),len(ty)));X[fullq[0],fullq[1]]=z['xt'];X+=X.T;CN=X@X;cn=CN[query[0],query[1]];absent=z['xt'][ids]==0
  P=np.zeros_like(ref);P[fam,query[0],query[1]]=pp;P[fam,query[1],query[0]]=pp;base=evaluate(P,en['dataset'])
  for r in np.unique(fam):
   ident=dict(dataset=en['dataset'],train_seed=en['train_seed'],sample_seed=en['sample_seed'],t=en['t'],family=int(r));m=(fam==r)&absent
   if not m.any() or cn[m].max()==cn[m].min():rows.append(dict(**ident,status='no_CN_contrast'));continue
   hi=m&(cn==cn[m].max());lo=m&(cn==cn[m].min());caps=[pp[hi].sum(),(1-pp[lo]).sum(),pp[lo].sum(),(1-pp[hi]).sum()];eps=min(.01,.1*min(caps))
   if eps<1e-12:rows.append(dict(**ident,status='tiny_capacity',epsilon=float(eps)));continue
   for direction,src,dst in [('high_to_low',hi,lo),('low_to_high',lo,hi)]:
    new=pp.copy();new[src]-=eps*pp[src]/pp[src].sum();new[dst]+=eps*(1-pp[dst])/(1-pp[dst]).sum();assert new.min()>=0 and new.max()<=1 and abs(new[fam==r].sum()-pp[fam==r].sum())<1e-10
    Q=np.zeros_like(P);Q[fam,query[0],query[1]]=new;Q[fam,query[1],query[0]]=new;res=evaluate(Q,en['dataset'])
    rows.append(dict(**ident,status='complete',direction=direction,epsilon=float(eps),max_CN=int(cn[hi][0]),min_CN=int(cn[lo][0]),high_n=int(hi.sum()),low_n=int(lo.sum()),**{'base_'+k:v for k,v in base.items()},**{'change_'+k:res[k]-v for k,v in base.items()}))
 a.table('direction_results',rows);a.dump('direction_receipt.json',dict(seconds=time.time()-start,rows=len(rows),complete=sum(r['status']=='complete' for r in rows),CPU_only=True,model_calls=0,autograd=0,no_network_realizability_claim=True));bind(__file__);bind(Path(a.__file__));a.dump('direction_sources.json',bindings)
if __name__=='__main__':main()
