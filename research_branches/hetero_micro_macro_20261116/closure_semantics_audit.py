"""CPU-only audit of typed closure opportunity and the frozen descriptor score.
No denoiser import, autograd, model forward, edits, sampling or training.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import json,csv,time,hashlib,itertools,importlib.util,resource,traceback,sys
from pathlib import Path
import numpy as np
import torch
torch.set_num_threads(1)
ROOT=Path(__file__).resolve().parents[2]
SRC=ROOT/'results/hetero_micro_macro_effect_run_20261116T083122'
PAIRED=ROOT/'results/paired_response_check_20261116'
OUT=ROOT/'results/closure_semantics_audit_20261116'
spec=importlib.util.spec_from_file_location('closure_objective',Path(__file__).with_name('objective.py'))
obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(n,x):(OUT/n).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False))
def table(n,rows):
 dump(n+'.json',rows)
 with (OUT/(n+'.csv')).open('w') as f:
  w=csv.DictWriter(f,list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
def counts(P,ty,plan):
 deg=P.sum(2).T;closed=[];op=[]
 for r,s,u in plan['closures']:
  div=2 if r==u else 1
  c=((P[r]@P[s])*P[u]).sum(1)/div
  eligible=np.zeros(len(ty),bool)
  for a in np.unique(ty):
   er=plan['schema'][r];eu=plan['schema'][u]
   if a not in er or a not in eu:continue
   b=er[1] if er[0]==a else er[0];d=eu[1] if eu[0]==a else eu[0]
   if sorted((b,d))==plan['schema'][s]:eligible[ty==a]=True
  o=(deg[:,r]*deg[:,u]-(P[r]*P[u]).sum(1))/div
  o*=eligible;assert np.min(o)>-1e-9 and np.all(c<=o+1e-8)
  closed.append(c);op.append(o)
 return deg,np.stack(closed,1),np.stack(op,1)
def main():
 start=time.time();OUT.mkdir(exist_ok=False);bindings={};contexts={};rows=[];groups=[];scores=[];layout=[]
 def bind(p):bindings[str(p)]=sha(p);return Path(p)
 def load(p):return np.load(bind(p))
 p=json.loads(bind(SRC/'protocol.json').read_text());inv=json.loads(bind(SRC/'inventory.json').read_text());manifest=json.loads(bind(PAIRED/'input_manifest.json').read_text())
 dump('protocol.json',dict(scope='32 saved inputs +96 saved p0 distributions +48 saved terminal graphs +2 references',GPU=0,model_calls=0,autograd=0,new_graphs=0,CPU_seconds_max=180,threads=1,groups='all nodes per type; low<=1 and top ceil(10%) by current hard input degree for probability records; no identity matching',closure='anchored unordered neighbor pair connected by specified third relation',opportunity='distinct neighbor pair of compatible types, ignoring closing-edge existence',scoring='original frozen 32 frequency joint degree+closure CF, no new fitted weights',no_new_candidate=True))
 # Finite independent enumeration: all64 undirected graphs on4 typed nodes.
 ty=np.array([0,0,1,2]);schema=list(itertools.combinations_with_replacement(range(3),2));plan=obj.make_plan(schema,3);edges=list(itertools.combinations(range(4),2));lookup={r:i for i,r in enumerate(schema)}
 for bits in itertools.product([0,1],repeat=6):
  P=np.zeros((6,4,4));A=np.zeros((4,4));F=np.full((4,4),-1)
  for (i,j),v in zip(edges,bits):
   f=lookup[tuple(sorted((ty[i],ty[j])))];P[f,i,j]=P[f,j,i]=v;A[i,j]=A[j,i]=v;F[i,j]=F[j,i]=f
  d,c,o=counts(P,ty,plan);ec=np.zeros_like(c);eo=np.zeros_like(o)
  for i in range(4):
   for j,k in itertools.combinations([x for x in range(4) if x!=i],2):
    if not A[i,j]*A[i,k]:continue
    r,u=sorted((F[i,j],F[i,k]));s=F[j,k];idx=plan['closures'].index((r,s,u));eo[i,idx]+=1;ec[i,idx]+=A[j,k]
  assert np.array_equal(c,ec) and np.array_equal(o,eo)
 dump('finite_checks.json',dict(graphs=64,closure_and_opportunity='passed',independent_node_neighbor_pair_enumeration=True))
 for ds in ['IMDB','PubMed']:
  row=next(r for r in p['parents'] if r['dataset']==ds);z=load(row['data']);ty=z['types'];schema=z['schema'];n=len(ty);ee=z['edges'];ee=ee if ee.shape[0]==2 else ee.T
  plan=obj.make_plan(schema,len(np.unique(ty)));P=np.zeros((len(schema),n,n));lookup={tuple(r):i for i,r in enumerate(schema)}
  for i,j in ee.T:P[lookup[tuple(sorted((ty[i],ty[j])))],i,j]=P[lookup[tuple(sorted((ty[i],ty[j])))],j,i]=1
  fixed=json.loads(bind(ROOT/'results/hetero_micro_macro_cpu_20261116_v2'/(ds+'_frozen.json')).read_text());d,c,o=counts(P,ty,plan)
  names=row['config']['datasets'][ds]['type_names'];contexts[ds]=(ty,schema,plan,P,fixed,names)
  # Verify this closure-only CPU implementation matches unchanged reference.
  with torch.no_grad():desc=obj.descriptors(torch.tensor(P),ty,plan)
  assert np.max(abs(desc['closure'].numpy()-c))<1e-9 and np.max(abs(desc['degree'].numpy()-d))<1e-9
  for k,(r,s,u) in enumerate(plan['closures']):
   layout.append(dict(dataset=ds,column=k,relations=[r,s,u],reference_closures=float(c[:,k].sum()),reference_opportunities=float(o[:,k].sum()),active_reference=bool(c[:,k].sum())))
 def score(ident,P,X):
  if time.time()-start>180:raise TimeoutError('CPU budget')
  ds=ident['dataset'];ty,schema,plan,ref,fixed,names=contexts[ds];d,c,o=counts(P,ty,plan);rd,rc,ro=counts(ref,ty,plan);xd=X.sum(1);xx=np.concatenate([d,c],1)
  sc=[]
  for a in np.unique(ty):
   mask=ty==a;b=fixed[str(a)]['closure'];scale=np.array(b['scale']);omega=np.array(b['omega']);target=np.array(b['target_real'])+1j*np.array(b['target_imag']);cf=np.exp(1j*((xx[mask]/scale)@omega)).mean(0);err=float(np.mean(abs(cf-target)**2));sc.append(err)
   scores.append(dict(**ident,type=int(a),type_name=names[int(a)],closure_CF_error=err))
   ix=np.flatnonzero(mask);hub=ix[np.argsort(xd[ix],kind='stable')[-max(1,int(np.ceil(.1*len(ix)))):]]
   for group,idx in [('all',ix),('low',ix[xd[ix]<=1]),('hub',hub)]:
    groups.append(dict(**ident,type=int(a),type_name=names[int(a)],group=group,n=len(idx),degree_mean=float(d[idx].sum(1).mean()) if len(idx) else None,opportunities=float(o[idx].sum()),closures=float(c[idx].sum()),ratio=float(c[idx].sum()/o[idx].sum()) if o[idx].sum() else None))
   for k,(r,s,u) in enumerate(plan['closures']):
    if ro[mask,k].sum()==0 and o[mask,k].sum()==0:continue
    oo=float(o[mask,k].sum());cc=float(c[mask,k].sum());rro=float(ro[mask,k].sum());rrc=float(rc[mask,k].sum())
    rows.append(dict(**ident,type=int(a),type_name=names[int(a)],column=k,r=int(r),s=int(s),u=int(u),opportunities=oo,closures=cc,ratio=cc/oo if oo else None,ref_opportunities=rro,ref_closures=rrc,ref_ratio=rrc/rro if rro else None,nodes=len(ix)))
  scores.append(dict(**ident,type=-1,type_name='type_equal',closure_CF_error=float(np.mean(sc))))
 for ds,(ty,schema,plan,ref,fixed,names) in contexts.items():score(dict(key=ds+'_ref',dataset=ds,train_seed=0,sample_seed=0,kind='reference',arm='reference',t=0),ref,ref.sum(0))
 for ent in manifest:
  z=load(ent['input']);ds=ent['dataset'];ty,schema,plan,ref,fixed,names=contexts[ds];n=len(ty);fullq=z['fullquery'];fullf=z['fullfamily'];P=np.zeros_like(ref);P[fullf,fullq[0],fullq[1]]=z['xt'];P[fullf,fullq[1],fullq[0]]=z['xt'];X=P.sum(0)
  base={k:ent[k] for k in ['key','dataset','train_seed','sample_seed','kind','t']};score(dict(**base,arm='input'),P,X)
  for arm in ['parent','CE','calibrated_micro_macro']:
   zz=load(PAIRED/(ent['key']+'_'+arm+'_prediction.npz'));Q=np.zeros_like(ref);qq=z['query'];ff=z['f'];Q[ff,qq[0],qq[1]]=zz['p'];Q[ff,qq[1],qq[0]]=zz['p'];score(dict(**base,arm=arm),Q,X)
 for ent in inv:
  z=load(ent['graph']);ty,schema,plan,ref,fixed,names=contexts[ent['dataset']];P=np.zeros_like(ref);e=z['edges'];e=e if e.shape[0]==2 else e.T;lookup={tuple(r):i for i,r in enumerate(schema)}
  for i,j in e.T:P[lookup[tuple(sorted((ty[i],ty[j])))],i,j]=P[lookup[tuple(sorted((ty[i],ty[j])))],j,i]=1
  score(dict(key=ent['model']+'_'+str(ent['sample_seed']),dataset=ent['dataset'],train_seed=ent['train_seed'],sample_seed=ent['sample_seed'],kind='terminal',arm=ent['arm'],t=0),P,P.sum(0))
 table('channel_layout',layout);table('closure_channels',rows);table('node_groups',groups);table('original_score',scores)
 summary=[]
 for seed in sorted(set(r['train_seed'] for r in scores)):
  for kind in ['reference','free','normal','terminal']:
   for arm in ['reference','input','parent','CE','calibrated_micro_macro']:
    ss=[r for r in scores if r['train_seed']==seed and r['kind']==kind and r['arm']==arm and r['type']==-1]
    if ss:summary.append(dict(dataset=ss[0]['dataset'],train_seed=seed,kind=kind,arm=arm,n=len(ss),mean_CF=float(np.mean([r['closure_CF_error'] for r in ss]))))
 table('score_summary',summary)
 pkg=Path(p['parents'][0]['data']).parent.parent
 for file in [pkg/'hsgd/common_neighbor.py',pkg/'hsgd/model.py',pkg/'hsgd/computational.py',pkg/'hsgd/slots.py',Path(__file__).with_name('objective.py'),Path(__file__).with_name('effect_worker.py'),Path(__file__)]:bind(file)
 for path,h in p['runtime_source_bindings'].items():assert sha(path)==h,path
 dump('sources.json',bindings);dump('receipt.json',dict(status='complete',states=178,model_calls=0,GPU_hours=0,autograd=0,new_graphs=0,seconds=time.time()-start,peak_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,source_bindings_passed=True))
def summarize_saved():
 """No descriptor recalculation; explicit fixed-zero support and dataset key."""
 start=time.time();ss=json.loads((OUT/'original_score.json').read_text());channels=json.loads((OUT/'closure_channels.json').read_text());p=json.loads((SRC/'protocol.json').read_text());support={}
 for ds in ['IMDB','PubMed']:
  row=next(r for r in p['parents'] if r['dataset']==ds);z=np.load(row['data']);ty=z['types'];e=z['edges'];e=e if e.shape[0]==2 else e.T
  support[ds]=[any(tuple(sorted(ty[x]))==tuple(t) for x in e.T) for t in z['schema']]
 for c in channels:c['closing_relation_active']=support[c['dataset']][c['s']]
 table('closure_channels_with_support',channels)
 summary=[]
 keys=sorted(set((r['dataset'],r['train_seed'],r['kind'],r['arm']) for r in ss))
 for ds,seed,kind,arm in keys:
  rr=[r for r in ss if (r['dataset'],r['train_seed'],r['kind'],r['arm'])==(ds,seed,kind,arm) and r['type']==-1]
  ch=[r for r in channels if (r['dataset'],r['train_seed'],r['kind'],r['arm'])==(ds,seed,kind,arm)];n=len(rr)
  W=sum(r['opportunities'] for r in ch)/n;Ws=sum(r['opportunities'] for r in ch if r['closing_relation_active'])/n;C=sum(r['closures'] for r in ch)/n
  summary.append(dict(dataset=ds,train_seed=seed,kind=kind,arm=arm,n=n,mean_CF=float(np.mean([r['closure_CF_error'] for r in rr])),declared_opportunities=W,active_closing_opportunities=Ws,anchored_closed=C,triangles=C/3,declared_ratio=C/W if W else None,active_ratio=C/Ws if Ws else None))
 table('summary_by_dataset_with_support',summary)
 dump('report_corrections.json',dict(original_summary_issue='seed0 reference summary grouped both datasets together; raw scores correct; old file retained, use summary_by_dataset_with_support',fixed_zero_note='Declared schema opportunities may have a forbidden closing relation in IMDB; added active closing relation denominator separately. PubMed all10 relations active.',source_after_report_correction_sha256=sha(__file__),seconds=time.time()-start,model_calls=0))
if __name__=='__main__':
 try:
  if '--summarize' in sys.argv:summarize_saved()
  else:main();summarize_saved()
 except BaseException:
  if OUT.exists():dump('failure.json',dict(traceback=traceback.format_exc()))
  raise
