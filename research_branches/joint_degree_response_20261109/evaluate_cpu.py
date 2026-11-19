"""Evaluate saved probabilities only; marginal proxies, never generated graphs."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import json,csv,time,itertools,hashlib
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/joint_degree_response_20261109';START=time.time()
def read(n):return json.loads((OUT/(n+'.json')).read_text())
def save(n,x):(OUT/(n+'.json')).write_text(json.dumps(x,ensure_ascii=False,indent=2))
def table(n,rows):
 save(n,rows)
 if rows:
  with (OUT/(n+'.csv')).open('w') as h:
   w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
assert read('receipt')['status']=='complete'
assert read('release')['worker_exited'] and read('release')['exit_code']==0
with np.load(OUT/'schema.npz') as z:types=z['types'].astype(int);qref=z['query'];fref=z['relations'];yref=z['reference']
n=len(types);R=int(fref.max())+1;classes=list(itertools.combinations_with_replacement(sorted(set(types.tolist())),3));keymap={v:i for i,v in enumerate(classes)}
ii=[];jj=[];kk=[]
for i in range(n-2):
 a,b=np.triu_indices(n-i-1,1);ii.append(np.full(len(a),i));jj.append(a+i+1);kk.append(b+i+1)
ii=np.concatenate(ii);jj=np.concatenate(jj);kk=np.concatenate(kk)
ts=np.sort(np.stack([types[ii],types[jj],types[kk]]),axis=0);codes=ts[0]*100+ts[1]*10+ts[2]
classcodes=[a*100+b*10+c for a,b,c in classes]
def degrees(q,f,p):
 d=np.zeros((n,R));np.add.at(d,(q[0],f),p);np.add.at(d,(q[1],f),p);return d
def triangles(q,p):
 a=np.zeros((n,n));a[q[0],q[1]]=p;a[q[1],q[0]]=p
 w=a[ii,jj]*a[jj,kk]*a[ii,kk];bc=np.bincount(codes,weights=w,minlength=1000)
 return bc[classcodes]
rd=degrees(qref,fref,yref);rt=triangles(qref,yref);refedges=float(yref.sum());refnode=rd.sum(1)
assert refedges==926 and rt.sum()==1937
rows=[];relations=[];typed=[];trirows=[];cache={}
arms=['NOOP','CE','FULL','COMMON','RESIDUAL','MATCH'];keys=list(x['key'] for x in read('evaluation_inputs'))+['update']
for parent in ['CE','PS']:
 for arm in arms:
  for key in keys:
   with np.load(OUT/f'{parent}_{arm}_{key}.npz') as z:p=z['probability'].astype(float);ce=z['ce'].astype(float);q=z['query'];f=z['relation'];y=z['reference'].astype(bool);t=int(z['t'])
   assert np.isfinite(p).all() and np.isfinite(ce).all() and len(p)==18336 and (q[0]<q[1]).all() and len(np.unique(q[0]*n+q[1]))==len(p)
   assert np.array_equal(degrees(q,f,y),rd)
   d=degrees(q,f,p);node=d.sum(1);tr=triangles(q,p)
   ident=dict(parent=parent,arm=arm,input=key,t=t,independent_input=key!='update' and t!=1)
   low=[];hub=[];joint=[];marg=[]
   for ty in sorted(set(types.tolist())):
    ids=np.flatnonzero(types==ty);dm=np.array([np.any(f[(types[q[0]]==ty)|(types[q[1]]==ty)]==r) for r in range(R)])
    scale=np.maximum(1.,rd[ids][:,dm].mean(0));cost=(abs(d[ids][:,None,dm]-rd[ids][None,:,dm])/scale).mean(-1);a,b=linear_sum_assignment(cost);jw=float(cost[a,b].mean())
    # As in ExplicitScore: bottom half and top two order statistics, not node-ID groups.
    sr=np.sort(refnode[ids]);sd=np.sort(node[ids]);k=max(1,len(ids)//2);h=min(2,len(ids));dl=abs(sd-sr)
    le=float(dl[:k].mean()/max(1.,sr[:k].mean()));he=float(dl[-h:].mean()/max(1.,sr[-h:].mean()))
    mw=float((abs(np.sort(d[ids][:,dm],axis=0)-np.sort(rd[ids][:,dm],axis=0))/scale).mean())
    low.append(le);hub.append(he);joint.append(jw);marg.append(mw)
    typed.append(dict(**ident,type=ty,nodes=len(ids),low_rank_count=k,hub_rank_count=h,expected_total_degree_MAE=float(abs(node[ids]-refnode[ids]).mean()),joint_expected_degree_W=jw,relation_expected_degree_W=mw,low_rank_error=le,hub_rank_error=he))
   rows.append(dict(**ident,CE=float(ce.sum()/18336),positive_CE=float(ce[y].mean()),negative_CE=float(ce[~y].mean()),positive_count=int(y.sum()),negative_count=int((~y).sum()),expected_edges=float(p.sum()),edge_abs_error=abs(float(p.sum())-refedges),missing_mass=float((1-p[y]).sum()),false_mass=float(p[~y].sum()),node_total_degree_MAE=float(abs(node-refnode).mean()),node_relation_degree_MAE=float(abs(d-rd).mean()),joint_expected_degree_W=float(np.mean(joint)),relation_expected_degree_W=float(np.mean(marg)),low_rank_error=float(np.mean(low)),hub_rank_error=float(np.mean(hub)),independent_triangle_proxy=float(tr.sum()),triangle_abs_error=abs(float(tr.sum()-rt.sum())),typed_triangle_L1=float(abs(tr-rt).sum()/rt.sum())))
   for r in range(R):
    sel=f==r;pos=sel&y;neg=sel&~y
    relations.append(dict(**ident,relation=r,slots=int(sel.sum()),positive_count=int(pos.sum()),negative_count=int(neg.sum()),CE=float(ce[sel].mean()),positive_CE=float(ce[pos].mean()) if pos.any() else None,negative_CE=float(ce[neg].mean()) if neg.any() else None,expected_edges=float(p[sel].sum()),reference_edges=int(pos.sum()),edge_abs_error=abs(float(p[sel].sum())-pos.sum()),missing_mass=float((1-p[pos]).sum()),false_mass=float(p[neg].sum())))
   for c,num,ref in zip(classes,tr,rt):trirows.append(dict(**ident,type_triple=str(c),triangle_product_sum=float(num),reference=float(ref),abs_error=float(abs(num-ref))))
   cache[parent,arm,key]=(p,q,f)
table('per_input',rows);table('per_relation',relations);table('per_type',typed);table('triangles',trirows)
summary=[]
metrics=[k for k in rows[0] if k not in ['parent','arm','input','t','independent_input','positive_count','negative_count']]
for parent in ['CE','PS']:
 for arm in arms:
  for phase in ['new_states','observed_t1','update','all_predeclared']:
   rr=[r for r in rows if r['parent']==parent and r['arm']==arm and ((phase=='new_states' and r['independent_input']) or (phase=='observed_t1' and r['t']==1) or (phase=='update' and r['input']=='update') or (phase=='all_predeclared' and r['input']!='update'))]
   summary.append(dict(parent=parent,arm=arm,panel=phase,inputs=len(rr),**{k:float(np.mean([r[k] for r in rr])) for k in metrics}))
table('summary',summary)
def cmp(a,b):
 den=np.linalg.norm(a)*np.linalg.norm(b)
 return dict(norm_a=float(np.linalg.norm(a)),norm_b=float(np.linalg.norm(b)),cosine=float(a@b/den) if den else None,difference_norm=float(np.linalg.norm(a-b)))
responses=[];parts=[]
for parent in ['CE','PS']:
 for key in keys:
  p0,q,f=cache[parent,'CE',key];diffs={arm:cache[parent,arm,key][0]-p0 for arm in arms}
  for a,b in [('FULL','COMMON'),('FULL','RESIDUAL'),('MATCH','RESIDUAL')]:responses.append(dict(parent=parent,input=key,comparison=a+'_vs_'+b,**cmp(diffs[a],diffs[b])))
  for arm in arms:
   v=diffs[arm];common=np.zeros_like(v)
   for r in range(R):common[f==r]=v[f==r].mean()
   parts.append(dict(parent=parent,input=key,arm=arm,probability_difference_norm=float(np.linalg.norm(v)),relation_mean_component_norm=float(np.linalg.norm(common)),within_relation_component_norm=float(np.linalg.norm(v-common)),net_expected_edges_change=float(v.sum())))
table('probability_response',responses);table('probability_components',parts)
save('CPU_evaluation',dict(seconds=time.time()-START,rows=len(rows),relations=len(relations),types=len(typed),triangles=len(trirows),reference_edges=refedges,reference_triangles=float(rt.sum()),definition='all quantities are conditional marginal probability proxies; triangle assumes conditional independent Bernoulli edges; no generated graph, LCC or hard degree distribution'))
print(json.dumps(summary,indent=2))
