"""Read-only CPU attribution of previously saved conditional probability changes.
No model imports/calls, training, new Bernoulli graphs or RNG draws.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']=os.environ['OPENBLAS_NUM_THREADS']='1'
import json,csv,time,hashlib,resource
from pathlib import Path
import numpy as np
import torch
torch.set_num_threads(1)
R=Path(__file__).resolve().parents[2];S=R/'results/direct_structure_response_20261108T205816';O=R/'results/direct_response_attribution_20261108'
O.mkdir(exist_ok=False);tick=time.time();bindings={}
def bind(p):
 p=Path(p);bindings[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest();return p
def read(p):return json.loads(bind(p).read_text())
def write(name,rows):
 with (O/name).open('w') as f:
  w=csv.DictWriter(f,list(dict.fromkeys(k for x in rows for k in x)));w.writeheader();w.writerows(rows)
protocol=dict(scope='all four historical parents x t1/500/1000, CPU only, no new random draws',
 contrasts=['start_to_CE','start_to_DIRECT','CE_to_DIRECT'],
 triangle_attribution='straight probability-line exact integral, not optimizer causal attribution',
 groups='relation x reference label; per-node current Xt degree <=Q25 low, >=Q90 and not low hub, otherwise middle; closure-high >=baseline Q75 of positive weighted CN',
 CPU_threads=1,GPU=0,timeout_seconds=120,memory_budget_GiB=2)
(O/'protocol.json').write_text(json.dumps(protocol,indent=2))
tasks=read(R/'results/output_boundary_replay_20261108T081654/tasks.json');probes=read(S/'probes.json')
panels=[];rels=[];nodes=[];tri_types=[];errors=[]
def tri(P):return float(np.sum(P*(P@P))/6)
def decompose(P,Q,i,j):
 D=Q-P;v0=(P@P)[i,j];v1=((P@D+D@P)/2)[i,j];v2=(D@D/3)[i,j]
 return D[i,j]*(v0+v1+v2),D[i,j]*v0,D[i,j]*v1,D[i,j]*v2,v0
# Independent unordered triangle enumeration checks identity on a deterministic fixture.
P=np.arange(49,dtype=float).reshape(7,7)%13/15;P=(P+P.T)/2;np.fill_diagonal(P,0)
Q=.8*P+.02;np.fill_diagonal(Q,0);i,j=np.triu_indices(7,1)
c,*_=decompose(P,Q,i,j)
enumerated=sum(Q[a,b]*Q[b,d]*Q[a,d]-P[a,b]*P[b,d]*P[a,d] for a in range(7) for b in range(a+1,7) for d in range(b+1,7))
assert abs(c.sum()-enumerated)<1e-12
for task in tasks:
 ds=task['dataset'];cfg=task['row']['config']['datasets'][ds];names=cfg['type_names']
 data=bind(R/'research_branches/typed_structure_early_cn_expansion_review/experiments/typed_structure_early_cn_expansion_v1/package'/cfg['file'])
 assert bindings[str(data)]==cfg['sha256']
 with np.load(data) as z:types=z['types'].copy()
 n=len(types);fixed=torch.load(bind(S/f'{ds}_fixed_inputs.pt'),map_location='cpu')['inputs']
 for t,seed,inp in fixed:
  i,j=inp['query'];families=inp['f'];y=inp['labels'];assert np.all(i<j) and len(set(zip(i,j)))==len(i)
  Y=np.zeros((n,n));Y[i,j]=Y[j,i]=y;xt=np.zeros((n,n));a,b=inp['edges'];xt[a,b]=1;xt[b,a]=1;xtdeg=xt.sum(1)
  labels={}
  for f in np.unique(families):
   ix=families==f;endtypes={(min(types[a],types[b]),max(types[a],types[b])) for a,b in zip(i[ix],j[ix])};assert len(endtypes)==1
   u,v=next(iter(endtypes));labels[int(f)]=names[u]+'-'+names[v]
  for parent in ['CE','PS']:
   condition=ds+'_'+parent;ps={}
   for arm in ['start','CE','CE_STRUCT']:
    with np.load(bind(S/f'{condition}_{arm}_t{t}.npz')) as z:p=z['probability'].copy()
    assert np.isfinite(p).all() and np.all((p>=0)&(p<=1));M=np.zeros((n,n));M[i,j]=M[j,i]=p;ps[arm]=M
    old=next(x for x in probes if x['condition']==condition and x['arm']==arm and x['t']==t)
    assert abs(tri(M)-old['expected_triangles'])<1e-8
   for source,dest,name in [('start','CE','start_to_CE'),('start','CE_STRUCT','start_to_DIRECT'),('CE','CE_STRUCT','CE_to_DIRECT')]:
    P,Q=ps[source],ps[dest];dp=(Q-P)[i,j];con,linear,quad,cubic,sens=decompose(P,Q,i,j)
    delta=tri(Q)-tri(P);err=abs(con.sum()-delta);errors.append(err);assert err<1e-8
    positive=sens[sens>0];cut=float(np.quantile(positive,.75)) if len(positive) else None;high=(sens>=cut) if cut is not None else np.zeros(len(sens),bool)
    base=dict(condition=condition,t=t,contrast=name)
    panels.append(dict(**base,expected_triangles_before=tri(P),expected_triangles_after=tri(Q),triangle_delta=delta,triangle_reference=tri(Y),exact_identity_error=err,linear=float(linear.sum()),quadratic=float(quad.sum()),cubic=float(cubic.sum()),edge_mass_delta=float(dp.sum()),true_edge_mass_delta=float(dp[y==1].sum()),nonedge_mass_delta=float(dp[y==0].sum()),triangle_true_edge_contribution=float(con[y==1].sum()),triangle_nonedge_contribution=float(con[y==0].sum()),probability_delta_L2=float(np.linalg.norm(dp)),probability_delta_max=float(abs(dp).max()),closure_high_support=int(high.sum()),closure_high_delta_mean=float(dp[high].mean()) if high.any() else None,closure_other_delta_mean=float(dp[~high].mean()) if (~high).any() else None,closure_high_triangle_contribution=float(con[high].sum())))
    for f in sorted(labels):
     for label in [0,1]:
      ix=(families==f)&(y==label);k=int(ix.sum());ref=float((y[families==f]==1).sum())
      rels.append(dict(**base,family=f,relation=labels[f],label=label,support=k,relation_reference_edges=ref,probability_before=float(P[i[ix],j[ix]].sum()),probability_after=float(Q[i[ix],j[ix]].sum()),mass_delta=float(dp[ix].sum()),triangle_contribution=float(con[ix].sum()),increased=int((dp[ix]>0).sum()),decreased=int((dp[ix]<0).sum()),mean_delta=float(dp[ix].mean()) if k else None))
    if name=='CE_to_DIRECT':
     # Per-node expected degree and independent-edge triangle participation.
     degP=P.sum(1);degQ=Q.sum(1);degY=Y.sum(1);partP=(P*(P@P)).sum(1)/2;partQ=(Q*(Q@Q)).sum(1)/2
     for node,ty in enumerate(types):
      degrees=xtdeg[types==ty];low,highq=np.quantile(degrees,[.25,.9]);group='low' if xtdeg[node]<=low else ('hub' if xtdeg[node]>=highq else 'middle')
      nodes.append(dict(**base,node=node,type=names[ty],Xt_degree=float(xtdeg[node]),Xt_group=group,reference_degree=float(degY[node]),degree_before=float(degP[node]),degree_after=float(degQ[node]),degree_abs_error_delta=float(abs(degQ[node]-degY[node])-abs(degP[node]-degY[node])),triangle_participation_delta=float(partQ[node]-partP[node])))
     # Independent vectorized enumeration checks matrix formula and typed triangle shifts.
     typemap={};total=0.
     for a in range(n-2):
      b,c=np.triu_indices(n-a-1,1);b=b+a+1;c=c+a+1
      shifts=Q[a,b]*Q[b,c]*Q[a,c]-P[a,b]*P[b,c]*P[a,c];total+=float(shifts.sum())
      for key in set(zip(types[b].tolist(),types[c].tolist())):
       ix=(types[b]==key[0])&(types[c]==key[1]);tp=tuple(sorted((int(types[a]),int(key[0]),int(key[1]))));typemap[tp]=typemap.get(tp,0.)+float(shifts[ix].sum())
     assert abs(total-delta)<1e-8
     tri_types.extend(dict(**base,triangle_types='-'.join(names[z] for z in tp),delta=v) for tp,v in sorted(typemap.items()))
write('panels.csv',panels);write('relations.csv',rels);write('nodes.csv',nodes);write('triangle_types.csv',tri_types)
updates=read(S/'updates.json');controls=read(S/'control_updates.json');training=[]
for x in updates:
 old=next(a for a in controls if a['condition']==x['condition'] and a['offset']==x['offset']);assert x['input_sha256']==old['input_sha256']
 training.append(dict(condition=x['condition'],offset=x['offset'],t=x['t'],CE_control=old['unweighted_CE'],CE_candidate=x['unweighted_CE'],CE_delta=x['unweighted_CE']-old['unweighted_CE'],candidate_aux_loss=x['auxiliary_loss'],CE_update_norm=old['parameter_moves']['all']['L2'],candidate_update_norm=x['parameter_moves']['all']['L2']))
write('updates.csv',training)
for path,h in bindings.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==h
(O/'source_binding.json').write_text(json.dumps({str(Path(__file__)):hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2))
(O/'asset_binding.json').write_text(json.dumps(bindings,indent=2))
receipt=dict(status='complete',seconds=time.time()-tick,max_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,GPU_hours=0,model_calls=0,updates=0,new_graphs=0,panels=len(panels),max_triangle_identity_error=max(errors),original_assets_unchanged=True)
(O/'receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
print(json.dumps([x for x in panels if x['condition'].startswith('PubMed') and x['t']==500 and x['contrast']=='CE_to_DIRECT'],indent=2))
