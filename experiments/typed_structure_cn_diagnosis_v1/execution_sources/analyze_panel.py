"""CPU reductions of the fixed eight outputs; mutually exclusive bins per axis."""
import pathlib,json,csv,collections,ast,math
import numpy as np
B=pathlib.Path(__file__).resolve().parent;ROOT=B.parents[1]
RUN=ROOT/'results/typed_structure_final_screen_formal_v1_20261029T0325/run'
def read(p):return json.loads(p.read_text())
def write(name,rows):
 keys=list(dict.fromkeys(k for x in rows for k in x))
 with (B/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def div(x,n):return float(x/n) if n else None
def bins(x,bounds):
 bounds=np.array([float(v) for v in bounds]);idx=np.searchsorted(bounds,x,side='right')-1
 return [(f'{int(a)}+' if math.isinf(b) else f'{int(a)}..{int(b)-1}',idx==i) for i,(a,b) in enumerate(zip(bounds[:-1],bounds[1:]))]
panel=read(B/'frozen_panel.json');receipt=read(B/'forward/receipt.json');assert receipt['status']=='complete' and receipt['completed_full_queries']==8
names=['Gene','Disease','Chemical','Species'];relations=list(__import__('itertools').combinations_with_replacement(range(4),2));rows=[];buckets=[];clean=[];slots=[]
for info in panel['inputs']:
 inp=info['name'];c=dict(np.load(B/'forward'/('C_'+inp+'.npz')));e=dict(np.load(B/'forward'/('E_'+inp+'.npz')))
 for k in ['ids','state','t','types','schema','query','family']:np.testing.assert_array_equal(c[k],e[k])
 t=int(c['t']);state=c['state'].astype(bool);u,v=c['query'];f=c['family'];A=np.zeros((192,192),dtype=np.int64);A[u[state],v[state]]=1;A[v[state],u[state]]=1
 degree=A.sum(1);rd=np.zeros((192,10),dtype=np.int64);np.add.at(rd,(u[state],f[state]),1);np.add.at(rd,(v[state],f[state]),1)
 common=A@A;axes={'cn_all':(common[u,v],panel['bins']['common_neighbors']),'min_degree':(np.minimum(degree[u],degree[v]),panel['bins']['endpoint_total_degree']),'max_degree':(np.maximum(degree[u],degree[v]),panel['bins']['endpoint_total_degree']),'min_relation_degree':(np.minimum(rd[u,f],rd[v,f]),panel['bins']['endpoint_relation_degree']),'max_relation_degree':(np.maximum(rd[u,f],rd[v,f]),panel['bins']['endpoint_relation_degree'])}
 for typ,name in enumerate(names):cn=A[:,c['types']==typ]@A[c['types']==typ,:];axes['cn_'+name]=(cn[u,v],panel['bins']['common_neighbors'])
 deltaP=e['p0']-c['p0'];deltaQ=e['q']-c['q'];deltaL=(e['logits'][:,1].astype(float)-e['logits'][:,0])-(c['logits'][:,1].astype(float)-c['logits'][:,0])
 np.savez_compressed(B/'forward'/('paired_'+inp+'.npz'),ids=c['ids'],state=c['state'],family=f,query=c['query'],delta_p0=deltaP,delta_q=deltaQ,delta_logit_edge_minus_none=deltaL,cn=common[u,v],endpoint_degree=np.stack([degree[u],degree[v]]),endpoint_relation_degree=np.stack([rd[u,f],rd[v,f]]))
 for fam,(a,b) in enumerate(relations):
  mask=f==fam;relation=names[a]+'--'+names[b]
  values={}
  for arm,x in [('C',c),('E',e)]:
   birth=float(x['q'][mask&~state].sum());death=float((1-x['q'][mask&state]).sum());values[arm]=dict(p0_sum=float(x['p0'][mask].sum()),q_sum=float(x['q'][mask].sum()),birth=birth,death=death,net=birth-death)
  rows.append(dict(input=inp,t=t,family=fam,relation=relation,slots=int(mask.sum()),current_edges=int((mask&state).sum()),**{arm+'_'+k:v for arm,z in values.items() for k,v in z.items()},**{'E_minus_C_'+k:values['E'][k]-values['C'][k] for k in values['C']}))
  for exists in [False,True]:
   base=mask&(state==exists)
   for axis,(vals,bounds) in axes.items():
    for label,selection in bins(vals,bounds):
     ix=base&selection;n=int(ix.sum());dq=deltaQ[ix];dp=deltaP[ix];dl=deltaL[ix]
     buckets.append(dict(input=inp,t=t,family=fam,relation=relation,current_edge=int(exists),axis=axis,bin=label,slots=n,C_p0_sum=float(c['p0'][ix].sum()),E_p0_sum=float(e['p0'][ix].sum()),C_q_sum=float(c['q'][ix].sum()),E_q_sum=float(e['q'][ix].sum()),delta_p0_sum=float(dp.sum()),delta_q_sum=float(dq.sum()),delta_q_positive_sum=float(dq[dq>0].sum()),delta_q_negative_sum=float(dq[dq<0].sum()),delta_logit_mean=div(dl.sum(),n),delta_p0_mean=div(dp.sum(),n),delta_q_mean=div(dq.sum(),n),delta_q_min=float(dq.min()) if n else None,delta_q_q25=float(np.quantile(dq,.25)) if n else None,delta_q_median=float(np.median(dq)) if n else None,delta_q_q75=float(np.quantile(dq,.75)) if n else None,delta_q_max=float(dq.max()) if n else None))
  if 'clean_label' in c:
   for exists in [False,True]:
    for truth in [0,1]:
     ix=mask&(state==exists)&(c['clean_label']==truth);n=int(ix.sum())
     clean.append(dict(input=inp,t=t,family=fam,relation=relation,current_edge=int(exists),clean_label=truth,slots=n,**{arm+'_'+k:div(x[k][ix].sum(),n) for arm,x in [('C',c),('E',e)] for k in ['p0','q','ce']}))
write('cross_relations.csv',rows);write('cross_slot_strata.csv',buckets);write('ordinary_label_groups.csv',clean)
# Algebraic opportunity/rate decomposition per stored graph, not a causal attribution.
ev=read(RUN/'unified_evaluation.json');closure=[];tri=[];index={(x['dataset'],x['arm'],x['sample_seed'],x['version']):x for x in ev['records']}
for d,left,right in [('PubMed','E','C'),('IMDB','D','A'),('IMDB','E','D')]:
 ref=read(RUN/'derived_lcc_evaluation'/f'{d}_reference.json')['statistics']
 for sample in [2026102821,2026102822]:
  for version in ['original','lcc']:
   l=index[d,left,sample,version]['statistics'];r=index[d,right,sample,version]['statistics']
   for key in l['typed_twohop_closure']:
    x,y,z=l['typed_twohop_closure'][key],r['typed_twohop_closure'][key],ref['typed_twohop_closure'][key]
    for basis in ['path','unique_pair']:
     nl,nr=x[basis+'_denominator'],y[basis+'_denominator'];pl,pr=x[basis+'_ratio'],y[basis+'_ratio'];numl,numr=x[basis+'_numerator'],y[basis+'_numerator'];op=rate=None
     if pl is not None and pr is not None:op=(nl-nr)*(pl+pr)/2;rate=(pl-pr)*(nl+nr)/2;assert np.isclose(op+rate,numl-numr,atol=1e-9)
     closure.append(dict(dataset=d,contrast=left+'-'+right,sample_seed=sample,version=version,pattern=key,basis=basis,left_opportunities=nl,right_opportunities=nr,reference_opportunities=z[basis+'_denominator'],left_closed=numl,right_closed=numr,reference_closed=z[basis+'_numerator'],left_ratio=pl,right_ratio=pr,reference_ratio=z[basis+'_ratio'],delta_closed=numl-numr,opportunity_component=op,rate_component=rate,reason=None if op is not None else 'undefined_zero_denominator',interpretation='symmetric exact algebraic decomposition, not causal'))
   for key in set(l['typed_triangles'])|set(r['typed_triangles'])|set(ref['typed_triangles']):
    x,y,z=l['typed_triangles'].get(key,0),r['typed_triangles'].get(key,0),ref['typed_triangles'].get(key,0)
    tri.append(dict(dataset=d,contrast=left+'-'+right,sample_seed=sample,version=version,pattern=key,left=x,right=y,reference=z,quantity_difference=x-y,absolute_error_difference=abs(x-z)-abs(y-z)))
write('closure_decomposition.csv',closure);write('triangle_contrasts.csv',tri)
summary=[]
for inp in [x['name'] for x in panel['inputs']]:
 rr=[x for x in rows if x['input']==inp];s=dict(input=inp,current_edges=sum(x['current_edges'] for x in rr))
 for arm in ['C','E']:
  for k in ['p0_sum','q_sum','birth','death','net']:s[arm+'_'+k]=sum(x[arm+'_'+k] for x in rr)
 for k in ['p0_sum','q_sum','birth','death','net']:s['E_minus_C_'+k]=s['E_'+k]-s['C_'+k]
 summary.append(s)
write('cross_totals.csv',summary)
print(json.dumps(summary,indent=2))
print('relation net deltas')
for inp in [x['name'] for x in panel['inputs']]:print(inp,[(x['relation'],round(x['E_minus_C_p0_sum'],4),round(x['E_minus_C_birth'],6),round(x['E_minus_C_death'],6),round(x['E_minus_C_net'],6)) for x in rows if x['input']==inp])
