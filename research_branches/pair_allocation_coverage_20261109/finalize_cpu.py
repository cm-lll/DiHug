"""Read-only saved-vector verification, relation diagnostics and cost accounting."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import json,csv,itertools,hashlib,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/pair_allocation_coverage_20261109';FULL=ROOT/'results/joint_degree_full_signal_20261109'
def read(n):return json.loads((OUT/(n+'.json')).read_text())
def save(n,x):
    (OUT/(n+'.json')).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    if isinstance(x,list) and x and isinstance(x[0],dict):
        with (OUT/(n+'.csv')).open('w',newline='') as f:
            w=csv.DictWriter(f,list(x[0]));w.writeheader();w.writerows(x)
def cos(a,b):
    den=float(np.linalg.norm(a)*np.linalg.norm(b));return float(a@b)/den if den else None
start=time.monotonic();assert read('receipt')['status']=='complete'
within=read('within_background');rels=read('relations');summaries=read('summary');pairs=read('pairs');plan=read('frozen_pairs')
detail=[];costs=[];checks=[];subsets=[]
for state in plan:
    tag=state['state'];rows=[x for x in pairs if x['state']==tag]
    with np.load(FULL/(tag+'.npz')) as z:f=z['relation'].copy()
    with np.load(OUT/(tag+'_vectors.npz')) as z:v={k:z[k].copy() for k in z.files}
    grouped={(rep,r):sorted([x for x in rows if x['rep']==rep and x['relation']==r],key=lambda x:x['position']) for rep in range(3) for r in range(10)}
    for k in [16,64,256]:
        means=np.array([v[f'rep{rep}_k{k}'] for rep in range(3)]).mean(0);total=float(means@means)
        mask=f!=9
        ss=[cos(v[f'rep{a}_k{k}'][mask],v[f'rep{b}_k{k}'][mask]) for a,b in itertools.combinations(range(3),2)]
        subsets.append(dict(state=tag,k=k,scope='exclude_SS_descriptive_only',cosine_mean=float(np.mean(ss)),cosine_min=min(ss),cosine_max=max(ss)))
        for r in range(10):
            vv=[v[f'rep{rep}_k{k}'][f==r] for rep in range(3)];cs=[cos(vv[a],vv[b]) for a,b in itertools.combinations(range(3),2)]
            detail.append(dict(state=tag,k=k,relation=r,cosine_mean=float(np.mean(cs)) if all(c is not None for c in cs) else None,
                cosine_min=min(cs) if all(c is not None for c in cs) else None,cosine_max=max(cs) if all(c is not None for c in cs) else None,
                mean_gradient_squared_share=float(means[f==r]@means[f==r])/total if total else None))
        for rep in range(3):
            g=np.zeros(len(f));variance=0.;reward=0.;cost={}
            for r in range(10):
                subset=grouped[(rep,r)][:k];m=len(subset);N=next(x['population'] for x in state['parts'] if x['rep']==rep and x['relation']==r)
                local=np.zeros(len(f));square=0.
                for row in subset:
                    local[row['remove']]-=row['h'];local[row['add']]+=row['h'];square+=2*row['h']**2
                avg=local/m;g+=avg/10;variance+=(1-m/N)*(square-m*float(avg@avg))/(m-1)/m/100
                good=[row for row in subset if row['score_change']<-1e-12];reward+=len(good)/m/10
                for metric in ['triangle_error','typed_triangle_L1','joint_distance','hub','low','isolates_error','LCC_error']:
                    cost[metric]=cost.get(metric,0)+sum(row[metric+'_change']>1e-12 for row in good)/m/10
            assert np.max(abs(g-v[f'rep{rep}_k{k}']))<1e-18
            orig=next(x for x in summaries if x['state']==tag and x['rep']==rep and x['requested_per_relation']==k)
            assert abs(variance-orig['estimated_logit_MSE'])<1e-16
            costs.append(dict(state=tag,k=k,rep=rep,relation_equal_reward_fraction=reward,
                **{name+'_conditional_cost_fraction':val/reward if reward else None for name,val in cost.items()}))
            checks.append(dict(state=tag,rep=rep,k=k,vector_matches=True,variance_matches=True))
assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in read('binding').items())
save('per_relation_stability',detail);save('without_SS_descriptive',subsets);save('relation_weighted_costs',costs)
aggregate=[]
for k in [16,64,256]:
    w=[r for r in within if r['requested_per_relation']==k];c=[r for r in costs if r['k']==k]
    aggregate.append(dict(k=k,within_cosine_mean=float(np.mean([r['cosine_mean'] for r in w])),
        within_cosine_min=min(r['cosine_min'] for r in w),within_cosine_max=max(r['cosine_max'] for r in w),
        **{name:float(np.mean([r[name] for r in c])) for name in c[0] if name.endswith('_fraction')}))
save('aggregate',aggregate);save('independent_checks',dict(vectors=36,variance_checks=36,all_pass=True,bindings_unchanged=True,seconds=time.monotonic()-start))
print(json.dumps(aggregate,ensure_ascii=False,indent=2));print(json.dumps(subsets,ensure_ascii=False,indent=2))
