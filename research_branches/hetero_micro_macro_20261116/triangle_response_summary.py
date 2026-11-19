"""Independent finite check and descriptive summaries of the saved CPU audit."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import itertools,json,statistics,csv
from pathlib import Path
import numpy as np
OUT=Path(__file__).resolve().parents[2]/'results/triangle_response_audit_20261116'
pairs=list(itertools.combinations(range(4),2));tris=list(itertools.combinations(range(4),3))
x=np.array([1,1,0,1,0,1]);q=np.array([.85,.8,.3,.6,.25,.9])
def matrix(a):
    X=np.zeros((4,4))
    for (u,v),b in zip(pairs,a):X[u,v]=X[v,u]=b
    return X
X=matrix(x);Q=matrix(q)
old=[(i,j,k) for i,j,k in tris if X[i,j]*X[j,k]*X[i,k]]
birth=death=after=0.
for bits in itertools.product((0,1),repeat=6):
    B=matrix(bits);weight=float(np.prod(np.where(bits,q,1-q)))
    actual=[(i,j,k) for i,j,k in tris if B[i,j]*B[j,k]*B[i,k]]
    birth+=weight*len(set(actual)-set(old));death+=weight*len(set(old)-set(actual));after+=weight*len(actual)
expected=lambda a:sum(a[i,j]*a[j,k]*a[i,k] for i,j,k in tris)
survive=sum(Q[i,j]*Q[j,k]*Q[i,k] for i,j,k in old)
np.testing.assert_allclose([after,birth,death],[expected(Q),expected(Q)-survive,len(old)-survive],atol=1e-12)
D=Q-X;linear=float(np.triu(D*(X@X),1).sum());quad=float(np.sum(X*(D@D))/2);cubic=float(np.sum((D@D)*D)/6)
np.testing.assert_allclose(linear+quad+cubic,after-len(old),atol=1e-12)
test=dict(passed=True,enumerated_outputs=64,expected_triangles=after,expected_births=birth,expected_destructions=death,
          polynomial_components=[linear,quad,cubic],model_calls=0,GPU_hours=0)
(OUT/'finite_decomposition_test.json').write_text(json.dumps(test,indent=2))
rows=json.loads((OUT/'terminal_decomposition.json').read_text());summary=[]
for ds,seed,arm in sorted({(r['dataset'],r['seed'],r['arm']) for r in rows}):
    v=[r for r in rows if (r['dataset'],r['seed'],r['arm'])==(ds,seed,arm)]
    r=dict(dataset=ds,seed=seed,arm=arm,n=len(v))
    for m in ['edges','triangles','wedges','closure','opportunity_component','closure_component','top10_triangle_share']:
        r[m+'_mean']=statistics.mean(x[m] for x in v);r[m+'_sample_SD']=statistics.stdev(x[m] for x in v)
    summary.append(r)
(OUT/'terminal_model_summary.json').write_text(json.dumps(summary,indent=2))
with (OUT/'terminal_model_summary.csv').open('w') as f:
    w=csv.DictWriter(f,summary[0].keys());w.writeheader();w.writerows(summary)
print(json.dumps(test,indent=2))
