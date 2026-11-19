"""Read saved derivatives only; does not load or execute a model."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
import numpy as np,json
from pathlib import Path
OUT=Path(__file__).resolve().parents[2]/'results/analytic_joint_model_check_20261113'
def cosine(a,b):
    x=np.linalg.norm(a)*np.linalg.norm(b)
    return float(np.sum(a*b)/x) if x else None
rows=[]
for path in sorted(OUT.glob('*_A_transport.json')):
    record=json.loads(path.read_text());job=record['job']
    a=np.load(OUT/(job+'_bias_gradients.npz'))['edge_logit_gradient']
    b=np.load(OUT/(job+'_risk_gradients.npz'))['edge_logit_gradient']
    rows.append(dict(job=job,stacked_edge_logit_cosine=cosine(a,b),
        times=[1,75,250,550,800,1000],
        per_time_cosines=[cosine(x,y) for x,y in zip(a,b)],
        bias_per_time_norm=np.linalg.norm(a,axis=1).tolist(),risk_per_time_norm=np.linalg.norm(b,axis=1).tolist(),
        scope='edge logit derivatives over six inputs are stacked, network parameter derivatives are summed through different Jacobians; their cosines are descriptive, not a causal decomposition'))
(OUT/'edge_gradient_readout.json').write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps(rows,indent=2))
