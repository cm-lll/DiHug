"""Independent NumPy verification of saved estimator samples. No model calls."""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:
    os.environ[key]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import json
import hashlib
import time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/whole_graph_feedback_cpu_20261110'
start=time.perf_counter()
fixture=np.load(OUT/'exact_fixture.npz')
bits=fixture['bits'];cost=fixture['costs']
profiles=json.loads((OUT/'profiles.json').read_text())
rows=[]
for info in profiles:
    p=np.array(info['probability']);mass=np.prod(np.where(bits,p,1-p),axis=1)
    exact=np.sum(mass[:,None]*cost[:,None]*(bits-p),axis=0)
    # Independence gives E[(C1-C2)(B1-p)] = E[C1(B1-p)] - E[C2]E[B1-p].
    expected_loo=exact-cost@mass*np.sum(mass[:,None]*(bits-p),axis=0)
    assert np.max(abs(expected_loo-exact))<1e-14
    assert np.max(abs(exact-np.array(info['exact_gradient'])))<1e-14
    for K in [4,16,64]:
        a=np.load(OUT/f"{info['profile']}_K{K}.npz")
        b=a['sampled_bits'];c=a['costs']
        indices=np.sum(b.astype(int)*2**np.arange(9,-1,-1),axis=-1)
        error=float(np.max(abs(c[...,0]-cost[indices])))
        assert error<1e-12
        lo=(c.sum(1,keepdims=True)-c)/(K-1)
        g=((c-lo)*(b-p)).mean(1)
        assert np.max(abs(g-a['leave_one_out']))<1e-14
        rows.append(dict(profile=info['profile'],K=K,hard_score_lookup_error=error,
                         LOO_recompute_error=float(np.max(abs(g-a['leave_one_out'])))))
    sg=np.array(info['soft_gradient'])
    info['soft_wrong_coordinate_signs']=int(np.sum(sg*exact<0))
    info['exact_gradient_norm']=float(np.linalg.norm(exact))
    info['soft_relative_bias']=float(np.linalg.norm(sg-exact)/np.linalg.norm(exact))
    # A fixed exact distribution calculation, not a new graph sample.
    info['probability_all_four_draws_empty']=float(np.prod(1-p)**4)
(OUT/'audit.json').write_text(json.dumps(dict(checks=rows,profiles=profiles,
       wall_seconds=time.perf_counter()-start,model_calls=0,GPU_hours=0),indent=2))
print(json.dumps(dict(checks=len(rows),max_score_error=max(r['hard_score_lookup_error'] for r in rows),
                      soft_wrong_signs=[p['soft_wrong_coordinate_signs'] for p in profiles])))
