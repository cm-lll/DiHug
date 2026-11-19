"""Small algebra/coverage tests; no model execution or extra graph generation."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import json,itertools,hashlib,time,sys
from pathlib import Path
import numpy as np
from reliability_analysis import coefficient,cosine,norm,write

def main():
    out=Path(sys.argv[1]);start=time.time();p=json.loads((out/'protocol.json').read_text())
    rng=np.random.default_rng(202611132603)
    v=rng.normal(size=(16,47));cost=rng.uniform(0,2,16);center=v-v.mean(0)
    G=center@center.T
    largest=0.
    for row in p['models']:
        assert sorted(sum(row['four_groups'],[]))==list(range(16))
        assert sorted(sum(row['eight_groups'],[]))==list(range(16))
        assert not set(row['eight_groups'][0])&set(row['eight_groups'][1])
        assert len(set(x['seed'] for x in row['new_trajectories']))==10
        assert all(len(set(x['times']))==8 and min(x['times'])>=1 and max(x['times'])<=1000 for x in row['new_trajectories'])
        for ix in row['four_groups']+row['eight_groups']+[list(range(16))]:
            c=coefficient(cost,ix);k=len(ix);a=(k*cost[ix]-cost[ix].sum())/(k-1)
            direct=.01*np.sum(a[:,None]*v[ix],axis=0)/k
            np.testing.assert_allclose(c@v,direct,atol=1e-15,rtol=1e-12)
            np.testing.assert_allclose(norm(c,G),np.linalg.norm(direct),atol=1e-15,rtol=1e-12)
            largest=max(largest,float(np.max(abs(c@v-direct))))
        a=coefficient(cost,row['four_groups'][0]);b=coefficient(cost,row['four_groups'][1])
        direct=float((a@v)@(b@v)/(np.linalg.norm(a@v)*np.linalg.norm(b@v)))
        assert abs(cosine(a,b,G)-direct)<1e-12
    # All subsets expectation is the full finite-population covariance estimator.
    smallcost=cost[:5];smallv=v[:5];target=coefficient(smallcost,list(range(5)))@smallv
    for k in (2,3,4):
        average=np.mean([coefficient(smallcost,list(ix))@smallv for ix in itertools.combinations(range(5),k)],axis=0)
        np.testing.assert_allclose(average,target,rtol=1e-12,atol=1e-15)
    assert cosine(np.zeros(16),np.ones(16),G) is None
    # Common parameter component cancels for zero-sum cost coefficients.
    c=coefficient(cost,list(range(16)))
    np.testing.assert_allclose(c@(v+100),c@v,rtol=1e-10,atol=1e-12)
    assert 'torch' not in sys.modules
    write(out/'CPU_algebra_checks.json',dict(status='passed',coefficient_direct_max_abs_error=largest,
          finite_population_covariance_verified=True,centered_gram_norm_cosine_verified=True,
          frozen_partition_coverage=True,zero_cosine_undefined=True,model_calls=0,GPU_hours=0,seconds=time.time()-start,
          limitation='Finite algebra tests do not certify eager gradient as unbiased terminal reward gradient or predict quality.'))
    paths=[Path(__file__),Path(__file__).with_name('reliability_analysis.py')]
    write(out/'analysis_source_bindings.json',{str(q.resolve()):hashlib.sha256(q.read_bytes()).hexdigest() for q in paths})
if __name__=='__main__':main()
