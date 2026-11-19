"""Analytical four-independent-path comparison, not a new baseline candidate.

For R=terminal cost, U=sum score, E U=0, g=E[(R-mu)U],
Var_trace(unbiased sample covariance / LOO) =
 Var_trace((R-mu)U)/n + (Var(R) E||U||² + ||g||²)/(n(n-1)).
Compare with n independent state-baseline paths; oracle constant is labeled oracle.
"""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import importlib.util, itertools, json, resource, time
from pathlib import Path
import numpy as np
from scipy.special import logit
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/terminal_baseline_cpu_20261110'
def load(name,p):
    spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def main():
    start=time.monotonic();n=4
    # Independent finite support test for sample covariance identity; not graph sampling.
    R=np.array([.1,.7,1.3,2.]);U=np.array([[1.,-.3],[-2.,.1],[.5,.8],[.5,-.6]])
    assert np.max(abs(U.mean(0)))<1e-15
    g=((R-R.mean())[:,None]*U).mean(0)
    v=(((R-R.mean())[:,None]*U-g)**2).sum(1).mean()
    formula=v/n+(R.var()*(U*U).sum(1).mean()+g@g)/(n*(n-1))
    estimates=[]
    for ids in itertools.product(range(4),repeat=n):
        rr=R[list(ids)];uu=U[list(ids)];estimates.append(((rr-rr.mean())[:,None]*uu).sum(0)/(n-1))
    observed=((np.array(estimates)-g)**2).sum(1).mean();assert abs(observed-formula)<1e-12
    m=load('finite_batch',ROOT/'research_branches/terminal_structure_audit_20261109/audit_cpu.py');m.kernel_module=load('kernel_batch',m.KERNEL)
    cfg=json.loads(m.PROTOCOL.read_text());f=m.FiniteDiffusion(cfg)
    moments=json.loads((OUT/'exact_moments.json').read_text());rows=[]
    for seed in cfg['initializations']:
        theta=np.random.default_rng(seed).normal(0,.15,f.D);theta[:3]+=logit(f.pi);r=f.evaluate(theta)
        marg=f.prior.copy();score_second=0.
        for t in range(f.T,0,-1):
            P,DP=r['P'][t],r['DP'][t];s=DP/P[:,:,None]
            score_second+=float(np.einsum('x,xzd,xzd,xz->',marg,s,s,P));marg=marg@P
        cost_var=float(marg@(f.cost-float(marg@f.cost))**2)
        v={x['method']:x['variance_trace'] for x in moments if x['seed']==seed}
        loo=v['population_constant']/n+(cost_var*score_second+float(r['g']@r['g']))/(n*(n-1))
        rows.append(dict(seed=seed,paths=4,LOO_variance_trace=loo,prestate_variance_trace=v['prestate_cost']/4,
            prestate_over_LOO=(v['prestate_cost']/4)/loo,uncentered_variance_trace=v['zero']/4,
            population_constant_oracle_variance_trace=v['population_constant']/4,
            conditional_value_oracle_variance_trace=v['conditional_value_oracle']/4))
    (OUT/'exact_four_path_comparison.json').write_text(json.dumps(rows,indent=2)+'\n')
    (OUT/'analytical_supplement_receipt.json').write_text(json.dumps(dict(seconds=time.monotonic()-start,RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        finite_support_identity_error=abs(formula-observed),finite_support_combinations=len(estimates),GPU_hours=0,model_calls=0,new_graphs=0,parameter_updates=0),indent=2)+'\n')
    print(json.dumps(rows))
if __name__=='__main__':main()
