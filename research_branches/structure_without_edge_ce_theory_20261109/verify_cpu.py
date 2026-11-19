"""Independent Bayes/production numpy-kernel checks. No torch/model imported."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import json,time,hashlib,resource,importlib.util
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/structure_without_edge_ce_theory_20261109';START=time.monotonic()
KPATH=ROOT/'research_branches/cn_multiseed_20261104/research_branches/typed_structure_early_cn_expansion_review/experiments/typed_structure_early_cn_expansion_v1/package/hsgd/kernel.py'
ARCHIVE=ROOT/'experiments/seed_coverage_repair_v1/package/hsgd/kernel.py'
assert KPATH.read_bytes()==ARCHIVE.read_bytes()
spec=importlib.util.spec_from_file_location('numpy_kernel',KPATH);K=importlib.util.module_from_spec(spec);spec.loader.exec_module(K)

def dump(name,obj):(OUT/(name+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def product_masses(bits,p):return np.prod(np.where(bits,p,1-p),axis=1)

def main():
    for p,h in json.loads((OUT/'binding.json').read_text()).items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
    with np.load(OUT/'finite_universe.npz') as z:bits=z['bits'];f=z['families'];ref=z['reference'];E=z['energy'];orbit=z['orbit'];priors={k:z[k] for k in ['reference_point','type_orbit','illustrative_gibbs']}
    pif=np.array([.25,.4,.35]);pi=pif[f];obs=[int(ref@(1<<np.arange(10))),int(next(x for x in orbit if x!=int(ref@(1<<np.arange(10))))) ]
    kernel=K.MarginalKernel(pif,np.array([1.,.95,.5,0.]))
    checks=[]
    # Independently build q_s over whole graphs; compare exact Bayes reverse marginals
    # with production per-edge x0-posterior mixture. Exact joint sampling is NOT claimed.
    for s,retention in [(0,1.),(1,.95)]:
        matrix=np.ones((1024,1024))
        for e in range(10):
            trans=kernel.matrix(retention,np.array([f[e]]))[0]
            matrix*=trans[bits[:,e,None],bits[None,:,e]]
        assert np.max(abs(matrix.sum(1)-1))<1e-12
        for name,prior in priors.items():
            marginal_s=prior@matrix
            for x in obs:
                qt=kernel.matrix(.5,f)
                likelihood0=np.prod(np.stack([qt[e,bits[:,e],bits[x,e]] for e in range(10)]),axis=0)
                p0=prior*likelihood0;p0/=p0.sum();m=p0@bits
                step=kernel.matrix(.5/retention,f)
                ls=np.prod(np.stack([step[e,bits[:,e],bits[x,e]] for e in range(10)]),axis=0)
                truth=marginal_s*ls;truth/=truth.sum()
                pred=kernel.reverse(m,bits[x],f,s,2);error=float(abs(pred-truth@bits).max());assert error<1e-12
                independent=product_masses(bits,pred)
                checks.append(dict(prior=name,observation_code=x,previous_retention=retention,observed_retention=.5,
                    max_marginal_error=error,independent_reverse_joint_TV=float(abs(truth-independent).sum()/2),
                    interpretation='Exact per-slot reverse marginal; factorized graph transition may differ from exact joint. This is not a production bug or proof about T1000 quality.'))
    fixed=K.MarginalKernel([0.,1.,.4],[1.,.5,0.]);p=fixed.reverse(np.array([0.,1.,.7]),np.array([0,1,1]),np.arange(3),0,1)
    assert np.array_equal(p,np.array([0.,1.,.7]))
    rejected=False
    try:fixed.reverse(np.array([.1,1.,.7]),np.array([0,1,1]),np.arange(3),0,1)
    except ValueError:rejected=True
    assert rejected
    dump('reverse_kernel_checks',checks)
    # Direct cardinality-risk exhaustive proof with exact rational values represented by quarters.
    little=np.array([[0,0],[1,0],[0,1],[1,1]]);cost=(little.sum(1)-1)**2
    indep=np.ones(4)/4;one=np.array([0.,.5,.5,0.]);det=np.array([0.,1.,0.,0.])
    assert indep@cost==.5 and one@cost==det@cost==0
    assert np.array_equal(indep@little,one@little)
    assert np.array_equal(indep@(cost[:,None]*(little-.5)),np.zeros(2))
    dump('verification',dict(status='passed',reverse_cases=len(checks),max_reverse_marginal_error=max(r['max_marginal_error'] for r in checks),
        fixed_support_preserved=True,invalid_fixed_prediction_rejected=True,two_slot_exact_check=True,
        kernel_source=str(KPATH),kernel_sha256=hashlib.sha256(KPATH.read_bytes()).hexdigest(),matches_archival_source=True,
        runtime_source=str(ROOT/'research_branches/joint_output_projected_review_20261108/runtime/science.py'),
        seconds=time.monotonic()-START,RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        model_calls=0,GPU_hours=0,updates=0,new_sampled_graphs=0,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    print((OUT/'verification.json').read_text())
if __name__=='__main__':main()
