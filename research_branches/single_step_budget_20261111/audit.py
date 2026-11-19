"""Independent covariance-estimator and RNG checks; no parameter changes."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import importlib.util,json,time,hashlib,resource
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/single_step_budget_20261111';start=time.time()
sp=importlib.util.spec_from_file_location('worker',ROOT/'research_branches/single_step_budget_20261111/run.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
f,C,*_=m.r.setup();f=m.m.augment(f);hist=np.load(ROOT/'results/toy_typed_endpoint_20261110/histories.npz');ev=np.load(OUT/'evidence.npz');rows=json.loads((OUT/'estimates.json').read_text());checks=[]
for seed in m.r.SEEDS:
    for step in (0,100):
        key=f'{seed}_{step}';w=hist[f'{seed}_typed_endpoint_parameters'][step];tx=ev[key+'_inputs'];Y=ev[key+'_outputs']
        for rep in (0,127):
            rng=np.random.default_rng(np.random.SeedSequence([2026111121,seed,step,rep]));ts=rng.integers(1,101,64);u=rng.random(64);xs=np.array([np.searchsorted(np.cumsum(f.qref[int(t)]),v) for t,v in zip(ts,u)]);np.testing.assert_array_equal(tx[rep],np.stack([ts,xs],1))
            scores=[];probs=[]
            for i,(t,x) in enumerate(tx[rep]):
                F=f.features[int(t)][int(x)];p=1/(1+np.exp(-(F@w)));nk=max(K for B,K in m.ALLOC if i<B);hrng=np.random.default_rng(np.random.SeedSequence([2026111122,seed,step,rep,i]));bits=(hrng.random((nk,6))<p).astype(int)
                np.testing.assert_array_equal(bits@ (2**np.arange(5,-1,-1)),Y[rep,i,:nk]);scores.append((bits-p)@F);probs.append(p)
            for B,K in m.ALLOC:
                g=np.zeros((5,10))
                for i in range(B):
                    c=C[Y[rep,i,:K]];s=scores[i][:K];g+=(c-c.mean(0)).T@(s-s.mean(0))/(K-1)/B
                row=next(a for a in rows if a['seed']==seed and a['update']==step and a['repeat']==rep and a['B']==B and a['K']==K);error=float(abs(g-np.array(row['gradient'])).max());assert error<1e-12
                checks.append(error)
        # Independent exact expectation using one input at a time, scalar probabilities.
        G=np.zeros((5,10));loss=np.zeros(5)
        for t in range(1,101):
            for x in range(64):
                F=f.features[t][x];p=1/(1+np.exp(-F@w));q=np.prod(np.where(f.B,p,1-p),axis=1);s=(f.B-p)@F;weight=f.qref[t][x]/100
                G+=weight*(q[:,None]*C).T@s;loss+=weight*q@C
        truth=next(a for a in json.loads((OUT/'truth.json').read_text()) if a['seed']==seed and a['update']==step)
        np.testing.assert_allclose(G,truth['gradient'],rtol=1e-11,atol=1e-12);np.testing.assert_allclose(loss,truth['cost'],rtol=1e-11,atol=1e-12)
for name,h in json.loads((OUT/'protocol.json').read_text())['bindings'].items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==h
result=dict(passed=True,covariance_checks=len(checks),max_error=max(checks),exact_population_checks=6,independent_input_rng_checks=12,independent_output_rng_checks=768,seconds=time.time()-start,RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,GPU_hours=0)
(OUT/'audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
