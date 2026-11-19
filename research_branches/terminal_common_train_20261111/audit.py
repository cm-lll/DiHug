"""No updates: verify saved steps, independent simplex solutions, scalar Bayes terminal laws."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import importlib.util,json,hashlib,time,resource,signal
from pathlib import Path
import numpy as np
from scipy.special import expit
from scipy.optimize import minimize
start=time.time();signal.alarm(120);resource.setrlimit(resource.RLIMIT_AS,(2*1024**3,2*1024**3))
s=importlib.util.spec_from_file_location('experiment',Path(__file__).with_name('run.py'));m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
out=m.OUT;protocol=json.loads((out/'protocol.json').read_text());hist=np.load(out/'histories.npz');D=np.load(out/'distributions.npz');records=json.loads((out/'updates.json').read_text())
f,C,tri,cfg,_=m.r.setup();f=m.a.m.augment(f);maxgap=0.;maxdist=0.
for row in records:
    seed,arm,n=row['seed'],row['arm'],row['update'];w=hist[f'{seed}_{arm}_parameters'];E=hist[f'{seed}_{arm}_EMA'];G=np.array(row['gradient']);delta=np.array(row['delta']);b=G.mean(0)
    np.testing.assert_array_equal(w[n-1]+delta,w[n]);np.testing.assert_array_equal(.99*E[n-1]+.01*w[n],E[n])
    if arm=='common':
        U=G/np.linalg.norm(G,axis=1)[:,None];cert=row['certificate'];a=np.array(cert['weights'])
        opt=minimize(lambda x:.5*np.linalg.norm(x@U)**2,np.ones(5)/5,jac=lambda x:U@(x@U),bounds=[(0,1)]*5,constraints=[{'type':'eq','fun':lambda x:x.sum()-1,'jac':lambda x:np.ones(5)}],method='SLSQP',options=dict(ftol=1e-14,maxiter=1000))
        assert opt.success;gap=abs(np.linalg.norm(opt.x@U)**2-cert['margin']**2);maxgap=max(maxgap,float(gap));assert gap<1e-8
        assert max(G@delta)<0;np.testing.assert_allclose(delta,-.2*np.linalg.norm(b)*(a@U)/np.linalg.norm(a@U),rtol=1e-12,atol=1e-14)
    elif arm=='mean':np.testing.assert_array_equal(delta,-.2*b)
    else:
        target=next(z['norm'] for z in records if z['seed']==seed and z['arm']=='common' and z['update']==n)
        np.testing.assert_allclose(delta,-target*b/np.linalg.norm(b),rtol=1e-12,atol=1e-14)
for seed in protocol['seeds']:
    for arm in protocol['arms']:
        for n in protocol['readouts']:
            for rd,key in [('raw','parameters'),('EMA','EMA')]:
                w=hist[f'{seed}_{arm}_{key}'][n];dist=f.prior.copy()
                for t in range(100,0,-1):
                    p=expit(f.features[t]@w);at,ap=f.alpha[t],f.alpha[t-1];ratio=at/ap;pi=f.pi[f.f];obs=f.B
                    lik1=ratio*obs+(1-ratio)*np.where(obs>0,pi,1-pi);lik0=ratio*(1-obs)+(1-ratio)*np.where(obs>0,pi,1-pi)
                    q=[]
                    for clean in (0,1):
                        prior=ap*clean+(1-ap)*pi;q.append(prior*lik1/(prior*lik1+(1-prior)*lik0))
                    prob=(1-p)*q[0]+p*q[1];P=np.prod(np.where(f.B[None,:,:]>0,prob[:,None,:],1-prob[:,None,:]),axis=-1);dist=dist@P
                err=float(abs(dist-D[f'{seed}_{arm}_{n}_{rd}']).max());maxdist=max(maxdist,err);assert err<1e-12
for p,h in protocol['bindings'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
result=dict(passed=True,saved_updates=len(records),independent_common_QPs=60,terminal_distributions=len(D.files),max_QP_objective_gap=maxgap,max_terminal_error=maxdist,bindings_unchanged=True,seconds=time.time()-start,RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,GPU_hours=0)
(out/'audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
