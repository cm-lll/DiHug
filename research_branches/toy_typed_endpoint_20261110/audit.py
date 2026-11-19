"""Independent scalar/enumeration recomputation of new feature gradients and reverse law."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import importlib.util,json,hashlib,time,resource
from pathlib import Path
import numpy as np
from scipy.special import expit
P=Path(__file__).with_name('run.py');s=importlib.util.spec_from_file_location('experiment',P);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
start=time.time();f,C,tri,cfg,_=m.r.setup();out=m.OUT
protocol=json.loads((out/'protocol.json').read_text());hist=np.load(out/'histories.npz');saved=np.load(out/'distributions.npz')
bank=np.load(m.ROOT/'results/full100_direction_long_20261110/input_banks.npz')
# Build the added scalar from direct neighbor counts, not augment().
extra=np.zeros((64,6))
for x,A in enumerate(f.adj):
    for e,(i,j) in enumerate(f.edges):
        if f.types[i]!=f.types[j]:
            lo,hi=sorted((i,j),key=lambda v:f.types[v]);extra[x,e]=(np.count_nonzero(A[lo])-np.count_nonzero(A[hi]))/3
checked=0;max_update=0.;max_terminal=0.
for seed in m.r.SEEDS:
    for arm in protocol['arms']:
        F={t:np.concatenate([v,extra[...,None]],axis=-1) if arm=='typed_endpoint' else v for t,v in f.features.items()}
        W=hist[f'{seed}_{arm}_parameters'];E=hist[f'{seed}_{arm}_EMA']
        for n in range(100):
            grads=[]
            for t,x in zip(bank[f'{seed}_t'][n],bank[f'{seed}_x'][n]):
                feat=F[int(t)][int(x)];p=expit(feat@W[n]);q=np.prod(np.where(f.B>0,p,1-p),axis=1)
                grads.append((C.T@(q[:,None]*(f.B-p)))@feat)
            G=np.mean(grads,axis=0);err=float(abs(W[n]-.2*G.mean(0)-W[n+1]).max());max_update=max(max_update,err);assert err<1e-12
            np.testing.assert_allclose(E[n+1],.99*E[n]+.01*W[n+1],atol=1e-14,rtol=1e-14);checked+=1
        for n in (0,20,50,100):
            for rd,ww in [('raw',W[n]),('EMA',E[n])]:
                dist=f.prior.copy()
                for t in range(100,0,-1):
                    p=expit(F[t]@ww)
                    # Independent Bayes component probabilities, each clean hypothesis normalized.
                    at,ap=f.alpha[t],f.alpha[t-1];ratio=at/ap;pi=f.pi[f.f]
                    obs=f.B;lik1=ratio*obs+(1-ratio)*np.where(obs>0,pi,1-pi)
                    lik0=ratio*(1-obs)+(1-ratio)*np.where(obs>0,pi,1-pi)
                    q=[]
                    for clean in (0,1):
                        prior=ap*clean+(1-ap)*pi;q.append(prior*lik1/(prior*lik1+(1-prior)*lik0))
                    prob=(1-p)*q[0]+p*q[1]
                    M=np.prod(np.where(f.B[None,:,:]>0,prob[:,None,:],1-prob[:,None,:]),axis=-1)
                    dist=dist@M
                err=float(abs(dist-saved[f'{seed}_{arm}_{n}_{rd}']).max());max_terminal=max(max_terminal,err);assert err<1e-12
for path,h in protocol['bindings'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==h
result=dict(passed=True,updates=checked,max_update_error=max_update,max_terminal_error=max_terminal,bindings_unchanged=True,seconds=time.time()-start,RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,GPU_hours=0)
(out/'audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
