"""Project a COMPLETE proposed displacement onto current gradient halfspaces.

min_d .5 ||d-delta||^2 subject to G d <= 0.
The dual is nonnegative least squares: min_lambda>=0 ||A.T lambda-u||^2,
with normalized rows A and normalized proposal u. No ridge or loss threshold.
Exact zero solution is permitted; failed solve is an exception, never a zero step.
"""
import itertools
import numpy as np

def project_displacement(G,delta):
    G=np.asarray(G,dtype=np.float64);delta=np.asarray(delta,dtype=np.float64)
    if G.ndim!=2 or G.shape[1]!=delta.size or not np.isfinite(G).all() or not np.isfinite(delta).all():raise ValueError('invalid inputs')
    dn=float(np.linalg.norm(delta));gn=np.linalg.norm(G,axis=1);nonzero=gn>0
    if dn==0 or not nonzero.any():
        return delta.copy(),dict(status='zero_proposal' if dn==0 else 'no_nonzero_constraints',residual=0.,rank=0,dual=[0.]*len(G),norm_ratio=1. if dn else 0.)
    A=G[nonzero]/gn[nonzero,None];u=delta/dn;H=A@A.T;b=A@u;n=len(A);candidates=[]
    for size in range(n+1):
        for subset in itertools.combinations(range(n),size):
            lam=np.zeros(n)
            if subset:
                ix=list(subset);sol=np.linalg.lstsq(H[np.ix_(ix,ix)],b[ix],rcond=1e-12)[0]
                if sol.min() < -1e-10:continue
                lam[ix]=np.maximum(sol,0)
            v=u-A.T@lam;slack=A@v
            residual=max(float(max(0,slack.max())),float(abs(lam*slack).max()))
            if residual>1e-9:continue
            candidates.append((float(np.sum((v-u)**2)),float(lam@lam),v,lam,residual))
    if not candidates:raise RuntimeError('no verified displacement projection')
    _,_,v,lam,residual=min(candidates,key=lambda x:(x[0],x[1]));d=dn*v
    assert np.linalg.norm(d)<=dn*(1+1e-9)
    dual=np.zeros(len(G));dual[nonzero]=lam*dn/gn[nonzero]
    return d,dict(status='verified',residual=residual,rank=int(np.linalg.matrix_rank(H,tol=1e-12)),
        dual=dual.tolist(),active=(np.flatnonzero(abs(A@v)<=1e-8)).tolist(),norm_ratio=float(np.linalg.norm(d)/dn),
        cosine=float(d@delta/(max(np.linalg.norm(d),1e-300)*dn)),
        normalized_tolerance=1e-9,rcond=1e-12)

def tests():
    from scipy.optimize import nnls
    rng=np.random.default_rng(2026111042);cases=[]
    cases.extend([(np.eye(2),np.array([1.,2.])),(np.array([[1.,0],[1.,0],[0,0]]),np.array([1.,2.])),
        (np.array([[1.,0],[-1.,0]]),np.array([1.,2.])),(np.zeros((5,9)),rng.normal(size=9)),(rng.normal(size=(5,9)),np.zeros(9))])
    for i in range(128):
        G=rng.normal(size=(5,9));d=rng.normal(size=9)
        if i%4==0:G[1]=G[0]
        if i%4==1:G[1]=-G[0]
        if i%4==2:G[1]=G[0]+1e-9*rng.normal(size=9)
        if i%4==3:G[2]=0
        cases.append((G,d))
    maximum=0.;rr=[]
    for i,(G,d) in enumerate(cases):
        got,info=project_displacement(G,d);norms=np.linalg.norm(G,axis=1);valid=norms>0
        if not valid.any() or np.linalg.norm(d)==0:expected=d.copy()
        else:
            A=G[valid]/norms[valid,None];u=d/np.linalg.norm(d);lam,_=nnls(A.T,u,maxiter=1000);expected=np.linalg.norm(d)*(u-A.T@lam)
        error=float(np.linalg.norm(got-expected));maximum=max(maximum,error)
        assert error<1e-7 and (G@got).max(initial=0)<1e-7
        rr.append(dict(case=i,error=error,status=info['status']))
    return dict(passed=True,cases=len(cases),max_independent_nnls_error=maximum,details=rr)
