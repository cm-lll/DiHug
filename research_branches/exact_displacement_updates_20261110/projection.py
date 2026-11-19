"""Euclidean projection of an actual AdamW displacement; no ridge or line search."""
import itertools
import numpy as np

def project(H,b):
    H=np.asarray(H,dtype=np.float64);b=np.asarray(b,dtype=np.float64)
    assert np.isfinite(H).all() and np.isfinite(b).all()
    scale=max(float(abs(H).max()),1e-30);hs=H/scale;bs=b/scale
    tol=1e-9*max(1.,float(abs(bs).max()));candidates=[]
    for n in range(len(b)+1):
        for active in itertools.combinations(range(len(b)),n):
            lam=np.zeros(len(b))
            if active:
                ix=list(active);v=np.linalg.lstsq(hs[np.ix_(ix,ix)],bs[ix],rcond=1e-12)[0]
                if min(v)<-tol:continue
                lam[ix]=np.maximum(v,0)
            slack=hs@lam-bs
            res=max(max(0.,float(-slack.min())),float(abs(slack[lam>0]).max()) if np.any(lam>0) else 0.)
            if res<=tol:candidates.append((float(lam@H@lam),float(lam@lam),lam,res))
    if not candidates:raise RuntimeError('No numerically certified displacement projection; do not replace by zero')
    corr,_,lam,res=min(candidates,key=lambda x:(x[0],x[1]))
    return lam,dict(residual=res,tolerance=tol,active=int((lam>0).sum()),correction_squared=corr,ideal_slopes=(b-H@lam).tolist(),rank=int(np.linalg.matrix_rank(hs,tol=1e-12)))
