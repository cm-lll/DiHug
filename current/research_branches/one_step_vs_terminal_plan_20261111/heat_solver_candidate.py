"""CPU-only numerical repair candidate. Not wired into any training worker.

Same heat signature and normalization as frozen typed_heat_audit_20261111.
Use divide-and-conquer eigh with explicit residual checks; retain all original
mass/positivity tolerances. No clipping/re-normalization hides solver errors.
"""
import time
import numpy as np
from scipy.linalg import eigh
TIMES=np.logspace(-1,2,9)

def heat(a,types):
    start=time.perf_counter();n=len(a);levels=np.unique(types);d=a.sum(1)
    inv=np.divide(1,np.sqrt(d),out=np.zeros(n),where=d>0)
    lap=np.diag((d>0).astype(float))-inv[:,None]*a*inv[None,:]
    eig,v=eigh(lap,check_finite=True,driver='evd')
    assert eig.min()>-1e-10 and eig.max()<2+1e-10
    assert np.max(abs(v.T@v-np.eye(n)))<1e-10
    assert np.max(abs(lap@v-v*eig))<1e-10
    h=np.exp(-TIMES[:,None]*eig);ret=(v*v)@h.T
    indicator=np.column_stack([types==t for t in levels]).astype(float)
    projected=v.T@(np.sqrt(d)[:,None]*indicator)
    reach=np.stack([inv[:,None]*(v@(h_s[:,None]*projected)) for h_s in h],axis=1)
    reach[d==0]=indicator[d==0,None,:]
    assert np.max(abs(reach.sum(2)-1))<1e-9
    assert reach.min()>-1e-9 and ret.min()>-1e-9
    feat=np.concatenate([ret/np.sqrt(2*len(TIMES)),reach.reshape(n,-1)/np.sqrt(2*len(TIMES)*len(levels))],1)
    return dict(global_trace=h.mean(1),ret=ret,reach=reach,feat=feat,seconds=time.perf_counter()-start,eigenvalues=eig)
