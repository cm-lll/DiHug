"""Entropy-regularized transportation. NOT exact capacity assignment marginals."""
import time
import numpy as np
from scipy.special import logsumexp
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import maximum_flow

TOL=1e-8
MAX_ITER=1000
MAX_ELEMENTS=2000000
MAX_WORKSPACE=256*1024**2

def support_feasible(support,counts):
    n,k=support.shape
    if support.all():return True
    if (~support.any(1)).any() or (~support.any(0)).any():return False
    i,j=np.nonzero(support);s=n+k;t=s+1
    rows=np.r_[np.full(n,s),i,n+np.arange(k)]
    cols=np.r_[np.arange(n),n+j,np.full(k,t)]
    val=np.r_[np.ones(n,dtype=np.int64),np.ones(len(i),dtype=np.int64),counts]
    graph=coo_matrix((val,(rows,cols)),shape=(t+1,t+1),dtype=np.int64).tocsr()
    return int(maximum_flow(graph,s,t).flow_value)==n

def scale(log_likelihood,counts,check=lambda:None):
    """Fixed float64, unit entropy; preserve -inf support and report failure.

    max <P,L> - sum P log P, P1=1, P^T1=counts. No posterior claim.
    Returned finite log probabilities distinguish representational underflow
    from mathematical zero. Such underflow is NOT accepted as convergence.
    """
    start=time.perf_counter();ll=np.asarray(log_likelihood,dtype=np.float64)
    c=np.asarray(counts)
    if ll.ndim!=2 or c.ndim!=1 or ll.shape[1]!=len(c) or not ll.shape[0] or not len(c):raise ValueError('nonempty matrix/count shape')
    if not np.issubdtype(c.dtype,np.integer) or (c<0).any() or c.sum()!=ll.shape[0]:raise ValueError('integer capacities sum to n')
    if np.isnan(ll).any() or np.isposinf(ll).any():raise ValueError('NaN/+inf likelihood')
    if ll.size>MAX_ELEMENTS or ll.size*128>MAX_WORKSPACE:raise MemoryError('fixed resource gate')
    keep=c>0;L=ll[:,keep].copy();cc=c[keep];support=np.isfinite(L)
    check();feasible=support_feasible(support,cc);flow_seconds=time.perf_counter()-start
    receipt=dict(status='support_infeasible' if not feasible else 'not_converged',iterations=0,
        support_density=float(support.mean()),positive_roles=int(keep.sum()),zero_frequency_roles=int((~keep).sum()),
        support_feasible=feasible,flow_seconds=flow_seconds,underflow_count=0)
    if not feasible:
        receipt['seconds']=time.perf_counter()-start;return None,receipt
    # These gauge changes add constants to a constrained objective only.
    with np.errstate(over='ignore',invalid='ignore'):
        L-=L.max(1,keepdims=True);L-=L.max(0,keepdims=True)
    if not np.isfinite(L[support]).all():
        receipt.update(status='numerical_range_exceeded',seconds=time.perf_counter()-start)
        return None,receipt
    v=np.zeros(len(cc));logc=np.log(cc);P=None
    for it in range(1,MAX_ITER+1):
        if it%16==1:check()
        u=-logsumexp(L+v,axis=1)
        v=logc-logsumexp(L+u[:,None],axis=0)
        lp=L+u[:,None]+v
        with np.errstate(under='ignore'):P=np.exp(lp)
        if not np.isfinite(P).all() or not np.isfinite(u).all() or not np.isfinite(v).all():
            receipt['status']='nonfinite';break
        row=float(abs(P.sum(1)-1).max());col=float((abs(P.sum(0)-cc)/np.maximum(1,cc)).max())
        receipt.update(iterations=it,row_residual=row,column_residual=col)
        if max(row,col)<=TOL:
            receipt['status']='converged';break
    under=support&(P==0)
    receipt['underflow_count']=int(under.sum())
    if under.any():receipt['status']='numerical_underflow'
    out=np.zeros_like(ll);out[:,keep]=P
    assert np.all(out[:,keep][~support]==0) and np.all(out[:,~keep]==0)
    receipt.update(seconds=time.perf_counter()-start,
        min_finite_log_probability=float(lp[support].min()),
        centered_objective=float(np.sum(P[support]*L[support])-np.sum(P[support]*lp[support])),
        objective_gauge='row/column constants removed; same constrained optimizer')
    return out,receipt

class ReadoutFailure(RuntimeError):pass

def required(log_likelihood,counts,check=lambda:None):
    p,r=scale(log_likelihood,counts,check)
    if r['status']!='converged':raise ReadoutFailure(str(r))
    return p,r
