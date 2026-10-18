"""Nonnegative absorbing-count polynomial arithmetic. No division or sampling."""
import numpy as np


def combine(a,b):
    """Convolve states 0,1,2,>=3. Broadcasting, only nonnegative operations."""
    out=np.empty(np.broadcast_shapes(a.shape,b.shape),dtype=np.float64)
    out[...,0]=a[...,0]*b[...,0]
    out[...,1]=a[...,0]*b[...,1]+a[...,1]*b[...,0]
    out[...,2]=a[...,0]*b[...,2]+a[...,1]*b[...,1]+a[...,2]*b[...,0]
    out[...,3]=a[...,3]*b.sum(-1)+b[...,3]*(a[...,0]+a[...,1]+a[...,2])+a[...,1]*b[...,2]+a[...,2]*(b[...,1]+b[...,2])
    return out


def binomial_tail_states(n,p):
    """All four masses of Binomial(n,p); exponentiation, no 1-CDF cancellation."""
    n=np.asarray(n,dtype=np.int64)
    if (n<0).any() or not 0<=p<=1:raise ValueError('invalid binomial')
    result=np.zeros(n.shape+(4,));result[...,0]=1
    base=np.zeros_like(result);base[...,0]=1-p;base[...,1]=p
    power=n.copy()
    while power.any():
        take=(power&1).astype(bool)
        result[take]=combine(result[take],base[take])
        power//=2
        if power.any():base=combine(base,base)
    return result


def segmented_prefix(y,nodes,check=None):
    """Inclusive degree<3/tail scan, O(S log(max degree)), O(S) memory.

    nodes must form contiguous segments; each doubling uses the previous array.
    Missing slots are never represented by fake occupied/query edges.
    """
    y=np.asarray(y,dtype=float);nodes=np.asarray(nodes,dtype=np.int64)
    if y.ndim!=1 or nodes.shape!=y.shape or ((y<0)|(y>1)).any() or not np.isfinite(y).all():
        raise ValueError('invalid thinned probabilities')
    if len(nodes)>1 and not ((np.diff(nodes)>=0).all() or (np.diff(nodes)<=0).all()):
        raise ValueError('segmented scan requires monotonic node groups')
    out=np.zeros((len(y),4));out[:,0]=1-y;out[:,1]=y
    distance=1
    while distance<len(y):
        if check:check()
        same=nodes[distance:]==nodes[:-distance]
        if not same.any():break
        ix=np.flatnonzero(same)+distance
        merged=combine(out[ix-distance],out[ix])
        out[ix]=merged
        distance*=2
    return out


def empirical_tail(q,ids,nodes,n,p,gradient=False,check=None):
    """E[mean h(D)] and per-incidence derivative, h(D)=Pr[Bin(D,p)>=3].

    Node degree thinning => independent Bernoulli(p*q). A derivative is
    p*Pr[the OTHER incident slots contain exactly two retained edges].
    Prefix/suffix coefficients compute it without quotient or subtraction.
    """
    if not 0<p<=.5:raise ValueError('frozen thinning p outside safe design range')
    if not len(ids):return 0.,np.zeros(0) if gradient else None,dict(mass_error=0.,underflow=0)
    forward=segmented_prefix(p*q[ids],nodes,check)
    end=np.r_[nodes[1:]!=nodes[:-1],True]
    masses=forward[end]
    mass_error=float(np.max(abs(masses.sum(-1)-1),initial=0))
    if not np.isfinite(masses).all() or mass_error>5e-11 or (masses<0).any():
        raise FloatingPointError('tail DP mass error exceeds frozen bound')
    mean=float(masses[:,3].sum()/n)
    underflow=int((masses[:,0]==0).sum()) # p<=.5: exact P0 must be positive
    derivative=None
    if gradient:
        backward=segmented_prefix((p*q[ids])[::-1],nodes[::-1],check)[::-1]
        left=np.zeros_like(forward);left[:,0]=1
        right=np.zeros_like(backward);right[:,0]=1
        ix=np.flatnonzero(nodes[1:]==nodes[:-1])+1
        left[ix]=forward[ix-1];right[ix-1]=backward[ix]
        derivative=p*(left[:,0]*right[:,2]+left[:,1]*right[:,1]+left[:,2]*right[:,0])
    return mean,derivative,dict(mass_error=mass_error,underflow=underflow)
