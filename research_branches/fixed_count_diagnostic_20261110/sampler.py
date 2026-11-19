"""Exact conditional Bernoulli sampling by log-domain elementary symmetric DP.

P(B=b | sum B=m) proportional to product odds_i**b_i. No clipping,
top-k approximation, rejection, or relaxation of exact zero/one support.
"""
import itertools
import numpy as np

def prepare(p, target):
    p=np.asarray(p,dtype=np.float64)
    if not np.isfinite(p).all() or ((p<0)|(p>1)).any():raise ValueError('invalid probability')
    target=int(target);certain=p==1;ix=np.flatnonzero((p>0)&(p<1))
    k=target-int(certain.sum());n=len(ix)
    if not 0<=k<=n:raise ValueError('conditioning event has zero probability')
    complement=k>n//2
    if complement:k=n-k
    odds=np.log(p[ix])-np.log1p(-p[ix]);odds=-odds if complement else odds
    shift=float(np.median(odds)) if n else 0.;w=odds-shift
    dp=np.full((n+1,k+1),-np.inf);dp[:,0]=0.
    for i in range(1,n+1):
        top=min(i,k)
        dp[i,1:top+1]=np.logaddexp(dp[i-1,1:top+1],w[i-1]+dp[i-1,:top])
    base=np.log(p[ix]).sum() if complement else np.log1p(-p[ix]).sum()
    log_event=float(base+dp[n,k]+k*shift)
    if log_event>1e-9 or not np.isfinite(log_event):raise FloatingPointError('invalid conditional normalizer')
    return dict(p=p,ix=ix,certain=certain,k=k,w=w,dp=dp,complement=complement,log_event=log_event)

def draw(state, uniforms):
    u=np.asarray(uniforms);p=state['p'];ix=state['ix'];dp=state['dp'];w=state['w']
    assert u.shape[1]==len(p) and np.all((u>=0)&(u<1))
    b=np.tile(state['certain'],(len(u),1));left=np.full(len(u),state['k'],dtype=int)
    chosen=np.zeros((len(u),len(ix)),dtype=bool)
    for i in range(len(ix),0,-1):
        take=left==i;active=(left>0)&~take
        if active.any():
            h=left[active];lp=w[i-1]+dp[i-1,h-1]-dp[i,h]
            if np.max(lp)>1e-9:raise FloatingPointError('conditional probability exceeds one')
            take[active]=u[active,ix[i-1]]<np.exp(np.minimum(lp,0))
        chosen[:,i-1]=take;left-=take
    assert not left.any()
    b[:,ix]=~chosen if state['complement'] else chosen
    return b.astype(np.uint8)

def path_mass(state,b):
    z=np.asarray(b)[state['ix']].astype(int)
    if state['complement']:z=1-z
    h=state['k'];mass=1.
    for i in range(len(z),0,-1):
        pr=0. if h==0 else (1. if h==i else np.exp(state['w'][i-1]+state['dp'][i-1,h-1]-state['dp'][i,h]))
        mass*=pr if z[i-1] else 1-pr;h-=z[i-1]
    return mass

def tests():
    maximum=0.;events=0;rng=np.random.default_rng(2026111017)
    for p in [np.array([.03,.17,.36,.51,.79,.98]),np.array([0.,1.,.2,.4,.8,.99]),np.full(6,.5),np.array([1e-15,1-1e-14,.01,.999,.3,.7])]:
        bits=np.array(list(itertools.product([0,1],repeat=6)));mass=np.prod(np.where(bits,p,1-p),axis=1)
        for m in range(7):
            ok=bits.sum(1)==m;Z=mass[ok].sum()
            if Z==0:
                try:prepare(p,m)
                except ValueError:continue
                raise AssertionError('zero event accepted')
            s=prepare(p,m);expected=mass[ok]/Z
            got=np.array([path_mass(s,b) if np.all(b[p==0]==0) and np.all(b[p==1]==1) else 0 for b in bits[ok]])
            err=float(abs(got-expected).max());maximum=max(maximum,err)
            np.testing.assert_allclose(got,expected,atol=2e-12,rtol=2e-12)
            np.testing.assert_allclose(s['log_event'],np.log(Z),atol=2e-12,rtol=2e-12)
            draws=draw(s,rng.random((32,6)));assert np.all(draws.sum(1)==m)
            assert np.all(draws[:,p==0]==0) and np.all(draws[:,p==1]==1)
            events+=1
    # Adding a constant to all log odds cannot change the conditional distribution.
    p=np.array([.1,.2,.3,.5,.7,.8]);q=1/(1+np.exp(-(np.log(p/(1-p))+1.7)))
    a,b=prepare(p,3),prepare(q,3);u=rng.random((128,6))
    np.testing.assert_array_equal(draw(a,u),draw(b,u))
    return dict(passed=True,events=events,exhaustive_states_per_case=64,max_probability_error=maximum,
                support_boundaries=True,complement=True,common_log_odds_shift_invariance=True)
