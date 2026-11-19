"""Use one positive scale on the existing combined gradient, preserve decay.

u = delta_Adam + lr*wd*theta is Adam's adaptive proposal excluding decay.
d = -||u|| g/||g|| - lr*wd*theta.
Moments select magnitude only, not the proposed direction. This is NOT AdamW.
Zero g commits only decay; no hidden momentum fallback or online acceptance.
"""
import numpy as np

def direction_step(g,native,theta,opt):
    g,native,theta=map(lambda x:np.asarray(x,dtype=np.float64),(g,native,theta))
    if g.shape!=native.shape or g.shape!=theta.shape or not all(np.isfinite(x).all() for x in (g,native,theta)):raise ValueError('invalid scalar-update inputs')
    decay=-opt['lr']*opt['weight_decay']*theta;adaptive=native-decay
    gn=float(np.linalg.norm(g));an=float(np.linalg.norm(adaptive));nn=float(np.linalg.norm(native))
    factor=an/gn if gn else 0.;d=-factor*g+decay
    if not np.isfinite(d).all() or not np.isfinite(factor):raise FloatingPointError('nonfinite scalar proposal')
    dn=float(np.linalg.norm(d))
    return d,dict(method='shared_positive_scalar',effective_scalar=factor,gradient_norm=gn,
        adaptive_norm=an,decay_norm=float(np.linalg.norm(decay)),zero_gradient=gn==0,
        norm_ratio=dn/nn if nn else 0.,cosine=float(d@native/(dn*nn)) if dn and nn else 0.,
        adaptive_direction_cosine=float((-g)@(d-decay)/(gn*np.linalg.norm(d-decay))) if gn and np.linalg.norm(d-decay) else None)

def tests():
    rng=np.random.default_rng(2026111043);maximum=0.;cases=0
    for wd in (0.,1e-12,.1):
        for i in range(32):
            g=rng.normal(size=9);theta=rng.normal(size=9);native=rng.normal(size=9)*1e-3;opt=dict(lr=.0002,weight_decay=wd)
            if i==0:g[:]=0
            if i==1:native=-opt['lr']*wd*theta
            got,info=direction_step(g,native,theta,opt);decay=-opt['lr']*wd*theta;adaptive=native-decay
            expected=decay.copy() if np.linalg.norm(g)==0 else -g/np.linalg.norm(g)*np.linalg.norm(adaptive)+decay
            err=float(abs(got-expected).max());maximum=max(maximum,err);assert err<1e-14
            scaled,_=direction_step(7*g,native,theta,opt);np.testing.assert_allclose(got,scaled,atol=1e-14,rtol=0)
            if np.linalg.norm(g):assert abs(np.linalg.norm(got-decay)-np.linalg.norm(adaptive))<1e-14
            cases+=1
    return dict(passed=True,cases=cases,max_formula_error=maximum,checks=['decay once','zero gradient','zero adaptive proposal','positive gradient rescaling invariance','norm and direction'])
