"""Reference-fixed, mass-preserving soft joint-distribution CPU score.

No model integration, no fitted predictor, no new random projections.
"""
import numpy as np
from conditional_score import ConditionalScore, cdf_distance, membership


def joint_discrepancy(x,y,q,qref,prior,u):
    """Group occupation plus probability measures with a zero-valued dummy atom.

    P_a = sum_i q_a(D_i)/n delta_Hi + (1-mean q_a) delta_0.
    Inverse REFERENCE mass gives the previous conditional coefficient when
    candidate and reference group masses coincide. No candidate group division.
    """
    p=q.mean(0)
    wx=np.concatenate([q/len(q),(1-p)[None]],axis=0)
    wy=np.concatenate([qref/len(qref),(1-prior)[None]],axis=0)
    assert wx.min()>-1e-12 and wy.min()>-1e-12
    assert np.max(abs(wx.sum(0)-1))<1e-12 and np.max(abs(wy.sum(0)-1))<1e-12
    x=np.concatenate([x,np.zeros((1,x.shape[1]))],axis=0)
    y=np.concatenate([y,np.zeros((1,y.shape[1]))],axis=0)
    shape=cdf_distance(x,y,wx,wy,1/prior,u)
    mass=float(np.sum((p-prior)**2/prior))
    return shape+mass,shape,mass


class FixedScore(ConditionalScore):
    def __init__(self,core):
        self.fixed_mu={}
        super().__init__(core)

    def local(self,raw,ids):
        c=self.core;key=tuple(ids)
        d=raw[ids,:c.r]
        p=raw[ids,c.base_width:].reshape(len(ids),c.r,c.r)
        if key not in self.fixed_mu:
            # First call is exclusively the reference during parent initialization.
            den=d.sum(0)
            self.fixed_mu[key]=np.divide(p.sum(0),den[:,None],out=np.zeros((c.r,c.r)),where=den[:,None]>0)
        residual=p-d[:,:,None]*self.fixed_mu[key][None]
        h=np.concatenate([raw[ids,c.r:c.base_width],residual.reshape(len(ids),-1)],axis=1)
        return d,h

    def terms(self,raw):
        raw=np.asarray(raw)
        if raw.ndim==2:raw=raw[None]
        degree=np.zeros(len(raw));joint=np.zeros(len(raw));mass=np.zeros(len(raw));shape=np.zeros(len(raw))
        for i,a in enumerate(raw):
            for c in self.cache.values():
                d,h=self.local(a,c['ids'])
                d=d[:,c['dc']]/c['ds'];h=h[:,c['hc']]/c['hs']
                q=membership(d,c['anchors'],c['h2'])
                qref=c['weights']*(len(d)*c['prior'])[None]
                uniform=np.ones((len(d),1))/len(d)
                degree[i]+=cdf_distance(d,c['d'],uniform,uniform,np.ones(1),c['ud'])
                j,s,m=joint_discrepancy(h,c['h'],q,qref,c['prior'],c['uh'])
                joint[i]+=j;shape[i]+=s;mass[i]+=m
        n=len(self.cache)
        degree/=n;joint/=n;shape/=n;mass/=n
        total=.5*(degree+joint)
        if not np.isfinite(total).all() or (total<-1e-12).any():raise FloatingPointError('invalid fixed score')
        return dict(total=total,degree=degree,joint=joint,occupation=mass,shape=shape)

    def configuration(self):
        out=super().configuration()
        for t,c in self.cache.items():out[str(t)]['fixed_path_coefficients']=self.fixed_mu[tuple(c['ids'])].tolist()
        return out
