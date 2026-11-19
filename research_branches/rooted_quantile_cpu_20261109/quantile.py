"""CPU candidate: fixed coordinate/mixed-projection quantile discrepancy.

Existing RootedScore features/scales/masks unchanged. No model integration.
Not full multivariate OT, not a new invention of sliced Wasserstein distance.
"""
import numpy as np


def directions(d, type_id):
    rng=np.random.default_rng(np.random.SeedSequence([2026110913,int(type_id),int(d)]))
    u=rng.standard_normal((d,32))
    return u/np.linalg.norm(u,axis=0,keepdims=True)


class QuantileScore:
    def __init__(self,core):
        self.core=core
        self.cache={}
        for t,ix in core.nodes.items():
            mask=core.masks[t].numpy();d=int(mask.sum())
            if not d:
                self.cache[t]=None;continue
            u=directions(d,t);y=core.targets[t].numpy()
            self.cache[t]=dict(nodes=ix.numpy(),mask=mask,scale=core.scales[t].numpy(),u=u,
                target_axis=np.sort(y,axis=0),target_mixed=np.sort(y@u,axis=0))

    def raw_score(self,raw):
        raw=np.asarray(raw);single=raw.ndim==2
        if single:raw=raw[None]
        axis=np.zeros(len(raw));mixed=np.zeros(len(raw))
        for c in self.cache.values():
            if c is None:continue
            z=np.log1p(raw[:,c['nodes']][:,:,c['mask']])/c['scale']
            axis+=np.square(np.sort(z,axis=1)-c['target_axis']).mean((1,2))
            mixed+=np.square(np.sort(z@c['u'],axis=1)-c['target_mixed']).mean((1,2))
        axis/=len(self.cache);mixed/=len(self.cache)
        total=.5*(axis+mixed)
        if not np.isfinite(total).all():raise FloatingPointError('nonfinite quantile score')
        return (float(total[0]) if single else total),axis,mixed

    def differences(self,inc,indices=None,batch_size=32):
        ix=np.arange(len(inc.bits)) if indices is None else np.asarray(indices,dtype=int)
        old=self.raw_score(inc.raw)[0];out=np.empty(len(ix))
        for start in range(0,len(ix),batch_size):
            ids=ix[start:start+batch_size];raw,_=inc.raw_flips(ids)
            values=self.raw_score(raw)[0]
            out[start:start+len(ids)]=(values-old)*(1-2*inc.bits[ids])
        return out
