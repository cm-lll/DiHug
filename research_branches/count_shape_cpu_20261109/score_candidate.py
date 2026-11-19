"""One CPU development candidate: count means plus centered joint profiles.
Not a production loss. Reference-only scales, no fitting or search.
"""
import numpy as np
from quantile import directions

class CountShapeScore:
    def __init__(self,core):
        self.core=core;self.cache={}
        ref=core.features(core.reference).numpy()
        for t,ids in core.nodes.items():
            ids=ids.numpy();mask=core.masks[t].numpy();cols=np.flatnonzero(mask);d=len(cols)
            if not d:continue
            y=ref[ids][:,mask];mu=y.mean(0)
            groups=[cols<core.r,(cols>=core.r)&(cols<core.base_width),cols>=core.base_width]
            groups=[g for g in groups if g.any()]
            scale2=np.maximum(1.,mu)
            for g in groups:scale2[g]*=len(groups)*g.sum()/d
            scale=np.sqrt(scale2);u=directions(d,t);z=y/scale;m=z.mean(0);c=z-m
            self.cache[t]=dict(ids=ids,mask=mask,cols=cols,scale=scale,u=u,mu=m,
                target_axis=np.sort(c,axis=0),target_mixed=np.sort(c@u,axis=0),reference_mean=mu)
    def terms(self,raw):
        raw=np.asarray(raw);raw=raw[None] if raw.ndim==2 else raw
        mass=np.zeros(len(raw));shape=np.zeros(len(raw));degree_mass=np.zeros(len(raw))
        for t,c in self.cache.items():
            z=raw[:,c['ids']][:,:,c['mask']]/c['scale'];m=z.mean(1);center=z-m[:,None,:]
            err=(m-c['mu'])**2
            mass+=err.mean(1)
            degree_mass+=err[:,c['cols']<self.core.r].sum(1)/len(c['cols'])
            shape+=.5*np.square(np.sort(center,axis=1)-c['target_axis']).mean((1,2))
            shape+=.5*np.square(np.sort(center@c['u'],axis=1)-c['target_mixed']).mean((1,2))
        n=len(self.core.nodes)
        return dict(total=(mass+shape)/n,mass=mass/n,shape=shape/n,degree_mass=degree_mass/n)
    def raw_score(self,raw):return self.terms(raw)['total']

    def relation_mass(self,raw):
        """Same degree-mean score, counted as ONE coefficient per relation.
        Supports only a unique unordered endpoint type pair for each relation.
        """
        raw=np.asarray(raw);raw=raw[None] if raw.ndim==2 else raw
        edgecounts=raw[:,:,:self.core.r].sum(1)/2
        refcounts=np.bincount(self.core.families.numpy(),weights=self.core.reference.numpy(),minlength=self.core.r)
        weights=np.zeros(self.core.r)
        for r in range(self.core.r):
            slots=np.flatnonzero(self.core.families.numpy()==r)
            if not len(slots):continue
            pairs={tuple(sorted(self.core.types[self.core.query[:,i].numpy()])) for i in slots}
            if len(pairs)!=1:raise ValueError('relation has multiple endpoint type pairs')
            pair=next(iter(pairs))
            for t in set(pair):
                c=self.cache[t];k=np.flatnonzero(c['cols']==r)
                if not len(k):continue
                incidence=pair.count(t)
                weights[r]+=incidence**2/(len(self.core.nodes)*len(c['cols'])*len(c['ids'])**2*c['scale'][k[0]]**2)
        return ((edgecounts-refcounts)**2*weights).sum(1),weights
