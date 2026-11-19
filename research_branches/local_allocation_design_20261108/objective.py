"""CPU prototype: supervised within-neighborhood ranking, not distribution matching.

No network, optimizer, graph generation or production integration. Full R1 support
only. The factor 1/2 gives a global free-logit gradient L1 upper bound of one.
"""
import numpy as np
import torch

class LocalAllocationRank:
    def __init__(self,types,schema,prior,query,families,labels):
        types=np.asarray(types,dtype=int);q=np.asarray(query,dtype=int)
        f=np.asarray(families,dtype=int);y=np.asarray(labels,dtype=int);prior=np.asarray(prior)
        n=len(types);schema=tuple(map(tuple,schema))
        assert q.shape==(2,len(y)) and len(f)==len(y)
        assert np.isin(y,[0,1]).all() and ((q>=0)&(q<n)).all() and (q[0]!=q[1]).all()
        keys=np.minimum(*q)*n+np.maximum(*q);assert len(set(keys))==len(keys)
        for j,(a,b) in enumerate(q.T):assert tuple(sorted((int(types[a]),int(types[b]))))==schema[f[j]]
        active=(prior[f]>0)&(prior[f]<1)
        legal={}
        for a in range(n):
            for b in range(a+1,n):
                pair=tuple(sorted((int(types[a]),int(types[b]))))
                if pair in schema:
                    r=schema.index(pair)
                    if 0<prior[r]<1:legal[a*n+b]=r
        assert {int(k):int(r) for k,r in zip(keys[active],f[active])}==legal
        self.groups=[];self.excluded=[];self.types=types
        for node in range(n):
            for r in range(len(schema)):
                ix=np.flatnonzero(active&(f==r)&((q[0]==node)|(q[1]==node)))
                if not len(ix):continue
                pos=ix[y[ix]==1];neg=ix[y[ix]==0]
                if not len(pos) or not len(neg):
                    self.excluded.append(dict(node=node,type=int(types[node]),relation=r,positive=len(pos),negative=len(neg)))
                    continue
                self.groups.append(dict(node=node,type=int(types[node]),relation=r,pos=pos,neg=neg))
        type_ids=sorted(set(types.tolist()))
        counts={t:sum(g['type']==t for g in self.groups) for t in type_ids}
        for g in self.groups:g['weight']=1/(len(type_ids)*counts[g['type']])
        self.active=active;self.labels=y;self.families=f;self.query=q

    def __call__(self,z):
        assert z.ndim==1 and z.numel()==len(self.labels) and bool(torch.isfinite(z).all())
        values=[];loss=z.sum()*0
        for g in self.groups:
            a=z[torch.as_tensor(g['neg'],device=z.device)];b=-z[torch.as_tensor(g['pos'],device=z.device)]
            v=torch.nn.functional.softplus(torch.logsumexp(a,0)-np.log(len(a))+torch.logsumexp(b,0)-np.log(len(b)))
            loss=loss+.5*g['weight']*v;values.append(v)
        return loss,values

class CEUnitsRank(LocalAllocationRank):
    """Final proposed normalization: positive-edge contributions in original CE units.

    Each undirected positive edge contributes at most once per endpoint. 1/(2S)
    accounts for those two endpoints. No adaptive weighting or dataset coefficient.
    The preceding type-mean formulation is retained only as a rejected scale draft.
    """
    def __init__(self,types,schema,prior,query,families,labels,*,schema_slots):
        super().__init__(types,schema,prior,query,families,labels)
        self.S=int(schema_slots)
        assert self.S>=len(labels)>0

    def __call__(self,z):
        assert z.ndim==1 and z.numel()==len(self.labels) and bool(torch.isfinite(z).all())
        loss=z.sum()*0;values=[]
        for g in self.groups:
            neg=z[torch.as_tensor(g['neg'],device=z.device)]
            pos=z[torch.as_tensor(g['pos'],device=z.device)]
            logmeanneg=torch.logsumexp(neg,0)-np.log(len(neg))
            v=torch.nn.functional.softplus(logmeanneg-pos)
            loss=loss+v.sum()/(2*self.S);values.append(v)
        return loss,values
