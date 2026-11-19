"""Experimental Xt-wedge closing-edge calibration; not a production runner.

One frozen hard Xt context; same denoiser's clean-edge probabilities; no second
model call, reverse-rule change, probe acceptance check or generated-edge quota.
Only exact full movable-slot R1 coverage is implemented. CPU reference context.
"""
import hashlib
import numpy as np
import torch

VERSION='xt_wedge_closing_edge_calibration_v1'
WEIGHT=1.0

def canonical(edges,n):
    edges=np.asarray(edges,dtype=np.int64)
    if edges.ndim!=2 or edges.shape[0]!=2:raise ValueError('edges require [2,E]')
    if np.any(edges<0) or np.any(edges>=n) or np.any(edges[0]==edges[1]):raise ValueError('invalid endpoint/self loop')
    keys=np.minimum(*edges)*n+np.maximum(*edges)
    if len(np.unique(keys))!=len(keys):raise ValueError('duplicate undirected pair')
    return keys

class WedgeCalibration:
    def __init__(self,types,schema,prior,clean_edges,*,data_sha256):
        self.types=np.asarray(types,dtype=np.int64).copy();self.n=len(self.types)
        self.schema=tuple(tuple(map(int,x)) for x in schema);self.prior=np.asarray(prior,dtype=float).copy()
        if self.types.ndim!=1 or self.n<1 or self.types.min()<0:raise ValueError('types')
        if len(set(self.schema))!=len(self.schema) or len(self.schema)!=len(self.prior):raise ValueError('schema/prior')
        if any(a<0 or a>b or b>int(self.types.max()) for a,b in self.schema):raise ValueError('canonical schema')
        if not np.isfinite(self.prior).all() or ((self.prior<0)|(self.prior>1)).any():raise ValueError('prior support')
        if len(data_sha256)!=64:raise ValueError('bound data SHA256 required')
        int(data_sha256,16);self.data_sha256=data_sha256
        pair={x:i for i,x in enumerate(self.schema)};u,v=np.triu_indices(self.n,1)
        f=np.array([pair.get(tuple(sorted((int(self.types[i]),int(self.types[j])))),-1) for i,j in zip(u,v)])
        legal=f>=0;self.u=u[legal];self.v=v[legal];self.f=f[legal];self.allkeys=self.u*self.n+self.v
        refkeys=canonical(clean_edges,self.n)
        if not np.isin(refkeys,self.allkeys).all():raise ValueError('clean illegal support')
        self.y=np.isin(self.allkeys,refkeys).astype(float)
        for r,pi in enumerate(self.prior):
            z=self.y[self.f==r]
            if not len(z):
                if pi!=0:raise ValueError('nonzero prior on no slots')
            elif not np.isclose(z.mean(),pi,rtol=1e-6,atol=1e-8):raise ValueError('reference/prior mismatch')
        self.movable=(self.prior[self.f]>0)&(self.prior[self.f]<1)
        self.keys=self.allkeys[self.movable];self.families=self.f[self.movable];self.labels=self.y[self.movable]

    def context(self,xt_edges):
        keys=canonical(xt_edges,self.n)
        if not np.isin(keys,self.allkeys).all():raise ValueError('Xt illegal support')
        x=np.isin(self.allkeys,keys)
        if np.any(x[self.prior[self.f]==0]) or not np.all(x[self.prior[self.f]==1]):raise ValueError('Xt fixed support')
        A=np.zeros((self.n,self.n),dtype=np.int64);A[self.u,self.v]=A[self.v,self.u]=x
        cn=(A@A)[self.u[self.movable],self.v[self.movable]].astype(float)
        status=x[self.movable];weights=[];targets=[];groups=[]
        # Group membership/weights use Xt and schema only; X0 supplies targets.
        for r in np.unique(self.families):
            for present in [0,1]:
                mask=(self.families==r)&(status==present);w=cn*mask;mass=float(w.sum())
                if mass==0:continue
                w=w/mass;weights.append(w);targets.append(float(w@self.labels))
                groups.append(dict(relation=int(r),Xt_present=present,slots=int(mask.sum()),positive_CN_slots=int((w>0).sum()),
                    wedge_mass=mass,target_closing_fraction=targets[-1],positive_label_slots=int((mask&(self.labels==1)).sum()),negative_label_slots=int((mask&(self.labels==0)).sum())))
        W=np.stack(weights) if weights else np.zeros((0,len(self.keys)))
        return FrozenWedgeContext(self,W,np.array(targets),groups,hashlib.sha256(np.sort(keys).tobytes()).hexdigest())

class FrozenWedgeContext:
    def __init__(self,owner,weights,targets,groups,state_sha256):
        self.owner=owner;self.weights=weights;self.targets=targets;self.groups=groups;self.state_sha256=state_sha256
    def __call__(self,p,query,families):
        if p.ndim!=1 or not p.is_floating_point() or not bool(torch.isfinite(p).all()) or bool(((p<0)|(p>1)).any()):raise ValueError('finite probability vector')
        keys=canonical(query,self.owner.n);order=np.argsort(keys)
        if len(keys)!=len(p) or not np.array_equal(keys[order],self.owner.keys):raise ValueError('exact full R1 movable support required')
        families=np.asarray(families,dtype=np.int64)
        if families.shape!=p.shape or not np.array_equal(families[order],self.owner.families):raise ValueError('ordered relation semantics')
        if len(self.groups)==0:return p.sum()*0,p.new_empty(0)
        weight=torch.as_tensor(self.weights,device=p.device,dtype=p.dtype);target=torch.as_tensor(self.targets,device=p.device,dtype=p.dtype)
        ordered=p[torch.as_tensor(order,device=p.device)];residual=weight@ordered-target
        values=.5*residual.square()
        return values.mean(),values

def combine(ce_sum,schema_slots,pb_loss,closing_loss):
    """Explicit scalar adapter; no training entry point or optimizer mutation.

    Adds a second auxiliary term: total auxiliary strength is NOT held constant.
    Counts are supplied by caller and must match its bound full-schema CE contract.
    """
    if schema_slots<=0:raise ValueError('full schema denominator')
    if any(v.ndim!=0 or not bool(torch.isfinite(v)) for v in [ce_sum,pb_loss,closing_loss]):raise ValueError('finite scalar losses')
    return ce_sum/schema_slots+pb_loss+WEIGHT*closing_loss
