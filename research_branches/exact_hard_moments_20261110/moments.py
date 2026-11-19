"""Exact expected EXISTING count/typed-triangle squared hard losses.

Independent Bernoulli clean edge law, fixed legal slots, canonical undirected
edges. Includes triangle variance and shared-edge covariance. No model calls.
"""
import itertools
import numpy as np
import torch

class ExactHardMoments:
    def __init__(self, core):
        self.core=core;self.m=len(core.reference)
        lookup=np.full((core.n,core.n),-1,dtype=np.int64)
        lookup[core.q[0],core.q[1]]=np.arange(self.m);lookup[core.q[1],core.q[0]]=np.arange(self.m)
        base=int(core.types.max())+1;codes=np.full(base**3,-1,dtype=np.int64)
        for j,(a,b,c) in enumerate(core.triples):codes[a*base*base+b*base+c]=j
        ix=[];groups=[]
        # Only legal movable slots; fixed-zero families cannot create triangles.
        for u in range(core.n):
            for v in np.flatnonzero(lookup[u]>-1):
                if v<=u:continue
                w=np.flatnonzero((lookup[u]>=0)&(lookup[v]>=0)&(np.arange(core.n)>v))
                if not len(w):continue
                ix.append(np.stack([np.full(len(w),lookup[u,v]),lookup[u,w],lookup[v,w]],1))
                ty=np.sort(np.stack([np.full(len(w),core.types[u]),np.full(len(w),core.types[v]),core.types[w]]),axis=0)
                groups.append(codes[ty[0]*base*base+ty[1]*base+ty[2]])
        tri=np.concatenate(ix) if ix else np.empty((0,3),dtype=np.int64)
        gi=np.concatenate(groups) if groups else np.empty(0,dtype=np.int64)
        self.triangle_edges=[torch.from_numpy(tri[gi==j].copy()) for j in range(len(core.triples))]
        assert all((len(x)>0)==bool(core.present[j] or core.absent[j]) for j,x in enumerate(self.triangle_edges))

    def value_gradient(self, probability):
        p0=np.asarray(probability,dtype=np.float64)
        if p0.shape!=(self.m,) or not np.isfinite(p0).all() or not ((p0>=0)&(p0<=1)).all():raise ValueError('invalid probability')
        c=self.core;p=torch.tensor(p0,dtype=torch.float64,requires_grad=True)
        losses=[];triangle_stats=[]
        for j,ix in enumerate(self.triangle_edges):
            if len(ix)==0:
                losses.append(p.sum()*0);triangle_stats.append(dict(index=j,mean=0.,variance=0.,squared_mean_error=0.,risk=0.,target=int(c.ref['triangles'][j]),capacity=0));continue
            edges=p[ix];q=edges.prod(1);mean=q.sum();s=torch.zeros(self.m,dtype=p.dtype);ss=torch.zeros_like(s)
            for e in range(3):
                other=edges[:,(e+1)%3]*edges[:,(e+2)%3]
                s=s.index_add(0,ix[:,e],other);ss=ss.index_add(0,ix[:,e],other.square())
            variance=(q-q.square()).sum()+(p*(1-p)*(s.square()-ss)).sum()
            target=int(c.ref['triangles'][j]);den=max(1,target)**2
            risk=((mean-target).square()+variance)/den
            if variance.item() < -1e-9:raise FloatingPointError('negative variance beyond roundoff')
            losses.append(risk);triangle_stats.append(dict(index=j,mean=float(mean.detach()),variance=float(variance.detach()),squared_mean_error=float((mean.detach()-target)**2),risk=float(risk.detach()),target=target,capacity=len(ix)))
        values=[];grads=[]
        count_value=0.;count_grad=np.zeros(self.m);count_stats=[]
        for f in range(c.r):
            mask=c.f==f;mu=p0[mask].sum();var=(p0[mask]*(1-p0[mask])).sum();target=int(c.ref['counts'][f]);den=max(1,target)**2
            risk=((mu-target)**2+var)/den
            count_value+=risk/c.r;count_grad[mask]=p0[mask]*(1-p0[mask])*(1-2*p0[mask]+2*(mu-target))/(den*c.r)
            count_stats.append(dict(family=f,mean=float(mu),variance=float(var),target=target,risk=float(risk)))
        values.append(count_value);grads.append(count_grad)
        for mask in (c.present,c.absent):
            ids=np.flatnonzero(mask)
            value=torch.stack([losses[j] for j in ids]).mean() if len(ids) else p.sum()*0
            grad=torch.autograd.grad(value,p,retain_graph=True)[0].detach().numpy()*p0*(1-p0)
            values.append(float(value.detach()));grads.append(grad)
        V=np.asarray(values);G=np.stack(grads)
        if not np.isfinite(V).all() or not np.isfinite(G).all():raise FloatingPointError('nonfinite exact risk')
        return V,G,dict(triangles=triangle_stats,relations=count_stats)
