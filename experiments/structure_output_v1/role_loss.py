"""Typed degree/triangle role-distribution supervision of clean-edge marginals.

No learned parameters, hard graph edits, model calls, or global RNG consumption.
Triangle products refer to the conditionally independent Bernoulli graph, not
the final reverse-chain distribution. Sorting expected summaries is a surrogate.
"""
import itertools,math
import numpy as np
import torch

LAMBDA=0.01
PROJECTION_SEED=2026110601
RANDOM_PROJECTIONS=16
VERSION='typed_degree_triangle_projected_quantiles_v1'

class RoleDistributionLoss:
    def __init__(self,types,pairs,prior,clean_edges,device='cpu',dtype=torch.float32):
        self.types=np.asarray(types,dtype=np.int64);self.n=len(types);self.nt=int(self.types.max())+1
        self.pairs=tuple(tuple(map(int,x)) for x in pairs);self.nf=len(pairs);self.channels=list(itertools.combinations_with_replacement(range(self.nt),2))
        if len(set(self.pairs))!=len(self.pairs):raise ValueError('unique unordered type-pair relation required')
        self.device=torch.device(device);self.dtype=dtype;self.prior=np.asarray(prior)
        self.groups=[torch.as_tensor(np.flatnonzero(self.types==t),device=device) for t in range(self.nt)]
        inventory=np.bincount(self.types,minlength=self.nt);allowed={p for p,v in zip(self.pairs,prior) if v>0}
        caps=[]
        for center in self.types:
            degree=[]
            for (a,b),pi in zip(self.pairs,prior):
                c=inventory[b]-(a==b) if center==a else inventory[a] if center==b else 0
                degree.append(c if pi>0 else 0)
            left=inventory.copy();left[center]-=1;tri=[]
            for a,b in self.channels:
                c=left[a]*(left[a]-1)//2 if a==b else left[a]*left[b]
                if any(tuple(sorted(x)) not in allowed for x in ((center,a),(center,b),(a,b))):c=0
                tri.append(c)
            caps.append(degree+tri)
        self.cap=torch.tensor(caps,device=device,dtype=dtype);self.den=torch.log1p(self.cap).clamp_min(1.)
        self.rel_masks=[]
        for a,b in self.pairs:
            mask=((self.types[:,None]==a)&(self.types[None,:]==b))|((self.types[:,None]==b)&(self.types[None,:]==a))
            np.fill_diagonal(mask,False);self.rel_masks.append(torch.tensor(mask,device=device,dtype=dtype))
        target=torch.zeros((self.n,self.n),device=device,dtype=dtype)
        e=torch.as_tensor(np.asarray(clean_edges),device=device,dtype=torch.long)
        if e.numel():target[e[0],e[1]]=1;target[e[1],e[0]]=1
        self.target=self.features(target).detach();self.indices=[];self.projections=[];self.target_sorted=[]
        for t,group in enumerate(self.groups):
            if not len(group):self.indices.append(None);self.projections.append(None);self.target_sorted.append(None);continue
            idx=torch.nonzero(self.cap[group[0]]>0).flatten();self.indices.append(idx)
            d=len(idx)
            if d==0:self.projections.append(None);self.target_sorted.append(None);continue
            # Independent local generator; same fixed rule for both datasets/types.
            rng=np.random.default_rng(np.random.SeedSequence([PROJECTION_SEED,t,d]))
            mix=rng.choice([-1.,1.],size=(d,RANDOM_PROJECTIONS))/math.sqrt(d)
            projection=torch.as_tensor(np.concatenate([np.eye(d),mix],axis=1),device=device,dtype=dtype)
            self.projections.append(projection)
            self.target_sorted.append(torch.sort(self.target[group][:,idx]@projection,dim=0).values)

    def counts(self,P):
        degrees=torch.stack([(P*mask).sum(1) for mask in self.rel_masks],1)
        triangle=[]
        for a,b in self.channels:
            ia,ib=self.groups[a],self.groups[b]
            v=((P[:,ia]@P[ia][:,ib])*P[:,ib]).sum(1)
            triangle.append(v*.5 if a==b else v)
        return torch.cat([degrees,torch.stack(triangle,1)],1)

    def features(self,P):
        counts=self.counts(P)
        return torch.where(self.cap>0,torch.log1p(counts)/self.den,torch.zeros_like(counts))

    def matrix(self,probabilities,query):
        q=torch.as_tensor(query,device=self.device,dtype=torch.long)
        P=torch.zeros((self.n,self.n),device=self.device,dtype=probabilities.dtype)
        P=P.index_put((q[0],q[1]),probabilities)
        return P+P.T

    def __call__(self,probabilities,query):
        P=self.matrix(probabilities,query);features=self.features(P);losses=[]
        for group,idx,projection,target in zip(self.groups,self.indices,self.projections,self.target_sorted):
            if projection is None:continue
            delta=torch.sort(features[group][:,idx]@projection,dim=0).values-target
            # Stable sqrt(1+d²)-1, bounded slope in projected residual d.
            phi=delta.square()/(torch.sqrt(1+delta.square())+1)
            losses.append(phi.mean())
        loss=torch.stack(losses).mean() if losses else probabilities.sum()*0
        return loss,dict(per_type=[float(x.detach()) for x in losses],projected_dimensions=[0 if p is None else p.shape[1] for p in self.projections])

    @classmethod
    def from_case(cls,case,device='cuda'):
        return cls(case['space'].types,case['space'].pairs,case['prior'],case['clean'],device)
