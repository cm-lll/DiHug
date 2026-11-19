"""Schema-sized learned state-potential differences and component-safe edits."""
import numpy as np
import torch
from candidate_consequences import Consequences


def scales(n,columns):return np.array([np.sqrt(n) if family=='typed_triangles' else n for family,key in columns],dtype=np.float64)


def encode(counts,delta,n,columns):
    assert np.array_equal(counts,counts.astype(np.float32).astype(np.int64))
    return np.concatenate([np.broadcast_to(counts,(len(delta),len(counts))).astype(np.float32),(delta/scales(n,columns)).astype(np.float32)],-1)


class Potential(torch.nn.Module):
    def __init__(self,n,columns,mean,sd):
        super().__init__();self.k=len(columns)
        self.register_buffer('mu',torch.tensor(mean,dtype=torch.float32));self.register_buffer('sd',torch.tensor(sd,dtype=torch.float32))
        self.register_buffer('delta_scale',torch.tensor(scales(n,columns),dtype=torch.float32));self.weights=torch.nn.Parameter(torch.zeros(2*self.k))
    def phi(self,x):
        z=(x[:,:,:self.k]-self.mu)/self.sd;d=x[:,:,self.k:]*self.delta_scale/self.sd
        return torch.cat([-d,-(2*z*d+d*d)],-1)
    def forward(self,x):return (self.phi(x)*self.weights).sum(-1)


class ComponentState(Consequences):
    def __init__(self,graph,columns,counts=None):
        super().__init__(graph,columns,counts);self.n_components=self.component_count()
    def component_count(self):
        unseen=set(range(self.n));count=0
        while unseen:
            count+=1;stack=[unseen.pop()]
            while stack:
                u=stack.pop()
                for v in self.nb[u]:
                    if v in unseen:unseen.remove(v);stack.append(v)
        return count
    def guarded(self,values,index):
        before=self.counts.copy();old=self.n_components;self.apply(values,index);now=self.component_count();reject=now>old
        if reject:
            self.apply(values,index,reverse=True);assert np.array_equal(self.counts,before) and self.component_count()==old
        else:self.n_components=now
        return reject
