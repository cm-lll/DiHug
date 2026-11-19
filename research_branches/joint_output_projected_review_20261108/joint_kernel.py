"""Exact full relation-degree product-kernel discrepancy; full R1 only.

No learned all-pair head, no model call, no sampler alteration. Node-pair
matrices aggregate the auxiliary loss only. Source: CPU protocol 20261108.
"""
import hashlib,json
import numpy as np
import torch
from pb_degree import RelationDegreeCDF
from joint_degree import _NodeCDF

class JointKernel(RelationDegreeCDF):
    def __init__(self,types,schema,prior,clean_edges,*,data_sha256):
        super().__init__(types,schema,prior,clean_edges,data_sha256=data_sha256)
        degree=np.zeros((self.n,len(self.schema)),dtype=np.int64)
        lookup={r:f for f,r in enumerate(self.schema)}
        for u,v in np.asarray(clean_edges).T:
            f=lookup[tuple(sorted((int(self.types[u]),int(self.types[v]))))]
            degree[u,f]+=1;degree[v,f]+=1
        self.type_groups=[];self._kernels={}
        for typ in sorted(set(self.types.tolist())):
            ids=[i for i,g in enumerate(self.groups) if g['node_type']==typ]
            if not ids:continue
            nodes=np.flatnonzero(self.types==typ);relations=[self.groups[i]['relation'] for i in ids]
            d=degree[np.ix_(nodes,relations)]
            self.type_groups.append(dict(node_type=typ,group_ids=ids,relations=relations,degrees=d,scales=np.maximum(1.,d.mean(0))))
        self.binding.update(version='full_relation_degree_product_kernel_v1',coefficient=1.,
            formula='CE+mean_type[0.5*MMD^2(mixture_node_product_PB,empirical_reference_joint_degrees)]',
            kernel='product_r exp(-abs(d_r-e_r)/s_type_r)',scale='max(1, training reference mean degree)',
            type_groups=[dict(node_type=g['node_type'],group_ids=g['group_ids'],relations=g['relations'],scales=g['scales'].tolist(),degrees=g['degrees'].tolist()) for g in self.type_groups],
            replaces_PB=True,extra_PB=False,independent_replicas_in_PP=True)
        self.binding_sha256=hashlib.sha256(json.dumps(self.binding,sort_keys=True).encode()).hexdigest()

    def __call__(self,p,query,families):
        # Reuse frozen complete support, query, relation, finite checks. The
        # returned marginal loss is deliberately NOT added to this objective.
        super().__call__(p,query,families)
        if not self.type_groups:return p.sum()*0,p.new_empty(0)
        q=np.asarray(query);order=np.argsort(np.minimum(*q)*self.n+np.maximum(*q))
        key=(str(p.device),p.dtype);ix,valid,_=self._device_cache[key]
        cdf=_NodeCDF.apply(p[torch.as_tensor(order,device=p.device)][ix]*valid)
        if key not in self._kernels:
            cache=[]
            for g in self.type_groups:
                entries=[];tt=None
                for j,gid in enumerate(g['group_ids']):
                    cap=self.capacities[gid];axis=torch.arange(cap+1,device=p.device,dtype=p.dtype)
                    K=torch.exp(-(axis[:,None]-axis[None,:]).abs()/float(g['scales'][j]))
                    d=torch.as_tensor(g['degrees'][:,j],device=p.device)
                    kt=K[d[:,None],d[None,:]];tt=kt if tt is None else tt*kt
                    entries.append((K,K[:,d]))
                cache.append((entries,tt.mean()))
            self._kernels[key]=cache
        vals=[]
        for g,(entries,tt) in zip(self.type_groups,self._kernels[key]):
            pp=None;pt=None
            for gid,(K,KT) in zip(g['group_ids'],entries):
                lo,hi=self.bounds[gid];f=cdf[lo:hi,:self.capacities[gid]+1]
                h=torch.cat((f[:,:1],f[:,1:]-f[:,:-1]),1)
                a=h@K@h.T;b=h@KT
                pp=a if pp is None else pp*a;pt=b if pt is None else pt*b
            vals.append(.5*(pp.mean()-2*pt.mean()+tt))
        typed=torch.stack(vals)
        if not bool(torch.isfinite(typed).all()) or float(typed.detach().min()) < -2e-6:
            raise FloatingPointError('nonfinite/negative kernel discrepancy beyond FP32 cancellation')
        # No clamping: preserve true analytic derivative and report cancellation.
        return typed.mean(),typed
