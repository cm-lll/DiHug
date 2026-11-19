"""One frozen extension: marginal PB CDF plus within-type pairwise degree CDF.

The joint distribution is an independent-edge conditional-clean-graph proxy.
It is not the reverse-chain law and does not impose an edge quota or a role label.
"""
import hashlib,itertools,json
import numpy as np
import torch
from torch.autograd.function import once_differentiable
from pb_degree import RelationDegreeCDF

VERSION='independent_bernoulli_marginal_plus_joint_relation_degree_cdf_v1'
PROPOSED_WEIGHT=1.0

class _NodeCDF(torch.autograd.Function):
    @staticmethod
    def forward(ctx,p):
        h=p.new_ones((len(p),1));history=[]
        for j in range(p.shape[1]):
            history.append(h);v=p[:,j:j+1]
            h=torch.cat(((1-v)*h,h.new_zeros((len(p),1))),1)+torch.cat((h.new_zeros((len(p),1)),v*h),1)
        ctx.save_for_backward(p,*history)
        return h.cumsum(1)

    @staticmethod
    @once_differentiable
    def backward(ctx,dCDF):
        p,*history=ctx.saved_tensors
        adj=dCDF.flip(1).cumsum(1).flip(1);gradient=torch.empty_like(p)
        for j in range(p.shape[1]-1,-1,-1):
            gradient[:,j]=(history[j]*(adj[:,1:]-adj[:,:-1])).sum(1)
            v=p[:,j:j+1];adj=(1-v)*adj[:,:-1]+v*adj[:,1:]
        return gradient

class JointRelationDegreeCDF(RelationDegreeCDF):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        edges=np.asarray(args[3] if len(args)>3 else kwargs['clean_edges'],dtype=np.int64)
        degree=np.zeros((self.n,len(self.schema)),dtype=np.int64)
        lookup={r:f for f,r in enumerate(self.schema)}
        for u,v in edges.T:
            f=lookup[tuple(sorted((int(self.types[u]),int(self.types[v]))))]
            degree[u,f]+=1;degree[v,f]+=1
        self.pair_groups=[];self.joint_targets=[];self._joint_cache={}
        for typ in sorted(set(self.types.tolist())):
            gids=[g for g,x in enumerate(self.groups) if x['node_type']==typ]
            nodes=np.flatnonzero(self.types==typ)
            for a,b in itertools.combinations(gids,2):
                ga,gb=self.groups[a],self.groups[b];ca,cb=ga['capacity'],gb['capacity']
                # Different relations have disjoint incident Bernoulli slots.
                assert ga['relation']!=gb['relation'] and ga['nodes']==gb['nodes']==len(nodes)
                da=degree[nodes,ga['relation']];db=degree[nodes,gb['relation']]
                ea=(da[:,None]<=np.arange(ca)[None,:]).astype(np.float64)
                eb=(db[:,None]<=np.arange(cb)[None,:]).astype(np.float64)
                self.joint_targets.append(ea.T@eb/len(nodes))
                self.pair_groups.append(dict(node_type=typ,left_group=a,right_group=b,left_relation=ga['relation'],right_relation=gb['relation'],nodes=len(nodes),left_capacity=ca,right_capacity=cb))
        self.binding.update(version=VERSION,joint_groups=self.pair_groups,joint_target_sha256=[hashlib.sha256(x.tobytes()).hexdigest() for x in self.joint_targets],objective='mean_marginal_half_squared_CDF_plus_mean_joint_half_squared_CDF',marginal_weight=1.0,joint_weight=1.0)
        self.binding_sha256=hashlib.sha256(json.dumps(self.binding,sort_keys=True).encode()).hexdigest()

    def __call__(self,p,query,families):
        # This keeps every old marginal/support check and its coefficient exactly.
        marginal,mg=super().__call__(p,query,families)
        if not self.pair_groups:return marginal,mg
        query=np.asarray(query,dtype=np.int64)
        order=np.argsort(np.minimum(*query)*self.n+np.maximum(*query))
        key=(str(p.device),p.dtype);ix,valid,_=self._device_cache[key]
        if key not in self._joint_cache:
            self._joint_cache[key]=[torch.as_tensor(x,device=p.device,dtype=p.dtype) for x in self.joint_targets]
        cdf=_NodeCDF.apply(p[torch.as_tensor(order,device=p.device)][ix]*valid)
        values=[]
        for pair,target in zip(self.pair_groups,self._joint_cache[key]):
            a,b=pair['left_group'],pair['right_group'];al,ah=self.bounds[a];bl,bh=self.bounds[b]
            fa=cdf[al:ah,:pair['left_capacity']];fb=cdf[bl:bh,:pair['right_capacity']]
            joint=fa.T@fb/pair['nodes']
            values.append(.5*(joint-target).square().mean())
        jg=torch.stack(values)
        return marginal+jg.mean(),torch.cat((mg,jg))
