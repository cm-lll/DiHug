"""Standalone exact independent-Bernoulli relation-degree CDF objective.

No denoiser, optimizer, sampler, RNG or automatic training entry point.
Requires full unique unordered mutable query coverage (R1); not a block estimator.
"""
import hashlib
import json
import numpy as np
import torch
from torch.autograd.function import once_differentiable

VERSION = 'independent_bernoulli_relation_degree_cdf_v1'
PROPOSED_WEIGHT = 1.0


class _GroupedCDF(torch.autograd.Function):
    """Batched polynomial DP; exact first derivative, no higher-order support."""
    @staticmethod
    def forward(ctx, probabilities, target_cdf, bounds, capacities):
        batch, width = probabilities.shape
        h = probabilities.new_ones((batch, 1))
        history = []
        for j in range(width):
            history.append(h)
            v = probabilities[:, j:j+1]
            h = torch.cat(((1-v)*h, h.new_zeros((batch, 1))), 1) + torch.cat((h.new_zeros((batch, 1)), v*h), 1)
        residuals, values = [], []
        for g, (lo, hi) in enumerate(bounds):
            cap = capacities[g]
            cdf = h[lo:hi].mean(0).cumsum(0)[:cap]
            e = cdf - target_cdf[g, :cap]
            residuals.append(e)
            values.append(.5 * e.square().mean())
        ctx.bounds, ctx.capacities = bounds, capacities
        ctx.save_for_backward(probabilities, *history, *residuals)
        ctx.width = width
        return torch.stack(values)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        saved = ctx.saved_tensors
        p = saved[0]; history = saved[1:1+ctx.width]; residuals = saved[1+ctx.width:]
        batch, width = p.shape
        adj = p.new_zeros((batch, width+1))
        for g, (lo, hi) in enumerate(ctx.bounds):
            cap = ctx.capacities[g]
            dcdf = grad_output[g]*residuals[g]/cap
            dh = dcdf.flip(0).cumsum(0).flip(0)/(hi-lo)
            adj[lo:hi, :cap] = dh
        gradient = torch.empty_like(p)
        for j in range(width-1, -1, -1):
            h = history[j]
            gradient[:, j] = (h*(adj[:, 1:]-adj[:, :-1])).sum(1)
            v = p[:, j:j+1]
            adj = (1-v)*adj[:, :-1]+v*adj[:, 1:]
        return gradient, None, None, None


class RelationDegreeCDF:
    def __init__(self, types, schema, prior, clean_edges, *, data_sha256):
        self.types = np.asarray(types, dtype=np.int64)
        self.schema = tuple(tuple(map(int, r)) for r in schema)
        self.prior = np.asarray(prior, dtype=np.float64)
        self.n = len(self.types)
        if not isinstance(data_sha256, str) or len(data_sha256) != 64:
            raise ValueError('caller must supply bound data SHA256')
        try:int(data_sha256,16)
        except ValueError:raise ValueError('data SHA256 must be hexadecimal')
        if len(set(self.schema)) != len(self.schema) or len(self.prior) != len(self.schema):
            raise ValueError('unique ordered schema and prior required')
        if any(a>b or a<0 or b>int(self.types.max()) for a,b in self.schema):
            raise ValueError('schema must contain canonical unordered valid types')
        if not np.isfinite(self.prior).all() or np.any((self.prior<0)|(self.prior>1)):
            raise ValueError('invalid prior')
        edges=np.asarray(clean_edges, dtype=np.int64)
        if edges.ndim != 2 or edges.shape[0] != 2:
            raise ValueError('clean_edges must be [2,E]')
        if np.any(edges<0) or np.any(edges>=self.n) or np.any(edges[0]==edges[1]):
            raise ValueError('invalid clean edge endpoint')
        ekeys=np.minimum(*edges)*self.n+np.maximum(*edges)
        if len(np.unique(ekeys))!=len(ekeys):raise ValueError('duplicate undirected target edge')
        ref=np.zeros((self.n,self.n));ref[edges[0],edges[1]]=ref[edges[1],edges[0]]=1
        u,v=np.triu_indices(self.n,1);pair_lookup={p:f for f,p in enumerate(self.schema)}
        families=np.array([pair_lookup.get(tuple(sorted((int(self.types[i]),int(self.types[j])))),-1) for i,j in zip(u,v)])
        if any(tuple(sorted((int(self.types[i]),int(self.types[j])))) not in pair_lookup for i,j in edges.T):raise ValueError('illegal target relation')
        active=[]
        for f,pi in enumerate(self.prior):
            m=families==f;count=int(ref[u[m],v[m]].sum());slots=int(m.sum())
            if not slots:
                if pi!=0:raise ValueError('nonzero prior on absent inventory')
                continue
            if not np.isclose(pi,count/slots,rtol=1e-6,atol=1e-8):raise ValueError('prior/target relationship mismatch')
            if 0<pi<1:active.append(f)
            elif count!=(slots if pi==1 else 0):raise ValueError('fixed support mismatch')
        mutable=np.isin(families,active)
        self.keys=u[mutable]*self.n+v[mutable]
        qu,qv=u[mutable],v[mutable];qf=families[mutable]
        indices=[];bounds=[];caps=[];targets=[];groups=[];cursor=0
        for f in active:
            for typ in sorted(set(self.schema[f])):
                nodes=np.flatnonzero(self.types==typ);nr=[];degree=[]
                for node in nodes:
                    ix=np.flatnonzero((qf==f)&((qu==node)|(qv==node)))
                    nr.append(ix);degree.append(int(ref[qu[ix],qv[ix]].sum()))
                cap=len(nr[0]);assert cap and all(len(x)==cap for x in nr)
                indices.extend(nr);bounds.append((cursor,cursor+len(nodes)));cursor+=len(nodes)
                caps.append(cap);targets.append(np.cumsum(np.bincount(degree,minlength=cap+1)/len(nodes))[:cap])
                groups.append(dict(relation=f,node_type=typ,nodes=len(nodes),capacity=cap))
        self.bounds=tuple(bounds);self.capacities=tuple(caps);self.groups=groups
        width=max(caps,default=0)
        self.indices=np.zeros((cursor,width),dtype=np.int64);self.valid=np.zeros((cursor,width),dtype=bool)
        for i,x in enumerate(indices):self.indices[i,:len(x)]=x;self.valid[i,:len(x)]=True
        self.target_cdf=np.zeros((len(groups),width))
        for i,x in enumerate(targets):self.target_cdf[i,:len(x)]=x
        self.binding=dict(version=VERSION,data_sha256=data_sha256,types=self.types.tolist(),ordered_schema=self.schema,prior=self.prior.tolist(),target_edges_sha256=hashlib.sha256(np.sort(ekeys).tobytes()).hexdigest(),groups=groups)
        self.binding_sha256=hashlib.sha256(json.dumps(self.binding,sort_keys=True).encode()).hexdigest()
        self._device_cache={}

    def __call__(self, probabilities, query, families):
        if probabilities.ndim!=1 or not probabilities.is_floating_point():raise ValueError('one floating probability per slot')
        if not bool(torch.isfinite(probabilities).all()) or bool(((probabilities<0)|(probabilities>1)).any()):raise ValueError('nonfinite/out-of-range probability')
        query=np.asarray(query,dtype=np.int64);families=np.asarray(families,dtype=np.int64)
        if query.shape!=(2,len(probabilities)) or families.shape!=(len(probabilities),):raise ValueError('query/family shape')
        if np.any(query<0) or np.any(query>=self.n) or np.any(query[0]==query[1]):raise ValueError('query endpoint')
        keys=np.minimum(*query)*self.n+np.maximum(*query);order=np.argsort(keys)
        if not np.array_equal(keys[order],self.keys):raise ValueError('requires exact full R1 unique mutable support; no subset/duplicate')
        if np.any(families<0) or np.any(families>=len(self.schema)):raise ValueError('family index')
        actual=np.sort(self.types[query].T,axis=1)
        if not np.array_equal(actual,np.asarray(self.schema)[families]):raise ValueError('family/schema semantic mismatch')
        if not self.groups:return probabilities.sum()*0, probabilities.new_empty((0,))
        key=(str(probabilities.device),probabilities.dtype)
        if key not in self._device_cache:
            self._device_cache[key]=(torch.as_tensor(self.indices,device=probabilities.device),torch.as_tensor(self.valid,device=probabilities.device),torch.as_tensor(self.target_cdf,device=probabilities.device,dtype=probabilities.dtype))
        ix,valid,target=self._device_cache[key]
        canonical=probabilities[torch.as_tensor(order,device=probabilities.device)]
        batched=canonical[ix]*valid
        values=_GroupedCDF.apply(batched,target,self.bounds,self.capacities)
        return values.mean(),values

    def objective_binding(self):
        return dict(self.binding,binding_sha256=self.binding_sha256,proposed_weight=PROPOSED_WEIGHT,CE_denominator=self.n*(self.n-1)//2,formal_authorized=False)

    def assert_binding(self, recorded):
        expected=self.objective_binding()
        if json.dumps(recorded,sort_keys=True)!=json.dumps(expected,sort_keys=True):
            raise ValueError('objective version/data/target/schema/weight mismatch')


def combine_normal1(ce_sum, probabilities, query, families, objective, *, schema_slots):
    """Candidate adapter only; caller owns model/optimizer/RNG/EMA transaction."""
    if schema_slots!=objective.n*(objective.n-1)//2:
        raise ValueError('this frozen 192-node protocol requires its full schema denominator')
    nt=int(objective.types.max())+1
    if set(objective.schema)!={(a,b) for a in range(nt) for b in range(a,nt)}:
        raise ValueError('current adapter requires declared complete type-pair schema')
    if ce_sum.ndim!=0 or not bool(torch.isfinite(ce_sum)):
        raise ValueError('finite scalar sumCE required')
    aux, per_group=objective(probabilities,query,families)
    ce=ce_sum/schema_slots
    return ce+PROPOSED_WEIGHT*aux, dict(unweighted_CE=ce,auxiliary=aux,per_group=per_group)
