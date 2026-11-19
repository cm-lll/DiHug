"""Experimental CPU-verifiable coupled reverse typed-triangle risk.

No denoiser, optimizer or sampling entry point. Exact full R1 mutable coverage.
Nonnegative Hoeffding/ANOVA decomposition avoids subtracting large covariances.
"""
import hashlib
import itertools
import numpy as np
import torch

VERSION = 'coupled_reverse_typed_triangle_anova_v1'


def product_difference(a, b):
    """prod(a)-prod(b), with no subtraction of complete products, width three."""
    return ((a[:, 0]-b[:, 0])*a[:, 1]*a[:, 2]
            + b[:, 0]*(a[:, 1]-b[:, 1])*a[:, 2]
            + b[:, 0]*b[:, 1]*(a[:, 2]-b[:, 2]))


def triangle_risk(q, k, triangles):
    """Exact E[(T(B_q)-T(B_k))²] for one triangle group and shared uniforms.

    q,k are per-undirected-slot probabilities; trials are independent across
    slots, but the two trials for one slot are monotonically coupled. All
    operations in risk other than the squared mean difference are nonnegative.
    q=k uses torch's zero subgradient for abs; zero exact risk and gradient.
    """
    if triangles.numel() == 0:
        z = q.sum()*0
        return z, dict(mean_q=z, mean_k=z, bias_squared=z, order1=z, order2=z, order3=z)
    lo, hi = torch.minimum(q, k), torch.maximum(q, k)
    gap = torch.abs(q-k)
    c = lo*(1-hi)
    rq = gap*torch.where(q <= k, q, 1-q)
    rk = gap*torch.where(k <= q, k, 1-k)
    qt, kt, ct = q[triangles], k[triangles], c[triangles]
    rqt, rkt = rq[triangles], rk[triangles]
    aq, ak = torch.zeros_like(q), torch.zeros_like(k)
    for j in range(3):
        a, b = (j+1) % 3, (j+2) % 3
        aq = aq.index_add(0, triangles[:, j], qt[:, a]*qt[:, b])
        ak = ak.index_add(0, triangles[:, j], kt[:, a]*kt[:, b])
    bias = product_difference(qt, kt).sum().square()
    order1 = (c*(aq-ak).square()+rq*aq.square()+rk*ak.square()).sum()
    order2 = q.sum()*0
    for j in range(3):
        a, b = (j+1) % 3, (j+2) % 3
        dq = rqt[:, a]*ct[:, b]+ct[:, a]*rqt[:, b]+rqt[:, a]*rqt[:, b]
        dk = rkt[:, a]*ct[:, b]+ct[:, a]*rkt[:, b]+rkt[:, a]*rkt[:, b]
        order2 = order2+(ct[:, a]*ct[:, b]*(qt[:, j]-kt[:, j]).square()
                         +dq*qt[:, j].square()+dk*kt[:, j].square()).sum()
    # v=c+r: write product(v)-product(c) directly with the known differences r.
    def residual_product(r):
        v = ct+r
        return r[:, 0]*v[:, 1]*v[:, 2]+ct[:, 0]*r[:, 1]*v[:, 2]+ct[:, 0]*ct[:, 1]*r[:, 2]
    order3 = (residual_product(rqt)+residual_product(rkt)).sum()
    risk = bias+order1+order2+order3
    return risk, dict(mean_q=qt.prod(1).sum(), mean_k=kt.prod(1).sum(),
                      bias_squared=bias, order1=order1, order2=order2, order3=order3)


def canonical(edges, n):
    e = np.asarray(edges, dtype=np.int64)
    if e.ndim != 2 or e.shape[0] != 2 or np.any(e < 0) or np.any(e >= n) or np.any(e[0] == e[1]):
        raise ValueError('valid non-self query/edges [2,E] required')
    keys = np.minimum(*e)*n+np.maximum(*e)
    if len(np.unique(keys)) != len(keys): raise ValueError('duplicate unordered edge')
    return keys


class CoupledTriangle:
    def __init__(self, types, schema, prior, clean_edges, *, data_sha256):
        if not isinstance(data_sha256, str) or len(data_sha256) != 64:
            raise ValueError('bound data SHA256 required')
        int(data_sha256, 16)
        self.types = np.array(types, dtype=np.int64, copy=True)
        if self.types.ndim != 1 or len(self.types) == 0 or np.any(self.types < 0): raise ValueError('types')
        self.n = len(self.types); self.schema = tuple(tuple(map(int, x)) for x in schema)
        self.prior = np.array(prior, dtype=float, copy=True)
        if len(self.schema) != len(self.prior) or len(set(self.schema)) != len(self.schema): raise ValueError('schema/prior')
        if any(a < 0 or a > b or b > int(self.types.max()) for a,b in self.schema): raise ValueError('canonical schema')
        if not np.isfinite(self.prior).all() or np.any((self.prior < 0)|(self.prior > 1)): raise ValueError('prior')
        u,v = np.triu_indices(self.n,1); lookup = {x:i for i,x in enumerate(self.schema)}
        f = np.array([lookup.get(tuple(sorted((self.types[i],self.types[j]))),-1) for i,j in zip(u,v)])
        good = f >= 0; self.u,self.v,self.f = u[good],v[good],f[good]
        self.keys = self.u*self.n+self.v
        ref = canonical(clean_edges,self.n)
        if not np.isin(ref,self.keys).all(): raise ValueError('illegal clean edges')
        self.y = np.isin(self.keys,ref).astype(float)
        for i,pi in enumerate(self.prior):
            z = self.y[self.f == i]
            if not np.isclose(z.mean() if len(z) else 0.,pi,rtol=1e-6,atol=1e-8): raise ValueError('prior vs data')
        self.mutable = (self.prior[self.f] > 0)&(self.prior[self.f] < 1)
        self.query_keys = self.keys[self.mutable]
        self.data_sha256 = data_sha256
        ids = np.full((self.n,self.n),-1,dtype=np.int64)
        ids[self.u,self.v] = ids[self.v,self.u] = np.arange(len(self.keys))
        nodes = np.array(list(itertools.combinations(range(self.n),3)), dtype=np.int64).reshape(-1,3)
        tris = np.stack([ids[nodes[:,0],nodes[:,1]],ids[nodes[:,0],nodes[:,2]],ids[nodes[:,1],nodes[:,2]]],1)
        valid = np.all(tris >= 0, axis=1)
        nodes,tris = nodes[valid],tris[valid]
        support = np.all(self.prior[self.f[tris]] > 0, axis=1)
        nodes,tris = nodes[support],tris[support]
        patterns = np.sort(self.types[nodes],axis=1)
        self.groups = []
        for typ in itertools.combinations_with_replacement(range(int(self.types.max())+1),3):
            take = np.all(patterns == typ,axis=1); tri = tris[take]
            target = int(self.y[tri].prod(1).sum())
            self.groups.append(dict(types=typ, triangles=tri, possible=len(tri), target=target,
                                    scale=max(target,1), active=bool(len(tri))))
        self.cache = {}

    def context(self, xt_edges, t, kernel):
        if not isinstance(t,(int,np.integer)) or not 1 <= t < len(kernel.alpha): raise ValueError('time')
        if not np.array_equal(kernel.prior,self.prior): raise ValueError('kernel prior binding')
        keys = canonical(xt_edges,self.n)
        if not np.isin(keys,self.keys).all(): raise ValueError('Xt illegal edge')
        x = np.isin(self.keys,keys).astype(np.int64)
        pi = self.prior[self.f]
        if np.any(x[pi == 0]) or not np.all(x[pi == 1]): raise ValueError('Xt fixed support')
        low = np.where(self.mutable,0.,pi); high = np.where(self.mutable,1.,pi)
        a = kernel.reverse(low,x,self.f,t-1,t)
        b = kernel.reverse(high,x,self.f,t-1,t)
        k = kernel.reverse(self.y,x,self.f,t-1,t)
        if not np.allclose(k,a+(b-a)*self.y,rtol=0,atol=1e-14): raise ValueError('posterior mixture identity')
        return Context(self,a,b,k,t,hashlib.sha256(np.sort(keys).tobytes()).hexdigest())


class Context:
    def __init__(self,owner,a,b,k,t,state_sha256):
        self.owner,self.a,self.b,self.k,self.t,self.state_sha256 = owner,a,b,k,t,state_sha256

    def evaluate(self,p,query,families,*,accumulation_dtype=torch.float64):
        if p.ndim != 1 or not p.is_floating_point() or not bool(torch.isfinite(p).all()) or bool(((p<0)|(p>1)).any()):
            raise ValueError('finite probability vector')
        o = self.owner; keys = canonical(query,o.n); order = np.argsort(keys)
        if len(keys) != len(p) or not np.array_equal(keys[order],o.query_keys): raise ValueError('full R1 mutable support required')
        f = np.asarray(families)
        if f.shape != p.shape or not np.array_equal(f[order],o.f[o.mutable]): raise ValueError('relation ordering')
        dev = p.device; typ = accumulation_dtype
        a,b,k = [torch.as_tensor(x,device=dev,dtype=typ) for x in [self.a,self.b,self.k]]
        ix = torch.as_tensor(np.flatnonzero(o.mutable),device=dev)
        pred = torch.as_tensor(o.y,device=dev,dtype=typ).index_copy(0,ix,p[torch.as_tensor(order,device=dev)].to(typ))
        q = a+(b-a)*pred
        values,rows = [],[]
        for g in o.groups:
            cachekey = (str(dev),g['types'])
            if cachekey not in o.cache: o.cache[cachekey] = torch.as_tensor(g['triangles'],device=dev)
            risk,parts = triangle_risk(q,k,o.cache[cachekey])
            scaled = risk/(g['scale']**2)
            if g['active']: values.append(scaled)
            rows.append(dict(types=g['types'],possible=g['possible'],target=g['target'],scale=g['scale'],
                             active=g['active'],risk=risk,scaled=scaled,**parts))
        total = torch.stack(values).mean() if values else p.sum()*0
        return total,rows
