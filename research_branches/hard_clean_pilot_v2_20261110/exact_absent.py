"""Exact hard triangle-count square expectation under independent Bernoulli edges.

No graph enumeration, edge editing, soft-count square substitution or objective
change. Includes the variance and shared-edge covariance of triangle indicators.
Only the already-defined reference-absent triangle group uses this estimator.
"""
import itertools
import numpy as np
import torch


class ExactAbsentTriangles:
    def __init__(self, core):
        self.m=len(core.reference)
        ids={tuple(pair):i for i,pair in enumerate(core.q.T.tolist())}
        self.triangle_edges=[]
        for idx in np.flatnonzero(core.absent):
            target=core.triples[idx]
            # Restrict nodes before enumeration; no probabilities/results select the set.
            nodes=np.flatnonzero(np.isin(core.types,target)).tolist();tri=[]
            for a,b,c in itertools.combinations(nodes,3):
                if tuple(sorted((int(core.types[a]),int(core.types[b]),int(core.types[c]))))!=target:continue
                edges=((a,b),(a,c),(b,c))
                if all(e in ids for e in edges):tri.append([ids[e] for e in edges])
            assert tri,'schema-feasible triangle group has no triples'
            self.triangle_edges.append(torch.tensor(tri,dtype=torch.long))

    def value_gradient(self, probability):
        prob=np.asarray(probability,dtype=np.float64)
        if not self.triangle_edges:return 0.,np.zeros_like(prob)
        p=torch.tensor(prob,dtype=torch.float64,requires_grad=True);values=[]
        for ix in self.triangle_edges:
            edges=p[ix];q=edges.prod(1);mean=q.sum()
            s=torch.zeros(self.m,dtype=p.dtype);ss=torch.zeros_like(s)
            for j in range(3):
                other=edges[:,(j+1)%3]*edges[:,(j+2)%3]
                s=s.index_add(0,ix[:,j],other)
                ss=ss.index_add(0,ix[:,j],other.square())
            # Distinct triangles can share at most one edge. Vertex-only overlap
            # is independent conditional on the predicted independent edge law.
            variance=(q-q.square()).sum()+(p*(1-p)*(s.square()-ss)).sum()
            values.append(mean.square()+variance)
        value=torch.stack(values).mean()
        derivative=torch.autograd.grad(value,p)[0].numpy()
        grad=derivative*prob*(1-prob)
        if not np.isfinite(grad).all() or not np.isfinite(value.item()):raise FloatingPointError('exact triangle moments')
        return float(value),grad
