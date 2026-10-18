"""Exact typed common-neighbor features from existing Xt edges ONLY.

Topology enumerates unordered wedges once; no candidate-edge Python loop, no
NxNxN tensor, no truncation. A query looks up all matching wedges with sorted
keys. Contextual node features use the shared first Transformer layer on Xt,
without queries. Query-independent features may be cached inside ONE frozen Xt.
"""
import torch
from torch import nn
from torch.nn import functional as F


class Wedges:
    def __init__(self, context):
        self.n = len(context.types)
        device = context.device
        directed, family = context.directed, context.directed_families
        order = torch.argsort(directed[0]*self.n+directed[1], stable=True)
        edges, family = directed[:,order], family[order]
        degree = torch.bincount(edges[0], minlength=self.n)
        starts = degree.cumsum(0)-degree
        positions = torch.arange(edges.shape[1], device=device)
        after = degree[edges[0]]-(positions-starts[edges[0]])-1
        # Exact workload size on CPU from already validated physical endpoints.
        import numpy as np
        dc = np.bincount(context.old.ravel(), minlength=self.n)
        self.raw_wedges = int(np.sum(dc*(dc-1)//2))
        if self.raw_wedges > context.model.max_wedges:
            raise RuntimeError('resource guard: exact Xt wedges %d exceed %d; no truncation' % (self.raw_wedges,context.model.max_wedges))
        self.max_query_paths=context.model.max_query_paths
        first = torch.repeat_interleave(positions, after, output_size=self.raw_wedges)
        repeats_start = (after.cumsum(0)-after)[first]
        second = first+1+torch.arange(self.raw_wedges,device=device)-repeats_start
        u,v = edges[1,first],edges[1,second]
        keys = u*self.n+v  # Neighbors sorted; u < v, no immediate returns.
        perm = torch.argsort(keys,stable=True)
        self.keys = keys[perm]
        self.middle = edges[0,first][perm]
        self.left,self.right = family[first][perm],family[second][perm]
        self.bytes = sum(x.numel()*x.element_size() for x in (self.keys,self.middle,self.left,self.right))

    def lookup(self, query):
        lo,hi = torch.minimum(query[0],query[1]),torch.maximum(query[0],query[1])
        keys = lo*self.n+hi
        begin=torch.searchsorted(self.keys,keys,right=False)
        end=torch.searchsorted(self.keys,keys,right=True)
        counts=end-begin
        matched=int(counts.sum())
        if matched>self.max_query_paths:
            raise RuntimeError('resource guard: matched query paths %d exceed %d; no truncation' % (matched,self.max_query_paths))
        qid=torch.repeat_interleave(torch.arange(len(keys),device=query.device),counts,output_size=matched)
        offset=(counts.cumsum(0)-counts)[qid]
        wid=begin[qid]+torch.arange(len(qid),device=query.device)-offset
        return qid,self.middle[wid],self.left[wid],self.right[wid]


class CommonNeighborFeatures(nn.Module):
    def __init__(self, node_width,n_types,n_families,hidden=32):
        super().__init__()
        self.n_types,self.n_families,self.hidden=n_types,n_families,hidden
        # h_w, type(w), unordered {r_uw,r_wv}, target family, t/T.
        dim=node_width+n_types+2*n_families+1
        self.phi=nn.Sequential(nn.Linear(dim,hidden),nn.SiLU(),nn.Linear(hidden,hidden),nn.SiLU())
        self.chunk_paths=65536  # Memory batching only: every matched path included.

    def contextual_nodes(self,ctx,time):
        b=ctx.model.backbone
        X=b.lin_in_X(ctx.x)
        if not ctx.directed.numel():return X
        fam=ctx.directed_families
        attrs=torch.cat([F.one_hot(fam+1,ctx.model.n_families+1).float(),
                         F.one_hot(fam,ctx.model.n_families).float()],-1)
        E=b.lin_in_E(attrs)
        y=b.lin_in_y(torch.tensor([[float(time)]],device=ctx.device))
        # Same learned layer, on existing edges alone. No query-edge input.
        return b.tf_layers[0](X,ctx.directed,E,y,ctx.batch)[0]

    def aggregate(self,ctx,query,qf,time):
        qid,w,r1,r2=ctx.cn_wedges.lookup(query)
        features=ctx.x.new_zeros((query.shape[1],self.hidden+self.n_types+1))
        if not len(w):return features
        learned=ctx.x.new_zeros((query.shape[1],self.hidden))
        type_counts=ctx.x.new_zeros((query.shape[1],self.n_types))
        if ctx.cn_nodes is None:
            ctx.cn_nodes=self.contextual_nodes(ctx,time)
            ctx.cn_time=float(time)
        elif float(time)!=ctx.cn_time:
            raise ValueError("common-neighbor hidden cache used at a different time")
        for start in range(0,len(w),self.chunk_paths):
            sl=slice(start,start+self.chunk_paths)
            iw,iq=w[sl],qid[sl]
            rel=F.one_hot(r1[sl],self.n_families)+F.one_hot(r2[sl],self.n_families)
            z=torch.cat([ctx.cn_nodes[iw],ctx.x[iw],rel.float(),
                F.one_hot(qf[iq],self.n_families).float(),
                ctx.x.new_full((len(iw),1),float(time))],-1)
            learned.index_add_(0,iq,self.phi(z))
            type_counts.index_add_(0,iq,ctx.x[iw])
        counts=type_counts.sum(-1)
        return torch.cat([learned/counts.clamp_min(1).unsqueeze(-1),torch.log1p(type_counts),
                          (counts>0).to(learned.dtype).unsqueeze(-1)],-1)
