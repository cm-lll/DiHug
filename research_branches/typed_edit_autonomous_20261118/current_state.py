"""Sparse exact current-graph motif counters updated by accepted legal edits."""
import numpy as np
from round06_fast_counts import context


class CurrentState:
    def __init__(self,graph,columns):
        self.columns=columns;self.index={(a,k):i for i,(a,k) in enumerate(columns)}
        self.types=graph.node_types.tolist();self.n=graph.num_nodes;self.specs=graph.family_specs;self.names=sorted(self.specs)
        self.edges={f:{tuple(sorted(e)) for e in es} for f,es in graph.family_edges.items()}
        self.nb=[set() for _ in range(self.n)];self.local={f:[set() for _ in range(self.n)] for f in self.names}
        for f,es in self.edges.items():
            for u,v in es:
                self.nb[u].add(v);self.nb[v].add(u);self.local[f][u].add(v);self.local[f][v].add(u)
        _,raw=context(graph,columns);self.counts=np.array(raw['raw'],dtype=np.int64)

    def features(self):return (np.log1p(self.counts)/np.log1p(self.n)).astype(np.float32)

    def connected(self):
        seen={0};stack=[0]
        while stack:
            u=stack.pop()
            for v in self.nb[u]:
                if v not in seen:seen.add(v);stack.append(v)
        return len(seen)==self.n

    def edge(self,f,u,v,sign):
        u,v=sorted((int(u),int(v)));assert u!=v
        spec=self.specs[f]
        assert sorted((self.types[u],self.types[v]))==sorted((spec.src_type,spec.dst_type))
        assert ((u,v) in self.edges[f])==(sign<0)
        for w in self.nb[u]&self.nb[v]:
            key=','.join(map(str,sorted((self.types[u],self.types[v],self.types[w]))))
            self.counts[self.index[('typed_triangles',key)]]+=sign
        local=self.local[f]
        if spec.src_type!=spec.dst_type:
            paths=sum(len((local[u]&local[w])-{v}) for w in local[v] if w!=u)
            self.counts[self.index[('cross_relation_four_cycles',f)]]+=sign*paths
        if sign<0:
            self.edges[f].remove((u,v));self.nb[u].remove(v);self.nb[v].remove(u);local[u].remove(v);local[v].remove(u)
        else:
            assert v not in self.nb[u]
            self.edges[f].add((u,v));self.nb[u].add(v);self.nb[v].add(u);local[u].add(v);local[v].add(u)
        assert np.all(self.counts>=0)

    def apply(self,values,index,reverse=False):
        if values['action'][index]==0:return
        assert values['action'][index]==4;f=self.names[values['family'][index]];degree={}
        fields=[('old_edges','old_valid',-1),('new_edges','new_valid',1)]
        if reverse:fields=[('new_edges','new_valid',-1),('old_edges','old_valid',1)]
        for field,valid,sign in fields:
            for uv,yes in zip(values[field][index],values[valid][index]):
                if not yes:continue
                u,v=uv;self.edge(f,u,v,sign);degree[u]=degree.get(u,0)+sign;degree[v]=degree.get(v,0)+sign
        assert all(x==0 for x in degree.values())

    def guarded(self,values,index):
        previous=self.counts.copy();self.apply(values,index);reject=not self.connected()
        if reject:
            self.apply(values,index,reverse=True);assert np.array_equal(self.counts,previous) and self.connected()
        return reject
