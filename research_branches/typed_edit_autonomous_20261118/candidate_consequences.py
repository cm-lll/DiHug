"""Exact bounded-candidate motif differences, using only the current graph."""
from current_state import CurrentState
import numpy as np


class Consequences(CurrentState):
    def __init__(self,graph,columns,counts=None):
        if counts is None:
            super().__init__(graph,columns)
        else:
            self.columns=columns;self.index={(a,k):i for i,(a,k) in enumerate(columns)}
            self.types=graph.node_types.tolist();self.n=graph.num_nodes;self.specs=graph.family_specs;self.names=sorted(self.specs)
            self.edges={f:{tuple(sorted(e)) for e in es} for f,es in graph.family_edges.items()}
            self.nb=[set() for _ in range(self.n)];self.local={f:[set() for _ in range(self.n)] for f in self.names}
            for f,es in self.edges.items():
                for u,v in es:
                    self.nb[u].add(v);self.nb[v].add(u);self.local[f][u].add(v);self.local[f][v].add(u)
            self.counts=np.array(counts,dtype=np.int64)
        self.scale=np.array([np.sqrt(self.n) if a=='typed_triangles' else self.n for a,k in columns],dtype=np.float64)

    def proposal(self,p,reverse=False):
        if p.action=='noop':return
        fields=[(p.old_edges,-1),(p.new_edges,1)]
        if reverse:fields=[(p.new_edges,-1),(p.old_edges,1)]
        for es,sign in fields:
            for u,v in es:self.edge(p.family,u,v,sign)

    def differences(self,proposals):
        before=self.counts.copy();out=[]
        for p in proposals:
            self.proposal(p);out.append(self.counts-before);self.proposal(p,True)
            assert np.array_equal(self.counts,before)
        return np.stack(out)

    def normalized(self,proposals):
        return (self.differences(proposals)/self.scale).astype(np.float32)
