"""Exact sparse graph summaries with an oriented triangle enumeration."""
import numpy as np
from scipy import sparse


def context(graph,columns):
    n=graph.num_nodes;types=graph.node_types.tolist();nt=len(graph.type_names);nb=graph.adjacency
    index={(family,key):i for i,(family,key) in enumerate(columns)};values=[0]*len(columns)
    tri_lookup=[]
    for a in range(nt):
        for b in range(nt):
            for c in range(nt):tri_lookup.append(index.get(('typed_triangles',','.join(map(str,sorted((a,b,c))))),-1))
    rank=np.empty(n,dtype=np.int64)
    for i,u in enumerate(sorted(range(n),key=lambda u:(len(nb[u]),u))):rank[u]=i
    forward=[{v for v in nb[u] if rank[u]<rank[v]} for u in range(n)]
    for u in range(n):
        for v in forward[u]:
            prefix=(types[u]*nt+types[v])*nt
            for w in forward[u]&forward[v]:
                k=tri_lookup[prefix+types[w]];assert k>=0;values[k]+=1
    ts=np.asarray(types)
    for f,spec in sorted(graph.family_specs.items()):
        if spec.src_type==spec.dst_type:continue
        aa=np.flatnonzero(ts==spec.src_type);bb=np.flatnonzero(ts==spec.dst_type)
        amap=np.full(n,-1,dtype=np.int64);bmap=amap.copy();amap[aa]=np.arange(len(aa));bmap[bb]=np.arange(len(bb))
        edges=np.asarray(graph.family_edges[f],dtype=np.int64).reshape(-1,2)
        u=edges[:,0].copy();v=edges[:,1].copy();swap=ts[u]!=spec.src_type
        u[swap]=edges[swap,1];v[swap]=edges[swap,0]
        matrix=sparse.csr_matrix((np.ones(len(u),dtype=np.int64),(amap[u],bmap[v])),shape=(len(aa),len(bb)))
        if len(aa)>len(bb):matrix=matrix.T.tocsr()
        q=sparse.triu(matrix@matrix.T,k=1).data
        values[index[('cross_relation_four_cycles',f)]]=int(np.sum(q*(q-1)//2))
    unseen=set(range(n));sizes=[]
    while unseen:
        root=unseen.pop();stack=[root];count=1
        while stack:
            u=stack.pop()
            for v in nb[u]:
                if v in unseen:unseen.remove(v);stack.append(v);count+=1
        sizes.append(count)
    raw=np.array(values,dtype=np.int64)
    return (np.log1p(raw)/np.log1p(n)).astype(np.float32),dict(raw=values,components=len(sizes),lcc_fraction=max(sizes)/n)
