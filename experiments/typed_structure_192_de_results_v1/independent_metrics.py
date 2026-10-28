"""CPU-only read-only independent checks of completed graphs and recorded contracts."""
import argparse
from collections import Counter
from pathlib import Path
import hashlib
import json
import math
import time
import numpy as np
import networkx as nx

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def close(a,b):
    assert math.isclose(float(a),float(b),abs_tol=1e-11,rel_tol=1e-11),(a,b)

def stats(types,edges,schema):
    edges=np.asarray(edges)
    assert edges.ndim==2 and edges.shape[0]==2
    assert np.all(edges[0]<edges[1]) and edges.min()>=0 and edges.max()<len(types)
    assert len(set(map(tuple,edges.T)))==edges.shape[1]
    lookup={tuple(map(int,p)):i for i,p in enumerate(schema)}
    fam=np.array([lookup[tuple(sorted((int(types[u]),int(types[v]))))] for u,v in edges.T])
    deg=np.zeros((len(types),len(schema)),dtype=np.int64)
    np.add.at(deg,(edges[0],fam),1);np.add.at(deg,(edges[1],fam),1)
    g=nx.Graph();g.add_nodes_from(range(len(types)));g.add_edges_from(edges.T)
    tr=Counter()
    for u in g:
        for v in g[u]:
            if v<=u:continue
            for w in set(g[u])&set(g[v]):
                if w>v:tr[str(tuple(sorted(map(int,types[[u,v,w]]))))]+=1
    comps=sorted(map(len,nx.connected_components(g)),reverse=True)
    # Number of unoriented simple length-3 paths: central-edge products minus triangles.
    mp2=sum(d*(d-1)//2 for _,d in g.degree())
    mp3=sum((g.degree(u)-1)*(g.degree(v)-1)-len(set(g[u])&set(g[v])) for u,v in g.edges())
    return dict(deg=deg,fam=fam,edges=edges.shape[1],triangles=sum(tr.values()),typed_triangles=tr,
                lcc=comps[0],isolates=len(list(nx.isolates(g))),component_sizes=comps,
                average_clustering=nx.average_clustering(g),maximum_degree=max(dict(g.degree()).values()),
                mp=[edges.shape[1],mp2,mp3])

def errors(types,schema,deg,ref):
    w1=joint=0.
    for t in np.unique(types):
        mask=types==t;rs=[r for r,p in enumerate(schema) if t in p];vals=[]
        for r in rs:
            a,b=deg[mask,r],ref[mask,r];size=int(max(a.max(),b.max()))+1
            vals.append(float(np.abs(np.cumsum(np.bincount(a,minlength=size)/len(a)-np.bincount(b,minlength=size)/len(b))).sum()))
        a,b=Counter(map(tuple,deg[mask])),Counter(map(tuple,ref[mask]));n=int(mask.sum())
        weight=n/len(types);w1+=weight*np.mean(vals)
        joint+=weight*.5*sum(abs(a[k]/n-b[k]/n) for k in a.keys()|b.keys())
    return w1,joint

