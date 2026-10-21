"""Read-only planning arithmetic and tiny proof witnesses; no neural inference."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import networkx as nx
import numpy as np


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for x in iter(lambda:f.read(1024*1024),b""):h.update(x)
    return h.hexdigest()


def checked(path, expected):
    if not expected or sha(path)!=expected:raise ValueError("pinned input hash mismatch: "+str(path))


def degrees(types, edges, families, pairs):
    types=np.asarray(types);edges=np.asarray(edges);families=np.asarray(families)
    if edges.shape!=(len(families),2) or types.ndim!=1:raise ValueError("array shape")
    if not np.issubdtype(edges.dtype,np.integer):raise ValueError("integer endpoints required")
    if ((edges<0)|(edges>=len(types))).any() or (edges[:,0]>=edges[:,1]).any():raise ValueError("canonical simple endpoints required")
    if len(set(map(tuple,edges)))!=len(edges):raise ValueError("duplicate edge")
    if ((families<0)|(families>=len(pairs))).any():raise ValueError("family bounds")
    if not np.array_equal(np.sort(types[edges],axis=1),np.asarray(pairs)[families]):raise ValueError("relation/type mismatch")
    d=np.zeros((len(types),len(pairs)),dtype=np.int64)
    np.add.at(d,(edges[:,0],families),1);np.add.at(d,(edges[:,1],families),1)
    return d


def role_rows(d,types):
    return tuple(tuple((tuple(map(int,v)),int(c)) for v,c in sorted(Counter(map(tuple,d[types==t])).items())) for t in range(int(types.max())+1))


def dp_cost(counts):
    c=sorted(map(int,counts))
    if not c or min(c)<=0:raise ValueError("positive role multiplicities required")
    n=sum(c);k=len(c)
    size=math.prod(x+1 for x in c[:-1]) if k>1 else 1
    cells=(n+1)*size
    # k=1 is an actual special case; no F/B tables or transitions allocated.
    dense_bytes=16*cells if k>1 else 0
    return dict(nodes=n,roles=k,frequency_histogram=[list(x) for x in sorted(Counter(c).items())],
        states=str(size),cells=str(cells),guard_applies=k>1,guard_passed=(k==1 or cells<=200000),
        forward_backward_bytes_lower_bound=str(dense_bytes),
        state_coordinates_bytes=str(8*size*(k-1) if k>1 else 0),
        transition_index_bytes_upper_bound=str(16*n*k*size if k>1 else 0),
        transition_opportunities_upper_bound=str(n*k*size if k>1 else 0),
        likelihood_matrix_bytes=8*n*k,output_marginal_bytes=8*n*k,
        special_case="k1: O(n) ones; receipt cells is bookkeeping" if k==1 else None)


def gale_ryser(rows,cols):
    r=sorted(map(int,rows),reverse=True);c=sorted(map(int,cols),reverse=True)
    if min(r+c,default=0)<0 or sum(r)!=sum(c):return False,[]
    checks=[dict(k=k,left=sum(c[:k]),right=sum(min(x,k) for x in r)) for k in range(1,len(c)+1)]
    return all(x["left"]<=x["right"] for x in checks),checks


def flow_witness(rows,cols):
    rows=list(map(int,rows));cols=list(map(int,cols));ok,gr=gale_ryser(rows,cols)
    if not ok:raise ValueError("bipartite degree sequence infeasible")
    G=nx.DiGraph()
    for i,d in enumerate(rows):G.add_edge("source",("M",i),capacity=d)
    for j,d in enumerate(cols):G.add_edge(("G",j),"sink",capacity=d)
    for i in range(len(rows)):
        for j in range(len(cols)):G.add_edge(("M",i),("G",j),capacity=1)
    value,flow=nx.maximum_flow(G,"source","sink",flow_func=nx.algorithms.flow.edmonds_karp)
    if value!=sum(rows):raise ValueError("incomplete feasibility flow")
    edges=np.array([(i,j) for i in range(len(rows)) for j in range(len(cols)) if flow[("M",i)][("G",j)]],dtype=np.int64).reshape(-1,2)
    if not np.array_equal(np.bincount(edges[:,0],minlength=len(rows)),rows):raise AssertionError("witness row degree")
    if not np.array_equal(np.bincount(edges[:,1],minlength=len(cols)),cols):raise AssertionError("witness column degree")
    return edges,dict(max_flow=int(value),required=sum(rows),unit_edge_capacity=1,gale_ryser=gr,
        algorithm="NetworkX Edmonds-Karp integral maxflow, proof only",feasible=True)


def inventory(types,edges,families,pairs,names):
    d=degrees(types,edges,families,pairs);sizes=np.bincount(types);roles=role_rows(d,types)
    counts=np.bincount(families,minlength=len(pairs));cap=[];nodecap=np.zeros((len(sizes),len(pairs)),dtype=int)
    for j,(a,b) in enumerate(pairs):
        cap.append(int(sizes[a]*(sizes[a]-1)//2 if a==b else sizes[a]*sizes[b]))
        nodecap[a,j]=sizes[b]-int(a==b);nodecap[b,j]=sizes[a]-int(a==b)
    prior=counts/np.maximum(cap,1);active=counts>0;tables=set()
    typesout=[]
    for t,rows in enumerate(roles):
        for v,c in rows:
            for r,N in enumerate(nodecap[t]):
                if N:tables.add((int(N),int(v[r]),float(prior[r])))
        typesout.append(dict(type=t,name=names[t],nodes=int(sizes[t]),role_kinds=len(rows),
            roles=[dict(vector=list(v),frequency=c) for v,c in rows],capacity_by_relation=nodecap[t].tolist(),DP=dp_cost([c for v,c in rows])))
    memory=8*sum(N+1 for N,d,p in tables)
    result=dict(nodes=len(types),physical_edges=len(edges),schema=np.asarray(pairs).tolist(),type_counts=sizes.tolist(),types=typesout,
        relations=[dict(id=j,types=list(map(int,p)),names=[names[int(p[0])],names[int(p[1])]],edges=int(counts[j]),slots=cap[j],prior=float(prior[j]),active=bool(active[j])) for j,p in enumerate(pairs)],
        all_slots=sum(cap),active_slots=sum(c for c,on in zip(cap,active) if on),deterministic_empty_slots=sum(c for c,on in zip(cap,active) if not on),
        likelihood_tables_per_time=len(tables),logpmf_calls_per_time=sum(N+1 for N,d,p in tables),
        convolution_terms_per_time=sum((d+1)*(N-d+1) for N,d,p in tables),
        likelihood_cache_bytes_per_time=memory,likelihood_cache_bytes_1000_times=1000*memory,
        LL_lookup_additions_per_time=sum(len(rows)*int(sizes[t])*int((nodecap[t]>0).sum()) for t,rows in enumerate(roles)),
        likelihood_table_keys=[dict(N=N,clean_degree=d,prior=p) for N,d,p in sorted(tables)],
        reference_wedges=int(sum(int(x)*(int(x)-1)//2 for x in d.sum(1))))
    return result
