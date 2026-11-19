"""Independent CPU endpoint counts; no scorer or GPU metric dependency."""
from common import *
from itertools import combinations,combinations_with_replacement
from collections import Counter
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.stats import wasserstein_distance

def counts(g):
    n=g.num_nodes;t=np.asarray(g.node_types);pairs={tuple(sorted((s.src_type,s.dst_type))) for s in g.family_specs.values()}
    classes=[x for x in combinations_with_replacement(range(len(g.type_names)),3)
             if all(tuple(sorted(p)) in pairs for p in combinations(x,2))]
    tri={','.join(map(str,x)):0 for x in classes};nb=[set() for _ in range(n)]
    for es in g.family_edges.values():
        for u,v in es:nb[u].add(v);nb[v].add(u)
    for u in range(n):
        for v in nb[u]:
            if v<=u:continue
            for w in nb[u]&nb[v]:
                if w>v:tri[','.join(map(str,sorted((int(t[u]),int(t[v]),int(t[w])))))]+=1
    cycles={}
    for f,s in sorted(g.family_specs.items()):
        if s.src_type==s.dst_type:continue
        aa=np.flatnonzero(t==s.src_type);bb=np.flatnonzero(t==s.dst_type)
        am={int(u):i for i,u in enumerate(aa)};bm={int(u):i for i,u in enumerate(bb)}
        rr=[];cc=[]
        for u,v in g.family_edges[f]:
            if t[u]!=s.src_type:u,v=v,u
            rr.append(am[u]);cc.append(bm[v])
        b=sparse.csr_matrix((np.ones(len(rr),dtype=np.int64),(rr,cc)),shape=(len(aa),len(bb)))
        if len(aa)>len(bb):b=b.T.tocsr()
        q=sparse.triu(b@b.T,k=1).data
        cycles[f]=int(np.sum(q*(q-1)//2))
    rows=[];cols=[]
    for u,vs in enumerate(nb):
        for v in vs:rows.append(u);cols.append(v)
    a=sparse.csr_matrix((np.ones(len(rows),dtype=np.int64),(rows,cols)),shape=(n,n))
    nc,labels=connected_components(a,directed=False)
    return {'typed_triangles':tri,'cross_relation_four_cycles':cycles,'triangles':sum(tri.values()),
        'components':int(nc),'connected':bool(nc==1),'lcc_fraction':float(np.bincount(labels).max()/n),
        'degrees':np.diff(a.indptr).tolist()}

def compare(g,ref,reference_counts=None):
    a=counts(g);b=reference_counts or counts(ref)
    errors=[]
    for key in ['typed_triangles','cross_relation_four_cycles']:
        assert a[key].keys()==b[key].keys()
        errors.append(sum(abs(a[key][k]-b[key][k]) for k in b[key])/max(1,sum(b[key].values())))
    av=np.array(list(a['typed_triangles'].values()),float);bv=np.array(list(b['typed_triangles'].values()),float)
    tv=.5*(np.abs(av/max(1,av.sum())-bv/max(1,bv.sum())).sum()+abs(float(av.sum()==0)-float(bv.sum()==0)))
    a.update(primary_error=float(np.mean(errors)),triangle_vector_relative_l1=errors[0],cycle_vector_relative_l1=errors[1],
        typed_triangle_tv=float(tv),
        degree_w1=float(wasserstein_distance(a['degrees'],b['degrees'])),
        relation_degree_w1={f:float(wasserstein_distance(g.family_degrees(f),ref.family_degrees(f))) for f in sorted(g.family_specs)})
    return a

def fixtures():
    # Complete small graph with repeated types and bipartite K2,3, then exhaustive subsets.
    for ts in [[0,0,1,1,1],[0,0,0,1,1],[0,1,2,2,2]]:
        n=len(ts);nodes={t:tuple(i for i,x in enumerate(ts) if x==t) for t in set(ts)}
        specs={f'{a}-{b}':FamilySpec(f'{a}-{b}',a,b,nodes[a],nodes[b]) for a,b in combinations_with_replacement(sorted(nodes),2)}
        allpairs=list(combinations(range(n),2))
        for mask in [0,1,31,341,682,1023]:
            edges={f:[] for f in specs};selected={p for i,p in enumerate(allpairs) if mask&(1<<i)}
            for u,v in selected:edges['-'.join(map(str,sorted((ts[u],ts[v]))))].append((u,v))
            g=TypedGraph(np.array(ts),[str(t) for t in sorted(nodes)],specs,edges);z=counts(g)
            brute=Counter(tuple(sorted(ts[u] for u in vs)) for vs in combinations(range(n),3) if all(p in selected for p in combinations(vs,2)))
            assert z['triangles']==sum(brute.values())
            for k,v in z['typed_triangles'].items():assert v==brute[tuple(map(int,k.split(',')))]
            for f,s in specs.items():
                if s.src_type==s.dst_type:continue
                expected=sum(all(tuple(sorted((u,v))) in selected for u in aa for v in bb) for aa in combinations(nodes[s.src_type],2) for bb in combinations(nodes[s.dst_type],2))
                assert z['cross_relation_four_cycles'][f]==expected
    return {'passed':True,'fixtures':18,'method':'independent exhaustive triple and bipartite quadruple enumeration'}

if __name__=='__main__':
    setup();put(RUN/'metric_fixtures.json',fixtures())
    for ds in ['imdb','pubmed','freebase','acm']:
        g=graph(ds);v=counts(g);deg=np.array(v['degrees']);fc=np.array([len(x) for x in g.family_edges.values()],float);prob=fc/fc.sum()
        v['description']={'degree_mean':float(deg.mean()),'degree_cv':float(deg.std()/deg.mean()),'degree_max':int(deg.max()),
            'density':2*g.num_edges/(g.num_nodes*(g.num_nodes-1)),'active_relations':int((fc>0).sum()),
            'relation_entropy_nats':float(-sum(p*math.log(p) for p in prob if p>0)),
            'nonzero_typed_triangle_classes':sum(x>0 for x in v['typed_triangles'].values()),'allowed_typed_triangle_classes':len(v['typed_triangles'])}
        put(RUN/'data'/ds/'reference_metrics.json',v)
