"""Exact raw-graph statistics. MP counts use category products, not path sampling."""
from collections import Counter
import time
import numpy as np
from .model import degrees as raw_degrees

def degrees(case,edges=None):
    edges=case['clean'] if edges is None else edges
    return raw_degrees(case['space'].types,edges,case['space'].encode(edges)[0],case['n_families'])

def canon(x):return min(tuple(x),tuple(reversed(x)))

def paths(types,edges,families,check=lambda:None,max_ops=20000000):
    n=len(types);adj=[{} for _ in types]
    for (u,v),r in zip(edges.T,families):adj[u][int(v)]=int(r);adj[v][int(u)]=int(r)
    counters=[Counter((int(types[v]),r) for v,r in a.items()) for a in adj]
    out=[Counter(),Counter(),Counter()];ops=0;triangles=Counter()
    for u,a in enumerate(adj):
        check();items=list(counters[u].items())
        for i,((ta,ra),ca) in enumerate(items):
            for j in range(i,len(items)):
                (tb,rb),cb=items[j];count=ca*cb if j!=i else ca*(ca-1)//2
                if count:out[1][canon((ta,ra,int(types[u]),rb,tb))]+=count
                ops+=1
    for k,((u,v),r) in enumerate(zip(edges.T,families)):
        if k%100==0:check()
        u,v,r=int(u),int(v),int(r)
        out[0][canon((int(types[u]),r,int(types[v])))]+=1
        left=counters[u].copy();right=counters[v].copy()
        left[(int(types[v]),r)]-=1;right[(int(types[u]),r)]-=1
        for (ta,ra),ca in left.items():
            if not ca:continue
            for (tb,rb),cb in right.items():
                if cb:out[2][canon((ta,ra,int(types[u]),r,int(types[v]),rb,tb))]+=ca*cb
                ops+=1
        # Remove repeated end node (a-u-v-a); all other nodes are distinct already.
        common=adj[u].keys() & adj[v].keys()
        for w in common:
            out[2][canon((int(types[w]),adj[u][w],int(types[u]),r,int(types[v]),adj[v][w],int(types[w]))) ]-=1
            if u<v<w:triangles[tuple(sorted((int(types[u]),int(types[v]),int(types[w]))))]+=1
        ops+=len(common)
        if ops>max_ops:raise RuntimeError('exact MP operation budget exceeded; no approximation/truncation')
    out=[+x for x in out]
    return out,triangles,adj

def tv(a,b):
    sa,sb=sum(a.values()),sum(b.values())
    if not sa or not sb:return 0. if not(sa or sb) else 1.
    return .5*sum(abs(a.get(k,0)/sa-b.get(k,0)/sb) for k in a.keys()|b.keys())

def graph(case,edges,type_names,check=lambda:None):
    edges=np.asarray(edges,dtype=np.int64).reshape(2,-1)
    fam,local=case['space'].encode(edges)
    if (edges[0]>=edges[1]).any() or len(set(zip(fam,local)))!=edges.shape[1]:raise ValueError('non-simple/canonical graph')
    types=case['space'].types;d=degrees(case,edges);total=d.sum(1)
    mp,tr,adj=paths(types,edges,fam,check)
    seen=set();cc=[]
    for start in range(len(types)):
        if start in seen:continue
        todo=[start];seen.add(start);size=0
        while todo:
            node=todo.pop();size+=1
            for v in adj[node]:
                if v not in seen:seen.add(v);todo.append(v)
        cc.append(size)
    typed={};joint={}
    for typ,name in enumerate(type_names):
        ix=types==typ;rows=d[ix]
        joint[name]={','.join(map(str,k)):v for k,v in sorted(Counter(map(tuple,rows)).items())}
        typed[name]=dict(count=int(ix.sum()),by_relation=[dict(histogram=np.bincount(col).tolist(),
                mean=float(col.mean()),max=int(col.max()),zero=int((col==0).sum()),
                low_le1=int((col<=1).sum()),p90=float(np.quantile(col,.9)),p99=float(np.quantile(col,.99))) for col in rows.T])
    closure=None
    if type_names==['Director','Actor','Movie','Genre']:
        numerator=denominator=0;pairs=set();closed=set()
        for m in np.flatnonzero(types==2):
            directors=[u for u in adj[m] if types[u]==0];actors=[u for u in adj[m] if types[u]==1]
            for u in directors:
                for v in actors:
                    denominator+=1;pairs.add((u,v))
                    if v in adj[u]:numerator+=1;closed.add((u,v))
        closure=dict(path_numerator=numerator,path_denominator=denominator,
                     path_ratio=numerator/denominator if denominator else None,
                     unique_pair_numerator=len(closed),unique_pair_denominator=len(pairs),
                     unique_pair_ratio=len(closed)/len(pairs) if pairs else None)
    triangle_total=sum(tr.values());cluster=[]
    for u,a in enumerate(adj):
        links=sum(len(adj[v].keys() & a.keys()) for v in a)//2
        cluster.append(2*links/(len(a)*(len(a)-1)) if len(a)>1 else 0.)
    assort=None
    if edges.shape[1]:
        x=total[edges].astype(float);mean=float(x.mean());var=float((x*x).mean()-mean*mean)
        if var>0:assort=float((np.mean(x[0]*x[1])-mean*mean)/var)
    return dict(nodes=len(types),edges=edges.shape[1],family_edges=np.bincount(fam,minlength=case['n_families']).tolist(),
                schema_legal=True,simple_undirected=True,component_sizes=sorted(cc,reverse=True),connected=len(cc)==1,
                lcc=max(cc),isolates=int((total==0).sum()),total_degree_histogram=np.bincount(total).tolist(),
                maximum_degree=int(total.max()),typed_degree=typed,joint_relation_degree=joint,
                triangles=triangle_total,typed_triangles={str(k):v for k,v in sorted(tr.items())},
                average_clustering=float(np.mean(cluster)),assortativity=assort,
                mp=[{'counts':{','.join(map(str,k)):int(v) for k,v in sorted(x.items())},'total':int(sum(x.values())),
                     'per_node':sum(x.values())/len(types)} for x in mp],imdb_closure=closure,
                role_guidance=False,postprocessing=False)

def compare(candidate,ref):
    marginal={};joint={}
    for typ,a in candidate['typed_degree'].items():
        b=ref['typed_degree'][typ];values=[]
        for x,y in zip(a['by_relation'],b['by_relation']):
            hx,hy=x['histogram'],y['histogram'];n=max(len(hx),len(hy))
            px=np.pad(np.asarray(hx,float)/sum(hx),(0,n-len(hx)))
            py=np.pad(np.asarray(hy,float)/sum(hy),(0,n-len(hy)))
            values.append(float(np.abs(np.cumsum(px-py)).sum()))
        marginal[typ]=values
        joint[typ]=tv(Counter(candidate['joint_relation_degree'][typ]),Counter(ref['joint_relation_degree'][typ]))
    return dict(family_edge_difference=(np.array(candidate['family_edges'])-ref['family_edges']).tolist(),
                typed_relation_degree_W1=marginal,joint_relation_degree_TV=joint,
                mp_frequency_TV=[tv(Counter(a['counts']),Counter(b['counts'])) for a,b in zip(candidate['mp'],ref['mp'])],
                mp_total_absolute_error=[abs(a['total']-b['total']) for a,b in zip(candidate['mp'],ref['mp'])],
                length_share_TV=tv(Counter({i:a['total'] for i,a in enumerate(candidate['mp'])}),Counter({i:a['total'] for i,a in enumerate(ref['mp'])})))
