from collections import Counter
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
