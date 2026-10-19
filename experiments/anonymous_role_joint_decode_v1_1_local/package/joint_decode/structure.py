"""Exact sparse discrete diagnostics, not constraints. No degree repair or graph draws."""
from collections import Counter
import time
import numpy as np


def evaluate(schema,bits,semantic_types,law=None,check=lambda:None):
    start=time.monotonic();bits=schema.bits(bits);s=schema;degree=s.degree(bits);adj=[set() for _ in range(s.n)]
    selected=np.flatnonzero(bits)
    for j,e in enumerate(selected):
        if j%1024==0:check()
        u,v=s.slots[:,e];adj[u].add(int(v));adj[v].add(int(u))
    triangles=Counter();triangle_total=0
    for u in range(s.n):
        check()
        for v in adj[u]:
            if v<=u:continue
            for w in adj[u].intersection(adj[v]):
                if w>v:
                    pattern=tuple(sorted([int(s.types[u]),int(s.types[v]),int(s.types[w])]))
                    triangles[pattern]+=1;triangle_total+=1
    seen=set();components=[]
    for u in range(s.n):
        check()
        if u in seen:continue
        stack=[u];seen.add(u);n=0
        while stack:
            v=stack.pop();n+=1
            for w in adj[v]:
                if w not in seen:seen.add(w);stack.append(w)
        components.append(n)
    director,actor,movie=(semantic_types[k] for k in ('director','actor','movie'))
    numerator=0;denominator=0
    for m in np.flatnonzero(s.types==movie):
        check();directors={v for v in adj[m] if s.types[v]==director};actors={v for v in adj[m] if s.types[v]==actor}
        denominator+=len(directors)*len(actors)
        numerator+=sum(len(adj[d].intersection(actors)) for d in directors)
    def distribution(values):
        count=Counter(map(int,values));n=len(values)
        return dict(counts={str(k):count[k] for k in sorted(count)},nodes=n,maximum=int(max(values,default=0)),
                    mean=float(np.mean(values)) if n else None,zero=int(count[0]),degree_le1=int(sum(v<=1 for v in values)),
                    P_gt153=float(sum(v>153 for v in values)/n) if n else None,
                    P90=float(np.quantile(values,.9,method='linear')) if n else None)
    total=degree.sum(1)
    joint={str(c):dict(rows=rows.tolist(),counts=counts.tolist()) for c in s.sizes
           for rows,counts in [np.unique(degree[s.types==c],axis=0,return_counts=True)]}
    result=dict(schema_simple_undirected=True,nodes=s.n,edges=int(bits.sum()),
        relation_edges=np.bincount(s.families,weights=bits,minlength=s.r).astype(int).tolist(),
        relation_endpoint_degree={str(r):distribution(degree[np.isin(s.types,[a,b]),r]) for r,(a,b) in enumerate(s.pairs)},
        total_degree=distribution(total),type_total_degree={str(c):distribution(total[s.types==c]) for c in s.sizes},
        type_relation_degree={str(c):{str(r):distribution(degree[s.types==c,r]) for r in range(s.r)} for c in s.sizes},
        joint_degree_frequencies=joint,triangles=triangle_total,
        typed_triangles={','.join(map(str,k)):v for k,v in sorted(triangles.items())},
        director_movie_actor_closure=dict(numerator=numerator,denominator=denominator,
            ratio=numerator/denominator if denominator else None,NA_reason=None if denominator else 'no D-M-A paths',
            definition='each D-M-A path once; closed iff its D-A edge exists; all edges from this discrete graph'),
        component_sizes=sorted(components,reverse=True),components=len(components),connected=len(components)==1,
        LCC=max(components,default=0),LCC_fraction=max(components,default=0)/s.n,
        isolates=int((total==0).sum()),diagnostic_seconds=time.monotonic()-start)
    if law is not None:
        supported=bool(law.allowed(bits));result['model_support']=supported
        result['log_probability']=float(law.log_prob(bits)) if supported else None
        result['log_probability_status']='finite' if supported else 'negative_infinity_support_violation'
    return result


def differences(schema,original,decoded):
    original=schema.bits(original).astype(bool);decoded=schema.bits(decoded).astype(bool)
    return {name:np.bincount(schema.families[mask],minlength=schema.r).tolist()
            for name,mask in dict(added=decoded&~original,deleted=original&~decoded,retained=original&decoded).items()}
