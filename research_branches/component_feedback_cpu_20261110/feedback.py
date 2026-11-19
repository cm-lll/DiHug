"""Same squared component-count risk, conditional Bernoulli edge derivatives.

No neural model, graph editing policy, new score, or full generation. A single
bridge/component traversal obtains all pivotal-edge differences without trying
M separate edited graphs. Zero difference remains a legitimate observation.
"""
import numpy as np


def topology(n, q, bits):
    adjacency=[[] for _ in range(n)]
    for e in np.flatnonzero(bits):
        u,v=q[:,e];adjacency[u].append((int(v),int(e)));adjacency[v].append((int(u),int(e)))
    entered=np.full(n,-1,dtype=np.int64);low=np.zeros(n,dtype=np.int64)
    component=np.full(n,-1,dtype=np.int64);bridge=np.zeros(len(bits),dtype=bool)
    clock=0
    def visit(u,parent_edge,c):
        nonlocal clock
        entered[u]=low[u]=clock;clock+=1;component[u]=c
        for v,e in adjacency[u]:
            if e==parent_edge:continue
            if entered[v]<0:
                visit(v,e,c);low[u]=min(low[u],low[v])
                if low[v]>entered[u]:bridge[e]=True
            else:low[u]=min(low[u],entered[v])
    count=0
    for u in range(n):
        if entered[u]<0:visit(u,-1,count);count+=1
    return count,component,bridge


def differences(n,q,bits,reference_components):
    """Return F(B_-e,1)-F(B_-e,0) for every legal query e.

    k is the component count of the OBSERVED graph. When a present edge is a
    bridge, its removal raises k; when an absent edge crosses components, its
    addition lowers k. Self loops, directed/multiple edges are not supported.
    """
    k,labels,bridges=topology(n,q,bits)
    cross=(labels[q[0]]!=labels[q[1]]) & ~bits.astype(bool)
    d=np.zeros(len(bits),dtype=np.float64);den=max(1,n-1)**2
    d[cross]=(1-2*(k-reference_components))/den
    d[bridges]=(-1-2*(k-reference_components))/den
    return d,dict(components=int(k),bridges=int(bridges.sum()),cross_slots=int(cross.sum()),
                  cost=(k-reference_components)**2/den)


def estimate(n,q,p,bits,reference_components):
    assert q.shape==(2,len(p)) and bits.shape[1]==len(p)
    assert np.all(q[0]<q[1]) and len(set(map(tuple,q.T)))==len(p)
    assert np.isfinite(p).all() and np.all((p>=0)&(p<=1))
    assert np.isin(bits,[0,1]).all()
    diffs=[];records=[]
    for b in bits:
        d,r=differences(n,q,b,reference_components);diffs.append(d);records.append(r)
    diffs=np.asarray(diffs);costs=np.array([r['cost'] for r in records]);k=len(bits)
    centered=costs-costs[0];adv=(k*centered-centered.sum())/(k-1)
    loo=(adv[:,None]*(bits-p)).mean(0)
    prob_gradient=diffs.mean(0)
    logit_samples=diffs*p*(1-p)
    return dict(prob_gradient=prob_gradient,logit_gradient=logit_samples.mean(0),loo_gradient=loo,
                logit_standard_error=logit_samples.std(0,ddof=1)/np.sqrt(k),
                diffs=diffs,costs=costs,records=records)
