import itertools
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


def load(p):
    with np.load(p) as z:
        t=z['types'].copy();s=z['schema'].copy();e=z['edges'].copy()
        # Reference NPZ relation labels reserve 0 for no-edge; generated families are zero-based.
        ef=z['relations'].copy()-1 if 'relations' in z else z['families'].copy()
    if e.shape[0]!=2:e=e.T
    pairs=[tuple(sorted(map(int,p))) for p in e.T]
    assert len(set(pairs))==len(pairs) and all(u!=v for u,v in pairs)
    for (u,v),r in zip(pairs,ef):assert tuple(sorted((int(t[u]),int(t[v]))))==tuple(s[r])
    a=np.zeros((len(t),len(t)),bool);a[e[0],e[1]]=True;a|=a.T
    return t,s,a

def bfs(a):
    lab=np.full(len(a),-1);cid=0
    for u in range(len(a)):
        if lab[u]>=0:continue
        stack=[u];lab[u]=cid
        while stack:
            v=stack.pop();new=np.flatnonzero(a[v]&(lab<0));lab[new]=cid;stack.extend(new)
        cid+=1
    return lab

def descriptor(t,lab):
    inventory=np.bincount(t);x=np.empty((len(t),len(inventory)))
    for c in np.unique(lab):
        ids=np.flatnonzero(lab==c);x[ids]=np.bincount(t[ids],minlength=len(inventory))/inventory
    return x

def dist(x,y,t):
    vals=[]
    for k in np.unique(t):
        ix=t==k;cost=cdist(x[ix],y[ix],metric='cityblock')/x.shape[1]
        i,j=linear_sum_assignment(cost);vals.append(cost[i,j].mean())
    return float(np.mean(vals))

PAIRS=list(itertools.combinations_with_replacement(range(4),2))

TRIPLES=list(itertools.combinations_with_replacement(range(4),3))

PI={p:i for i,p in enumerate(PAIRS)}

TI={p:i for i,p in enumerate(TRIPLES)}

def features(a,t,s):
    d=np.zeros((len(t),len(s)));tr=np.zeros((len(t),len(PAIRS)));tc=np.zeros(len(TRIPLES))
    rm={tuple(p):r for r,p in enumerate(s)}
    for u,v in zip(*np.nonzero(np.triu(a,1))):
        r=rm[tuple(sorted((t[u],t[v])))];d[u,r]+=1;d[v,r]+=1
        for w in np.flatnonzero(a[u]&a[v]&(np.arange(len(t))>v)):
            tc[TI[tuple(sorted((t[u],t[v],t[w])))]]+=1
            for n,i,j in ((u,v,w),(v,u,w),(w,u,v)):tr[n,PI[tuple(sorted((t[i],t[j])))]]+=1
    return d,tr,tc

def terms(f,ref,t,s):
    d,tr,tc=f;rd,rt,rtc=ref
    out={k:0. for k in ('joint','relation_W1','low','hub','node_triangle_W1')}
    for k in np.unique(t):
        ids=np.flatnonzero(t==k);cols=np.flatnonzero((s==k).any(1))
        x=d[ids][:,cols];r=rd[ids][:,cols];scale=np.maximum(1.,r.mean(0))
        cost=cdist(x/scale,r/scale,metric='cityblock')/len(cols);i,j=linear_sum_assignment(cost)
        out['joint']+=cost[i,j].mean()/4
        out['relation_W1']+=(abs(np.sort(x,axis=0)-np.sort(r,axis=0))/scale).mean()/4
        sr=np.sort(rd[ids].sum(1));diff=abs(np.sort(d[ids].sum(1))-sr);n=max(1,len(ids)//2)
        out['low']+=diff[:n].mean()/max(1.,sr[:n].mean())/4
        out['hub']+=diff[-2:].mean()/max(1.,sr[-2:].mean())/4
        out['node_triangle_W1']+=(abs(np.sort(tr[ids],axis=0)-np.sort(rt[ids],axis=0))/np.maximum(1.,rt[ids].mean(0))).mean()/4
    out['typed_triangle_L1']=float(abs(tc-rtc).sum()/max(1.,rtc.sum()))
    out['old_F']=(out['joint']+out['relation_W1']+out['typed_triangle_L1']+out['node_triangle_W1'])/8+(out['low']+out['hub'])/4
    return {k:float(v) for k,v in out.items()}
