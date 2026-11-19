"""Single CPU candidate: interpretable penalties on hard-graph statistics.

Soft scalar penalty, not quotas or node-identity matching. No model integration.
"""
import itertools
from collections import Counter
import numpy as np


class ExplicitScore:
    def __init__(self,core):
        self.core=core;self.ref=core.features(core.reference).numpy();self.cache={}
        self.classes=list(itertools.combinations_with_replacement(core.type_values,3))
        self.class_index={c:i for i,c in enumerate(self.classes)}
        self.mapping=[]
        for t,ids in core.nodes.items():
            ids=ids.numpy();d=self.ref[ids,:core.r];td=np.sort(d.sum(1));k=max(1,len(ids)//2)
            dm=core.masks[t][:core.r].numpy();tm=core.masks[t][core.r:core.base_width].numpy()
            tr=self.ref[ids,core.r:core.base_width][:,tm]
            self.cache[t]=dict(ids=ids,dm=dm,tm=tm,rd=np.sort(d[:,dm],axis=0),counter=Counter(map(tuple,d)),
                relation_scale=np.maximum(1.,d[:,dm].mean(0)),tri_scale=np.maximum(1.,tr.mean(0)),
                rt=np.sort(tr,axis=0),sorted_degree=td,k=k,
                low_scale=max(1.,float(td[:k].mean())),hub_scale=max(1.,float(td[-min(2,len(td)):].mean())))
            for j,pair in enumerate(core.triangle_pairs):
                self.mapping.append((ids,core.r+j,self.class_index[tuple(sorted((t,*pair)))]))
        self.reftri=self.triangle_counts(self.ref)[0]
        self.triangle_scale=max(1.,float(self.reftri.sum()))
        self.old_degree_scale=max(1.,float(self.ref[:,:core.r].sum()/core.n))

    def triangle_counts(self,raw):
        raw=np.asarray(raw);raw=raw[None] if raw.ndim==2 else raw
        out=np.zeros((len(raw),len(self.classes)))
        for ids,col,dest in self.mapping:out[:,dest]+=raw[:,ids,col].sum(1)/3
        return out

    def terms(self,raw):
        raw=np.asarray(raw);raw=raw[None] if raw.ndim==2 else raw
        b=len(raw);nt=len(self.cache)
        v={k:np.zeros(b) for k in ['joint_TV','relation_W1','node_triangle_W1','low','hub','total_degree_W1']}
        for c in self.cache.values():
            d=raw[:,c['ids'],:self.core.r];sd=np.sort(d.sum(-1),axis=1);diff=abs(sd-c['sorted_degree'])
            v['total_degree_W1']+=diff.mean(1)/nt
            v['relation_W1']+=(abs(np.sort(d[:,:,c['dm']],axis=1)-c['rd'])/c['relation_scale']).mean((1,2))/nt
            for i in range(b):
                z=Counter(map(tuple,d[i]));target=c['counter']
                v['joint_TV'][i]+=.5*sum(abs(z.get(k,0)-target.get(k,0)) for k in z.keys()|target.keys())/len(c['ids'])/nt
            tr=raw[:,c['ids'],self.core.r:self.core.base_width][:,:,c['tm']]
            if tr.shape[-1]:v['node_triangle_W1']+=(abs(np.sort(tr,axis=1)-c['rt'])/c['tri_scale']).mean((1,2))/nt
            v['low']+=diff[:,:c['k']].mean(1)/c['low_scale']/nt
            v['hub']+=diff[:,-min(2,len(c['ids'])):].mean(1)/c['hub_scale']/nt
        tri=self.triangle_counts(raw)
        v['typed_triangle_L1']=abs(tri-self.reftri).sum(1)/self.triangle_scale
        v['joint']=.5*(v['joint_TV']+v['relation_W1'])
        v['closure']=.5*(v['typed_triangle_L1']+v['node_triangle_W1'])
        v['total']=.25*(v['joint']+v['closure']+v['low']+v['hub'])
        v['old_DIRECT']=(v['total_degree_W1']/self.old_degree_scale+v['joint_TV']+abs(tri.sum(1)-self.reftri.sum())/self.triangle_scale)/3
        if not all(np.isfinite(z).all() for z in v.values()):raise FloatingPointError('nonfinite explicit score')
        return v

    def configuration(self):
        return dict(triangle_classes=self.classes,reference_triangles=self.reftri.tolist(),triangle_scale=self.triangle_scale,
            old_DIRECT_degree_scale=self.old_degree_scale,
            types={str(t):dict(nodes=len(c['ids']),low_ranks=c['k'],hub_ranks=min(2,len(c['ids'])),
                relation_scale=c['relation_scale'].tolist(),triangle_participation_scale=c['tri_scale'].tolist(),
                low_scale=c['low_scale'],hub_scale=c['hub_scale']) for t,c in self.cache.items()})


def connectivity(adj):
    n=len(adj);seen=np.zeros(n,dtype=bool);sizes=[]
    for v in range(n):
        if seen[v]:continue
        seen[v]=True;stack=[v];count=0
        while stack:
            u=stack.pop();count+=1
            nxt=np.flatnonzero(adj[u]&~seen);seen[nxt]=True;stack.extend(nxt.tolist())
        sizes.append(count)
    return dict(LCC=max(sizes,default=0),components=len(sizes),isolates=int((~adj.any(1)).sum()))


def connectivity_changes(core,bits,slots):
    q=core.query.numpy();n=core.n;a=np.zeros((n,n),dtype=bool);ref=np.zeros_like(a)
    a[q[0],q[1]]=bits.astype(bool);a|=a.T
    ref[q[0],q[1]]=core.reference.numpy().astype(bool);ref|=ref.T
    base=connectivity(a);target=connectivity(ref);out=[]
    for e in slots:
        i,j=q[:,e];a[i,j]=a[j,i]=not a[i,j];cur=connectivity(a);a[i,j]=a[j,i]=not a[i,j]
        out.append({k+'_error_change':abs(cur[k]-target[k])-abs(base[k]-target[k]) for k in base})
    return out,base,target
