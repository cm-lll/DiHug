"""Independent implementation of the baseline's typed binary-slot/statistic definitions."""
import hashlib
import json
import itertools
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
CONFIG=json.loads((ROOT/'frozen_config.json').read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


class Schema:
    def __init__(self,types,pairs):
        types=np.asarray(types);pairs=np.asarray(pairs)
        if types.ndim!=1 or not len(types) or not np.issubdtype(types.dtype,np.integer) or (types<0).any():raise ValueError('integer node types required')
        if pairs.ndim!=2 or pairs.shape[1]!=2 or not np.issubdtype(pairs.dtype,np.integer) or (pairs<0).any() or (pairs[:,0]>pairs[:,1]).any() or len(np.unique(pairs,axis=0))!=len(pairs):raise ValueError('unique unordered one-relation-per-type-pair schema required')
        self.types=types.astype(np.int64).copy();self.pairs=pairs.astype(np.int64).copy();self.n=len(types);self.r=len(pairs)
        slots=[];families=[]
        for r,(a,b) in enumerate(pairs):
            u=np.flatnonzero(types==a);v=np.flatnonzero(types==b)
            if a==b:
                i,j=np.triu_indices(len(u),1);e=np.stack([u[i],u[j]])
            else:e=np.sort(np.stack([np.repeat(u,len(v)),np.tile(v,len(u))]),axis=0)
            slots.append(e);families.append(np.full(e.shape[1],r,np.int64))
        self.slots=np.concatenate(slots,axis=1);self.families=np.concatenate(families);self.s=len(self.families)
        self.sizes={int(c):int((types==c).sum()) for c in np.unique(types)}
        for v in (self.types,self.pairs,self.slots,self.families):v.flags.writeable=False

    def bits(self,x):
        x=np.asarray(x)
        if x.shape!=(self.s,) or not np.isin(x,[0,1]).all():raise ValueError('complete binary canonical slot vector required')
        return x.astype(np.uint8)

    def encode_edges(self,edges):
        e=np.asarray(edges)
        if e.ndim!=2 or e.shape[0]!=2 or not np.issubdtype(e.dtype,np.integer) or ((e<0)|(e>=self.n)).any() or (e[0]>=e[1]).any():raise ValueError('canonical simple physical edges required')
        keys=self.slots[0]*self.n+self.slots[1];lookup={int(k):i for i,k in enumerate(keys)}
        if len(np.unique(e[0]*self.n+e[1]))!=e.shape[1]:raise ValueError('duplicate edge')
        try:return np.array([lookup[int(k)] for k in e[0]*self.n+e[1]],np.int64)
        except KeyError:raise ValueError('edge outside schema')

    def degree(self,x):
        x=self.bits(x);d=np.zeros((self.n,self.r),np.int64)
        for endpoint in self.slots:np.add.at(d,(endpoint,self.families),x.astype(np.int64))
        return d

    def profile(self,x):return profile_from_degrees(self,self.degree(x))


def profile_from_degrees(schema,degrees):
    d=np.asarray(degrees)
    if d.shape!=(schema.n,schema.r) or not np.issubdtype(d.dtype,np.integer) or (d<0).any():raise ValueError('joint integer degree matrix required')
    groups={}
    for c,n in schema.sizes.items():
        rows,count=np.unique(d[schema.types==c],axis=0,return_counts=True)
        groups[str(c)]=dict(nodes=n,rows=rows.tolist(),counts=count.tolist())
    return dict(format='anonymous-joint-relation-degree-v1',family_pairs=schema.pairs.tolist(),groups=groups)


def validate_target(schema,target):
    if target.get('format')!='anonymous-joint-relation-degree-v1' or target.get('family_pairs')!=schema.pairs.tolist() or set(target.get('groups',{}))!={str(c) for c in schema.sizes}:raise ValueError('target schema/type inventory mismatch')
    totals={}
    for c,n in schema.sizes.items():
        g=target['groups'][str(c)];v=np.asarray(g['rows']);k=np.asarray(g['counts'])
        if v.ndim!=2 or v.shape[1]!=schema.r or k.shape!=(len(v),) or not np.issubdtype(v.dtype,np.integer) or not np.issubdtype(k.dtype,np.integer) or (v<0).any() or (k<=0).any() or int(k.sum())!=n or g['nodes']!=n or len(np.unique(v,axis=0))!=len(v):raise ValueError('invalid complete integer role multiset')
        caps=np.array([schema.sizes.get(int(b if c==a else a),0)-(a==b) if c in (a,b) else 0 for a,b in schema.pairs])
        if (v>caps).any():raise ValueError('role exceeds legal capacity')
        totals[c]=(v*k[:,None]).sum(0)
    for r,(a,b) in enumerate(schema.pairs):
        if a==b:
            if totals[int(a)][r]%2:raise ValueError('odd same-type degree total')
        elif totals[int(a)][r]!=totals[int(b)][r]:raise ValueError('cross-type degree totals disagree')
    return True


def canonical_assignment(schema,target):
    validate_target(schema,target);d=np.zeros((schema.n,schema.r),np.int64)
    for c in schema.sizes:
        g=target['groups'][str(c)];rows=np.asarray(g['rows']);count=np.asarray(g['counts']);order=sorted(range(len(rows)),key=lambda j:tuple(rows[j]))
        d[schema.types==c]=np.repeat(rows[order],count[order],axis=0)
    return d


def validate_assignment(schema,target,d):
    validate_target(schema,target)
    normalized=dict(format=target['format'],family_pairs=target['family_pairs'],groups={})
    for c,g in target['groups'].items():
        order=sorted(range(len(g['rows'])),key=lambda j:tuple(g['rows'][j]))
        normalized['groups'][c]=dict(nodes=g['nodes'],rows=[g['rows'][j] for j in order],counts=[g['counts'][j] for j in order])
    if profile_from_degrees(schema,d)!=normalized:raise ValueError('one shared COMPLETE role assignment required across all relations')


class EdgeLaw:
    """Finite logits remain free even if a rounded sigmoid is exactly0/1.

    Actual structural zeros/ones use an EXPLICIT support vector (-1 free,0,1).
    from_probabilities interprets supplied exact0/1 as actual hard support.
    """
    def __init__(self,logits,support=None):
        z=np.asarray(logits,dtype=np.float64)
        if z.ndim!=1 or not np.isfinite(z).all():raise ValueError('finite logits required; hard support is separate')
        hard=np.full(z.shape,-1,np.int8) if support is None else np.asarray(support)
        if hard.shape!=z.shape or not np.isin(hard,[-1,0,1]).all():raise ValueError('invalid hard support')
        self.logits=z.copy();self.support=hard.astype(np.int8).copy();self.free=hard==-1
        self.logp=-np.logaddexp(0.,-z);self.log1mp=-np.logaddexp(0.,z)
        self.logp[hard==0]=-np.inf;self.log1mp[hard==0]=0.
        self.logp[hard==1]=0.;self.log1mp[hard==1]=-np.inf
        self.log_base=float(self.log1mp[self.free].sum())

    @classmethod
    def from_probabilities(cls,p):
        p=np.asarray(p,dtype=np.float64)
        if p.ndim!=1 or not np.isfinite(p).all() or ((p<0)|(p>1)).any():raise ValueError('invalid probabilities')
        hard=np.where(p==0,0,np.where(p==1,1,-1));z=np.zeros_like(p);free=hard==-1
        z[free]=np.log(p[free])-np.log1p(-p[free]);return cls(z,hard)

    def log_prob(self,x):
        x=np.asarray(x)
        if x.ndim not in (1,2) or x.shape[-1]!=len(self.logits) or not np.isin(x,[0,1]).all():raise ValueError('binary complete graph vector required')
        value=np.where(x,self.logp,self.log1mp).sum(axis=-1)
        if not np.isfinite(np.asarray(value)[np.asarray(self.allowed(x))]).all():
            raise FloatingPointError('finite-logit score overflow; not a structural support zero')
        return value

    def allowed(self,x):
        x=np.asarray(x);return np.all((self.support==-1)|(x==self.support),axis=-1)
