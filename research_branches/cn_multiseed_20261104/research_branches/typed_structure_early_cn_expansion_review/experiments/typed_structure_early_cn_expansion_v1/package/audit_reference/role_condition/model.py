import hashlib
import json
import math
import time
from collections import Counter
from dataclasses import dataclass
import numpy as np
import torch
from torch import nn
from hsgd.model import TypedSparseDenoiser

ATOL=2e-6
RTOL=1e-5

def sha(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True,allow_nan=False).encode()).hexdigest()

def capacities(types,pairs):
    sizes=np.bincount(types);cap=np.zeros((len(sizes),len(pairs)),dtype=np.int64)
    for r,(a,b) in enumerate(pairs):
        cap[a,r]=sizes[b]-int(a==b);cap[b,r]=sizes[a]-int(a==b)
    return cap

def degrees(types,edges,families,R):
    d=np.zeros((len(types),R),dtype=np.int64)
    np.add.at(d,(edges[0],families),1);np.add.at(d,(edges[1],families),1)
    return d

@dataclass(frozen=True)
class RoleTarget:
    # Immutable tuple payload, NO original node IDs, row order or edge identity.
    roles: tuple
    type_counts: tuple
    schema: tuple

    def __post_init__(self):
        object.__setattr__(self,'type_counts',tuple(self.type_counts))
        object.__setattr__(self,'schema',tuple(tuple(x) for x in self.schema))
        if len(self.roles)!=len(self.type_counts):raise ValueError('type count mismatch')
        cap=capacities(np.repeat(np.arange(len(self.type_counts)),self.type_counts),self.schema)
        canonical=[]
        for typ,rows in enumerate(self.roles):
            merged=Counter()
            for vec,count in rows:
                if type(count) is not int or count<=0:raise ValueError('positive integer role frequencies required')
                if len(vec)!=len(self.schema) or any(type(x) is not int or x<0 for x in vec) or (np.array(vec)>cap[typ]).any():
                    raise ValueError('role vector/schema capacity mismatch')
                merged[tuple(vec)]+=count
            if sum(merged.values())!=self.type_counts[typ]:raise ValueError('role frequencies do not sum to type inventory')
            canonical.append(tuple(sorted(merged.items())))
        object.__setattr__(self,'roles',tuple(canonical))
        # Undirected endpoint stubs must agree across types, same-type totals even.
        sums=np.array([[sum(vec[r]*count for vec,count in rows) for r in range(len(self.schema))] for rows in canonical])
        for r,(a,b) in enumerate(self.schema):
            if (a==b and sums[a,r]%2) or (a!=b and sums[a,r]!=sums[b,r]):raise ValueError('role stub totals conflict')

    @property
    def fingerprint(self):return sha(self.as_dict())
    def as_dict(self):return dict(roles=self.roles,type_counts=self.type_counts,schema=self.schema)

def extract_target(case):
    d=degrees(case['space'].types,case['clean'],case['space'].encode(case['clean'])[0],case['n_families'])
    rows=tuple(tuple((tuple(map(int,v)),int(count)) for v,count in sorted(Counter(map(tuple,d[case['space'].types==t])).items()))
               for t in range(case['n_types']))
    return RoleTarget(rows,tuple(map(int,np.bincount(case['space'].types))),case['space'].pairs)

def features(d,cap):
    # ONLY schema capacity: no role-dependent normalization, RMS, or fitted mean.
    safe=np.maximum(cap,1)
    return np.concatenate([d/safe,np.log1p(d)/np.log1p(safe)],axis=-1).astype(np.float32)

class NodeCondition(nn.Module):
    def __init__(self,R,K,width,with_roles):
        super().__init__();self.with_roles=with_roles;self.h=32
        self.degree_encoder=nn.Sequential(nn.Linear(2*R+K+1,32),nn.SiLU(),nn.Linear(32,32),nn.SiLU())
        self.degree_out=nn.Linear(32,width);nn.init.zeros_(self.degree_out.weight);nn.init.zeros_(self.degree_out.bias)
        if with_roles:
            self.role_encoder=nn.Sequential(nn.Linear(2*R+2,32),nn.SiLU(),nn.Linear(32,32))
            self.query=nn.Linear(2*R+K+1,32);self.key=nn.Linear(32,32);self.value=nn.Linear(32,32)
            self.role_out=nn.Linear(32,width);nn.init.zeros_(self.role_out.weight);nn.init.zeros_(self.role_out.bias)

    def forward(self,ctx,t,target):
        start=time.perf_counter();types=ctx.types;K=ctx.model.n_types;R=ctx.model.n_families
        cap=capacities(types,ctx.model.family_pairs)
        d=degrees(types,ctx.old,ctx.of,R)
        xf=features(d,cap[types]);x=torch.as_tensor(np.c_[xf,np.eye(K,dtype=np.float32)[types],np.full((len(types),1),t,dtype=np.float32)],device=ctx.device)
        stats_seconds=time.perf_counter()-start;start=time.perf_counter()
        degree_delta=self.degree_out(self.degree_encoder(x));degree_seconds=time.perf_counter()-start
        role_context=None;role_seconds=0.;delta=degree_delta
        if self.with_roles:
            if target is None:raise ValueError('C requires an explicit anonymous target')
            if target.type_counts!=tuple(map(int,np.bincount(types))) or target.schema!=ctx.model.family_pairs:raise ValueError('condition inventory/schema mismatch')
            start=time.perf_counter();role_context=x.new_zeros((len(types),32))
            for typ,rows in enumerate(target.roles):
                ix=torch.as_tensor(np.flatnonzero(types==typ),device=ctx.device)
                vec=np.array([v for v,c in rows]);counts=np.array([c for v,c in rows]);freq=counts/target.type_counts[typ]
                inputs=np.c_[features(vec,cap[typ]),freq,np.log1p(counts)/np.log1p(target.type_counts[typ])].astype(np.float32)
                encoded=self.role_encoder(torch.as_tensor(inputs,device=ctx.device))
                q=self.query(x[ix]);k=self.key(encoded);v=self.value(encoded)
                # Explicit frequency prior; canonical duplicate merging is performed before encoding.
                attention=(q@k.T/math.sqrt(32)+torch.as_tensor(np.log(freq),device=ctx.device,dtype=x.dtype)).softmax(-1)
                role_context=role_context.index_copy(0,ix,attention@v)
            delta=delta+self.role_out(role_context);role_seconds=time.perf_counter()-start
        if not torch.isfinite(delta).all():raise FloatingPointError('nonfinite node condition')
        return delta,dict(degree=d.copy(),role_context=role_context,statistics_seconds=stats_seconds,
                         degree_network_seconds=degree_seconds,role_attention_seconds=role_seconds)

class ConditionalDenoiser(TypedSparseDenoiser):
    def __init__(self,*args,mode='A',**kwargs):
        super().__init__(*args,**kwargs)
        if mode not in ('A','B','C'):raise ValueError('A/B/C only')
        self.mode=mode;self._target=None;self.branch=None
        if mode!='A':self.branch=NodeCondition(self.n_families,self.n_types,self.backbone.lin_in_X.out_features,mode=='C')
        self.branch_calls=0;self.branch_timings=[]

    def set_target(self,target):
        if self.mode!='C' and target is not None:raise ValueError('A/B must not read target information')
        if target is not None:
            if target.schema!=self.family_pairs:raise ValueError('target schema mismatch')
            cap=capacities(np.repeat(np.arange(len(target.type_counts)),target.type_counts),target.schema)
            pi=self.prior.detach().cpu().numpy()
            for typ,rows in enumerate(target.roles):
                for vec,count in rows:
                    v=np.asarray(vec)
                    if (v[pi==0]!=0).any() or (v[pi==1]!=cap[typ,pi==1]).any():raise ValueError('target violates exact prior0/1 support')
        self._target=target

    def node_condition(self,ctx,t):
        if self.mode=='A':return None
        ctx.check(ctx.state_version)
        key=(float(t),self._target.fingerprint if self._target else None,self.training,torch.is_grad_enabled())
        if hasattr(ctx,'condition_key'):
            if ctx.condition_key!=key:raise ValueError('target/time/mode changed inside frozen context')
            return ctx.node_residual
        delta,diagnostic=self.branch(ctx,t,self._target)
        ctx.condition_key=key;ctx.node_residual=delta;ctx.condition_diagnostic=diagnostic
        self.branch_calls+=1
        self.branch_timings.append({k:diagnostic[k] for k in ('statistics_seconds','degree_network_seconds','role_attention_seconds')})
        return delta

    def joint_predict(self,ctx,graph,time,**kwargs):
        self.node_condition(ctx,time)
        return super().joint_predict(ctx,graph,time,**kwargs)

def paired_models(case,cfg,seed):
    from hsgd.bw_preflight import common as c
    c.seed_all(seed)
    a=ConditionalDenoiser(case['n_types'],case['n_families'],family_pairs=case['space'].pairs,prior=case['prior'],mode='A',**cfg['model'])
    b=ConditionalDenoiser(case['n_types'],case['n_families'],family_pairs=case['space'].pairs,prior=case['prior'],mode='B',**cfg['model'])
    cmodel=ConditionalDenoiser(case['n_types'],case['n_families'],family_pairs=case['space'].pairs,prior=case['prior'],mode='C',**cfg['model'])
    base=a.state_dict()
    for model in (b,cmodel):
        state=model.state_dict()
        for name,value in base.items():state[name]=value.clone()
        model.load_state_dict(state,strict=True)
    bc=cmodel.state_dict()
    for name,value in b.state_dict().items():bc[name]=value.clone()
    cmodel.load_state_dict(bc,strict=True)
    sb,sc=b.state_dict(),cmodel.state_dict()
    for name,value in base.items():
        if not torch.equal(value,sb[name]) or not torch.equal(value,sc[name]):raise AssertionError('shared parameters differ')
    degree_c=cmodel.branch.state_dict()
    for name,value in b.branch.state_dict().items():
        if not torch.equal(value,degree_c[name]):raise AssertionError('degree branch differs')
    return dict(A=a,B=b,C=cmodel)
