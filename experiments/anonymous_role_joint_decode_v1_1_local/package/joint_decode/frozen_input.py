"""Explicit LOCAL standard input only. No discovery, network, model or legacy auto-detection."""
import hashlib
import io
import json
from pathlib import Path
import numpy as np
from .contracts import Schema,EdgeLaw,validate_target

POSTERIOR_ATOL=5e-14  # unchanged old saved-q verification absolute tolerance
FORMAT='local_frozen_role_decode_v1'


def valid_sha(value):
    if not isinstance(value,str) or len(value)!=64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('explicit expected SHA256 required')
    return value


def checked(path,expected,check=lambda:None):
    valid_sha(expected);h=hashlib.sha256();chunks=[]
    with Path(path).open('rb') as f:
        while True:
            check();b=f.read(1024*1024)
            if not b:break
            h.update(b);chunks.append(b)
    check()
    if h.hexdigest()!=expected:raise ValueError('expected SHA256 mismatch: '+str(path))
    return b''.join(chunks)


def array_sha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def query_plan(schema,active,seed,check=lambda:None):
    """Exact local old UniformQueryPlan recipe: spawned families + balanced20 rounds.

    Complete query digest is raw int64 ids/families/2xQ endpoints in worker order.
    No importing Torch/common.py or evaluating old producer code.
    """
    permutations=[];cuts=[];offsets=np.r_[0,np.bincount(schema.families,minlength=schema.r).cumsum()]
    seeds=np.random.SeedSequence([seed,1,61]).spawn(schema.r)
    for r,child in enumerate(seeds):
        check();n=int(offsets[r+1]-offsets[r]) if active[r] else 0
        rng=np.random.default_rng(child);perm=rng.permutation(n);quota=np.full(20,n//20,dtype=np.int64)
        quota[rng.permutation(20)[:n%20]]+=1;permutations.append(perm);cuts.append(np.r_[0,quota.cumsum()])
    ids=[];families=[];queries=[];boundaries=[0];block=np.full(schema.s,-1,np.int64);digest=hashlib.sha256()
    for b in range(20):
        check();ii=np.concatenate([offsets[r]+p[cuts[r][b]:cuts[r][b+1]] for r,p in enumerate(permutations)]).astype(np.int64)
        ff=schema.families[ii].astype(np.int64);qq=np.ascontiguousarray(schema.slots[:,ii],dtype=np.int64)
        for a in (ii,ff,qq):digest.update(a.tobytes())
        if (block[ii]>=0).any():raise ValueError('duplicate reconstructed query')
        block[ii]=b;ids.append(ii);families.append(ff);queries.append(qq);boundaries.append(boundaries[-1]+len(ii))
    return dict(query_ids=np.concatenate(ids),query_families=np.concatenate(families),
                query_slots=np.concatenate(queries,axis=1),query_offsets=np.array(boundaries,np.int64),
                query_block=block,query_sha256=digest.hexdigest())


def canonical_map(schema,slots,families):
    slots=np.asarray(slots);families=np.asarray(families)
    if slots.shape!=(2,schema.s) or families.shape!=(schema.s,) or not np.issubdtype(slots.dtype,np.integer) or not np.issubdtype(families.dtype,np.integer):
        raise ValueError('complete slot/family dimensions and integer labels required')
    if ((slots<0)|(slots>=schema.n)).any() or (slots[0]>=slots[1]).any():raise ValueError('canonical unordered u<v endpoints required')
    keys=slots[0]*schema.n+slots[1];base=schema.slots[0]*schema.n+schema.slots[1]
    if len(np.unique(keys))!=schema.s:raise ValueError('duplicate input slots')
    order=np.argsort(base);locations=np.searchsorted(base[order],keys)
    if (locations>=schema.s).any() or not np.array_equal(base[order[locations]],keys):raise ValueError('missing/illegal schema slots')
    mapping=order[locations]
    if not np.array_equal(families,schema.families[mapping]):raise ValueError('wrong slot family')
    return mapping


class Frozen:
    def __init__(self,path,expected,arm,check=lambda:None):
        self.check=check;self.path=Path(path).resolve();self.root=self.path.parent;self.hashes={}
        self.meta=json.loads(self.file(self.path,expected));m=self.meta
        if m.get('format')!=FORMAT or arm not in ('A','B') or m.get('arm')!=arm or type(m.get('t')) is not int or m['t']!=1 or m.get('unguided') is not True:
            raise ValueError('only unguided A/B at t=1 is supported')
        if type(m.get('seed')) is not int or m['seed']<0 or type(m.get('T')) is not int or m['T']!=1000 or type(m.get('query_rounds')) is not int or m['query_rounds']!=20:
            raise ValueError('seed/T/query configuration required')
        if m.get('p0_semantics')!='clean_edge_present_probability' or m.get('endpoint_convention')!='unordered_u_lt_v' or m.get('relation_ids')!='zero_based_family_pairs_order':
            raise ValueError('probability/endpoint/relation semantics differs')
        for key in ('payload','target','source_receipt'):
            entry=m['files'][key];rel=Path(entry['path'])
            if rel.is_absolute():raise ValueError('relative local payload path required')
            p=(self.root/rel).resolve()
            if self.root not in p.parents:raise ValueError('payload escapes explicit local input root')
            self.file(p,entry['sha256'])
        def path(key):return (self.root/m['files'][key]['path']).resolve()
        self.receipt=json.loads(self.file(path('source_receipt'),m['files']['source_receipt']['sha256']))
        receipt=self.receipt
        for k in ('arm','seed','t','unguided','query_sha256'):
            if receipt.get(k)!=m.get(k):raise ValueError('source receipt binding differs: '+k)
        if receipt.get('source_kind') not in ('synthetic_fixture','historical_frozen_output'):
            raise ValueError('explicit source kind required')
        for k in ('weight_sha256','source_manifest_sha256','config_sha256','reference_sha256'):
            valid_sha(receipt.get(k))
        if receipt.get('weight_authentication')!='historical_receipt_only':raise ValueError('this offline reader does not authenticate private weight bytes')
        self.source_limit='沿用历史来源记录，未重新认证权重字节'
        if receipt['source_kind']=='synthetic_fixture':self.source_limit='合成工程夹具；不是实际模型输出或真实来源认证'
        artifact=json.loads(self.file(path('target'),m['files']['target']['sha256']));self.target=artifact['profile']
        if artifact.get('reference_sha256')!=receipt['reference_sha256']:raise ValueError('target/reference receipt differs')
        payload=self.file(path('payload'),m['files']['payload']['sha256'])
        with np.load(io.BytesIO(payload),allow_pickle=False) as z:
            required=('types','pairs','slots','families','state','p0','support','prior','alpha','query_block',
                      'query_ids','query_families','query_slots','query_offsets')
            if not set(required).issubset(z.files):raise ValueError('missing required standard arrays: '+str(set(required)-set(z.files)))
            a={k:z[k].copy() for k in z.files}
        for name in ('p0','prior','alpha',*(['q'] if 'q' in a else []),*(['clean_logits'] if 'clean_logits' in a else [])):
            if a[name].dtype!=np.dtype('float64'):raise ValueError('FP64 probability/schedule array required: '+name)
        if not np.issubdtype(a['support'].dtype,np.integer) or a['query_block'].dtype!=np.dtype('int64'):
            raise ValueError('integer support and int64 query blocks required')
        self.schema=s=Schema(a['types'],a['pairs']);validate_target(s,self.target)
        mapping=canonical_map(s,a['slots'],a['families']);self.mapping=mapping
        def canon(name):
            if a[name].shape!=(s.s,):raise ValueError('wrong complete slot dimension: '+name)
            out=np.empty_like(a[name]);out[mapping]=a[name];return out
        self.p0=np.asarray(canon('p0'),dtype=np.float64);self.state=canon('state');self.support=canon('support')
        self.prior=np.asarray(a['prior']);alpha=np.asarray(a['alpha'])
        if self.prior.shape!=(s.r,) or not np.isfinite(self.prior).all() or ((self.prior<0)|(self.prior>1)).any():raise ValueError('prior dimensions/range')
        if alpha.shape!=(1001,) or not np.isfinite(alpha).all() or alpha[0]!=1 or ((alpha<0)|(alpha>1)).any() or (np.diff(alpha)>0).any():raise ValueError('T1000 cumulative schedule contract')
        if receipt.get('alpha_sha256')!=array_sha(alpha):raise ValueError('schedule array binding')
        counts=np.bincount(s.families,minlength=s.r);edge_counts=np.zeros(s.r)
        for c,g in self.target['groups'].items():edge_counts+=(np.asarray(g['rows'])*np.asarray(g['counts'])[:,None]).sum(0)/2
        expected_prior=np.divide(edge_counts,counts,out=np.zeros(s.r),where=counts>0)
        if not np.array_equal(self.prior,expected_prior):raise ValueError('prior not target reference family density')
        structural=np.where(self.prior[s.families]==0,0,np.where(self.prior[s.families]==1,1,-1))
        if not np.array_equal(self.support,structural):raise ValueError('structural support not explained by exact prior')
        if not np.isfinite(self.p0).all() or ((self.p0<0)|(self.p0>1)).any() or not np.isin(self.state,[0,1]).all():raise ValueError('p0/state range')
        fixed=structural!=-1
        if not np.array_equal(self.p0[fixed],structural[fixed]) or not np.array_equal(self.state[fixed],structural[fixed]):raise ValueError('fixed support conflicts with p0/X1')
        free=~fixed;self.saturated=int(np.sum(free&((self.p0==0)|(self.p0==1))))
        score_kind=m.get('score_semantics')
        if score_kind=='recorded_probability':
            if self.saturated:raise ValueError('unexplained numerical saturation: finite clean logits are required')
            logits=np.zeros(s.s);logits[free]=np.log(self.p0[free])-np.log1p(-self.p0[free])
        elif score_kind=='finite_clean_logits':
            logits=np.asarray(canon('clean_logits'),dtype=np.float64)
            if not np.isfinite(logits).all() or receipt.get('logits_semantics')!='finite_clean_present_minus_absent':raise ValueError('finite clean-logit evidence missing')
            prob=np.exp(-np.logaddexp(0.,-logits[free]))
            if np.max(abs(prob-self.p0[free]),initial=0)>POSTERIOR_ATOL:raise ValueError('logits/p0 mismatch')
        else:raise ValueError('explicit scoring semantics required')
        self.law=EdgeLaw(logits,structural)
        self.q_error=None
        if 'q' in a:
            q=np.asarray(canon('q'),dtype=np.float64)
            if m.get('q_semantics')!='next_state_edge_present_probability' or not np.isfinite(q).all() or ((q<0)|(q>1)).any():raise ValueError('q semantics/range')
            self.q_error=float(np.max(abs(q-self.p0),initial=0))
            if self.q_error>POSTERIOR_ATOL:raise ValueError('t=1 saved q differs from clean p0')
        plan=query_plan(s,edge_counts>0,m['seed'],check)
        for name in ('query_ids','query_families','query_slots','query_offsets'):
            if a[name].dtype!=np.dtype('int64') or not np.array_equal(a[name],plan[name]):raise ValueError('full query order/coverage differs: '+name)
        if not np.array_equal(canon('query_block'),plan['query_block']) or m.get('query_sha256')!=plan['query_sha256']:
            raise ValueError('query full byte digest/block differs')
        self.semantic_types=m['semantic_types']
        semantic=[self.semantic_types[k] for k in ('director','actor','movie')]
        if len(set(semantic))!=3 or any(not isinstance(t,int) or t not in s.sizes for t in semantic):raise ValueError('explicit director/actor/movie type mapping required')
        self.rehash()

    def file(self,path,expected):
        path=Path(path).resolve();raw=checked(path,expected,self.check)
        if path in self.hashes and self.hashes[path]!=expected:raise ValueError('conflicting file pins')
        self.hashes[path]=expected;return raw

    def rehash(self):
        for path,h in self.hashes.items():checked(path,h,self.check)

    def original(self,path,sha,receipt_path,receipt_sha):
        receipt=json.loads(self.file(receipt_path,receipt_sha))
        expected=dict(arm=self.meta['arm'],seed=self.meta['seed'],t=0,unguided=True,
                      graph_stage='raw_final_t0',input_manifest_sha256=self.hashes[self.path],
                      source_receipt_sha256=self.meta['files']['source_receipt']['sha256'],graph_sha256=sha,
                      weight_sha256=self.receipt['weight_sha256'])
        if any(receipt.get(k)!=v for k,v in expected.items()):raise ValueError('original graph arm/seed/source binding differs')
        raw=self.file(path,sha)
        with np.load(io.BytesIO(raw),allow_pickle=False) as z:
            if not np.array_equal(z['types'],self.schema.types):raise ValueError('original graph type inventory differs')
            ids=self.schema.encode_edges(z['edges'])
            if not np.array_equal(z['families'],self.schema.families[ids]):raise ValueError('original graph families differ')
        bits=np.zeros(self.schema.s,np.uint8);bits[ids]=1
        if not self.law.allowed(bits):raise ValueError('original raw graph violates frozen model support')
        return bits
