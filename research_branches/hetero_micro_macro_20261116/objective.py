"""CPU reference for anonymous, relation-ordered micro/macro descriptors.

No network, sampler or production loss changes. Independent categorical slots
permit multiple alternative relations on one pair, but not simultaneous edges.
Expected counts are exact; nonlinear distribution losses on those expectations
are soft plug-ins, NOT expected hard-graph losses.
"""
from itertools import product
import numpy as np
import torch


def make_plan(schema, ntypes):
    schema = np.asarray(schema, int)
    support = np.zeros((len(schema), ntypes, ntypes), dtype=np.int64)
    for r, (a,b) in enumerate(schema):
        support[r,a,b] = support[r,b,a] = 1
    pairs, triples, closures = [], [], []
    for r,s in product(range(len(schema)), repeat=2):
        reach = support[r] @ support[s]
        ends = np.flatnonzero(reach.sum(0)).tolist()
        if ends:
            pairs.append((r,s,ends))
        for u in range(len(schema)):
            reach3 = reach @ support[u]
            ends3 = np.flatnonzero(reach3.sum(0)).tolist()
            if ends3:
                triples.append((r,s,u,ends3))
            if r <= u and np.trace(reach3):
                closures.append((r,s,u))
    return dict(schema=schema.tolist(), ntypes=ntypes, pairs=pairs,
                triples=triples, closures=closures)


def validate(P, types, plan, fixed_zero=None):
    R,n,_=P.shape
    assert P.shape==(len(plan['schema']),len(types),len(types))
    assert torch.isfinite(P).all() and P.min()>=0 and P.max()<=1
    assert torch.allclose(P,P.transpose(1,2),atol=1e-12,rtol=0)
    assert torch.count_nonzero(torch.diagonal(P,dim1=1,dim2=2))==0
    assert torch.max(P.sum(0))<=1+1e-12, 'categorical relation mass exceeds one'
    ty=np.asarray(types)
    for r,(a,b) in enumerate(plan['schema']):
        mask=((ty[:,None]==a)&(ty[None,:]==b))|((ty[:,None]==b)&(ty[None,:]==a))
        assert torch.count_nonzero(P[r][torch.as_tensor(~mask)])==0, 'illegal schema slot'
    if fixed_zero is not None:
        assert torch.count_nonzero(P[torch.as_tensor(fixed_zero)])==0, 'fixed-zero support violated'


def descriptors(P, types, plan, check=True):
    """Relation degrees, typed anchored triangles, relation-ordered simple paths 2/3.

    Opposite orientations of anchored triangles are merged by r<=u; r==u
    counts are halved. Paths are directed from the anchor to a terminal type.
    Repeated vertices are removed algebraically, hence no repeated slot factors.
    All outputs equal their hard-feature conditional expectations under the
    independent-slot categorical distribution. Slots may share endpoints.
    """
    if check: validate(P,types,plan)
    n=len(types)
    T=torch.as_tensor(np.eye(plan['ntypes'])[types],dtype=P.dtype,device=P.device)
    degree=P.sum(2).T
    one=P@T
    two={}
    out2=[]; out3=[]; closed=[]
    for r,s,ends in plan['pairs']:
        raw=P[r]@one[s]
        back=(P[r]*P[s]).sum(1)
        two[r,s]=(raw,back)
        out2.append((raw-back[:,None]*T)[:,ends])
    # A product of the same edge probabilities is only used inside cancellation
    # of repeated-slot walks, never as an independence assumption on that slot.
    for r,s,u,ends in plan['triples']:
        raw_su,back_su=two[s,u]
        back_rs=two[r,s][1]
        triangle=((P[r]@P[s])*P[u]).sum(1)
        simple=(P[r]@raw_su-back_rs[:,None]*one[u]
                -P[r]@(back_su[:,None]*T)+(P[r]*P[s]*P[u])@T
                -triangle[:,None]*T)
        out3.append(simple[:,ends])
    for r,s,u in plan['closures']:
        c=((P[r]@P[s])*P[u]).sum(1)
        closed.append((c/(2 if r==u else 1))[:,None])
    empty=P.new_zeros((n,0))
    return dict(degree=degree,
                closure=torch.cat(closed,1) if closed else empty,
                propagation=torch.cat(out2+out3,1) if out2+out3 else empty)


def column_manifest(plan):
    cols={'degree':[dict(relation=r) for r in range(len(plan['schema']))],
          'closure':[dict(relations=[r,s,u],anchor='source node type',orientation_divisor=2 if r==u else 1)
                     for r,s,u in plan['closures']], 'propagation':[]}
    for r,s,ends in plan['pairs']:
        cols['propagation'] += [dict(length=2,relations=[r,s],terminal_type=b) for b in ends]
    for r,s,u,ends in plan['triples']:
        cols['propagation'] += [dict(length=3,relations=[r,s,u],terminal_type=b) for b in ends]
    return cols


def freeze_distribution(reference, types, seed=2026111601, frequencies=32):
    """Reference-only RMS scales and fixed random Fourier projections.

    Count floor 1 means one motif/path, not graph-size capacity. Joint degree
    is included in closure/path blocks to retain same-node associations.
    Frozen seed is a design choice, not selected on outputs.
    """
    rng=np.random.default_rng(seed); frozen={}
    for typ in np.unique(types):
        frozen[int(typ)]={}
        select=torch.as_tensor(np.asarray(types)==typ)
        for name in ('degree','closure','propagation'):
            x=reference[name]
            if name!='degree': x=torch.cat([reference['degree'],x],1)
            x=x[select].detach()
            scale=x.square().mean(0).sqrt().clamp_min(1)
            w=torch.as_tensor(rng.normal(size=(x.shape[1],frequencies))/np.sqrt(x.shape[1]),dtype=x.dtype)
            cf=torch.exp(1j*((x/scale)@w)).mean(0)
            frozen[int(typ)][name]=dict(scale=scale,omega=w,target=cf)
    return frozen


def distribution_errors(desc, types, frozen):
    """Anonymous joint distributions of expected node descriptors: soft plug-in."""
    rows=[]
    for typ,blocks in frozen.items():
        select=torch.as_tensor(np.asarray(types)==typ)
        for name,b in blocks.items():
            x=desc[name]
            if name!='degree': x=torch.cat([desc['degree'],x],1)
            cf=torch.exp(1j*((x[select]/b['scale'])@b['omega'])).mean(0)
            rows.append(dict(type=typ,block=name,error=(cf-b['target']).abs().square().mean()))
    return rows


def gaussian_profile(mean_square_error, min_log_std=-6.0):
    """GraphVAE-MM-inspired residual scale with finite exact-match limit.

    Inputs are real-coordinate MSEs (complex CF errors require dividing by 2).
    This is a descriptor pseudo-likelihood; not a diffusion ELBO or hard risk.
    Numerical clamp is explicit and is NOT inherited from upstream verbatim.
    It deliberately does not enforce a no-compensation constraint.
    """
    e=mean_square_error
    log_sigma=min_log_std+torch.nn.functional.softplus(.5*torch.log(e.clamp_min(1e-30))-min_log_std)
    return .5*e*torch.exp(-2*log_sigma)+log_sigma
