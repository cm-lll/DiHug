"""Independent held-out CF bank and full expected degree-PMF W1, CPU only.

No new evaluation frequencies are generated. Full probability recurrences,
not expected-degree substitutions or scientific graph sampling.
"""
import numpy as np
from stepwise_role.core import signature

CF_CONTRACT='squared_distance_of_expected_empirical_joint_CF_to_forward_target'
W1_CONTRACT='W1_of_expected_empirical_integer_degree_PMF_unscaled'


def validate_spec(spec,guide,reference_sha256):
    if spec.get('reference_sha256')!=reference_sha256 or spec.get('profile_sha256')!=signature(guide.frozen['payload']['profile']):
        raise ValueError('old evaluation bank reference/profile differs')
    if spec.get('CF_contract')!=CF_CONTRACT or spec.get('W1_contract')!=W1_CONTRACT:
        raise ValueError('historical metric definition missing/different; cannot silently substitute')
    c=str(spec['Genre_type']);w=np.asarray(spec['omega64'],float)
    if c not in guide.omega or w.shape!=(64,len(guide.layout.pairs)) or not np.isfinite(w).all():
        raise ValueError('actual original Genre64 frequency array required')
    if spec.get('CF_normalizer') not in (1.,4.):raise ValueError('explicit historical CF normalizer required')
    if not spec.get('original_artifact_sha256') or not spec.get('original_artifact_source'):
        raise ValueError('missing historical evaluation artifact provenance')
    return c,w


def full_pmf_rows(probabilities,check=None):
    """Exact Poisson-binomial coefficient DP in FP64, no FFT/MC/truncation.

    O(nodes*capacity^2) arithmetic, O(nodes*capacity) memory. Can be expensive;
    check() is called at every column. No hidden fallback or renormalization.
    """
    values=np.asarray(probabilities,float)
    if values.ndim!=2 or not np.isfinite(values).all() or ((values<0)|(values>1)).any():
        raise ValueError('invalid full PMF probabilities')
    n,m=values.shape;pmf=np.zeros((n,m+1));pmf[:,0]=1
    for k in range(m):
        if check:check()
        v=values[:,k,None]
        pmf[:,1:k+2]=pmf[:,1:k+2]*(1-v)+pmf[:,:k+1]*v
        pmf[:,0]*=1-v[:,0]
    error=float(np.max(abs(pmf.sum(1)-1),initial=0))
    if error>5e-11 or (pmf<0).any() or not np.isfinite(pmf).all():
        raise FloatingPointError('full PMF failed mass/positivity check')
    return pmf,error


def degree_w1(q,guide,s,check=None):
    l=guide.layout;result={};a=guide.alpha[s];pi=l.prior
    b=a+(1-a)*pi;z=(1-a)*pi
    for c,channels in guide.channels.items():
        n=l.sizes[int(c)];g=guide.frozen['payload']['profile']['groups'][c]
        rows=np.asarray(g['rows']);weights=np.asarray(g['counts'])/g['nodes']
        cap=np.asarray(guide.frozen['payload']['capacities'][c]);result[c]=[]
        for j,ch in enumerate(channels):
            if check:check()
            ids,node=guide.tail_layout[c][j];counts=np.bincount(node,minlength=n)
            capacity=int(cap[ch['families']].sum())
            if not (counts==capacity).all():raise ValueError('node-specific capacity differs; not uniformly masked')
            mat=np.zeros((n,capacity))
            start=np.r_[0,np.cumsum(counts)[:-1]]
            pos=np.arange(len(ids))-np.repeat(start,counts)
            mat[node,pos]=q[ids]
            predicted,ep=full_pmf_rows(mat,check);predicted=predicted.mean(0)
            clean_rows=np.zeros((len(rows),capacity));offset=0
            for r in ch['families']:
                for row,d in enumerate(rows[:,r]):
                    clean_rows[row,offset:offset+d]=b[r]
                    clean_rows[row,offset+d:offset+cap[r]]=z[r]
                offset+=cap[r]
            target,et=full_pmf_rows(clean_rows,check);target=weights@target
            w1=float(np.abs(np.cumsum(predicted)[:-1]-np.cumsum(target)[:-1]).sum())
            result[c].append(dict(families=ch['families'],kind=ch['kind'],W1=w1,
                                  predicted_mass_error=ep,target_mass_error=et,capacity=capacity,
                                  numerical_allowance=float(64*np.finfo(float).eps*capacity*(capacity+n+len(rows)+1)),
                                  allowance_scope='conservative FP64 operation-count allowance, not interval certification',
                                  expected_degree=float(predicted@np.arange(capacity+1)),
                                  target_expected_degree=float(target@np.arange(capacity+1))))
    return result


def genre64(q,guide,s,spec,check=None):
    c=str(spec['Genre_type']);w=np.asarray(spec['omega64'],float);l=guide.layout
    nodes=np.flatnonzero(l.types==int(c));local=np.full(len(l.types),-1,int);local[nodes]=np.arange(len(nodes))
    phi=np.ones((len(nodes),64),complex)
    for k in range(64):
        if check:check()
        factor=1+q*np.expm1(1j*w[k,l.families])
        for endpoint in l.slots:
            take=l.types[endpoint]==int(c)
            np.multiply.at(phi[:,k],local[endpoint[take]],factor[take])
    g=guide.frozen['payload']['profile']['groups'][c];rows=np.asarray(g['rows']);counts=np.asarray(g['counts'])/g['nodes']
    cap=np.asarray(guide.frozen['payload']['capacities'][c]);a=guide.alpha[s];pi=l.prior
    b=a+(1-a)*pi;z=(1-a)*pi;h=np.expm1(1j*w)
    target=(counts[:,None]*((1+b*h)[None,:,:]**rows[:,None,:]*(1+z*h)[None,:,:]**(cap-rows)[:,None,:]).prod(-1)).sum(0)
    value=float(np.mean(abs(phi.mean(0)-target)**2)/spec['CF_normalizer'])
    return dict(error=value,underflow_node_frequencies=int((phi==0).sum()),frequency_count=64)


def evaluate(q,guide,s,spec,check=None):
    guide.layout.probability(q)
    return dict(Genre64=genre64(q,guide,s,spec,check),expected_degree_PMF_W1=degree_w1(q,guide,s,check),
                expected_relation_edges=np.bincount(guide.layout.families,weights=q,minlength=len(guide.layout.pairs)).tolist(),
                meaning='W1 of expected empirical degree PMF, NOT expected W1 of a sampled graph')
