import itertools
import sys
from pathlib import Path
import numpy as np
import pytest
import torch
from scipy.special import logsumexp
PREV=Path(__file__).resolve().parent.parent/"role_condition_diagnosis_v1"
sys.path[:0]=[str(PREV/"baseline"),str(PREV),str(Path(__file__).resolve().parent)]
from capacity_readout import allocation_marginals,degree_logpmf,CapacityDenoiser
from test_late_fusion import fixture,SMALL,activate,all_queries,context,infer
from role_condition.model import extract_target
from late_fusion import LateConditionalDenoiser
from hsgd.bw_preflight import common as c

def make(case):
    torch.manual_seed(120)
    return CapacityDenoiser(case["n_types"],case["n_families"],
        family_pairs=case["space"].pairs,prior=case["prior"],T=64,**SMALL)

def brute(ll,counts):
    n,k=ll.shape
    assignments=[x for x in itertools.product(range(k),repeat=n)
                 if np.array_equal(np.bincount(x,minlength=k),counts)]
    scores=np.array([sum(ll[i,j] for i,j in enumerate(x)) for x in assignments])
    weights=np.exp(scores-logsumexp(scores));out=np.zeros_like(ll)
    for w,x in zip(weights,assignments):
        for i,j in enumerate(x):out[i,j]+=w
    return out

@pytest.mark.parametrize("counts",[[1,1],[2,1,1],[1,2,2]])
def test_exact_capacity_enumeration(counts):
    rng=np.random.default_rng(33);ll=rng.normal(size=(sum(counts),len(counts)))
    actual,_=allocation_marginals(ll,np.array(counts))
    np.testing.assert_allclose(actual,brute(ll,counts),atol=1e-12,rtol=0)

def test_node_and_token_permutation_including_ties():
    ll=np.array([[1.,3.,0.],[1.,3.,0.],[2.,0.,4.],[2.,0.,4.]])
    counts=np.array([2,1,1]);a,_=allocation_marginals(ll,counts)
    nodes=np.array([3,1,0,2]);cols=np.array([2,0,1])
    b,_=allocation_marginals(ll[nodes][:,cols],counts[cols])
    np.testing.assert_allclose(b,a[nodes][:,cols],atol=1e-12,rtol=0)
    np.testing.assert_allclose(a[0],a[1],atol=1e-12,rtol=0)

def test_fully_symmetric_counterexample_returns_frequency_not_artificial_hub():
    a,_=allocation_marginals(np.zeros((6,3)),np.array([4,1,1]))
    np.testing.assert_allclose(a,np.tile([4/6,1/6,1/6],(6,1)),atol=1e-12,rtol=0)

@pytest.mark.parametrize("retention",[0.,.4,1.])
def test_degree_law_exhaustive_and_exact_support(retention):
    N,d,pi=4,2,.3;prob=np.r_[np.repeat(retention+(1-retention)*pi,d),np.repeat((1-retention)*pi,N-d)]
    expected=np.zeros(N+1)
    for x in itertools.product([0,1],repeat=N):
        x=np.array(x);expected[x.sum()]+=np.prod(np.where(x,prob,1-prob))
    actual=np.exp(degree_logpmf(N,d,retention,pi))
    np.testing.assert_allclose(actual,expected,atol=1e-12,rtol=0)

def test_zero_projection_matches_old_late():
    a,b,xt=fixture();m=make(a);base=LateConditionalDenoiser(2,1,family_pairs=a["space"].pairs,prior=a["prior"],**SMALL)
    state=base.state_dict()
    for k,v in m.state_dict().items():state[k]=v.clone()
    base.load_state_dict(state);m.eval();base.eval()
    with torch.no_grad():
        x=infer(m,a,xt,extract_target(b));y=infer(base,a,xt,extract_target(b))
    for l,r in zip(x,y):assert torch.equal(l,r)

def test_nonzero_prediction_response_and_query_independent_context():
    a,b,xt=fixture();m=make(a);activate(m);m.eval();target=extract_target(b)
    with torch.no_grad():
        x=infer(m,a,xt,extract_target(a));y=infer(m,a,xt,target)
    assert (x[1]-y[1]).abs().max()>1e-5
    m.set_target(target);ctx=context(m,a,xt);q,f=all_queries(a)
    with torch.no_grad():
        ctx.predict(q[:,:8],f[:8],.5,"fixed")
        z=ctx.condition_diagnostic["role_context"].clone()
        ctx.predict(q[:,8:],f[8:],.5,"fixed")
        assert torch.equal(z,ctx.condition_diagnostic["role_context"])
    ctx.close()

def test_renumber_and_training_sampling_same_features():
    a,b,xt=fixture();m=make(a);activate(m);m.set_target(extract_target(b))
    q,f=all_queries(a);perm=np.array([2,0,3,1,7,5,4,6])
    from hsgd.runtime import predict
    m.train()
    x=predict(m,a["space"].types,xt,np.zeros(3,dtype=int),q,f,.5,details=True)
    m.eval()
    with torch.no_grad():
        same=infer(m,a,xt,extract_target(b))
        y=predict(m,a["space"].types,np.sort(perm[xt],axis=0),np.zeros(3,dtype=int),np.sort(perm[q],axis=0),f,.5,details=True)
    for l,r in zip(x,same):torch.testing.assert_close(l,r,atol=2e-6,rtol=1e-5)
    for l,r in zip(x,y):torch.testing.assert_close(l,r,atol=2e-6,rtol=1e-5)

def test_encoder_receives_nonzero_gradient_after_zero_projection_updates():
    a,b,xt=fixture();m=make(a);m.set_target(extract_target(b))
    optimizer=torch.optim.AdamW(m.parameters(),lr=2e-4,weight_decay=1e-12)
    inp=c.train_input(b,64,4,121,0,np.random.default_rng(121),force_t=32)
    seen=False
    for i in range(4):
        c.update(m,optimizer,b,inp,64,"cpu")
        g=m.branch.role_encoder[0].weight.grad
        seen=seen or (g is not None and torch.isfinite(g).all() and bool(torch.any(g!=0)))
    assert seen

def test_impossible_support_and_resource_limit_fail_explicitly():
    with pytest.raises(ValueError):allocation_marginals(np.full((2,2),-np.inf),np.array([1,1]))
    with pytest.raises(RuntimeError):allocation_marginals(np.zeros((20,20)),np.ones(20,dtype=int))

def test_joint_vectors_not_separate_marginal_assignment():
    # Two observations: same univariate role histograms, different joint roles.
    obs=np.array([[1,1],[0,0]])
    joint=np.array([[1,1],[0,0]]);cross=np.array([[1,0],[0,1]])
    def score(role):return -((obs[:,None,:]-role[None,:,:])**2).sum(2)*3
    a,_=allocation_marginals(score(joint),np.ones(2,dtype=int))
    b,_=allocation_marginals(score(cross),np.ones(2,dtype=int))
    assert a[0,0]>.99
    np.testing.assert_allclose(b,.5,atol=1e-12)
