import sys
from pathlib import Path
import numpy as np
import pytest
import torch
HERE = Path(__file__).resolve().parent
PREV = HERE.parent/"role_condition_diagnosis_v1"
sys.path[:0] = [str(HERE), str(PREV), str(PREV/"baseline"), str(HERE.parent/"role_specialization_v1")]
from residual_model import DegreeResidualDenoiser, degree_inputs
from capacity_readout import CapacityDenoiser
from test_late_fusion import fixture, SMALL, all_queries, infer, context
from role_condition.model import extract_target
from hsgd.bw_preflight import common as c
from hsgd.runtime import predict, reverse_step
from hsgd.kernel import MarginalKernel

def models(case):
    torch.manual_seed(120)
    kw = dict(family_pairs=case["space"].pairs, prior=case["prior"], T=64, **SMALL)
    a = CapacityDenoiser(2, 1, **kw); b = DegreeResidualDenoiser(2, 1, **kw)
    b.load_state_dict({**b.state_dict(), **a.state_dict()}, strict=True)
    a.eval(); b.eval()
    return a, b

def activate(b):
    with torch.no_grad():
        b.degree_fusion.weight.copy_(torch.linspace(-.3, .3, b.degree_fusion.weight.numel()).reshape_as(b.degree_fusion.weight))

def test_noise_alignment_zero_capacity_and_extreme_time():
    cur=np.array([[2.,0.]]); soft=np.array([[4.,0.]]); cap=np.array([[8.,0.]])
    prior=np.array([.25,0.])
    np.testing.assert_allclose(degree_inputs(cur,soft,cap,prior,0),[[.5,0,0,0]],atol=0,rtol=0)
    np.testing.assert_allclose(degree_inputs(cur,soft,cap,prior,1),[[.5,0,-.25,0]],atol=0,rtol=0)
    with pytest.raises(ValueError):degree_inputs(cur*10,soft,cap,prior,.5)

def test_zero_projection_exact_logits_loss_and_reverse_rng():
    a,b,xt=fixture(); old,new=models(a); target=extract_target(b)
    with torch.no_grad():
        l=infer(old,a,xt,target);r=infer(new,a,xt,target)
    for x,y in zip(l,r): assert torch.equal(x,y)
    kernel=MarginalKernel(a["prior"],c.schedule(64)); inp=c.train_input(a,64,4,5,0,np.random.default_rng(5),force_t=32)
    old.set_target(target);new.set_target(target)
    # Build full canonical state from production forward-noise input.
    state=np.zeros(a["space"].total,dtype=np.uint8)
    fam, ids=a["space"].encode(xt); state[ids]=1
    ra=np.random.default_rng(19);rb=np.random.default_rng(19)
    with torch.no_grad():
        x,xi=reverse_step(old,a,state,kernel,32,4,19,ra,"cpu",trace=True)
        y,yi=reverse_step(new,a,state,kernel,32,4,19,rb,"cpu",trace=True)
    assert np.array_equal(x,y) and ra.bit_generator.state==rb.bit_generator.state
    np.testing.assert_array_equal(xi["trace"]["p0"],yi["trace"]["p0"])

def test_nonzero_response_renumber_train_eval_and_endpoint_symmetry():
    a,b,xt=fixture();old,m=models(a);activate(m);target=extract_target(b)
    q,f=all_queries(a);perm=np.array([2,0,3,1,7,5,4,6])
    m.set_target(target);m.train()
    x=predict(m,a["space"].types,xt,np.zeros(3,dtype=int),q,f,.5,details=True)
    m.eval()
    with torch.no_grad():
        same=infer(m,a,xt,target)
        y=predict(m,a["space"].types,np.sort(perm[xt],axis=0),np.zeros(3,dtype=int),np.sort(perm[q],axis=0),f,.5,details=True)
        other=infer(m,a,xt,extract_target(a))
    for l,r in zip(x,same):torch.testing.assert_close(l,r,atol=2e-6,rtol=1e-5)
    for l,r in zip(x,y):torch.testing.assert_close(l,r,atol=2e-6,rtol=1e-5)
    assert (same[1]-other[1]).abs().max()>1e-6
    z=torch.randn(8,2)
    assert torch.equal(m.endpoint_roles(z,q),m.endpoint_roles(z,q[::-1].copy()))

def test_cache_queries_target_switch_and_gradients():
    a,b,xt=fixture();old,m=models(a);m.set_target(extract_target(b));q,f=all_queries(a)
    ctx=context(m,a,xt)
    ctx.predict(q[:,:8],f[:8],.5,"fixed")
    saved=ctx.condition_diagnostic["explicit_degree_inputs"].clone()
    ctx.predict(q[:,8:],f[8:],.5,"fixed")
    assert torch.equal(saved,ctx.condition_diagnostic["explicit_degree_inputs"])
    assert np.array_equal(ctx.condition_diagnostic["degree"].sum(0),[6])
    m.set_target(extract_target(a))
    with pytest.raises(ValueError):ctx.predict(q,f,.5,"fixed")
    ctx.close()
    m.set_target(extract_target(b))
    inp=c.train_input(b,64,4,9,0,np.random.default_rng(9),force_t=32)
    op=torch.optim.AdamW(m.parameters(),lr=2e-4,weight_decay=1e-12)
    c.update(m,op,b,inp,64,"cpu")
    g=m.degree_fusion.weight.grad
    assert g is not None and torch.isfinite(g).all() and torch.any(g!=0)

def test_engineering_direction_not_a_learned_guarantee():
    a,b,xt=fixture();old,m=models(a)
    # Deliberately construct one MLP channel: over-connected residual suppresses
    # edge gap. This proves representability, not the learned direction.
    with torch.no_grad():
        for p in m.joint_head.parameters():p.zero_()
        m.role_fusion.weight.zero_();m.degree_fusion.weight.zero_()
        m.joint_head[0].bias[0]=2.
        m.degree_fusion.weight[0,1]=-1.
        m.joint_head[2].weight[1,0]=1.
    x=torch.zeros((1,m.joint_head[0].in_features));r=torch.zeros((1,64))
    d=torch.zeros((1,4));d[0,1]=.5
    low=m.readout(x,r,d);neutral=m.readout(x,r,torch.zeros_like(d))
    assert (low[:,1]-low[:,0]).item()<(neutral[:,1]-neutral[:,0]).item()
    # Zero projection is a valid counterexample: feature differences need not
    # change predictions. No automatic correct sign is built into the model.
    with torch.no_grad():m.degree_fusion.weight.zero_()
    assert torch.equal(m.readout(x,r,d),m.readout(x,r,torch.zeros_like(d)))
