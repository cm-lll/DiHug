"""CPU mechanism tests; no optimizer steps or complete sampling chains."""
import itertools
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent/"baseline"))
import numpy as np
import pytest
import torch
from hsgd.bw_preflight import common as c
from hsgd.computational import FrozenTypedContext
from hsgd.runtime import predict
from role_condition.model import ConditionalDenoiser, extract_target, RoleTarget
from late_fusion import LateConditionalDenoiser

torch.set_num_threads(1)
ATOL,RTOL=2e-6,1e-5
SMALL=dict(layers=1,width=16,edge_width=8,global_width=8,heads=2,
           ff_node=16,ff_edge=8,ff_global=8,cn_hidden=32,joint_width=8,dropout=0.)


def fixture():
    types=np.repeat([0,1],4);pairs=((0,1),)
    balanced=c.make_case(types,np.array([(i,4+i) for i in range(4)]).T,pairs)
    extreme=c.make_case(types,np.array([(i,4) for i in range(4)]).T,pairs)
    xt=np.array([(0,4),(1,4),(2,5)]).T
    return balanced,extreme,xt


def model(case):
    torch.manual_seed(120)
    return LateConditionalDenoiser(case["n_types"],case["n_families"],
        family_pairs=case["space"].pairs,prior=case["prior"],**SMALL)


def context(m,case,edges):
    return FrozenTypedContext(m,case["space"].types,edges,case["space"].encode(edges)[0],"fixed")


def activate(m):
    # Engineering weights ONLY, not a trained correction.
    with torch.no_grad():
        m.role_fusion.weight.copy_(torch.linspace(-.15,.15,m.role_fusion.weight.numel()).reshape_as(m.role_fusion.weight))


def all_queries(case):
    return np.array([(u,v) for u in range(4) for v in range(4,8)]).T, np.zeros(16,dtype=int)


def infer(m,case,xt,target):
    m.set_target(target);q,f=all_queries(case)
    return predict(m,case["space"].types,xt,np.zeros(xt.shape[1],dtype=int),q,f,.5,details=True)


def test_zero_projection_matches_matched_B_and_explicit_parameter_copy():
    a,b,xt=fixture();m=model(a)
    baseline=ConditionalDenoiser(2,1,mode="B",family_pairs=a["space"].pairs,prior=a["prior"],**SMALL)
    state=baseline.state_dict()
    for k in state:state[k]=m.state_dict()[k].clone()
    baseline.load_state_dict(state,strict=True)
    m.eval();baseline.eval();q,f=all_queries(a)
    with torch.no_grad():
        x=infer(m,a,xt,extract_target(a))
        y=predict(baseline,a["space"].types,xt,np.zeros(3,dtype=int),q,f,.5,details=True)
    assert all(torch.equal(u,v) for u,v in zip(x,y))


def test_nonzero_branch_actual_logits_gap_and_p0_response():
    a,b,xt=fixture();m=model(a);activate(m);m.eval()
    with torch.no_grad():
        x=infer(m,a,xt,extract_target(a));y=infer(m,a,xt,extract_target(b))
    assert float(((x[0][:,1]-x[0][:,0])-(y[0][:,1]-y[0][:,0])).abs().max())>1e-5
    assert float((x[1]-y[1]).abs().max())>1e-5


def test_type_preserving_renumber_nonzero_path():
    a,b,xt=fixture();m=model(a);activate(m);m.eval();target=extract_target(b)
    perm=np.array([2,0,3,1,7,5,4,6]);q,f=all_queries(a)
    with torch.no_grad():
        x=infer(m,a,xt,target)
        m.set_target(target)
        y=predict(m,a["space"].types,np.sort(perm[xt],axis=0),np.zeros(3,dtype=int),
                  np.sort(perm[q],axis=0),f,.5,details=True)
    for u,v in zip(x,y):torch.testing.assert_close(u,v,atol=ATOL,rtol=RTOL)


def test_endpoint_symmetry_and_nonlinear_decision_mixed_response():
    a,b,xt=fixture();m=model(a);activate(m)
    node=torch.randn(8,32);q,_=all_queries(a)
    z=m.endpoint_roles(node,q);zz=m.endpoint_roles(node,q[::-1].copy())
    assert torch.equal(z,zz)
    f=torch.randn(16,m.joint_head[0].in_features);f2=f+torch.randn_like(f)*.2
    other=z+.2
    def gap(x,y):
        l=m.readout(x,y);return l[:,1]-l[:,0]
    mixed=(gap(f2,other)-gap(f2,z))-(gap(f,other)-gap(f,z))
    assert float(mixed.abs().max())>1e-5


def test_Xt_degrees_queries_and_cached_condition_block_invariance():
    a,b,xt=fixture();m=model(a);activate(m);m.set_target(extract_target(b));m.eval()
    ctx=context(m,a,xt);q,f=all_queries(a)
    with torch.no_grad():
        ctx.predict(q[:,:8],f[:8],.5,"fixed")
        x=ctx.condition_diagnostic["role_context"].clone()
        ctx.predict(q[:,8:],f[8:],.5,"fixed")
        assert torch.equal(x,ctx.condition_diagnostic["role_context"])
        assert m.branch_calls==1
        independent=np.bincount(xt.ravel(),minlength=8)
        assert np.array_equal(ctx.condition_diagnostic["degree"][:,0],independent)
    ctx.close()


@pytest.mark.parametrize("change",["target","time","parameter","closed"])
def test_cache_invalidations(change):
    a,b,xt=fixture();m=model(a);m.set_target(extract_target(a));ctx=context(m,a,xt)
    m.node_condition(ctx,.5);t=.5
    if change=="target":m.set_target(extract_target(b))
    elif change=="time":t=.4
    elif change=="parameter":
        with torch.no_grad():m.role_fusion.weight.add_(.01)
    else:ctx.close()
    with pytest.raises(ValueError):m.node_condition(ctx,t)


def test_training_sampling_inputs_and_gradient_no_updates():
    a,b,xt=fixture();m=model(a);activate(m);target=extract_target(b)
    m.train();x=infer(m,a,xt,target)
    loss=x[0].square().sum();loss.backward()
    for key in ["role_fusion.weight","branch.role_encoder.0.weight","branch.query.weight","branch.value.weight"]:
        p=dict(m.named_parameters())[key]
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().max()>0,key
    m.eval()
    with torch.no_grad():y=infer(m,a,xt,target)
    for u,v in zip(x,y):assert torch.equal(u,v)


def test_anonymous_token_order_duplicate_merge_and_actual_frequency_response():
    a,b,xt=fixture();m=model(a);activate(m);m.eval();target=extract_target(b)
    rows=[]
    for group in target.roles:
        rr=[]
        for vec,n in reversed(group):rr.extend([(vec,1)]*n)
        rows.append(tuple(rr))
    duplicate=RoleTarget(tuple(rows),target.type_counts,target.schema)
    assert duplicate.fingerprint==target.fingerprint
    assert set(target.as_dict())=={"roles","type_counts","schema"}
    with torch.no_grad():
        x=infer(m,a,xt,target);y=infer(m,a,xt,duplicate)
    for u,v in zip(x,y):assert torch.equal(u,v)


def test_complete_joint_roles_not_equal_marginal_histograms():
    types=np.array([0,0,1,1,2,2]);pairs=((0,1),(0,2))
    cases=[c.make_case(types,e,pairs) for e in [np.array([[0,0],[2,4]]),np.array([[0,1],[2,4]])]]
    targets=list(map(extract_target,cases));m=model(cases[0]);activate(m);m.eval()
    expanded=[np.repeat(np.array([v for v,n in t.roles[0]]),[n for v,n in t.roles[0]],axis=0) for t in targets]
    assert np.array_equal(np.sort(expanded[0],axis=0),np.sort(expanded[1],axis=0))
    values=[]
    with torch.no_grad():
        for target in targets:
            m.set_target(target);ctx=context(m,cases[0],cases[0]["clean"]);m.node_condition(ctx,.5)
            values.append(ctx.condition_diagnostic["role_context"].clone());ctx.close()
    assert float((values[0]-values[1]).abs().max())>1e-5


def oracle(condition,xt,alpha=.5):
    """Exact uniform clean prior on each anonymous role set, independent forward noise."""
    candidates=[]
    for assignment in itertools.product(range(4),repeat=4):
        hist=np.bincount(assignment,minlength=4)
        if (sorted(hist)==([1,1,1,1] if condition=="balanced" else [0,0,0,4])):
            z=np.zeros((4,4));z[np.arange(4),assignment]=1;candidates.append(z.ravel())
    bank=np.array(candidates);obs=np.zeros((4,4));obs[xt[0],xt[1]-4]=1
    p=alpha*bank+(1-alpha)*.25
    loglik=np.where(obs.ravel(),np.log(p),np.log1p(-p)).sum(1)
    weights=np.exp(loglik-loglik.max());weights/=weights.sum()
    return weights@bank


def directional_witness():
    a,b,xt=fixture();m=model(a)
    # Anonymous engineering encoder: SiLU(normalized joint degree), NOT IDs.
    # Fixed hand weights witness nonlinear interface capability, never learning.
    with torch.no_grad():
        for layer in [m.branch.role_encoder[0],m.branch.role_encoder[2],m.branch.query,m.branch.key,m.branch.value]:
            layer.weight.zero_();layer.bias.zero_()
        m.branch.role_encoder[0].weight[0,0]=1
        m.branch.role_encoder[2].weight[0,0]=1;m.branch.value.weight[0,0]=1
        m.joint_head[0].weight.zero_();m.joint_head[0].bias.zero_()
        m.joint_head[2].weight.zero_();m.joint_head[2].bias.zero_()
        m.role_fusion.weight.zero_()
        # g(x,c)=SiLU(x+c)-SiLU(x-c)-SiLU(c)+SiLU(-c).
        # Local evidence x is observed opposite-type degree minus 1.
        m.joint_head[0].weight[0,0]=1;m.joint_head[0].weight[1,0]=1
        m.role_fusion.weight[:4,0]=torch.tensor([1.,-1.,1.,-1.])
        m.joint_head[2].weight[1,:4]=torch.tensor([1.,-1.,-1.,1.])
    q,_=all_queries(a);f=torch.zeros(16,m.joint_head[0].in_features)
    d=np.bincount(xt.ravel(),minlength=8);f[:,0]=torch.tensor(d[q[1]]-1,dtype=torch.float32)
    p=[]
    with torch.no_grad():
        for target in [extract_target(a),extract_target(b)]:
            m.set_target(target);ctx=context(m,a,xt);m.node_condition(ctx,.5)
            z=m.endpoint_roles(ctx.condition_diagnostic["role_context"],q)
            p.append(m.readout(f,z).softmax(-1)[:,1].numpy());ctx.close()
    exact=oracle("extreme",xt)-oracle("balanced",xt);observed=p[1]-p[0]
    # These two degree-evidence classes have unambiguous oracle directions.
    high=d[q[1]]>1;low=d[q[1]]<1
    assert np.all(exact[high]>0) and np.all(exact[low]<0)
    assert np.all(observed[high]>1e-5) and np.all(observed[low]<-1e-5)
    return dict(oracle_difference=exact.tolist(),engineered_p0_difference=observed.tolist(),
                high_min=float(observed[high].min()),low_max=float(observed[low].max()),
                description="hand-weight nonlinear readout capability witness; NOT trained model efficacy")


def test_directional_response_against_enumerated_posterior():
    directional_witness()


def test_zero_fusion_opens_encoder_and_attention_after_engineering_updates():
    a,b,xt=fixture();m=model(a);m.set_target(extract_target(b))
    q,f=all_queries(a)
    inp=dict(t=2,ids=np.arange(16),f=f,query=q,
             labels=np.isin(np.arange(16),b["clean_ids"]).astype(np.float32),
             edges=xt,edge_families=np.zeros(xt.shape[1],dtype=int),ce_weight=1/16,
             round_id=0,preparation_seconds=0.,noise_seconds=0.)
    optimizer=torch.optim.AdamW(m.parameters(),lr=2e-4,weight_decay=1e-12)
    norms=[]
    for i in range(3):
        c.update(m,optimizer,a,inp,4,"cpu")
        row={}
        for name in ["branch.role_encoder.0.weight","branch.query.weight","branch.key.weight","role_fusion.weight"]:
            grad=dict(m.named_parameters())[name].grad
            assert grad is not None and torch.isfinite(grad).all()
            row[name]=float(grad.norm())
        norms.append(row)
    assert norms[0]["branch.role_encoder.0.weight"]==0
    for name in norms[-1]:assert norms[-1][name]>0


def test_symmetric_input_counterexample_no_forced_role_assignment():
    a,b,xt=fixture();empty=np.empty((2,0),dtype=int)
    # With alpha=0 and equal edge counts, both exact per-edge marginals are 1/4.
    np.testing.assert_allclose(oracle("balanced",empty,alpha=0),oracle("extreme",empty,alpha=0),atol=1e-14)
    m=model(a);activate(m);m.eval()
    with torch.no_grad():
        _,p=infer(m,a,empty,extract_target(b))
    assert torch.max(p)-torch.min(p)<ATOL
