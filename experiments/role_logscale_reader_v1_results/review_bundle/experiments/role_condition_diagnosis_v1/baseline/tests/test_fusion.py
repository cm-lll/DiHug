import itertools
import numpy as np
import pytest
import torch
from hsgd.fusion_checks import fixture,make,context,mechanism
from hsgd.computational import FrozenTypedContext


def test_joint_mechanism():
    assert mechanism()['status']=='passed'


def test_cache_time_parameter_and_wedge_limits():
    case,q,f=fixture();m,_=make(case,True);ctx=context(m,case)
    with torch.no_grad():ctx.predict(q,f,.2,'fixture')
    with pytest.raises(ValueError,match='different time'):ctx.predict(q,f,.3,'fixture')
    ctx.close()
    with pytest.raises(ValueError,match='closed'):ctx.predict(q,f,.2,'fixture')
    ctx=context(m,case)
    with torch.no_grad():m.joint_head[0].weight.add_(.1)
    with pytest.raises(ValueError,match='parameters changed'):ctx.predict(q,f,.2,'fixture')
    ctx.close();m.max_wedges=1
    with pytest.raises(RuntimeError,match='resource guard'):context(m,case)
    m.max_wedges=2000000;m.max_query_paths=1;ctx=context(m,case)
    with pytest.raises(RuntimeError,match='resource guard'):ctx.predict(q,f,.2,'fixture')
    ctx.close()


def test_mean_content_separate_from_log_count():
    types=np.array([0,0,1,1]);pairs=tuple(itertools.combinations_with_replacement(range(2),2))
    from hsgd.bw_preflight import common as c
    case=c.make_case(types,np.array([[0,1],[2,2]]),pairs);m,_=make(case,True);m.eval()
    one=context(m,case);two=context(m,case,np.array([[0,1,0,1],[2,2,3,3]]))
    with torch.no_grad():
        a=m.common_neighbor.aggregate(one,torch.tensor([[0],[1]]),torch.tensor([0]),.1)
        b=m.common_neighbor.aggregate(two,torch.tensor([[0],[1]]),torch.tensor([0]),.1)
    assert torch.allclose(a[:,:32],b[:,:32],atol=2e-6,rtol=0)
    assert a[0,33].item()==pytest.approx(np.log(2),abs=2e-6)
    assert b[0,33].item()==pytest.approx(np.log(3),abs=2e-6)
    one.close();two.close()


def test_old_checkpoint_rejected_and_empty_query():
    case,q,f=fixture();m,_=make(case,True)
    old={k:v for k,v in m.state_dict().items() if not k.startswith('joint_head.')}
    old['common_neighbor.readout.weight']=torch.zeros(1,35)
    with pytest.raises(RuntimeError):m.load_state_dict(old,strict=True)
    ctx=context(m,case)
    graph,l,p=ctx.predict(np.empty((2,0),dtype=int),np.empty(0,dtype=int),.1,'fixture',details=True)
    assert l.shape==(0,2) and p.shape==(0,)
    ctx.close()


def test_existing_role_guidance_remains_a_separate_t1_option():
    from pathlib import Path
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'role_degree'))
    from role_guidance import Layout,RoleGuide,freeze_target
    from role_statistics import joint_profile
    from role_sampling import reverse_step as guided_step
    from hsgd.kernel import MarginalKernel
    from hsgd.bw_preflight import common as c
    from hsgd.runtime import reverse_step
    case,_,_=fixture();layout=Layout.canonical(case['space'].types,case['space'].pairs,case['prior'])
    target=joint_profile(case['space'].types,case['clean'],case['space'].encode(case['clean'])[0],case['space'].pairs)
    guide=RoleGuide(layout,freeze_target(target,layout));kernel=MarginalKernel(case['prior'],c.schedule(1000))
    state=np.zeros(case['space'].total,dtype=np.uint8);state[case['clean_ids']]=1
    for enabled,off,on in [(False,'A','C'),(True,'B','D')]:
        m,_=make(case,enabled);m.eval()
        a,ra=reverse_step(m,case,state,kernel,1,3,17,np.random.default_rng(83),'cpu',trace=True)
        b,rb=guided_step(m,case,state,kernel,1,3,17,np.random.default_rng(83),'cpu',arm=off,trace=True)
        d,rd=guided_step(m,case,state,kernel,1,3,17,np.random.default_rng(83),'cpu',arm=on,guide=guide,trace=True)
        assert np.array_equal(a,b) and np.array_equal(ra['trace']['p0'],rb['trace']['p0'])
        assert np.array_equal(ra['trace']['p0'],rd['trace']['p0'])
        assert np.array_equal(ra['trace']['uniform'],rd['trace']['uniform'])
        assert np.max(abs(rd['trace']['q']-rd['trace']['p_guided']))<=1e-12
        assert all(p.grad is None for p in m.parameters())


def test_sampling_cannot_read_clean_identity_and_hidden_exposure_preserves_backbone():
    from hsgd.kernel import MarginalKernel
    from hsgd.bw_preflight import common as c
    from hsgd.runtime import reverse_step
    case,q,f=fixture();m,_=make(case,True);m.eval();ctx=context(m,case)
    graph=ctx.assemble(ctx.transfer_query(ctx.prepare_query(q,f,'fixture'),'fixture'),'fixture')
    with torch.no_grad():
        plain=m.backbone(ctx.x,graph.attrs,graph.edges,ctx.x.new_tensor([[.03]]),ctx.batch)
        exposed,hidden=m.backbone(ctx.x,graph.attrs,graph.edges,ctx.x.new_tensor([[.03]]),ctx.batch,return_hidden=True)
    assert torch.equal(plain.edge_attr,exposed.edge_attr) and torch.equal(plain.node,exposed.node)
    assert torch.equal(plain.y,exposed.y);ctx.close()
    state=np.zeros(case['space'].total,dtype=np.uint8);state[case['clean_ids']]=1
    class Forbidden:
        def __array__(self,*args,**kwargs):raise AssertionError('clean identities used in inference')
        def __iter__(self):raise AssertionError('clean identities used in inference')
    alt=dict(case,clean=Forbidden(),clean_ids=Forbidden())
    kernel=MarginalKernel(case['prior'],c.schedule(1000))
    x,a=reverse_step(m,case,state,kernel,30,3,17,np.random.default_rng(83),'cpu',trace=True)
    y,b=reverse_step(m,alt,state,kernel,30,3,17,np.random.default_rng(83),'cpu',trace=True)
    assert np.array_equal(x,y) and np.array_equal(a['trace']['p0'],b['trace']['p0'])


def test_path_memory_chunks_preserve_values_and_gradients():
    case,q,f=fixture();m,_=make(case,True);records=[]
    from hsgd.runtime import predict
    for chunk in [65536,1]:
        m.zero_grad(set_to_none=True);m.common_neighbor.chunk_paths=chunk
        logits,p=predict(m,case['space'].types,case['clean'],case['space'].encode(case['clean'])[0],q,f,.03,details=True)
        p.sum().backward()
        records.append((logits.detach().clone(),{n:x.grad.detach().clone() for n,x in m.named_parameters() if x.grad is not None}))
    assert torch.allclose(records[0][0],records[1][0],atol=2e-6,rtol=0)
    assert records[0][1].keys()==records[1][1].keys()
    for name in records[0][1]:
        assert torch.allclose(records[0][1][name],records[1][1][name],atol=2e-6,rtol=0),name
