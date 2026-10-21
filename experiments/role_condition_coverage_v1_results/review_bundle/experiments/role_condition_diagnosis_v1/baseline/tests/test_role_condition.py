import copy
import itertools
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from hsgd.bw_preflight import common as c
from hsgd.computational import FrozenTypedContext
from hsgd.runtime import predict,reverse_step
from hsgd.kernel import MarginalKernel
from hsgd.model import TypedSparseDenoiser
from role_condition.model import (RoleTarget,extract_target,capacities,degrees,paired_models,ATOL,RTOL)
from role_condition.panel import make_panel,augment,isomorphic,NAMES
from role_condition.experiment import opt,validate,fixed_validation

CFG=json.loads((Path(__file__).resolve().parents[1]/'role_condition/config.json').read_text())
torch.set_num_threads(1)

@pytest.fixture(scope='module')
def panel():return make_panel()

@pytest.fixture
def models(panel):return paired_models(panel[0][0]['case'],CFG,1221)

def ctx(model,case):
    return FrozenTypedContext(model,case['space'].types,case['clean'],case['space'].encode(case['clean'])[0],'test')

def nonzero(model):
    with torch.no_grad():
        if model.mode!='A':model.branch.degree_out.weight.fill_(.015)
        if model.mode=='C':torch.nn.init.normal_(model.branch.role_out.weight,std=.03)

def test_panel_degrees_CN_gap_and_disjoint_topologies(panel):
    records,targets=panel;assert len(records)==12
    for condition in ['balanced','extreme']:
        cases=[r['case'] for r in records if r['condition']==condition]
        for a,b in itertools.combinations(cases,2):assert not isomorphic(a,b)
        for case in cases:
            d=degrees(case['space'].types,case['clean'],case['space'].encode(case['clean'])[0],10)
            assert sorted(d[28:,8])==([4]*6 if condition=='balanced' else [1,1,1,1,2,18])
            assert sorted(d[10:28,8])==[1]*12+[2]*6
            assert case['counts'][8]==24
            g=[set() for _ in range(34)]
            for u,v in case['clean'].T:g[u].add(v);g[v].add(u)
            assert all(not(g[u]&g[v]) for u in range(10,28) for v in range(28,34))
    assert records[0]['case']['counts'].tolist()==records[-1]['case']['counts'].tolist()

def test_actual_shared_initialization_and_degree_branch_copy(models):
    a,b,d=models.values()
    for k,v in a.state_dict().items():assert torch.equal(v,b.state_dict()[k]) and torch.equal(v,d.state_dict()[k])
    for k,v in b.branch.state_dict().items():assert torch.equal(v,d.branch.state_dict()[k])

def test_zero_projection_matches_original_CN_exactly(panel,models):
    case=panel[0][0]['case'];target=panel[1]['balanced'];q=c.plan_for(case,11,4).block(0)
    baseline=TypedSparseDenoiser(4,10,family_pairs=case['space'].pairs,prior=case['prior'],**CFG['model'])
    baseline.load_state_dict(models['A'].state_dict());baseline.eval()
    values=[]
    with torch.no_grad():
        for model in [baseline,*models.values()]:
            model.eval()
            if hasattr(model,'set_target'):model.set_target(target if model.mode=='C' else None)
            values.append(predict(model,case['space'].types,case['clean'],case['space'].encode(case['clean'])[0],q[2],q[1],.5,details=True))
    for z,p in values[1:]:assert torch.equal(z,values[0][0]) and torch.equal(p,values[0][1])

def test_Xt_degrees_independent_query_and_overlap(panel,models):
    case=panel[0][0]['case'];model=models['B'];nonzero(model);model.eval();context=ctx(model,case)
    value=model.node_condition(context,.2)
    independent=np.zeros((34,10),int)
    for u,v in case['clean'].T:
        r=case['space'].encode(np.array([[u],[v]]))[0][0];independent[u,r]+=1;independent[v,r]+=1
    assert np.array_equal(independent,context.condition_diagnostic['degree'])
    # Existing true edge remains counted even when queried; absent candidates do not enter degrees.
    for ids,f,q in [c.plan_for(case,3,4).block(0),c.plan_for(case,17,4).block(1)]:
        context.predict(q,f,.2,'test')
        assert context.node_residual is value
        assert np.array_equal(independent,context.condition_diagnostic['degree'])
    assert model.branch_calls==1;context.close()

def test_roles_no_original_IDs_and_permutation(panel):
    records,targets=panel
    for rec in records:
        other=augment(rec['case'],np.random.default_rng(99))
        assert extract_target(other).fingerprint==targets[rec['condition']].fingerprint
    assert set(targets['extreme'].as_dict())=={'roles','type_counts','schema'}

def test_role_token_order_and_duplicate_merge(panel,models):
    target=panel[1]['extreme'];rows=[]
    for group in target.roles:
        modified=[]
        for vec,n in reversed(group):
            modified.extend([(vec,1),(vec,n-1)] if n>1 else [(vec,n)])
        rows.append(tuple(modified))
    other=RoleTarget(tuple(rows),target.type_counts,target.schema)
    assert target==other and target.fingerprint==other.fingerprint
    case=panel[0][0]['case'];model=models['C'];nonzero(model);model.eval()
    model.set_target(target)
    with torch.no_grad():
        left=model.node_condition(ctx(model,case),.3)
        model.set_target(other);right=model.node_condition(ctx(model,case),.3)
    assert torch.equal(left,right)

def test_role_condition_has_nonzero_actual_probability_response(panel,models):
    model=models['C'];nonzero(model);model.eval();case=panel[0][0]['case'];ids,f,q=c.plan_for(case,17,4).block(0);p=[]
    with torch.no_grad():
        for target in panel[1].values():
            model.set_target(target);p.append(predict(model,case['space'].types,case['clean'],case['space'].encode(case['clean'])[0],q,f,.3))
    assert float((p[0]-p[1]).abs().max())>ATOL

def test_permutation_equivariance_nonzero_branches(panel,models):
    case=panel[0][0]['case'];model=models['C'];nonzero(model);model.eval();model.set_target(panel[1]['extreme'])
    rng=np.random.default_rng(24);perm=np.arange(34)
    for t in range(4):ix=np.flatnonzero(case['space'].types==t);perm[ix]=rng.permutation(ix)
    other=c.make_case(case['space'].types,np.sort(perm[case['clean']],axis=0),case['space'].pairs)
    _,f,q=c.plan_for(case,17,4).block(0);qq=np.sort(perm[q],axis=0)
    with torch.no_grad():
        a=predict(model,case['space'].types,case['clean'],case['space'].encode(case['clean'])[0],q,f,.4)
        b=predict(model,other['space'].types,other['clean'],other['space'].encode(other['clean'])[0],qq,f,.4)
    assert torch.allclose(a,b,atol=ATOL,rtol=RTOL)

def test_cache_target_time_parameter_and_state_invalidation(panel,models):
    case=panel[0][0]['case'];model=models['C'];model.set_target(panel[1]['balanced']);context=ctx(model,case)
    model.node_condition(context,.3);model.set_target(panel[1]['extreme'])
    with pytest.raises(ValueError):model.node_condition(context,.3)
    model.set_target(panel[1]['balanced'])
    with pytest.raises(ValueError):model.node_condition(context,.4)
    with torch.no_grad():next(model.parameters()).add_(.001)
    with pytest.raises(ValueError):model.node_condition(context,.3)
    another=ctx(model,case);assert model.node_condition(another,.3) is not None
    another.close()
    with pytest.raises(ValueError):model.node_condition(another,.3)

def test_gradient_reaches_branch_without_extra_loss(panel,models):
    case=panel[0][0]['case'];model=models['C'];nonzero(model);model.set_target(panel[1]['balanced'])
    inp=c.train_input(case,64,4,17,1,np.random.default_rng(17),force_t=32)
    result=c.update(model,opt(model,CFG),case,inp,64,'cpu')
    assert result['optimizer_update']
    for name in ['degree_encoder.0.weight','degree_out.weight','role_encoder.0.weight','query.weight','role_out.weight']:
        p=dict(model.branch.named_parameters())[name]
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().max()>0,name

def test_joint_information_not_just_marginals():
    types=np.array([0,0,1,1,2,2]);pairs=((0,1),(0,2));cases=[]
    for e in [np.array([[0,0],[2,4]]),np.array([[0,1],[2,4]])]:cases.append(c.make_case(types,e,pairs))
    a,b=map(extract_target,cases)
    va=np.repeat(np.array([v for v,n in a.roles[0]]),[n for v,n in a.roles[0]],axis=0)
    vb=np.repeat(np.array([v for v,n in b.roles[0]]),[n for v,n in b.roles[0]],axis=0)
    assert np.array_equal(np.sort(va,axis=0),np.sort(vb,axis=0)) and a.fingerprint!=b.fingerprint
    cfg=dict(CFG['model']);from role_condition.model import ConditionalDenoiser
    model=ConditionalDenoiser(3,2,family_pairs=pairs,prior=cases[0]['prior'],mode='C',**cfg);nonzero(model);model.eval()
    with torch.no_grad():
        model.set_target(a);x=model.node_condition(ctx(model,cases[0]),.2).clone()
        model.set_target(b);y=model.node_condition(ctx(model,cases[0]),.2).clone()
    assert float((x-y).abs().max())>ATOL

def test_A_B_reject_condition_and_capacity_no_self_loops(panel,models):
    for arm in ['A','B']:
        with pytest.raises(ValueError):models[arm].set_target(panel[1]['balanced'])
    cap=capacities(panel[0][0]['case']['space'].types,panel[0][0]['case']['space'].pairs)
    assert cap[0,0]==3 and cap[3,9]==5 and cap[3,8]==18

def test_training_sampling_same_inputs_no_clean_identity(panel,models):
    case=panel[0][0]['case'];model=models['C'];nonzero(model);model.set_target(panel[1]['balanced'])
    _,f,q=c.plan_for(case,44,4).block(0);context=ctx(model,case)
    model.train()
    train=model.node_condition(context,.4).detach().clone();context.close();model.eval()
    with torch.no_grad():infer=model.node_condition(ctx(model,case),.4)
    assert torch.equal(train,infer)
    # Target object has no per-node clean degree array; frozen context accepts only Xt.
    model.set_target(panel[1]['extreme']);context=ctx(model,case)
    assert context.old.shape[0]==2
    assert not hasattr(model._target,'edges') and not hasattr(model._target,'node_ids')

def test_T1000_20_blocks_engineering_interface_only(panel,models):
    case=panel[0][0]['case'];state=(np.random.default_rng(17).random(case['space'].total)<np.repeat(case['prior'],case['space'].counts)).astype(np.uint8)
    for arm,model in models.items():
        model.eval();model.set_target(panel[1]['extreme'] if arm=='C' else None);before=model.branch_calls
        with torch.no_grad():result,report=reverse_step(model,case,state,MarginalKernel(case['prior'],c.schedule(1000)),1000,20,17,np.random.default_rng(17),'cpu')
        assert report['blocks']==20 and report['exact_coverage'] and report['remaining_time']==999
        assert model.branch_calls-before==int(arm!='A')

def test_validation_preserves_training_rng(panel,models):
    records,targets=panel;model=models['C'];torch.manual_seed(123);before=torch.get_rng_state().clone()
    values=fixed_validation(records,CFG)[:1];result=validate(model,values,CFG,targets,lambda:None)
    assert torch.equal(before,torch.get_rng_state()) and result['rows'][0]['counts_by_family_label']

@pytest.mark.parametrize('fixed',[0,1])
def test_target_exact_support_boundaries(fixed):
    from role_condition.model import ConditionalDenoiser
    types=np.array([0,0,1,1,2,2]);pairs=((0,1),(0,2));edges={(0,2),(0,3),(1,2),(1,3)}
    case=c.make_case(types,np.array(sorted(edges)).T,pairs)
    model=ConditionalDenoiser(3,2,family_pairs=pairs,prior=case['prior'],mode='C',**CFG['model'])
    model.set_target(extract_target(case))
    invalid=edges|{(0,4)} if fixed==0 else edges-{(0,2)}
    other=c.make_case(types,np.array(sorted(invalid)).T,pairs)
    with pytest.raises(ValueError):model.set_target(extract_target(other))
