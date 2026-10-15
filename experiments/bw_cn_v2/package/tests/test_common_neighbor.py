"""Frozen tolerances inherited v9. Small graphs are engineering fixtures only."""
import copy
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
import torch
from hsgd.bw_preflight import common as c
from hsgd.cn_checks import small_model,mechanism
from hsgd.computational import FrozenTypedContext
from hsgd.perf_checks import compare_float,compare_rng


def context(model,case,edges=None,token="test"):
    edges=case["clean"] if edges is None else edges
    return FrozenTypedContext(model,case["space"].types,edges,case["space"].encode(edges)[0],token)


def test_mechanism_smoke():assert mechanism()["status"]=="passed"


def test_exact_shared_vs_distinct_nodes():
    m,case,cfg=small_model();query=torch.tensor([[0],[1]])
    shared=context(m,case);a=shared.cn_wedges.lookup(query)
    assert a[1].tolist()==[2]
    e=case["clean"].copy();e[:,1]=[1,3]
    separate=context(m,case,e);b=separate.cn_wedges.lookup(query)
    assert b[1].numel()==0
    shared.close();separate.close()


def test_independent_wedge_enumeration_and_labels():
    m,case,cfg=small_model();ctx=context(m,case)
    query=np.array(list(itertools.combinations(range(len(case["space"].types)),2))).T
    qid,w,left,right=ctx.cn_wedges.lookup(torch.tensor(query))
    got=sorted(zip(qid.tolist(),w.tolist(),left.tolist(),right.tolist()))
    adj={i:{} for i in range(len(case["space"].types))}
    for (u,v),r in zip(case["clean"].T,case["space"].encode(case["clean"])[0]):adj[u][v]=int(r);adj[v][u]=int(r)
    expected=[]
    for qi,(u,v) in enumerate(query.T):
        for mid in adj[u].keys()&adj[v].keys():expected.append((qi,mid,adj[u][mid],adj[v][mid]))
    assert got==sorted(expected)
    assert len(got)==ctx.cn_wedges.raw_wedges
    ctx.close()


def test_contextual_features_not_just_type_counts():
    m,case,cfg=small_model();m.eval();q=torch.tensor([[0],[1]]);qf=torch.tensor(case["space"].encode(q.numpy())[0])
    with torch.no_grad():
        ctx=context(m,case);first=m.common_neighbor.aggregate(ctx,q,qf,.4)
        e=case["clean"].copy();e[:,3]=[3,5]  # Change node2's other-neighbor context; preserve common node2.
        other=context(m,case,e);second=m.common_neighbor.aggregate(other,q,qf,.4)
    assert torch.equal(first[:,-3:],second[:,-3:])
    assert not torch.allclose(first[:,:32],second[:,:32],atol=1e-7,rtol=0)
    ctx.close();other.close()


def test_absent_queries_do_not_enter_topology_and_partition_branch_invariant():
    m,case,cfg=small_model();m.eval();ctx=context(m,case);q=np.array([[0],[1]]);qf=case["space"].encode(q)[0]
    with torch.no_grad():
        a=m.common_neighbor.aggregate(ctx,torch.tensor(q),torch.tensor(qf),.5)
        extra=np.array([[0,0,1],[1,4,5]]);ef=case["space"].encode(extra)[0]
        ctx.predict(extra,ef,.5,"test")
        b=m.common_neighbor.aggregate(ctx,torch.tensor(q),torch.tensor(qf),.5)
    assert torch.equal(a,b)
    assert ctx.cn_wedges.lookup(torch.tensor(q))[1].tolist()==[2]
    ctx.close()


def test_endpoint_swap_features_exact():
    m,case,cfg=small_model();ctx=context(m,case)
    q=torch.tensor([[0,0],[1,4]]);f=torch.tensor(case["space"].encode(q.numpy())[0])
    a=m.common_neighbor.aggregate(ctx,q,f,.5);b=m.common_neighbor.aggregate(ctx,q.flip(0),f,.5)
    assert torch.equal(a,b);ctx.close()


def test_zero_common_neighbors_preserves_base_logit():
    m,case,cfg=small_model();q=np.array([[0],[3]]);f=case["space"].encode(q)[0]
    with torch.no_grad():m.common_neighbor.readout.weight.fill_(.03)
    ctx=context(m,case)
    assert not ctx.cn_wedges.lookup(torch.tensor(q))[1].numel()
    g=ctx.assemble(ctx.transfer_query(ctx.prepare_query(q,f,"test"),"test"),"test")
    assert torch.equal(m.common_neighbor(ctx,g,.5),torch.zeros(1))
    ctx.close()


def test_empty_graph_and_query():
    m,case,cfg=small_model();e=np.zeros((2,0),dtype=np.int64);ctx=context(m,case,e)
    assert ctx.cn_wedges.raw_wedges==0
    assert ctx.predict(e,np.zeros(0,dtype=np.int64),.5,"test").numel()==0
    q=np.array([[0],[1]]);f=case["space"].encode(q)[0]
    p=ctx.predict(q,f,.5,"test");assert p.isfinite().all()
    ctx.close()


def test_gradient_and_parameter_update():
    m,case,cfg=small_model();opt=torch.optim.AdamW(m.parameters(),**cfg["optimizer"])
    rng=np.random.default_rng(77)
    before=m.common_neighbor.readout.weight.detach().clone()
    e=case["clean"];q=np.array([[0,0],[1,4]]);f=case["space"].encode(q)[0]
    inp=dict(t=500,query=q,f=f,labels=np.array([0,1],dtype=np.float32),ids=np.array([0,1]),edges=e,
        edge_families=case["space"].encode(e)[0],ce_weight=.5,round_id=0,preparation_seconds=0,noise_seconds=0)
    r=c.update(m,opt,case,inp,1000,"cpu",require_parameter_change=False)
    assert not torch.equal(before,m.common_neighbor.readout.weight)
    r=c.update(m,opt,case,inp,1000,"cpu",require_parameter_change=False)
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in m.common_neighbor.phi.parameters())
    assert r["positive_count"]==1 and r["negative_count"]==1
    assert np.isfinite(r["positive_ce"]) and np.isfinite(r["negative_ce"])


def test_node_renumbering_with_nonzero_readout():
    m,case,cfg=small_model();m.eval()
    with torch.no_grad():m.common_neighbor.readout.weight.fill_(.013)
    perm=np.array([7,1,3,2,6,4,5,0]);types=np.empty_like(case["space"].types);types[perm]=case["space"].types
    assert np.array_equal(types,case["space"].types)
    q=np.array([[0,0],[1,4]]);f=case["space"].encode(q)[0]
    edges=np.sort(perm[case["clean"]],axis=0);newquery=np.sort(perm[q],axis=0)
    other=c.make_case(types,edges,case["space"].pairs)
    a=context(m,case);b=context(m,other)
    with torch.no_grad():
        _,la,pa=a.predict(q,f,.5,"test",details=True)
        _,lb,pb=b.predict(newquery,f,.5,"test",details=True)
    assert compare_float(la,lb,"logits")<5e-6
    assert compare_float(pa,pb,"renumber_p0")<2e-6
    a.close();b.close()


def test_cache_lifecycle_and_parameter_guard():
    m,case,cfg=small_model();ctx=context(m,case);q=np.array([[0],[1]]);f=case["space"].encode(q)[0]
    ctx.predict(q,f,.5,"test")
    with pytest.raises(ValueError,match="different time"):ctx.predict(q,f,.4,"test")
    with torch.no_grad():next(m.parameters()).add_(.01)
    with pytest.raises(ValueError,match="parameters changed"):ctx.predict(q,f,.5,"test")
    ctx.close()
    with pytest.raises(ValueError):ctx.predict(q,f,.5,"test")


def test_baseline_initialization_and_zero_readout_equivalence():
    cn,case,cfg=small_model();basecfg=copy.deepcopy(cfg);basecfg["model"]["cn_hidden"]=0
    c.seed_all(1601);base=c.make_model(case,basecfg,"cpu")
    for name,value in base.state_dict().items():assert torch.equal(value,cn.state_dict()[name])
    q=np.array([[0,0],[1,4]]);f=case["space"].encode(q)[0]
    a=context(cn,case);b=context(base,case)
    with torch.no_grad():
        _,la,pa=a.predict(q,f,.5,"test",details=True);_,lb,pb=b.predict(q,f,.5,"test",details=True)
    assert torch.equal(la,lb) and torch.equal(pa,pb)
    assert sum(p.numel() for p in cn.common_neighbor.parameters())==sum(p.numel() for p in cn.parameters())-sum(p.numel() for p in base.parameters())
    a.close();b.close()


def test_duplicate_illegal_and_fixed_support():
    m,case,cfg=small_model()
    with pytest.raises(ValueError):context(m,case,np.array([[0,0],[2,2]]))
    with pytest.raises(ValueError):context(m,case,np.array([[0],[0]]))
    # Type2-type2 family has exact zero prior. No epsilon opening via CN.
    ctx=context(m,case);q=np.array([[4],[5]]);f=case["space"].encode(q)[0]
    assert torch.equal(ctx.predict(q,f,.5,"test"),torch.zeros(1));ctx.close()


def test_metadata_role_triangle_closure_evaluator():
    from hsgd.cn_metrics import evaluate,imdb_roles
    types=np.array([0,1,2]);edges=np.array([[0,0,1],[1,2,2]]);pairs=((0,1),(0,2),(1,2))
    s=evaluate(types,edges,np.arange(3),pairs,(0,2,1))
    assert s["triangles"]==1 and s["director_movie_actor_closure"]["rate"]==1
    t=evaluate(types,edges[:,1:],np.array([1,2]),pairs,(0,2,1))
    assert t["triangles"]==0 and t["director_movie_actor_closure"]["rate"]==0
    assert imdb_roles(json.loads((c.PACKAGE/"data/IMDB_preprocessing_provenance.json").read_text()))==(0,2,1)


def test_empty_branch_adam_counts_are_participation_not_total():
    from hsgd.cn_optimizer import verify_steps
    m,case,cfg=small_model();opt=torch.optim.AdamW(m.parameters(),**cfg["optimizer"])
    history=[]
    for q in (np.array([[0],[3]]),np.array([[0],[1]])):
        inp=dict(t=500,query=q,f=case["space"].encode(q)[0],labels=np.zeros(q.shape[1],np.float32),
            ids=np.arange(q.shape[1]),edges=case["clean"],edge_families=case["space"].encode(case["clean"])[0],
            ce_weight=1.,round_id=0,preparation_seconds=0,noise_seconds=0)
        history.append(c.update(m,opt,case,inp,1000,"cpu",require_parameter_change=False))
    ck=c.checkpoint(m,opt,np.random.default_rng(1),"cpu",2,{},history)
    r=verify_steps(ck,m)
    assert r["participating_step_max"]==2 and r["participating_step_min"]==0
    actual=sorted(set(int(x["step"]) for x in opt.state.values()))
    assert actual==[1,2]
    corrupt=copy.deepcopy(ck)
    for value in corrupt["optimizer"]["state"].values():value["step"]+=1
    with pytest.raises(RuntimeError,match="participation count"):verify_steps(corrupt,m)


def test_training_sampling_same_input_path_and_hidden_cache_once():
    from hsgd.runtime import predict,StepPredictor,reverse_step
    from hsgd.kernel import MarginalKernel
    m,case,cfg=small_model();m.eval()
    with torch.no_grad():m.common_neighbor.readout.weight.fill_(.007)
    state=np.isin(np.arange(case["space"].total),case["clean_ids"]).astype(np.uint8)
    q=np.array([[0,0],[1,4]]);f,loc=case["space"].encode(q);ids=case["space"].offsets[f]+loc
    kernel=MarginalKernel(case["prior"],c.schedule(1000))
    with torch.no_grad():
        train_style=predict(m,case["space"].types,case["clean"],case["space"].encode(case["clean"])[0],q,f,.5)
        cb=StepPredictor(m,case,state,kernel,500)
        sample_style=cb(state,ids,f,q)
        # Sampling exports FP32 p0 to float64 for the original probability kernel.
        compare_float(train_style.double(),sample_style,"p0");cb.close()
        calls=[];hook=m.backbone.tf_layers[0].register_forward_hook(lambda *args:calls.append(1))
        _,rec=reverse_step(m,case,state,kernel,500,3,11,np.random.default_rng(12),"cpu")
        hook.remove()
        assert len(calls)==rec["blocks"]+1  # Main query passes + one Xt-only context pass.
