import json
import random
import numpy as np
import pytest
import torch
from pairing import config,construct,input_digest,uniform_contract,c,sha
from role_condition.experiment import opt,validate
from role_condition.model import extract_target
from run import activate


def small():
    cfg=config();cfg["model"]=dict(layers=1,width=16,edge_width=8,global_width=8,
        heads=2,ff_node=16,ff_edge=8,ff_global=8,dropout=.1,cn_hidden=32,joint_width=8)
    return c.toy_case(),cfg


def test_frozen_scope():
    cfg=config()
    assert cfg["train_seeds"]==[2026102011,2026102012,2026102013]
    assert cfg["updates"]==1000 and cfg["T"]==64 and cfg["rounds"]==4
    assert cfg["budget_seconds"]==3600 and cfg["expected_chains"]==96
    assert cfg["model"]["layers"]==8 and cfg["model"]["width"]==256
    assert cfg["optimizer"]["lr"]==2e-4 and cfg["gradient_clipping"] is None


def test_actual_parameter_copy_independent_optimizers_saved_weights(tmp_path):
    case,cfg=small();models,receipt=construct(case,cfg,122)
    optimizers={a:opt(m,cfg) for a,m in models.items()}
    assert not any(o.state for o in optimizers.values())
    assert len({id(o) for o in optimizers.values()})==2
    for arm,m in models.items():
        path=tmp_path/(arm+".pt");torch.save(m.state_dict(),path)
        loaded=torch.load(path,weights_only=True)
        assert all(torch.equal(v,loaded[k]) for k,v in m.state_dict().items())
    left,right=[m.state_dict() for m in models.values()]
    assert all(torch.equal(left[k],right[k]) for k in receipt["shared_tensors"])
    assert all(left[k].data_ptr()!=right[k].data_ptr() for k in receipt["shared_tensors"] if left[k].numel())


def test_train_same_inputs_and_rng_one_real_update():
    case,cfg=small();models,_=construct(case,cfg,123)
    inp=c.train_input(case,64,4,123,0,np.random.default_rng(123),force_t=32)
    before=input_digest(case,inp);rng=torch.get_rng_state();end=None
    for arm,m in models.items():
        m.set_target(extract_target(case));torch.set_rng_state(rng)
        result=c.update(m,opt(m,cfg),case,inp,64,"cpu")
        assert result["optimizer_update"] and input_digest(case,inp)==before
        if end is None:end=torch.get_rng_state()
        else:assert torch.equal(end,torch.get_rng_state())
    changed=dict(inp,labels=1-inp["labels"])
    assert input_digest(case,changed)!=before


def test_sampling_uniform_and_plan_pairing():
    case,cfg=small();rng=np.random.default_rng(77);other=np.random.default_rng(77)
    for t in range(4,0,-1):
        h,expected=uniform_contract(rng,case["space"].total)
        assert (h,expected)==uniform_contract(other,case["space"].total)
        np.testing.assert_array_equal(rng.random(case["space"].total),other.random(case["space"].total))
        assert sha(rng.bit_generator.state)==expected
        a=c.plan_for(case,[77,t,61],4);b=c.plan_for(case,[77,t,61],4)
        for j in range(4):
            for x,y in zip(a.block(j),b.block(j)):np.testing.assert_array_equal(x,y)


def test_validation_does_not_advance_training_rng():
    case,cfg=small();models,_=construct(case,cfg,124)
    inp=c.train_input(case,64,4,124,0,np.random.default_rng(124),force_t=32)
    values=[("toy","balanced",case,inp,0)]
    for m in models.values():
        m.set_target(extract_target(case))
        before=(random.getstate(),np.random.get_state(),torch.get_rng_state())
        validate(m,values,cfg,{"balanced":extract_target(case)},lambda:None)
        assert before[0]==random.getstate()
        assert before[1][0]==np.random.get_state()[0]
        np.testing.assert_array_equal(before[1][1],np.random.get_state()[1])
        assert torch.equal(before[2],torch.get_rng_state())


def test_activation_is_exclusive_no_budget_reset(tmp_path):
    out=tmp_path/"unique";ledger,epoch=activate(out,12)
    before=ledger.read_bytes()
    with pytest.raises(FileExistsError):activate(out,3600)
    assert before==ledger.read_bytes()
    x=json.loads(before);assert x["hard_cutoff_epoch"]==epoch+12
