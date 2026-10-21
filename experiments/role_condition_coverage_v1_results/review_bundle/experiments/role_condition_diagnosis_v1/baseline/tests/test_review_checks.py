"""Engineering checks only. Never invokes the 3-seed learning panel."""
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from hsgd.bw_preflight import common as c
from hsgd.computational import FrozenTypedContext
from hsgd.runtime import predict
from role_condition.model import RoleTarget,paired_models,ATOL
from role_condition.panel import make_panel
from role_condition.experiment import opt

CFG=json.loads((Path(__file__).resolve().parents[1]/'role_condition/config.json').read_text())

def test_zero_projection_naturally_opens_role_encoder_and_attention_gradients(record_property):
    records,targets=make_panel();case=next(r['case'] for r in records if r['condition']=='extreme')
    model=paired_models(case,CFG,1221)['C'];model.set_target(targets['extreme'])
    assert torch.count_nonzero(model.branch.role_out.weight)==0
    assert torch.count_nonzero(model.branch.degree_out.weight)==0
    optimizer=opt(model,CFG);rng=np.random.default_rng(2026102041);rows=[]
    names=['role_encoder.0.weight','role_encoder.2.weight','query.weight','key.weight','value.weight','role_out.weight']
    params=dict(model.branch.named_parameters())
    for update in range(5):
        inp=c.train_input(case,64,4,2026102041,update,rng,force_t=32)
        receipt=c.update(model,optimizer,case,inp,64,'cpu')
        assert receipt['optimizer_update']
        magnitudes={}
        for name in names:
            grad=params[name].grad
            assert grad is not None and torch.isfinite(grad).all(),name
            magnitudes[name]=float(grad.abs().max())
        rows.append(dict(update=update+1,loss=receipt['loss'],gradient_max_abs=magnitudes))
    # First-step zero upstream gradients are EXPECTED; do not alter initialization.
    for name in names[:-1]:assert rows[0]['gradient_max_abs'][name]==0,name
    assert rows[0]['gradient_max_abs']['role_out.weight']>0
    for name in names:
        assert any(row['gradient_max_abs'][name]>0 for row in rows[1:]),name
    record_property('natural_zero_projection_updates',json.dumps(rows,allow_nan=False))

def test_frequency_response_duplicate_merge_and_cache_switch(record_property):
    records,targets=make_panel();case=records[0]['case'];base=targets['balanced'];conditions=[]
    # Same token vectors, same type inventory, same total MG edges and mean degree.
    # Only integer frequencies differ: [1,4,1] versus [2,2,2] on degrees [1,4,7].
    for counts in [(1,4,1),(2,2,2)]:
        roles=list(base.roles);rows=[]
        for degree,count in zip([1,4,7],counts):
            vec=[0]*10;vec[8]=degree;rows.append((tuple(vec),count))
        roles[3]=tuple(rows);conditions.append(RoleTarget(tuple(roles),base.type_counts,base.schema))
    a,b=conditions;assert a.fingerprint!=b.fingerprint
    roles=list(a.roles);roles[3]=tuple((v,1) for v,n in reversed(a.roles[3]) for _ in range(n))
    duplicate=RoleTarget(tuple(roles),a.type_counts,a.schema)
    assert duplicate==a and duplicate.fingerprint==a.fingerprint
    model=paired_models(case,CFG,1221)['C'];model.eval()
    # This separate interface-response test intentionally uses nonzero projections.
    # It is NOT the natural-zero-init training check above.
    with torch.no_grad():torch.nn.init.normal_(model.branch.role_out.weight,std=.03)
    family=case['space'].encode(case['clean'])[0];_,f,q=c.plan_for(case,17,4).block(0)
    with torch.no_grad():
        model.set_target(a);context=FrozenTypedContext(model,case['space'].types,case['clean'],family,'freq')
        before=model.branch_calls;left=model.node_condition(context,.5).clone()
        model.set_target(duplicate);same=model.node_condition(context,.5)
        assert torch.equal(left,same) and model.branch_calls==before+1
        model.set_target(b)
        with pytest.raises(ValueError,match='target/time/mode'):model.node_condition(context,.5)
        context.close();context=FrozenTypedContext(model,case['space'].types,case['clean'],family,'freq2')
        right=model.node_condition(context,.5).clone();context.close()
        assert float((left-right).abs().max())>ATOL
        probabilities=[]
        for target in [a,b,a]:
            model.set_target(target)
            probabilities.append(predict(model,case['space'].types,case['clean'],family,q,f,.5))
        assert torch.equal(probabilities[0],probabilities[2])
        response=float((probabilities[0]-probabilities[1]).abs().max());assert response>ATOL
    record_property('frequency_cache_response',json.dumps(dict(probability_max_abs=response,
        same_token_vectors=True,same_mean_degree=4,merged_target_exact=True,stale_context_rejected=True,
        switch_back_probabilities_exact=True),allow_nan=False))

def test_only_version_and_authorized_wall_budget_changed():
    original=Path(__file__).resolve().parents[1]/'role_condition/previous_config.json'
    old=json.loads(original.read_text());current=dict(CFG)
    assert current.pop('budget_seconds')==5400 and old.pop('budget_seconds')==1800
    current.pop('version');old.pop('version');assert current==old

def test_cumulative_budget_receipt_rejects_restart_without_overwrite(tmp_path):
    import importlib.util
    spec=importlib.util.spec_from_file_location('panel_launcher',Path(__file__).resolve().parents[1]/'run.py')
    launcher=importlib.util.module_from_spec(spec);spec.loader.exec_module(launcher)
    out=tmp_path/'panel';ledger=launcher.reserve_budget(out,CFG,'source-hash',1000.)
    before=ledger.read_bytes();data=json.loads(before)
    assert data['hard_cutoff_epoch']==6400. and data['budget_seconds']==5400
    assert data['no_budget_reset'] and not data['resume_supported']
    with pytest.raises(FileExistsError):launcher.reserve_budget(out,CFG,'source-hash',50000.)
    assert ledger.read_bytes()==before and not out.exists()
