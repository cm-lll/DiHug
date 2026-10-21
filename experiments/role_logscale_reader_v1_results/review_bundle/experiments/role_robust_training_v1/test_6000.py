import copy
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import pytest
import torch
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from core import *
from transaction import UpdateBoundary,save_pair,restore_pair
from test_robust import case_for,model_for

def test_authorized_scope_exact():
    cfg=config()
    assert cfg["updates"]==6000 and cfg["expected_updates"]==36000
    assert cfg["expected_chains"]==288 and cfg["learning_budget_seconds"]==28800
    assert cfg["checkpoint_updates"]==list(range(1000,6001,1000))
    assert cfg["evaluation_updates"]==[1000,3000,6000]
    assert cfg["validation_updates"]==[0,250]+list(range(500,6001,500))
    assert cfg["optimizer"]["lr"]==.0002 and cfg["gradient_accumulation"]==2

def test_stop_waits_for_both_arms_and_both_microbatches(tmp_path):
    stop=UpdateBoundary(tmp_path,time.time()+60,reserve=0);steps=[]
    with pytest.raises(InterruptedError):
        with stop.pair():
            steps.append("normal2_micro0");(tmp_path/"STOP").touch();stop.check()
            steps.append("normal2_micro1");stop.check();steps.append("normal2_step")
            steps.append("robust_micro0");stop.check();steps.append("robust_micro1");stop.check();steps.append("robust_step")
    assert steps==["normal2_micro0","normal2_micro1","normal2_step","robust_micro0","robust_micro1","robust_step"]
    assert not stop.in_pair

def test_hard_deadline_interrupts_inside_pair(tmp_path):
    gate=UpdateBoundary(tmp_path,time.time()+60,reserve=0)
    with pytest.raises(TimeoutError):
        with gate.pair():
            gate.deadline=time.time()-1;gate.check()
    assert not gate.in_pair

def test_save_restore_two_update_trajectory_and_binding(tmp_path):
    setup();case=case_for("balanced");cfg=config()
    models={a:model_for(case) for a in cfg["arms"]};opts={a:opt(m,cfg) for a,m in models.items()}
    normal,robust,mask,_=paired_inputs(case,cfg,33,0)
    for arm,m in models.items():double_update(m,opts[arm],case,normal if arm=="normal2" else robust,64,mask)
    rng=np.random.default_rng(3);rng.random(5);binding={"frozen":"test"};budget={"budget_seconds":28800,"activation_epoch":1,"deadline_epoch":28801}
    path=tmp_path/"checkpoint.pt"
    r=save_pair(path,models,opts,rng,33,{a:1 for a in models},binding,budget)
    assert fingerprint(path)==r["sha256"]
    ck=torch.load(path,map_location="cpu",weights_only=False)
    restored={a:model_for(case) for a in cfg["arms"]};newopts={a:opt(m,cfg) for a,m in restored.items()}
    newrng=np.random.default_rng(999);restore_pair(ck,restored,newopts,newrng,binding)
    assert rng.bit_generator.state==newrng.bit_generator.state
    assert ck["budget"]==budget
    normal,robust,mask,_=paired_inputs(case,cfg,33,1)
    for arm in models:
        inp=normal if arm=="normal2" else robust
        double_update(models[arm],opts[arm],case,inp,64,mask)
        double_update(restored[arm],newopts[arm],case,inp,64,mask)
        assert all(torch.equal(p,q) for p,q in zip(models[arm].parameters(),restored[arm].parameters()))
        assert int(newopts[arm].state[restored[arm].backbone.lin_out_E.weight]["step"])==2
    with pytest.raises(ValueError,match="binding"):restore_pair(ck,restored,newopts,newrng,{"frozen":"tampered"})
    with pytest.raises(FileExistsError):save_pair(path,models,opts,rng,33,{a:2 for a in models},binding,budget)

def test_mismatched_pair_counters_refused(tmp_path):
    with pytest.raises(ValueError,match="paired"):
        save_pair(tmp_path/"bad.pt",{}, {},np.random.default_rng(),1,{"normal2":2,"robust":1},{},{})
    assert not (tmp_path/"bad.pt").exists()

def test_read_only_evaluation_rng_and_optimizer_unchanged():
    setup();case=case_for("extreme");m=model_for(case);cfg=config();o=opt(m,cfg)
    inp,_,mask,_=paired_inputs(case,cfg,2,2);double_update(m,o,case,inp,64,mask)
    before={k:v.clone() for k,v in m.state_dict().items()};rng=np.random.default_rng(8)
    saved=c.rng_save(rng,"cpu");state=copy.deepcopy(o.state_dict())
    evaluator=copy.deepcopy(m);evaluator.eval();evaluator.set_target(extract_target(case))
    with torch.no_grad():
        x=inp[0];predict(evaluator,case["space"].types,x["edges"],x["edge_families"],x["query"],x["f"],x["t"]/64)
    c.seed_all(51);rng.random(90);c.rng_restore(saved,rng,"cpu")
    assert torch.equal(saved["torch"],torch.get_rng_state()) and rng.bit_generator.state==saved["generator"]
    assert all(torch.equal(v,m.state_dict()[k]) for k,v in before.items())
    for p,v in state["state"].items():
        for k,t in v.items():assert torch.equal(t,o.state_dict()["state"][p][k])

def test_real_graceful_parent_allows_pair_to_commit(tmp_path):
    out=tmp_path/"out";out.mkdir()
    child=tmp_path/"child.py"
    child.write_text("import sys,time,json,pathlib\n"+f"sys.path.insert(0,{str(HERE)!r})\nfrom transaction import UpdateBoundary\n"+
        f"p=pathlib.Path({str(out)!r});g=UpdateBoundary(p,time.time()+30,0)\n"+
        "try:\n with g.pair():\n  (p/'entered').touch()\n  time.sleep(.6)\n  g.check()\n  (p/'both_committed').touch()\n"+
        "except InterruptedError:\n (p/'worker_receipt.json').write_text(json.dumps({'status':'stopped','reason':'STOP after pair'}))\n sys.exit(1)\n")
    parent=tmp_path/"parent.py"
    parent.write_text("import sys,pathlib,time\n"+f"sys.path.insert(0,{str(HERE)!r})\nfrom run import supervise\n"+
        f"sys.exit(supervise([{sys.executable!r},{str(child)!r}],pathlib.Path({str(out)!r}),time.time()+30,graceful_stop=True))\n")
    p=subprocess.Popen([sys.executable,str(parent)],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    end=time.monotonic()+10
    while not (out/"entered").exists() and p.poll() is None and time.monotonic()<end:time.sleep(.02)
    assert (out/"entered").exists();(out/"STOP").touch();p.communicate(timeout=10)
    assert p.returncode!=0 and (out/"both_committed").exists()
    assert json.loads((out/"controller.json").read_text())["status"]=="stopped"

def test_device_validation_export_matches_original_CPU_rows_and_rng():
    from device_eval import validate_device
    from role_condition.experiment import validate as original
    panel=HERE.parent/"role_error_recovery_v1_results/evidence/archived_panel"
    records,targets=load_panel(panel);cfg=config();case=records[0]["case"];m=model_for(case)
    values=fixed_validation(records,cfg);state=torch.get_rng_state().clone()
    old=original(m,values,cfg,targets,lambda:None)
    new=validate_device(m,values,cfg,targets,lambda:None)
    assert old["rows"]==new["rows"] and old["mean_ce"]==new["mean_ce"]
    assert torch.equal(state,torch.get_rng_state())

def test_parallel_merge_keeps_paths_counts_and_checkpoint_identity(tmp_path):
    import importlib.util
    spec=importlib.util.spec_from_file_location("robust_parallel_test",HERE/"parallel.py")
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    child=tmp_path/"seedjobs"/"seed33";child.mkdir(parents=True)
    (child/"updates.jsonl").write_text(json.dumps({"seed":33,"arm":"normal2","update":1})+"\n")
    (child/"generation.jsonl").write_text(json.dumps({"seed":33,"graph_file":"seed33/a.npz"})+"\n")
    (child/"weight_hashes.json").write_text(json.dumps({"seed33/a.pt":"frozenhash"}))
    (child/"checkpoint_index.json").write_text(json.dumps([{"path":"seed33/a.pt","sha256":"frozenhash"}]))
    counts=module.merge(tmp_path,[dict(folder=child,seed=33,gpu="CPU slot0")],False)
    assert counts==(1,1)
    r=json.loads((tmp_path/"generation.json").read_text())[0]
    assert r["graph_file"]=="seedjobs/seed33/seed33/a.npz"
    assert json.loads((tmp_path/"checkpoint_index.json").read_text())[0]["sha256"]=="frozenhash"
