import copy
import json
import os
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
from role_condition.panel import mg_initial
from test_late_fusion import SMALL
import importlib.util
_spec=importlib.util.spec_from_file_location("robust_parent_under_test",HERE/"run.py")
_parent=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_parent)
new_output,supervise=_parent.new_output,_parent.supervise

def case_for(condition):return graph_case(skeleton()|mg_initial(condition))
def model_for(case):
    torch.manual_seed(921)
    m=CapacityDenoiser(4,10,family_pairs=case["space"].pairs,prior=case["prior"],T=64,**SMALL)
    m.set_target(extract_target(case));return m

@pytest.mark.parametrize("condition",["balanced","extreme"])
@pytest.mark.parametrize("direction,k",[("add",1),("add",4),("add",5),("delete",1),("delete",2),("delete",3)])
def test_exact_edits_query_labels_and_no_input_mutation(condition,direction,k):
    case=case_for(condition);cfg=config()
    inp=c.train_input(case,64,4,32,0,np.random.default_rng(32),force_t=1)
    before=input_digest(case,inp)
    out,edit,receipt=perturb(case,inp,np.random.default_rng(13),cfg["perturbation"],direction,k)
    assert receipt["actual"]==k and edit.sum()==k
    assert input_digest(case,inp)==before
    for key in ("labels","ids","f","query"):assert np.array_equal(inp[key],out[key])
    for key in ("t","round_id","ce_weight"):assert inp[key]==out[key]
    x=input_state(case,inp);xx=input_state(case,out);clean=np.isin(np.arange(len(x)),case["clean_ids"])
    assert np.array_equal(x!=xx,edit)
    assert (clean[edit]==(direction=="delete")).all()
    assert (xx[edit]==(direction=="add")).all()
    ee,ff=canonical(case);assert (ff[edit]==8).all()
    assert (ee[:,edit]==receipt["selected_node"]).any(0).all()
    assert (case["prior"][ff[edit]]>0).all() and (case["prior"][ff[edit]]<1).all()

def test_infeasible_is_identity_without_strength_reduction():
    case=case_for("balanced");inp=c.train_input(case,64,4,5,0,np.random.default_rng(8),force_t=64)
    inp["edges"]=np.empty((2,0),dtype=np.int64);inp["edge_families"]=np.empty(0,dtype=int)
    out,mask,r=perturb(case,inp,np.random.default_rng(8),config()["perturbation"],"delete",3)
    assert r["status"]=="infeasible_identity" and r["requested"]==3 and r["actual"]==0
    assert not mask.any() and input_digest(case,inp)==input_digest(case,out)

def test_stream_pairing_both_conditions_and_both_directions():
    cfg=config();seen={condition:set() for condition in ("balanced","extreme")}
    for condition in seen:
        case=case_for(condition)
        for u in range(40):
            a,b,mask,r=paired_inputs(case,cfg,2026102011,u)
            aa,bb,mm,rr=paired_inputs(case,cfg,2026102011,u)
            assert [input_digest(case,x) for x in a]==[input_digest(case,x) for x in aa]
            assert input_digest(case,b[0])==input_digest(case,a[0]);assert np.array_equal(mask,mm) and r==rr
            seen[condition].add(r["direction"])
            for l,h in zip(a,b):
                assert l["t"]==h["t"] and np.array_equal(l["query"],h["query"]) and np.array_equal(l["labels"],h["labels"])
    assert all(v=={"add","delete"} for v in seen.values())

def test_double_gradient_is_mean_of_two_original_ce_gradients():
    setup();case=case_for("extreme");cfg=config();a,b,mask,_=paired_inputs(case,cfg,2026102011,3)
    m=model_for(case);ref=model_for(case);optimizer=opt(m,cfg);ropt=opt(ref,cfg)
    diag={};r=double_update(m,optimizer,case,b,64,mask,diagnostics=diag,step=False)
    ropt.zero_grad(set_to_none=True);terms=[]
    for inp in b:
        logits,p=predict(ref,case["space"].types,inp["edges"],inp["edge_families"],inp["query"],inp["f"],inp["t"]/64,details=True)
        item=supported_binary_ce(logits,torch.as_tensor(inp["labels"]),ref.prior[torch.as_tensor(inp["f"])])
        terms.append(item.sum()*inp["ce_weight"])
    expected=(terms[0]+terms[1])/2;expected.backward()
    assert r["loss"]==pytest.approx(float(expected.detach()),abs=1e-7)
    for n,p in ref.named_parameters():
        if p.grad is not None:torch.testing.assert_close(diag["gradients"][n],p.grad,atol=2e-6,rtol=1e-5)
    for detail in r["parts"]:
        assert detail["splits"]["edited"]["n"]+detail["splits"]["untouched"]["n"]==detail["query_count"]
    assert not optimizer.state
    optimizer.step();ropt.step()
    for p,q in zip(m.parameters(),ref.parameters()):torch.testing.assert_close(p,q,atol=2e-6,rtol=1e-5)
    assert int(optimizer.state[m.backbone.lin_out_E.weight]["step"])==1

def test_identity_second_input_same_update_and_rng():
    setup();case=case_for("balanced");cfg=config();normal,_,_,_=paired_inputs(case,cfg,4,2)
    m=model_for(case);ref=model_for(case);o=opt(m,cfg);p=opt(ref,cfg)
    before=torch.get_rng_state().clone()
    l=double_update(m,o,case,normal,64)
    r=double_update(ref,p,case,[normal[0],dict(normal[1])],64)
    assert torch.equal(before,torch.get_rng_state()) and l["loss"]==r["loss"]
    assert all(torch.equal(x,y) for x,y in zip(m.parameters(),ref.parameters()))
    assert l["forward_count"]==2 and l["optimizer_updates"]==1

def test_clean_identity_and_edit_mask_are_not_prediction_inputs():
    setup();case=case_for("balanced");cfg=config();_,inp,mask,_=paired_inputs(case,cfg,3,2)
    m=model_for(case);m.eval();x=inp[1]
    with torch.no_grad():
        a=predict(m,case["space"].types,x["edges"],x["edge_families"],x["query"],x["f"],x["t"]/64)
        # No clean graph, mask, selected node or role assignment argument exists.
        altered=dict(case,clean_ids=np.array([],int),clean=np.empty((2,0),int))
        b=predict(m,altered["space"].types,x["edges"],x["edge_families"],x["query"],x["f"],x["t"]/64)
    assert torch.equal(a,b)

def test_holdout_topologies_are_new_joint_roles_preserved_and_reproducible():
    panel=HERE.parent/"role_error_recovery_v1_results/evidence/archived_panel"
    records,targets=load_panel(panel);cfg=config()
    first=make_holdout(records,cfg);second=make_holdout(records,cfg)
    assert len(first)==4
    for i,r in enumerate(first):
        assert extract_target(r["case"]).fingerprint==targets[r["condition"]].fingerprint
        assert np.array_equal(r["case"]["clean"],second[i]["case"]["clean"])
        for old in records+first[:i]:
            if old["condition"]==r["condition"]:assert not isomorphic(r["case"],old["case"])

def test_incomplete_second_microbatch_does_not_step_optimizer(monkeypatch):
    import core
    setup();case=case_for("balanced");normal,_,mask,_=paired_inputs(case,config(),3,1)
    m=model_for(case);o=opt(m,config());state={k:v.clone() for k,v in m.state_dict().items()}
    original=core.predict;calls=[]
    def failing(*args,**kwargs):
        calls.append(1)
        if len(calls)==2:raise RuntimeError("injected second-input failure")
        return original(*args,**kwargs)
    monkeypatch.setattr(core,"predict",failing)
    with pytest.raises(RuntimeError,match="second-input"):double_update(m,o,case,normal,64,mask)
    assert not o.state and all(torch.equal(m.state_dict()[k],v) for k,v in state.items())

def test_output_protection_before_mkdir(tmp_path):
    source=tmp_path/"input";source.mkdir();(source/"old").write_text("untouched")
    with pytest.raises(ValueError):new_output(source/"new",[source])
    assert sorted(p.name for p in source.iterdir())==["old"]
    with pytest.raises(ValueError):new_output(tmp_path,[source])

def test_learning_requires_explicit_cli_flag_before_any_output(tmp_path):
    out=tmp_path/"must_not_exist"
    command=[sys.executable,str(HERE/"run.py"),"panel","--source",str(tmp_path/"source"),"--panel",str(tmp_path/"old"),"--output",str(out)]
    r=subprocess.run(command,capture_output=True,text=True,timeout=10)
    assert r.returncode!=0 and "--execute-learning-panel" in r.stderr and not out.exists()

@pytest.mark.parametrize("mode",["success","failure","timeout","STOP","SIGTERM","SIGHUP","SIGINT"])
def test_real_supervisor_exit_and_reaping(tmp_path,mode):
    out=tmp_path/mode;out.mkdir()
    script=tmp_path/(mode+".py")
    script.write_text("import pathlib,time,json,sys,os\n"
        f"p=pathlib.Path({str(out)!r})\n"
        "(p/'pid').write_text(str(os.getpid()))\n"+
        ("(p/'worker_receipt.json').write_text(json.dumps({'status':'completed'}))\n" if mode=="success" else
         "sys.exit(3)\n" if mode=="failure" else "time.sleep(60)\n"))
    helper=tmp_path/(mode+"_parent.py")
    helper.write_text("import sys,time,pathlib\n"+f"sys.path.insert(0,{str(HERE)!r})\nfrom run import supervise\n"+
        f"sys.exit(supervise([{sys.executable!r},{str(script)!r}],pathlib.Path({str(out)!r}),time.time()+{.6 if mode=='timeout' else 10}))\n")
    parent=subprocess.Popen([sys.executable,str(helper)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    end=time.monotonic()+5
    while not (out/"pid").exists() and parent.poll() is None and time.monotonic()<end:time.sleep(.02)
    assert (out/"pid").exists()
    if mode=="STOP":(out/"STOP").touch()
    elif mode.startswith("SIG"):
        import signal
        parent.send_signal(getattr(signal,mode))
    parent.communicate(timeout=12)
    receipt=json.loads((out/"controller.json").read_text())
    assert (parent.returncode==0)==(mode=="success")
    assert (receipt["status"]=="completed")== (mode=="success")
    pid=int((out/"pid").read_text())
    with pytest.raises(ProcessLookupError):os.kill(pid,0)
