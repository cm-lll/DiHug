"""One robust objective; no architecture, kernel, or sampler modification."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
EXP=HERE.parent
sys.path.insert(0,str(EXP/"role_specialization_v1"))
from panel import CapacityDenoiser,load_panel,c,fingerprint
from pairing import input_digest,atomic_weights,sha,uniform_contract
from role_condition.experiment import opt,save,emit,fixed_validation,validate
from role_condition.panel import augment,NAMES,swap,graph_case,skeleton,isomorphic,save_panel
from role_condition.model import extract_target
from hsgd.runtime import predict,reverse_step
from hsgd.losses import supported_binary_ce
from hsgd.kernel import MarginalKernel

def config():return json.loads((HERE/"config.json").read_text())

def file_manifest():
    paths=list(HERE.glob("*.py"))+list(HERE.glob("*.json"))+list(HERE.glob("*.md"))
    for directory in (EXP/"role_specialization_v1",EXP/"role_late_paired_v1"):
        paths += list(directory.glob("*.py"))+list(directory.glob("*.md"))
    prior=EXP/"role_condition_diagnosis_v1"
    paths += [prior/n for n in ("diagnose.py","late_fusion.py","test_late_fusion.py")]
    paths += [p for p in (prior/"baseline").rglob("*") if p.is_file() and p.suffix in (".py",".json")]
    return {str(p.relative_to(EXP)):fingerprint(p) for p in sorted(set(paths))}

def setup():
    torch.set_num_threads(1);torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(False)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False

def canonical(case):
    sp=case["space"]
    return np.concatenate([sp.decode(r,np.arange(n)) for r,n in enumerate(sp.counts)],axis=1),np.repeat(np.arange(case["n_families"]),sp.counts)

def input_state(case,inp):
    state=np.zeros(case["space"].total,dtype=np.uint8)
    f,local=case["space"].encode(inp["edges"])
    if not np.array_equal(f,inp["edge_families"]):raise ValueError("state relation labels mismatch")
    state[case["space"].offsets[f]+local]=1
    return state

def perturb(case,inp,rng,settings,direction=None,k=None):
    """Clean identities ONLY construct perturbation; not passed to the network."""
    before=input_state(case,inp);edges,families=canonical(case)
    pair=tuple(settings["family_pair"]);rel=case["space"].pairs.index(pair)
    direction=settings["directions"][int(rng.integers(2))] if direction is None else direction
    if direction not in ("add","delete"):raise ValueError("invalid perturbation direction")
    options=settings["add_counts" if direction=="add" else "delete_counts"]
    k=int(options[int(rng.integers(len(options)))]) if k is None else int(k)
    if k<=0:raise ValueError("positive edit count required")
    clean=np.zeros(len(before),bool);clean[case["clean_ids"]]=True
    pi=case["prior"][families]
    candidates=(families==rel)&(pi>0)&(pi<1)
    candidates &= (~clean&(before==0)) if direction=="add" else (clean&(before==1))
    nodes=np.flatnonzero(case["space"].types==settings["selected_endpoint_type"])
    eligible=[int(n) for n in nodes if int((candidates&(edges==n).any(0)).sum())>=k]
    out=dict(inp);edit=np.zeros(len(before),bool);selected=None;state=before.copy()
    if eligible:
        selected=eligible[int(rng.integers(len(eligible)))]
        ids=np.flatnonzero(candidates&(edges==selected).any(0))
        edit[rng.permutation(ids)[:k]]=True;state[edit]=int(direction=="add")
    out["edges"],out["edge_families"]=c.state_edges(case,state)
    assert np.array_equal(out["ids"],inp["ids"]) and np.array_equal(out["labels"],inp["labels"])
    assert np.count_nonzero(state!=before)==int(edit.sum())
    queried=edit[inp["ids"]]
    receipt=dict(direction=direction,requested=k,actual=int(edit.sum()),selected_node=selected,
        status="edited" if selected is not None else "infeasible_identity",
        reason=None if selected is not None else "no node supports exact requested legal edits",
        eligible_nodes=len(eligible),family=rel,queried_edited=int(queried.sum()),
        queried_edited_positive=int(np.sum(queried&(inp["labels"]==1))),
        queried_edited_negative=int(np.sum(queried&(inp["labels"]==0))))
    if selected is not None:
        incident=(edges==selected).any(0)&(families==rel)
        receipt.update(selected_degree_before=int(before[incident].sum()),selected_degree_after=int(state[incident].sum()))
    return out,edit,receipt

def paired_inputs(case,cfg,seed,update):
    normal=[c.train_input(case,cfg["T"],cfg["rounds"],seed,2*update+j,
             np.random.default_rng([cfg["input_seed"],seed,update,j])) for j in (0,1)]
    changed,mask,record=perturb(case,normal[1],np.random.default_rng([cfg["perturbation"]["seed"],seed,update]),cfg["perturbation"])
    return normal,[normal[0],changed],mask,record

def pair_models(case,cfg,seed,source):
    c.seed_all(seed)
    old=EXP/"role_specialization_v1_results/evidence/learning_run01"
    hashes=json.loads((old/"weight_hashes.json").read_text())
    path=source/f"seed{seed}/capacity_C_initial.pt"
    assert fingerprint(path)==hashes[str(path.relative_to(source))]
    initial=torch.load(path,map_location="cpu",weights_only=True)
    models={}
    for arm in cfg["arms"]:
        m=CapacityDenoiser(case["n_types"],case["n_families"],family_pairs=case["space"].pairs,
                          prior=case["prior"],T=cfg["T"],**cfg["model"])
        m.load_state_dict({n:v.clone() for n,v in initial.items()},strict=True);models[arm]=m
    left,right=models.values()
    for n,v in left.state_dict().items():assert torch.equal(v,right.state_dict()[n])
    assert all(l.data_ptr()!=r.data_ptr() for l,r in zip(left.parameters(),right.parameters()))
    return models,dict(seed=seed,source=str(path),sha256=fingerprint(path),all_tensors_exact=True,
                       tensors=len(initial),independent_storage=True,parameters=sum(p.numel() for p in left.parameters()))

def double_update(model,optimizer,case,inputs,T,edit=None,check=lambda:None,step=True,diagnostics=None):
    """Two original R/S CE estimates, ordinary mean, ONE optimizer step."""
    if len(inputs)!=2:raise ValueError("exactly two inputs required")
    start=time.monotonic();optimizer.zero_grad(set_to_none=True);model.train();parts=[]
    before=model.backbone.lin_out_E.weight.detach().clone()
    rng=torch.get_rng_state().clone()
    for j,inp in enumerate(inputs):
        check();tick=time.monotonic();timings={}
        logits,p=predict(model,case["space"].types,inp["edges"],inp["edge_families"],inp["query"],inp["f"],
                         inp["t"]/T,details=True,times=timings)
        c.check_prob_device(p,inp["f"],case,device_prior=model.prior)
        labels=torch.as_tensor(inp["labels"],device=logits.device)
        prior=model.prior[torch.as_tensor(inp["f"],dtype=torch.long,device=logits.device)]
        item=supported_binary_ce(logits,labels,prior)
        ce=item.sum()*inp["ce_weight"]
        if not torch.isfinite(ce):raise FloatingPointError("nonfinite CE")
        forward=time.monotonic()-tick;tick=time.monotonic()
        (.5*ce).backward();backward=time.monotonic()-tick
        if any(v.grad is not None and not torch.isfinite(v.grad).all() for v in model.parameters()):raise FloatingPointError("nonfinite accumulated gradient")
        counts={}
        for label in (0,1):
            ix=labels==label
            counts[str(label)]=dict(n=int(ix.sum()),ce_sum=float(item[ix].detach().sum()))
        changed=np.zeros(len(inp["ids"]),bool) if edit is None or j==0 else edit[inp["ids"]]
        splits={}
        for changed_value,name in ((True,"edited"),(False,"untouched")):
            ix=torch.as_tensor(changed==changed_value)
            splits[name]=dict(n=int(ix.sum()),ce_sum=float(item[ix].detach().sum()))
        parts.append(dict(loss=float(ce.detach()),t=inp["t"],query_count=len(inp["ids"]),query_block=inp["round_id"],
            forward_seconds=forward,backward_seconds=backward,timings=timings,labels=counts,splits=splits,
            input_sha256=input_digest(case,inp)))
    named=[(n,p) for n,p in model.named_parameters() if p.grad is not None]
    if not named:raise RuntimeError("no participating gradient")
    norm=torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(p.grad) for _,p in named]))
    if not torch.isfinite(norm):raise FloatingPointError("nonfinite gradient norm")
    if diagnostics is not None:diagnostics.update(gradients={n:p.grad.clone() for n,p in named})
    assert torch.equal(rng,torch.get_rng_state()),"original forward unexpectedly consumed Torch RNG"
    tick=time.monotonic()
    if step:optimizer.step()
    opt_seconds=time.monotonic()-tick
    if step and any(not torch.isfinite(p).all() for p in model.parameters()):raise FloatingPointError("nonfinite updated parameter")
    if step and torch.equal(before,model.backbone.lin_out_E.weight):raise RuntimeError("edge head did not update")
    return dict(loss=.5*(parts[0]["loss"]+parts[1]["loss"]),parts=parts,gradient_norm=float(norm),
                seconds=time.monotonic()-start,optimizer_seconds=opt_seconds,forward_count=2,optimizer_updates=int(step))

def make_holdout(records,cfg,check=lambda:None):
    """Separate topologies, SAME anonymous roles; bounded deterministic search."""
    result=[];rng=np.random.default_rng(cfg["holdout_seed"])
    for condition in ("balanced","extreme"):
        source=next(r["case"] for r in records if r["condition"]==condition)
        f=source["space"].encode(source["clean"])[0];mg=source["space"].pairs.index((2,3))
        edges=set(map(tuple,source["clean"][:,f==mg].T))
        old=[r["case"] for r in records if r["condition"]==condition]
        selected=[]
        for attempt in range(cfg["holdout_max_attempts_per_condition"]):
            check();edges=swap(edges,rng);candidate=graph_case(skeleton()|edges)
            if any(isomorphic(candidate,x) for x in old+selected):continue
            assert extract_target(candidate).fingerprint==extract_target(source).fingerprint
            selected.append(candidate)
            result.append(dict(id=f"heldout_{condition}_{len(selected)-1}",condition=condition,split="heldout",case=candidate))
            if len(selected)==cfg["holdout_graphs_per_condition"]:break
        if len(selected)!=cfg["holdout_graphs_per_condition"]:raise RuntimeError("bounded holdout topology search failed; no replacement panel")
    return result
