"""Paired construction/input/RNG contracts; no experiment starts on import."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent
PREVIOUS=ROOT.parent/"role_condition_diagnosis_v1"
BASE=PREVIOUS/"baseline"
sys.path[:0]=[str(BASE),str(PREVIOUS)]
from hsgd.bw_preflight import common as c
from role_condition.model import paired_models,sha
from late_fusion import LateConditionalDenoiser


def config():
    cfg=json.loads((BASE/"role_condition/config.json").read_text())
    cfg.update(version="role_late_paired_v1",base_commit="1613be145b095f238a805b6a9513561191ef3bd1",
               arms=["original_C","late_C"],budget_seconds=3600,exit_reserve_seconds=30,
               updates=1000,expected_updates=6000,expected_chains=96,
               cost_forecast_safety_factor=1.15)
    return cfg


def source_manifest():
    files=list(ROOT.glob("*.py"))+list(ROOT.glob("*.md"))
    files+=[p for p in BASE.rglob("*") if p.is_file() and p.suffix in (".py",".json")]
    files+=[PREVIOUS/"late_fusion.py",PREVIOUS/"test_late_fusion.py"]
    return {str(p.relative_to(ROOT.parent)):c.sha(p) for p in sorted(files)}


def construct(case,cfg,seed):
    old=paired_models(case,cfg,seed)["C"]  # reproduce original C initialization exactly
    new=LateConditionalDenoiser(case["n_types"],case["n_families"],family_pairs=case["space"].pairs,
                               prior=case["prior"],**cfg["model"])
    left=old.state_dict();right=new.state_dict();shared=sorted(set(left)&set(right))
    assert set(left)-set(right)=={"branch.role_out.weight","branch.role_out.bias"}
    assert set(right)-set(left)=={"role_fusion.weight"}
    for name in shared:
        assert left[name].shape==right[name].shape
        right[name]=left[name].detach().clone()
    new.load_state_dict(right,strict=True)
    for name in shared:assert torch.equal(old.state_dict()[name],new.state_dict()[name])
    assert torch.count_nonzero(old.branch.role_out.weight)==0
    assert torch.count_nonzero(new.role_fusion.weight)==0
    receipt=dict(seed=seed,shared_tensors=shared,shared_exact=True,
        original_only=sorted(set(left)-set(right)),late_only=sorted(set(right)-set(left)),
        parameter_counts={a:sum(p.numel() for p in m.parameters()) for a,m in [("original_C",old),("late_C",new)]})
    return {"original_C":old,"late_C":new},receipt


def input_digest(case,inp):
    h=hashlib.sha256()
    for name,x in [("types",case["space"].types),("clean",case["clean"])]+[
            (k,inp[k]) for k in ["ids","f","query","labels","edges","edge_families"]]:
        x=np.ascontiguousarray(x);h.update(name.encode());h.update(str(x.dtype).encode())
        h.update(str(x.shape).encode());h.update(x.tobytes())
    h.update(json.dumps({k:inp[k] for k in ["t","round_id","ce_weight"]},sort_keys=True).encode())
    return h.hexdigest()


def uniform_contract(rng,size):
    clone=np.random.default_rng();clone.bit_generator.state=rng.bit_generator.state
    u=clone.random(size)
    return hashlib.sha256(u.tobytes()).hexdigest(),sha(clone.bit_generator.state)


def atomic_weights(path,value):
    tmp=path.with_name(path.name+".tmp");torch.save(value,tmp);tmp.replace(path)
    return c.sha(path)
