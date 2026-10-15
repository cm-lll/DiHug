"""Exact saved-state LOAD checks; no ordinary-mode trajectory-equality claim."""
import numpy as np
import torch
from hsgd.bw_preflight.common import cpu_tree,rng_save
from hsgd.perf_checks import compare_rng
from hsgd.learning_budget import model_fingerprint


def exact(a,b):
    if torch.is_tensor(a):
        return torch.is_tensor(b) and a.dtype==b.dtype and a.shape==b.shape and torch.equal(a.cpu(),b.cpu())
    if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and a.dtype==b.dtype and np.array_equal(a,b)
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(exact(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return type(a)==type(b) and len(a)==len(b) and all(exact(x,y) for x,y in zip(a,b))
    return a==b


def verify_loaded(ck,model,opt,rng,device):
    if model_fingerprint(model.state_dict())!=ck["model_sha256"]:raise RuntimeError("loaded weight content differs")
    if not exact(ck["optimizer"],cpu_tree(opt.state_dict())):raise RuntimeError("loaded optimizer content differs")
    saved,current=ck["rng"],rng_save(rng,device)
    if saved.keys()!=current.keys():raise RuntimeError("loaded RNG keys differ")
    if saved["accelerator"] is not None and len(saved["accelerator"])!=len(current["accelerator"]):raise RuntimeError("loaded accelerator RNG inventory differs")
    compare_rng(saved,current)
    return dict(loaded_weights_exact=True,loaded_optimizer_exact=True,loaded_RNG_exact=True,
                completed_updates=ck["completed_updates"],post_resume_trajectory_identity_required=False)
