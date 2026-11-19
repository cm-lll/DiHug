"""Bounded engineering only: two datasets, fourteen temporary updates, no chain.

Reuses real single-input update, EMA validation and transactional reverse_step.
Does not call/modify any formal launcher or accept a training-length argument.
Private asset root and output must be passed explicitly; no default locations.
"""
import argparse
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback

ROOT = (Path(__file__).resolve().parents[3] / 'research_branches/cn_multiseed_20261104')
SCI = ROOT/'research_branches/typed_structure_early_cn_expansion_review/experiments/typed_structure_early_cn_expansion_v1'
sys.path[:0] = [str(ROOT),str(ROOT/'experiments/o_cn_matched_execute_v1'),str(SCI/'package'),str(SCI)]
import numpy as np
import torch
from hsgd.bw_preflight import common
from hsgd.computational import FrozenTypedContext
from hsgd.runtime import reverse_step
from hsgd.kernel import MarginalKernel
from model_variants import StructuralDenoiser
from single_input import one_input, update
from ema import ParameterEMA
from boundary_transactions import TrainingBoundary, SamplingBoundary, rng_capture, rng_restore
from integration import restore_payload, ema_validation


def digest(value):
    h=hashlib.sha256()
    def visit(v):
        if torch.is_tensor(v):
            h.update(str((v.shape,v.dtype)).encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(v,np.ndarray):
            h.update(str((v.shape,v.dtype)).encode());h.update(v.tobytes())
        elif isinstance(v,dict):
            for k in sorted(v):h.update(str(k).encode());visit(v[k])
        elif isinstance(v,(tuple,list)):
            for item in v:visit(item)
        else:h.update(repr(v).encode())
    visit(value);return h.hexdigest()


def save(path,obj):
    path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')


def compare_state(left,right,atol,rtol):
    """Integer/RNG metadata exact; float continuation uses existing cfg tolerance."""
    rows=[]
    def visit(a,b,path):
        if torch.is_tensor(a):
            if a.shape!=b.shape or a.dtype!=b.dtype:raise AssertionError(path+' shape/dtype')
            exact=torch.equal(a,b)
            ok=exact or (a.is_floating_point() and torch.allclose(a,b,atol=atol,rtol=rtol))
            rows.append(dict(path=path,exact=exact,within_tolerance=bool(ok),
                             max_abs=float((a-b).abs().max()) if a.numel() else 0.))
        elif isinstance(a,np.ndarray):
            assert np.array_equal(a,b),path
        elif isinstance(a,dict):
            assert a.keys()==b.keys(),path
            for k in a:visit(a[k],b[k],path+'/'+str(k))
        elif isinstance(a,(list,tuple)):
            assert len(a)==len(b),path
            for k,(x,y) in enumerate(zip(a,b)):visit(x,y,path+'/'+str(k))
        else:assert a==b,path
    for key in ('model','optimizer','ema','rng','updates'):visit(left[key],right[key],key)
    return dict(bitwise_equal=all(x['exact'] for x in rows),
                within_existing_tolerance=all(x['within_tolerance'] for x in rows),
                atol=atol,rtol=rtol,largest_differences=sorted(rows,key=lambda x:x['max_abs'],reverse=True)[:10])

