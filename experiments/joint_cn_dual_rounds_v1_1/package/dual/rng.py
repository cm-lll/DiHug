"""Adapter-only RNG ownership: CPU and ONE selected accelerator, never all visible cards."""
import copy
import random
from contextlib import contextmanager
import numpy as np
import torch
from hsgd.bw_preflight import common as c

def seed_all(seed,device='cpu'):
    random.seed(seed);np.random.seed(seed)
    # torch.manual_seed seeds ALL accelerators too; use the CPU generator directly.
    torch.random.default_generator.manual_seed(seed)
    if device=='cuda':torch.cuda.manual_seed(seed)

def rng_save(rng,device):
    return dict(generator=copy.deepcopy(rng.bit_generator.state),python=random.getstate(),
        numpy=np.random.get_state(),torch=torch.get_rng_state(),
        accelerator=torch.cuda.get_rng_state(torch.cuda.current_device()) if device=='cuda' else None,
        accelerator_scope='selected_device_v1',
        selected_device_index=torch.cuda.current_device() if device=='cuda' else None)

def rng_restore(saved,rng,device):
    if saved.get('accelerator_scope')!='selected_device_v1':
        raise ValueError('RNG scope not certified as selected-device only')
    if device=='cuda' and (not torch.is_tensor(saved['accelerator']) or saved['accelerator'].dtype!=torch.uint8):
        raise ValueError('selected accelerator RNG missing/invalid')
    if device=='cpu' and saved['accelerator'] is not None:raise ValueError('accelerator RNG cannot be restored on CPU')
    rng.bit_generator.state=copy.deepcopy(saved['generator']);random.setstate(saved['python'])
    np.random.set_state(saved['numpy']);torch.set_rng_state(saved['torch'])
    if device=='cuda':torch.cuda.set_rng_state(saved['accelerator'],device=torch.cuda.current_device())

@contextmanager
def selected_core_rng():
    """Narrow synchronous bridge for unmodified reverse_step's transaction rollback.

    Only replaces accounting helpers during this call; no kernel/formula/query change.
    Production adapter runs a single worker thread; this scope is not a threaded API.
    """
    before=c.rng_save,c.rng_restore
    c.rng_save,c.rng_restore=rng_save,rng_restore
    try:yield
    finally:c.rng_save,c.rng_restore=before

def reverse_step(*args,**kwargs):
    from hsgd.runtime import reverse_step as original
    with selected_core_rng():return original(*args,**kwargs)
