"""Explicit single-process, single-thread diagnostic dispatch; no fallback."""
from contextlib import contextmanager
from scaling import required

@contextmanager
def reader_context(module,mode='scaled',check=lambda:None):
    if mode not in ('exact','scaled'):raise ValueError('explicit reader mode required')
    old=module.allocation_marginals
    if mode=='scaled':module.allocation_marginals=lambda L,c:required(L,c,check)
    try:yield
    finally:module.allocation_marginals=old
