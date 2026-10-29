"""Intermediate readout isolation; training model/optimizer/EMA/RNG restored."""
from contextlib import contextmanager
from ema import cpu_copy
from probe import rng_get,rng_set,equal

@contextmanager
def isolated_readout(model,opt,ema,rng,device):
    raw=cpu_copy(model.state_dict());optimizer=cpu_copy(opt.state_dict())
    averaged=ema.state_dict();random=rng_get(rng,device);mode=model.training
    try:
        yield
    finally:
        model.load_state_dict(raw,strict=True);model.train(mode)
        rng_set(random,rng,device)
        if not equal(opt.state_dict(),optimizer) or not equal(ema.state_dict(),averaged):
            raise AssertionError('intermediate inference modified optimizer/EMA')
        if not equal(model.state_dict(),raw) or not equal(rng_get(rng,device),random):
            raise AssertionError('intermediate inference restore failed')
