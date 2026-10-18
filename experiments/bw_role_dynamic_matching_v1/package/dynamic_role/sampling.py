"""Reuse original frozen-Xt 20-block adapter; replace ONLY its guide object.

No runnable scientific chain here. A/C share native no-CN weights; B/D share
native joint-CN weights. Model import/provenance remains the caller's duty.
"""
from .core import MatchingGuide


def reverse_step(*args,**kwargs):
    from stepwise_role.sampling import reverse_step as original
    if kwargs.get('arm') in ('C','D') and not isinstance(kwargs.get('guide'),MatchingGuide):
        raise ValueError('dynamic matching guide required for enabled arms')
    return original(*args,**kwargs)


def binding(model,case,kernel,guide,arm,rounds,seed,receipt=None):
    from stepwise_role.sampling import binding as original
    from .runner import fingerprint
    result=original(model,case,kernel,guide,arm,rounds,seed,receipt)
    result['dynamic_module_sha256']=fingerprint()
    return result


from stepwise_role.sampling import save_checkpoint,load_checkpoint
