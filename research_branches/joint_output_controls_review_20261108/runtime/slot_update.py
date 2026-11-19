"""Delegate unchanged to production CE update; add detached log aliases."""
from . import science as g
def update(model,opt,case,inp,T,device,ema,completed,transaction=None,*,objective,diagnostic=False):
    if objective.objective_binding()['kind']!='CE':raise ValueError('CE control only')
    rec=g.update(model,opt,case,inp,T,device,ema,completed,transaction=transaction)
    rec.update(unweighted_CE=rec['loss'],weighted_CE=rec['loss'],auxiliary_loss=0.,weighted_auxiliary=0.,
               auxiliary_seconds=0.,degree_group_loss=[],weight_min=1.,weight_max=1.,weight_mean_error=0.,
               objective_kind='CE',pure_CE_original_update=True)
    return rec
