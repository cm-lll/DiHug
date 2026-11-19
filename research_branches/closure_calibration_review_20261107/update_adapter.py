"""Explicit opt-in adapter around the existing transactional PB update.

No launcher and no changes to production files. The caller supplies the original
update callable and its PB objective. Requires the frozen PB coefficient 1.0.
The original update still owns forward, backward, AdamW, EMA and recovery phases.
"""
import numpy as np
import torch
from closure_calibration import VERSION, WEIGHT

def update(base_update,model,opt,case,inp,T,device,ema,completed,transaction=None,*,pb_objective,closure_objective,diagnostic=False,pb_coefficient=1.0):
    if pb_coefficient!=1.0:raise ValueError('frozen PB weight must remain 1')
    if not np.array_equal(case['space'].types,closure_objective.types) or tuple(map(tuple,case['space'].pairs))!=closure_objective.schema or not np.array_equal(case['prior'],closure_objective.prior):
        raise ValueError('case/schema/prior binding')
    if len(inp['query'].T)!=len(closure_objective.keys):raise ValueError('complete query required')
    q=np.asarray(inp['query']);keys=np.minimum(*q)*closure_objective.n+np.maximum(*q);order=np.argsort(keys)
    if not np.array_equal(keys[order],closure_objective.keys):raise ValueError('exact complete movable query')
    if not np.array_equal(np.asarray(inp['labels'])[order],closure_objective.labels):raise ValueError('same X0 labels required')
    context=closure_objective.context(inp['edges']);observed={}
    def objective(p,query,families):
        if observed:raise RuntimeError('must use single existing model prediction')
        pb,pb_groups=pb_objective(p,query,families);closing,c_groups=context(p,query,families)
        observed.update(pb=pb.detach(),closing=closing.detach(),pb_groups=pb_groups.detach(),c_groups=c_groups.detach())
        return pb+WEIGHT*closing,torch.cat([pb_groups,c_groups])
    result=base_update(model,opt,case,inp,T,device,ema,completed,transaction,objective=objective,diagnostic=diagnostic)
    result.pop('degree_group_loss',None)
    result.update(output_objective_version=VERSION,pb_loss=float(observed['pb']),closure_loss=float(observed['closing']),
        degree_group_loss=observed['pb_groups'].cpu().tolist(),closure_group_loss=observed['c_groups'].cpu().tolist(),
        closure_groups=context.groups,closure_context_sha256=context.state_sha256,closure_weight=WEIGHT)
    return result
