"""Adam counts follow gradient participation, including empty CN batches."""
import numpy as np


def verify_steps(ck,model=None):
    names=ck["optimizer_parameter_names"]
    if model is not None and names!=[n for n,p in model.named_parameters()]:
        raise RuntimeError("optimizer parameter order mismatch")
    groups=ck["optimizer"]["param_groups"]
    if len(groups)!=1 or len(groups[0]["params"])!=len(names) or len(set(names))!=len(names):
        raise RuntimeError("optimizer inventory mismatch")
    count=np.zeros(len(names),dtype=np.int64)
    history=ck["history"]
    if len(history)!=ck["completed_updates"]:raise RuntimeError("update history count mismatch")
    for r in history:
        absent=np.asarray(r["no_gradient_indices"],dtype=np.int64)
        if ((absent<0)|(absent>=len(names))).any() or len(np.unique(absent))!=len(absent):
            raise RuntimeError("gradient participation indices invalid")
        count+=1;count[absent]-=1
    states=ck["optimizer"]["state"]
    if set(states)-set(groups[0]["params"]):raise RuntimeError("unknown Adam parameter state")
    for i,pid in enumerate(groups[0]["params"]):
        if count[i]==0:
            if pid in states:raise RuntimeError("Adam state for a never-participating parameter")
        elif pid not in states or int(states[pid]["step"])!=int(count[i]):
            raise RuntimeError("Adam participation count differs: "+names[i])
    return dict(effective_updates=len(history),parameters=len(names),
        participating_step_min=int(count.min()),participating_step_max=int(count.max()))
