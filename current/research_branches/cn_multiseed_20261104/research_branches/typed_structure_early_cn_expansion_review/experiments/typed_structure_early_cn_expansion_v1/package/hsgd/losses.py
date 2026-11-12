"""Stable per-slot binary CE with exact, checked prior support.

The training objective is sum(items) * original query ce_weight (R/S).
Sampling probabilities and their exact-support projection remain unchanged.
"""
import torch
from torch.nn import functional as F


def supported_binary_ce(logits, labels, prior):
    if logits.ndim != 2 or logits.shape[1] != 2 or not logits.is_floating_point():
        raise ValueError('two floating-point logits per slot required')
    labels=torch.as_tensor(labels,device=logits.device)
    prior=torch.as_tensor(prior,device=logits.device)
    if labels.shape != logits.shape[:1] or prior.shape != labels.shape:
        raise ValueError('slot/label/prior dimensions differ')
    if not torch.isfinite(logits).all() or not torch.isfinite(labels).all() or not torch.isfinite(prior).all():
        raise ValueError('nonfinite logits/labels/prior')
    if not ((labels==0)|(labels==1)).all():raise ValueError('binary labels required')
    if not ((prior>=0)&(prior<=1)).all():raise ValueError('prior outside [0,1]')
    fixed0,fixed1=prior==0,prior==1
    if (labels[fixed0]!=0).any() or (labels[fixed1]!=1).any():
        raise ValueError('label violates exact prior support')
    free=~(fixed0|fixed1)
    # Multiply each finite logit by zero before summing: all-fixed/empty input
    # stays differentiable with exactly zero gradient and no 0*inf from a sum.
    zero=(logits*0).sum(-1)
    ids=torch.nonzero(free,as_tuple=False).flatten()
    if not len(ids):return zero
    item=F.cross_entropy(logits[ids],labels[ids].long(),reduction='none')
    return zero.index_copy(0,ids,item)
