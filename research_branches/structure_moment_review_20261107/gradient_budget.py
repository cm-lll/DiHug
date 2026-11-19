"""Output-space gradient budget; research prototype, no model integration.

This is a detached adaptive optimization coefficient, not a fixed loss weight
and not a parameter/AdamW update guarantee. Coordinates are binary logit gaps.
"""
import torch

VERSION='detached_global_logit_gradient_budget_v1'
RHO=0.1


def budget(base,structure,rho=RHO):
    if base.shape!=structure.shape or base.ndim!=1:
        raise ValueError('matching one-dimensional logit-gap gradients required')
    if not (0 < rho < 1):raise ValueError('rho must be in (0,1)')
    if not torch.isfinite(base).all() or not torch.isfinite(structure).all():
        raise FloatingPointError('nonfinite gradient')
    # FP64 norms avoid FP32 norm overflow/underflow. No epsilon changes units.
    b=base.detach().to(torch.float64);h=structure.detach().to(torch.float64)
    nb=torch.linalg.vector_norm(b);nh=torch.linalg.vector_norm(h)
    if float(nb)==0:
        weight=nb*0
    elif float(nh)==0:
        weight=nb*0+1
    else:
        weight=torch.minimum(torch.ones_like(nb),rho*nb/nh)
    return weight.detach(),dict(base_norm=float(nb),structure_norm=float(nh),
        coefficient=float(weight),capped=bool(float(weight)<1),
        structure_over_base=float(nh/nb) if float(nb)>0 else None,
        zero_base=bool(float(nb)==0),zero_structure=bool(float(nh)==0),rho=rho)


def compose(base_loss,structure_loss,logit_gap,rho=RHO):
    """One same-output measurement; NO extra neural forward or model probe.

    Caller's eventual backward traverses the model once. This reference version
    retains output loss graphs, so it repeats their differentiation on backward.
    A production VJP optimization would need separate validation, not assumed.
    """
    b=torch.autograd.grad(base_loss,logit_gap,retain_graph=True)[0]
    h=torch.autograd.grad(structure_loss,logit_gap,retain_graph=True)[0]
    weight,record=budget(b,h,rho)
    return base_loss+weight*structure_loss,record,b.detach(),h.detach()
