"""Single candidate: sqrt-target scaled, pseudo-Huber relation count loss.

Independent CPU-reviewed function only. No model import or training entrypoint.
"""
import torch

LAMBDA = 0.001

def smooth_count_penalty(delta, target):
    """Unweighted per-family penalty and derivative d rho / d delta."""
    scale = target.clamp_min(1).sqrt()
    u = delta / scale
    h = torch.hypot(torch.ones_like(u), u)
    # Algebraically sqrt(1+u*u)-1, without cancellation near zero.
    rho = (u / (h + 1)) * u
    slope = (u / h) / scale
    return rho, scale, u, slope

def count_loss(p0, families, targets, movable_counts, fixed_present=None):
    # Caller must provide each full active UNORDERED slot once. Family counts
    # alone cannot detect duplicate slot IDs: retain normal2's existing ID gate.
    p = p0.to(torch.float64)
    f = torch.as_tensor(families, device=p.device, dtype=torch.long)
    target = torch.as_tensor(targets, device=p.device, dtype=torch.float64)
    slots = torch.as_tensor(movable_counts, device=p.device, dtype=torch.long)
    fixed = torch.zeros_like(target) if fixed_present is None else torch.as_tensor(fixed_present, device=p.device, dtype=torch.float64)
    if p.ndim != 1 or target.ndim != 1 or f.shape != p.shape or slots.shape != target.shape or fixed.shape != target.shape:
        raise ValueError('count shapes')
    if not torch.isfinite(p).all() or not torch.isfinite(target).all() or not torch.isfinite(fixed).all():
        raise FloatingPointError('nonfinite count input')
    if (p < 0).any() or (p > 1).any() or (slots < 0).any() or (target < 0).any() or (fixed < 0).any():
        raise ValueError('invalid probability/count')
    if f.numel() and ((f < 0).any() or (f >= target.numel()).any()):
        raise ValueError('family range')
    if not torch.equal(torch.bincount(f, minlength=len(target)), slots):
        raise ValueError('not full relation coverage')
    if ((target < fixed) | (target > fixed + slots)).any():
        raise ValueError('target inconsistent with support')
    mu = torch.stack([p[f == r].sum() for r in range(len(target))]) + fixed if len(target) else p.new_empty((0,))
    included = slots > 0
    delta = mu - target
    penalty, scale, u, slope = smooth_count_penalty(delta, target)
    loss = penalty[included].mean() if included.any() else p.sum() * 0
    return loss, {
        'expected_edges': mu, 'target_edges': target, 'signed_error': delta,
        'relative_error': delta / target.clamp_min(1),
        'residual_scale': scale, 'standardized_residual': u,
        'normalized_penalty': penalty, 'penalty_count_derivative': slope,
        'included': included, 'relation_denominator': int(included.sum()),
    }

def case_contract(case, inp):
    import numpy as np
    fixed = np.where((~case['active']) & (case['prior'] == 1), case['space'].counts, 0)
    movable = np.where(case['active'], case['space'].counts, 0)
    target = np.bincount(inp['f'], weights=inp['labels'], minlength=case['n_families']) + fixed
    return target, movable, fixed
