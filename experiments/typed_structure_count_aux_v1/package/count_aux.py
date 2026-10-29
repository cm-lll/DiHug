"""Differentiable whole-query p0 relation counts. No inference modification."""
import torch

LAMBDA = 0.01

def count_loss(p0, families, targets, movable_counts, fixed_present=None):
    # Each probability MUST represent one unique unordered active slot. The
    # caller verifies complete unique slot IDs before constructing this tensor.
    p = p0.to(torch.float64)
    f = torch.as_tensor(families, device=p.device, dtype=torch.long)
    target = torch.as_tensor(targets, device=p.device, dtype=torch.float64)
    slots = torch.as_tensor(movable_counts, device=p.device, dtype=torch.long)
    fixed = torch.zeros_like(target) if fixed_present is None else torch.as_tensor(fixed_present, device=p.device, dtype=torch.float64)
    if p.ndim != 1 or f.shape != p.shape or slots.shape != target.shape or fixed.shape != target.shape:
        raise ValueError('count shapes')
    if not torch.isfinite(p).all() or not torch.isfinite(target).all() or not torch.isfinite(fixed).all():
        raise FloatingPointError('nonfinite count input')
    if (p < 0).any() or (p > 1).any() or (slots < 0).any() or (target < 0).any() or (fixed < 0).any():
        raise ValueError('invalid probability/count')
    if f.numel() and ((f < 0).any() or (f >= target.numel()).any()):raise ValueError('family range')
    if not torch.equal(torch.bincount(f, minlength=len(target)), slots):raise ValueError('not full relation coverage')
    if ((target < fixed) | (target > fixed + slots)).any():raise ValueError('target inconsistent with support')
    # Per-family reductions avoid a GPU atomic floating scatter dependency.
    mu = torch.stack([p[f == r].sum() for r in range(len(target))]) + fixed
    included = slots > 0
    error = (mu - target) / target.clamp_min(1)
    # Squared density error divided by target density (floor one edge/K).
    # This prevents rare relations with huge candidate sets from dominating
    # solely through an unbounded relative count error at initialization.
    terms = (mu-target).square() / (slots.clamp_min(1)*target.clamp_min(1))
    loss = terms[included].mean() if included.any() else p.sum() * 0
    return loss, {'expected_edges': mu, 'target_edges': target, 'signed_error': mu-target,
                  'relative_error': error, 'normalized_square': terms,
                  'included': included, 'relation_denominator': int(included.sum())}

def case_contract(case, inp):
    import numpy as np
    # train_input labels are full clean X0 labels for active slots; fixed
    # relations retain their exact supported presence count.
    fixed = np.where((~case['active']) & (case['prior'] == 1), case['space'].counts, 0)
    movable = np.where(case['active'], case['space'].counts, 0)
    target = np.bincount(inp['f'], weights=inp['labels'], minlength=case['n_families']) + fixed
    return target, movable, fixed
