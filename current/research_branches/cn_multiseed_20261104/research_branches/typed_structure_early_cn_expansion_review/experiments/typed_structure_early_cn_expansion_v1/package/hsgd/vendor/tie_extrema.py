"""Exact grouped extrema; an explicitly equal-share subgradient at exact ties.

Forward delegates to the existing PyG pool (including zero-valued empty groups).
Autograd's no-grad forward selects PyG's native reduction; no initial fill entry
is counted as an input. Integer counts are exact. No epsilon or ID tie breaking.
"""
import torch
from torch_geometric.nn.pool import global_max_pool


class _EqualTieMax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, batch, size):
        if x.ndim != 2 or batch.ndim != 1 or len(batch) != len(x):
            raise ValueError('expected [items, channels] and [items] batch')
        if batch.dtype != torch.long or batch.device != x.device:
            raise ValueError('batch must be same-device int64')
        if not torch.is_floating_point(x) or not bool(torch.isfinite(x).all()):
            raise FloatingPointError('nonfinite / nonfloating extrema input')
        out = global_max_pool(x, batch, size=size)
        if not bool(torch.isfinite(out).all()):
            raise FloatingPointError('nonfinite extrema output')
        tied = x == out[batch]
        counts = torch.zeros(out.shape, device=x.device, dtype=torch.long)
        counts.scatter_add_(0, batch[:, None].expand_as(tied), tied.long())
        ctx.save_for_backward(batch, tied, counts)
        return out

    @staticmethod
    def backward(ctx, upstream):
        if not bool(torch.isfinite(upstream).all()):
            raise FloatingPointError('nonfinite extrema upstream gradient')
        batch, tied, counts = ctx.saved_tensors
        # Every input belongs to a nonempty group. Channel counts there >= 1.
        shares = upstream[batch] / counts[batch].to(upstream.dtype)
        return torch.where(tied, shares, torch.zeros_like(shares)), None, None


def equal_tie_max_pool(x, batch, size=None):
    return _EqualTieMax.apply(x, batch, size)
