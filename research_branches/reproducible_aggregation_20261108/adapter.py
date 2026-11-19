"""Opt-in diagnostic execution policy; no model/loss/optimizer definition change.

PyTorch's strict deterministic kernels cover gather/scatter/index accumulation
and backward as a whole. This is broader than replacing one attention sum.
The sole explicit numerical adapter is a CPU scan for the short CDF vectors
which CUDA cumsum cannot execute under Torch 2.1 strict mode.
Never install this in a running production process. Scope is single-threaded.
"""
import contextlib
import os
import torch
from torch.autograd.function import once_differentiable

VERSION = 'strict_native_reductions_cpu_cdf_prefix_v1'
MAX_PREFIX = 4096

def prefix_sum(x):
    if x.ndim != 1 or x.numel() > MAX_PREFIX:
        raise ValueError('registered short one-dimensional CDF vector required')
    # Same dtype, inclusive prefix, no rescaling, clipping or CPU FP64 promotion.
    return x.to('cpu').cumsum(0).to(x.device)

class ReproducibleGroupedCDF(torch.autograd.Function):
    """Original _GroupedCDF math, changing only its two prefix-sum executions."""
    @staticmethod
    def forward(ctx, probabilities, target_cdf, bounds, capacities):
        batch, width = probabilities.shape
        h = probabilities.new_ones((batch, 1)); history = []
        for j in range(width):
            history.append(h); v = probabilities[:, j:j+1]
            h = torch.cat(((1-v)*h, h.new_zeros((batch, 1))), 1) + torch.cat((h.new_zeros((batch, 1)), v*h), 1)
        residuals, values = [], []
        for g, (lo, hi) in enumerate(bounds):
            cap = capacities[g]
            cdf = prefix_sum(h[lo:hi].mean(0))[:cap]
            e = cdf - target_cdf[g, :cap]
            residuals.append(e); values.append(.5 * e.square().mean())
        ctx.bounds, ctx.capacities = bounds, capacities
        ctx.save_for_backward(probabilities, *history, *residuals); ctx.width = width
        return torch.stack(values)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        saved = ctx.saved_tensors; p = saved[0]
        history = saved[1:1+ctx.width]; residuals = saved[1+ctx.width:]
        batch, width = p.shape; adj = p.new_zeros((batch, width+1))
        for g, (lo, hi) in enumerate(ctx.bounds):
            cap = ctx.capacities[g]; dcdf = grad_output[g]*residuals[g]/cap
            dh = prefix_sum(dcdf.flip(0)).flip(0)/(hi-lo)
            adj[lo:hi, :cap] = dh
        gradient = torch.empty_like(p)
        for j in range(width-1, -1, -1):
            h = history[j]
            gradient[:, j] = (h*(adj[:, 1:]-adj[:, :-1])).sum(1)
            v = p[:, j:j+1]; adj = (1-v)*adj[:, :-1]+v*adj[:, 1:]
        return gradient, None, None, None

@contextlib.contextmanager
def execution_policy(enabled):
    """Explicit temporary module binding and runtime policy, restored on failure."""
    import pb_degree
    import projected_degree
    mode = torch.are_deterministic_algorithms_enabled()
    warn = torch.is_deterministic_algorithms_warn_only_enabled()
    old_pb,old_ps = pb_degree._GroupedCDF,projected_degree._GroupedCDF
    if enabled and os.environ.get('CUBLAS_WORKSPACE_CONFIG') not in (':4096:8', ':16:8'):
        raise RuntimeError('set CUBLAS_WORKSPACE_CONFIG before process/CUDA startup')
    try:
        torch.use_deterministic_algorithms(bool(enabled), warn_only=False)
        if enabled:
            pb_degree._GroupedCDF = projected_degree._GroupedCDF = ReproducibleGroupedCDF
        yield dict(version=VERSION,strict=bool(enabled),warn_only=False,
                   cdf_prefix='CPU same-dtype' if enabled else 'original device',
                   max_prefix=MAX_PREFIX)
    finally:
        pb_degree._GroupedCDF,projected_degree._GroupedCDF=old_pb,old_ps
        torch.use_deterministic_algorithms(mode, warn_only=warn)
