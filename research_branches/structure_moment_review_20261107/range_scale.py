"""Isolated CPU candidate: attainable-range scaling with weak-channel gating.

No learned model, optimizer, lambda calibration or production integration.
Endpoints must come from the same frozen reverse kernel and fixed support.
"""
import torch
from moments import moments

VERSION = 'typed_moment_attainable_range_soft_gate_v1'


def prepare(a, b, groups):
    if a.requires_grad or b.requires_grad:
        raise ValueError('kernel endpoints must be frozen')
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError('finite endpoints required')
    if bool(((a < 0) | (b > 1) | (a > b)).any()):
        raise ValueError('ordered Bernoulli endpoint support required')
    prepared = []
    with torch.no_grad():
        for group in groups:
            tri = group['triangles']
            if tri.numel() == 0:
                continue
            lo = moments(a, tri)[:2]
            hi = moments(b, tri)[:2]
            widths = tuple(h-l for h,l in zip(hi,lo))
            for width,h,l in zip(widths,hi,lo):
                if float(width) < 0:
                    raise FloatingPointError('negative attainable interval')
                # Reject unresolved cancellation instead of pretending it is a
                # well-measured small channel and dividing by an arbitrary eps.
                if bool(((b-a)[tri] > 0).any()) and float(width) == 0 and float(h) > 0:
                    raise FloatingPointError('unresolved zero interval width')
            prepared.append(dict(types=group['types'],triangles=tri,widths=widths))
    return prepared


def evaluate(q, k, prepared):
    values, rows = [], []
    for group in prepared:
        mq = moments(q, group['triangles'])[:2]
        mk = moments(k, group['triangles'])[:2]
        parts = []
        for order,(x,y,r) in enumerate(zip(mq,mk,group['widths']),1):
            diff = x-y
            if float(r) == 0:
                if float(diff) != 0:
                    raise FloatingPointError('nonzero residual on constant channel')
                part = q.sum()*0
                gate = r
            else:
                gate = r/(1+r)
                relative = diff/r
                if abs(float(relative.detach())) > 1+1e-8:
                    raise FloatingPointError('moment outside registered attainable range')
                part = gate*relative.square()
            parts.append(part)
            rows.append(dict(types=group['types'],order=order,width=float(r),gate=float(gate),
                             residual=float(diff.detach()),loss=float(part.detach())))
        values.append(parts[0]+parts[1])
    return (torch.stack(values).mean() if values else q.sum()*0),rows
