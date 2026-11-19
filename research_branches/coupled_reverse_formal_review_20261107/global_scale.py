"""One documented scale revision; keep the original diagnostic version intact.

Sum absolute squared typed-count risks with ONE reference-volume denominator.
No per-type inverse-frequency weighting, online gradient scaling or fitted gain.
This changes the structural metric; it is not merely a numerical optimization.
"""
import torch

VERSION = 'coupled_reverse_typed_triangle_global_volume_v2'


def evaluate(context, p, query, families, *, accumulation_dtype=torch.float64):
    _, rows = context.evaluate(p, query, families, accumulation_dtype=accumulation_dtype)
    reference_triangles = sum(g['target'] for g in context.owner.groups)
    denominator = max(reference_triangles, 1)**2
    values = [g['risk'] for g in rows if g['active']]
    loss = torch.stack(values).sum()/denominator if values else p.sum()*0
    return loss, rows, dict(version=VERSION, reference_triangles=reference_triangles,
                           denominator=denominator, aggregation='sum of typed risks',
                           changes_metric_relative_to_v1=True)
