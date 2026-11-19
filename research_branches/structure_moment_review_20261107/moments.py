"""CPU research prototype; no model, optimizer, or sampling entry point.

Exact first/second moments of typed triangle counts under independent edges.
Triangles can share edges: their covariance is retained, not approximated away.
"""
import torch


def moments(p, triangles):
    if triangles.numel() == 0:
        z = p.sum() * 0
        return z, z, z
    pt = p[triangles]
    vt = pt * (1 - pt)
    a = torch.zeros_like(p)
    for j in range(3):
        a = a.index_add(0, triangles[:, j], pt[:, (j+1) % 3] * pt[:, (j+2) % 3])
    # Orthogonal Bernoulli ANOVA terms, including all shared-edge covariance.
    variance = (p * (1-p) * a.square()).sum()
    for j in range(3):
        variance = variance + (pt[:, j].square() * vt[:, (j+1) % 3] * vt[:, (j+2) % 3]).sum()
    variance = variance + vt.prod(1).sum()
    mean = pt.prod(1).sum()
    return mean, mean.square() + variance, variance


def moment_loss(q, k, groups, scale):
    """Equal active-type weighting; scale fixed independently of prediction.

    This is a moment discrepancy, NOT a full distribution distance. No lambda
    is chosen and this function is not connected to any production entry point.
    """
    if scale < 1:
        raise ValueError('fixed scale must be >= 1')
    values, details = [], []
    for group in groups:
        tri = group['triangles']
        if tri.numel() == 0:
            continue
        mq, nq, vq = moments(q, tri)
        mk, nk, vk = moments(k, tri)
        first = ((mq-mk)/scale).square()
        second = ((nq-nk)/(scale*scale)).square()
        values.append(first+second)
        details.append(dict(types=group['types'], mean_q=mq, mean_k=mk,
                            variance_q=vq, variance_k=vk, second_q=nq, second_k=nk,
                            first_loss=first, second_loss=second))
    return (torch.stack(values).mean() if values else q.sum()*0), details
