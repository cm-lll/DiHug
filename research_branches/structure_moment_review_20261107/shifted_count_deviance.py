"""One-count-shifted triangle mean deviance; isolated CPU candidate.

The common +1 is a count unit, not a fitted scale or gradient limit.
Does not represent the KL of the true graph triangle-count law.
"""
import torch
from count_deviance import reverse_log_probability

VERSION = 'typed_triangle_mean_one_count_deviance_v1'

def scalar_deviance(log_mu, m):
    if not torch.isfinite(m) or float(m) < 0:
        raise ValueError('invalid teacher mean')
    if torch.isnan(log_mu) or torch.isposinf(log_mu):
        raise ValueError('invalid predicted log mean')
    # log(mu + 1), without clipping small positive probabilities.
    d = torch.logaddexp(log_mu, torch.zeros_like(log_mu)) - torch.log1p(m)
    if abs(float(d.detach())) < 1e-4:
        # Stable expm1(d)-d; truncation error < 2e-30 in this interval.
        remainder = d*d*(.5+d*(1/6+d*(1/24+d*(1/120+d/720))))
        return (m+1)*remainder
    if abs(float(d.detach())) < .5:
        return (m+1)*(torch.expm1(d)-d)
    return log_mu.exp()-m-(m+1)*d

def evaluate(logq, k, groups):
    if k.requires_grad or not torch.isfinite(k).all() or bool(((k<0)|(k>1)).any()):
        raise ValueError('frozen valid teacher probabilities required')
    if torch.isnan(logq).any() or bool((logq>1e-14).any()):
        raise ValueError('invalid log probability')
    terms, rows = [], []
    for group in groups:
        tri = group['triangles']
        if not tri.numel():
            continue
        products = logq[tri].sum(1)
        possible = torch.isfinite(products)
        m = k[tri].prod(1).sum()
        if not possible.any():
            # Smoothing must not conceal an inconsistent kernel/support contract.
            if float(m)>0:
                raise ValueError('positive teacher count outside predicted support')
            term = logq[torch.isfinite(logq)].sum()*0
            log_mu, mu = None, term
        else:
            lm = torch.logsumexp(products[possible], dim=0)
            term = scalar_deviance(lm, m)
            log_mu, mu = float(lm.detach()), lm.exp()
        if not torch.isfinite(term):
            raise FloatingPointError('nonfinite shifted deviance')
        terms.append(term)
        rows.append(dict(types=group['types'], triangle_slots=len(tri),
                         mean_q=float(mu.detach()), mean_k=float(m),
                         log_mean_q=log_mu, loss=float(term.detach())))
    if not terms:
        return logq[torch.isfinite(logq)].sum()*0, rows
    return torch.stack(terms).mean(), rows
