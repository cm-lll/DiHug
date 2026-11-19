"""CPU prototype: typed triangle MEAN deviance, not the true count-law KL.

No gradient budgets, model calls, optimizer, learned scales or fitted dispersion.
Use log probabilities to preserve very small nonzero triangle intensities.
"""
import torch
import torch.nn.functional as F

VERSION='typed_triangle_mean_poisson_deviance_v1'

def reverse_log_probability(z,a,delta):
    if z.shape!=a.shape or z.shape!=delta.shape:
        raise ValueError('shape mismatch')
    if not all(torch.isfinite(x).all() for x in (z,a,delta)):
        raise FloatingPointError('nonfinite input')
    if bool(((a<0)|(delta<0)|(a+delta>1+1e-14)).any()):
        raise ValueError('invalid reverse support')
    live=(a>0)|(delta>0)
    # Preserve a zero derivative path even if every slot is fixed impossible.
    out=z*0+torch.full_like(z,-float('inf'))
    # Constants may be -inf. Never evaluate logaddexp(-inf,-inf), whose
    # derivative is undefined, for a structurally impossible edge.
    out[live]=torch.logaddexp(a[live].log(),delta[live].log()+F.logsigmoid(z[live]))
    return out

def scalar_deviance(log_mu,m):
    if not torch.isfinite(m) or float(m)<0:raise ValueError('invalid teacher mean')
    if float(m)==0:return log_mu.exp()
    if not torch.isfinite(log_mu):raise ValueError('positive teacher count outside predicted support')
    d=log_mu-m.log()
    # Equivalent expressions, with expm1 near equality to avoid cancellation.
    return m*(torch.expm1(d)-d) if abs(float(d.detach()))<.5 else log_mu.exp()-m-m*d

def evaluate(logq,k,groups):
    if k.requires_grad or not torch.isfinite(k).all() or bool(((k<0)|(k>1)).any()):
        raise ValueError('frozen valid teacher probabilities required')
    if torch.isnan(logq).any() or bool((logq>1e-14).any()):raise ValueError('invalid log probability')
    terms=[];rows=[]
    for group in groups:
        tri=group['triangles']
        if not tri.numel():continue
        log_products=logq[tri].sum(1)
        possible=torch.isfinite(log_products)
        m=k[tri].prod(1).sum()
        if not possible.any():
            if float(m)>0:raise ValueError('positive teacher count outside predicted support')
            # Impossible at this kernel boundary, not an epsilon-smoothed event.
            term=logq[torch.isfinite(logq)].sum()*0
            log_mu=None;mu=term
        else:
            lm=torch.logsumexp(log_products[possible],dim=0)
            term=scalar_deviance(lm,m);log_mu=float(lm.detach());mu=lm.exp()
        if not torch.isfinite(term):raise FloatingPointError('nonfinite deviance')
        terms.append(term)
        rows.append(dict(types=group['types'],triangle_slots=len(tri),mean_q=float(mu.detach()),
                         mean_k=float(m),log_mean_q=log_mu,loss=float(term.detach())))
    if not terms:return logq[torch.isfinite(logq)].sum()*0,rows
    return torch.stack(terms).mean(),rows
