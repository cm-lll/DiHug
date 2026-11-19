"""Isolated differentiable likelihood of the existing binary reverse kernel.

Consumes actual clean probabilities, not reinterpreted logits. No epsilon, reward
clipping, slot averaging, optimizer, model, or changed generation rule. Float64
matches the production NumPy kernel's conversion. This is not wired to training.
"""
import torch


def reverse_probability(p0, xt, families, prior, alpha_bar, s, t):
    if p0.ndim != 1 or xt.shape != p0.shape or families.shape != p0.shape:
        raise ValueError('flat active/fixed slot arrays required')
    if not p0.is_floating_point():raise ValueError('floating clean probabilities required')
    dev=p0.device
    p=p0.to(torch.float64)
    prior=torch.as_tensor(prior,device=dev,dtype=torch.float64)
    alpha=torch.as_tensor(alpha_bar,device=dev,dtype=torch.float64)
    if prior.ndim!=1 or not bool(torch.isfinite(prior).all()) or bool(((prior<0)|(prior>1)).any()):
        raise ValueError('invalid prior')
    if alpha.ndim!=1 or not bool(torch.isfinite(alpha).all()) or alpha[0]!=1 or bool((torch.diff(alpha)>0).any()) or bool((alpha<0).any()) or bool((alpha[1:-1]<=0).any()):
        raise ValueError('invalid cumulative retention')
    if not 0<=s<t<len(alpha):raise ValueError('require 0<=s<t<=T')
    if families.dtype!=torch.long or bool(((families<0)|(families>=len(prior))).any()):
        raise ValueError('invalid families')
    if not bool(torch.isfinite(p).all()) or bool(((p<0)|(p>1)).any()) or bool(((xt!=0)&(xt!=1)).any()):
        raise ValueError('invalid p0 or observation')
    pi=prior[families];fixed=(pi==0)|(pi==1);active=~fixed
    if bool(((p[fixed]!=pi[fixed])|(xt[fixed]!=pi[fixed])).any()):
        raise ValueError('outside fixed support')
    # Fixed coordinates have zero gradient; no epsilon mass is introduced.
    result=pi.clone()+p*0
    if not bool(active.any()):return result
    qpi=pi[active];obs=xt[active].long();pp=p[active]
    stat=torch.stack((1-qpi,qpi),dim=-1)
    eye=torch.eye(2,device=dev,dtype=torch.float64)
    qs=alpha[s]*eye+(1-alpha[s])*stat[:,None,:]
    qt=alpha[t]*eye+(1-alpha[t])*stat[:,None,:]
    ratio=alpha[t]/alpha[s]
    transition=ratio*eye+(1-ratio)*stat[:,None,:]
    idx=obs[:,None,None].expand(-1,2,1)
    lik=torch.gather(transition,2,idx).squeeze(-1)
    den=torch.gather(qt,2,idx).squeeze(-1)
    if bool((den<=0).any()):raise ValueError('impossible active observation')
    posterior=qs*lik[:,None,:]/den[:,:,None]
    mixed=(torch.stack((1-pp,pp),-1)[:,:,None]*posterior).sum(1)[:,1]
    if not bool(torch.isfinite(mixed).all()) or bool(((mixed<0)|(mixed>1)).any()):
        raise FloatingPointError('invalid reverse probability')
    result[active]=mixed
    return result


def transition_log_probability(p0, xt, xprev, families, prior, alpha_bar, s, t):
    """Joint log probability over unordered legal slots, each counted once.

Caller must certify slot identity, coverage, frozen context and stochastic model
state. An impossible realized transition is an error, never silently floored.
"""
    if xprev.shape!=xt.shape or bool(((xprev!=0)&(xprev!=1)).any()):
        raise ValueError('invalid next graph')
    q=reverse_probability(p0,xt,families,prior,alpha_bar,s,t)
    chosen=torch.where(xprev.bool(),q,1-q)
    if bool((chosen<=0).any()):raise ValueError('zero-probability realized transition')
    return torch.log(chosen).sum()


def score_function_loss(logps, costs, total_steps, sampled_steps):
    """One update from independent on-policy paths; unbiased time subsampling.

logps: [paths, uniformly selected distinct times]. costs: [paths], terminal
reference-structure error. Leave-one-path-out baseline; no batch SD/clipping.
Baseline independence requires independent paths, NOT shared-noise paired paths.
Trajectory/parameter-dependent costs and off-policy reuse are not supported.
"""
    if logps.ndim!=2 or costs.shape!=(logps.shape[0],):raise ValueError('shape mismatch')
    if sampled_steps!=logps.shape[1] or not 1<=sampled_steps<=total_steps:
        raise ValueError('invalid uniform time subset')
    if not bool(torch.isfinite(logps).all()) or not bool(torch.isfinite(costs).all()):
        raise FloatingPointError('nonfinite trajectory log probability or cost')
    c=costs.detach()
    baseline=(c.sum()-c)/(len(c)-1) if len(c)>1 else torch.zeros_like(c)
    return ((c-baseline).detach()*logps.sum(1)*(total_steps/sampled_steps)).mean()
