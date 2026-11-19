"""Dataset-agnostic local scale prototype, not connected to training.

Inputs are kernel endpoints and an independent-prior clean posterior anchor.
No dataset identifiers, clean labels, reference triangle counts, model outputs,
or generated quality scores are used to prepare the scale.
"""
import torch
from moments import moments

VERSION = 'local_posterior_linear_variance_units_v1'


def prepare(a, b, anchor, groups):
    for x in (a,b,anchor):
        if x.requires_grad or not torch.isfinite(x).all():
            raise ValueError('finite frozen inputs required')
    if a.shape!=b.shape or a.shape!=anchor.shape:
        raise ValueError('shape mismatch')
    if bool(((a<0)|(b>1)|(a>b)|(anchor<0)|(anchor>1)).any()):
        raise ValueError('support violation')
    delta=b-a
    reference=(a+delta*anchor).detach()
    uncertainty=anchor*(1-anchor)
    prepared=[]
    for group in groups:
        tri=group['triangles']
        if not tri.numel():continue
        # Only mathematical moment differentiation; never a denoiser call.
        r=reference.clone().requires_grad_(True)
        mu,nu,_=moments(r,tri)
        j1=torch.autograd.grad(mu,r,retain_graph=True)[0].detach()*delta
        j2=torch.autograd.grad(nu,r)[0].detach()*delta
        v1=(j1.square()*uncertainty).sum()
        v2=(j2.square()*uncertainty).sum()
        unit1=torch.ones_like(mu).detach()
        unit2=(2*mu.detach()+1)
        scale1=(unit1.square()+v1).detach()
        scale2=(unit2.square()+v2).detach()
        if not torch.isfinite(scale1+scale2) or float(scale1)<1 or float(scale2)<1:
            raise FloatingPointError('invalid local scale')
        prepared.append(dict(types=group['types'],triangles=tri,
            scales=(scale1,scale2),anchor_mean=float(mu.detach()),
            anchor_second=float(nu.detach()),linear_variances=(float(v1),float(v2)),
            units=(float(unit1),float(unit2))))
    return prepared


def evaluate(q,k,prepared):
    values=[];rows=[]
    for group in prepared:
        u=moments(q,group['triangles'])[:2];v=moments(k,group['triangles'])[:2]
        terms=[]
        for order,(x,y,s) in enumerate(zip(u,v,group['scales']),1):
            part=(x-y).square()/s
            if not torch.isfinite(part):raise FloatingPointError('nonfinite moment loss')
            terms.append(part)
            rows.append(dict(types=group['types'],order=order,scale_squared=float(s),
                linear_variance=group['linear_variances'][order-1],unit=group['units'][order-1],
                anchor_mean=group['anchor_mean'],residual=float((x-y).detach()),loss=float(part.detach())))
        values.append(sum(terms))
    return (torch.stack(values).mean() if values else q.sum()*0),rows
