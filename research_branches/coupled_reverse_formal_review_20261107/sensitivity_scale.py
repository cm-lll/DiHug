"""Candidate local structural-sensitivity normalization, no model/RNG calls.

For independent teacher edges and monotonically coupled trials, near q=k:
R = sum_{e,c} |q_e-k_e| E_k[C_ec**2] + O(||q-k||**2).
Since q-k=d*(p-y), the summed inward p-coordinate slopes equal A below.
"""
import numpy as np
import torch
from global_scale import evaluate as volume_evaluate

VERSION = 'coupled_reverse_teacher_local_sensitivity_v1'

def denominator(context):
    k=np.asarray(context.k,dtype=np.float64)
    d=np.asarray(context.b-context.a,dtype=np.float64)
    if not np.isfinite(d).all() or (d < -1e-12).any():
        raise ValueError('non-monotone or nonfinite binary posterior sensitivity')
    rounded_negative=int((d<0).sum()); d=np.maximum(d,0.)
    owner=context.owner; total=0.; rows=[]; edge_contributions=np.zeros(len(k))
    for group in owner.groups:
        tri=group['triangles']; mean=np.zeros(len(k)); variance=np.zeros(len(k))
        for j in range(3):
            z=k[tri[:,(j+1)%3]]*k[tri[:,(j+2)%3]]
            mean+=np.bincount(tri[:,j],weights=z,minlength=len(k))
            variance+=np.bincount(tri[:,j],weights=z*(1-z),minlength=len(k))
        second=mean*mean+variance
        contribution=d*second
        value=float(contribution.sum());total+=value;edge_contributions+=contribution
        rows.append(dict(types=group['types'],active=group['active'],possible=group['possible'],
                         target=group['target'],sensitivity=value,
                         mean_square_part=float((d*mean*mean).sum()),
                         variance_part=float((d*variance).sum())))
    if not np.isfinite(total) or total<0:raise FloatingPointError('invalid local sensitivity')
    return max(total,1.),dict(version=VERSION,raw_sensitivity=total,denominator=max(total,1.),
        floor_applied=total<1.,rounded_negative_deltas=rounded_negative,groups=rows,
        edge_contributions=edge_contributions,delta=d)

def evaluate(context,p,query,families,*,accumulation_dtype=torch.float64):
    denom,detail=denominator(context)
    old,groups,scale=volume_evaluate(context,p,query,families,accumulation_dtype=accumulation_dtype)
    # denom is detached analytically: depends only on teacher kernel and frozen Xt.
    return old*(scale['denominator']/denom),groups,detail
