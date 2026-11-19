"""Isolated feedback contract and ONE untrained count-score candidate.

No production imports, model calls, optimizer or sampling. Fixed scales come
from the reference and legal-slot inventory, never from a generated panel.
The bounded count score changes the objective and is NOT a bug fix to GDPO.
"""
import numpy as np


def loo(costs, standardize=False):
    """Cost advantage: positive means discourage; retain all samples."""
    x = np.asarray(costs, dtype=np.float64)
    if x.ndim != 1 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError('At least two finite scalar costs required')
    if standardize and len(x) < 4:
        raise ValueError('K<4 standardization disallowed: K=2 erases gap magnitude')
    a = (len(x)*x-x.sum())/(len(x)-1)
    if standardize:
        sd = float(x.std())
        if sd == 0: return np.zeros_like(a)
        a = a/sd
    return a


def count_terms(counts, reference, slots, mode='legacy'):
    """Return per-relation contributions; fixed support stays in the denominator.

    Candidate: mean_r [1-exp(-0.5*(m-m_ref)^2/v_ref)], where
    v_ref = S*p_ref*(1-p_ref), p_ref=m_ref/S. This is a bounded
    variance-scaled discrepancy, not a calibrated Gaussian test or likelihood.
    Zero/full fixed supports must match exactly; violations raise an error.
    """
    m, ref, S = (np.asarray(v, dtype=np.float64) for v in (counts, reference, slots))
    if m.shape != ref.shape or S.shape != ref.shape or m.ndim != 1 or len(m)==0:
        raise ValueError('Aligned nonempty relation vectors required')
    if not all(np.isfinite(v).all() for v in (m, ref, S)) or np.any(S<0):
        raise ValueError('Nonfinite or negative capacity')
    if np.any(m<0) or np.any(m>S) or np.any(ref<0) or np.any(ref>S):
        raise ValueError('Counts outside legal capacity')
    fixed = (ref==0)|(ref==S)
    if np.any(m[fixed]!=ref[fixed]): raise ValueError('Fixed-support violation')
    if mode=='legacy': return ((m-ref)/np.maximum(1,ref))**2/len(m)
    if mode!='bounded_variance': raise ValueError('Unknown score mode')
    active=~fixed; out=np.zeros_like(m)
    variance=ref[active]*(1-ref[active]/S[active])
    z2=(m[active]-ref[active])**2/variance
    out[active]=-np.expm1(-.5*z2)
    return out/len(m)


def feedback_record(joint_costs, count_costs, active_slots, *, standardize=False):
    h,n=(np.asarray(v,dtype=np.float64) for v in (joint_costs,count_costs))
    if h.shape!=n.shape or active_slots<=0: raise ValueError('Invalid shapes or slot denominator')
    ah,an=loo(h),loo(n)
    total=loo(h+n,standardize=standardize)
    return dict(joint_costs=h.tolist(),count_costs=n.tolist(),costs=(h+n).tolist(),
                joint_advantages=ah.tolist(),count_advantages=an.tolist(),advantages=total.tolist(),
                K=len(h),active_slots=int(active_slots),standardized=standardize,
                logscore_reduction='mean over active slots',terminal_time_reduction='mean',
                K2_components_collinear=len(h)==2,estimator='biased eager clean-head surrogate')


def eager_loss(log_prob_of_terminal_labels, advantage, *, trajectory_count, time_count, scale=.01):
    """Tensor-neutral explicit mean reduction; no trainable-parameter changes.

    This changes the historical sum scaling, not the sampling distribution.
    AdamW does not generally scale its step by this same factor.
    """
    if trajectory_count<2 or time_count<1 or log_prob_of_terminal_labels.numel()==0:
        raise ValueError('Invalid reduction support')
    return scale*float(advantage)*log_prob_of_terminal_labels.mean()/(trajectory_count*time_count)
