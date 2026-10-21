"""Two-state family marginal kernel; clean t=0; exact Bayes s<t bridge.

Separate from legacy T+1 cosine indexing and fixed-count initialization.
No epsilon probability support, MASK, exact-degree selection, or absorbing kernel.
"""
import numpy as np


class MarginalKernel:
    def __init__(self, prior, alpha_bar):
        self.prior = np.asarray(prior, dtype=np.float64)
        self.alpha = np.asarray(alpha_bar, dtype=np.float64)
        if self.prior.ndim != 1 or not np.isfinite(self.prior).all() or ((self.prior < 0) | (self.prior > 1)).any():
            raise ValueError("invalid Bernoulli family prior")
        if self.alpha.ndim != 1 or not np.isfinite(self.alpha).all() or self.alpha[0] != 1 or (np.diff(self.alpha) > 0).any() or (self.alpha < 0).any():
            raise ValueError("alpha_bar must start exactly at 1 and decrease to >=0")
        if (self.alpha[1:-1] <= 0).any():
            raise ValueError("zero cumulative retention only allowed at terminal time")

    def matrix(self, retention, families):
        pi = self.prior[np.asarray(families, dtype=np.int64)]
        stationary = np.stack([1 - pi, pi], axis=-1)
        return retention * np.eye(2) + (1 - retention) * stationary[..., None, :]

    def reverse(self, p0, xt, families, s, t):
        p0, xt = np.asarray(p0, dtype=np.float64), np.asarray(xt, dtype=np.int64)
        f = np.asarray(families, dtype=np.int64)
        if not 0 <= s < t < len(self.alpha):
            raise ValueError("require 0<=s<t<=T")
        if p0.shape != xt.shape or f.shape != xt.shape or ((p0 < 0) | (p0 > 1)).any() or not np.isfinite(p0).all() or ((xt != 0) & (xt != 1)).any():
            raise ValueError("invalid prediction/observation")
        fixed0, fixed1 = self.prior[f] == 0, self.prior[f] == 1
        if (xt[fixed0] != 0).any() or (xt[fixed1] != 1).any() or (p0[fixed0] != 0).any() or (p0[fixed1] != 1).any():
            raise ValueError("prediction/observation outside exact fixed-family support")
        q_s = self.matrix(self.alpha[s], f)
        q_t = self.matrix(self.alpha[t], f)
        transition = self.matrix(self.alpha[t] / self.alpha[s], f)
        likelihood = np.take_along_axis(transition, xt[:, None, None], axis=2)[..., 0]
        denominator = np.take_along_axis(q_t, xt[:, None, None], axis=2)[..., 0]
        weights = np.stack([1 - p0, p0], axis=-1)
        if ((denominator == 0) & (weights > 0)).any():
            raise ValueError("positive prediction weight on impossible conditional event")
        posterior = np.divide(q_s * likelihood[:, None, :], denominator[:, :, None],
                              out=np.zeros_like(q_s), where=denominator[:, :, None] > 0)
        return np.einsum("ni,nij->nj", weights, posterior)[:, 1]


def full_sweep(plan, state, kernel, predict_clean, uniforms, s, t):
    """Frozen X_t for EVERY query; commit X_s only after exact full coverage.

    callback receives (copy of entire state, queried IDs, families, endpoints).
    Used for deterministic probability tests; not a scientific sampling command.
    Conditional independence applies given X_t AND random query plan. Integrating
    over random plans need not remain independent (queries participate in MP).
    """
    x = np.asarray(state, dtype=np.int64)
    u = np.asarray(uniforms, dtype=np.float64)
    if x.shape != (plan.space.total,) or u.shape != x.shape or ((u < 0) | (u >= 1)).any():
        raise ValueError("state/uniform shape or range")
    result = x.copy()
    visited = np.zeros(len(x), dtype=bool)
    for f, active in enumerate(plan.active):
        if active:
            continue
        prior = kernel.prior[f]
        interval = slice(plan.space.offsets[f], plan.space.offsets[f + 1])
        if prior not in (0., 1.) or not (x[interval] == int(prior)).all():
            raise ValueError("cannot skip a nonfixed family or inconsistent fixed state")
    for r in range(plan.rounds):
        ids, f, edges = plan.block(r)
        if not len(ids):
            continue
        if visited[ids].any():
            raise AssertionError("duplicate slot update")
        p0 = predict_clean(x.copy(), ids, f, edges)
        q = kernel.reverse(p0, x[ids], f, s, t)
        result[ids] = u[ids] < q
        visited[ids] = True
    for f, active in enumerate(plan.active):
        interval = slice(plan.space.offsets[f], plan.space.offsets[f + 1])
        if active and not visited[interval].all() or not active and visited[interval].any():
            raise AssertionError("global time advanced before complete active-slot coverage")
    return result
