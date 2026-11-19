"""Zero-feedback-safe combination, not a new loss or an AdamW safeguard.

Given estimated parameter gradients g_i, project their unweighted mean b onto
the cone C={v: <g_i,v> >= 0 for all i}. The intended gradient direction is -v.
A zero g_i remains in the problem but imposes no constraint. This differs from
simplex minimum-norm MGDA, where a zero vertex makes the solution zero.

Only the supplied estimates satisfy the first-order inequalities. Sampling
error, finite steps, and AdamW preconditioning invalidate stronger guarantees.
"""
import itertools
import numpy as np


def loo_gradient(probability, bits, costs):
    """Algebraically identical LOO; constant costs become EXACT floating zero.

    Subtracting an anchor before forming the leave-one-out difference cancels
    algebraically, including the anchor sample. It is not a new baseline or a
    change to the estimator's expectation. No estimated-zero group is dropped.
    """
    p, b, c = map(np.asarray, (probability, bits, costs))
    k = len(b)
    assert k > 1 and b.shape[1] == len(p) and c.shape[0] == k
    centered = c - c[0]
    advantage = (k * centered - centered.sum(0)) / (k - 1)
    return advantage.T @ (b - p) / k


def project_mean(gram):
    """Enumerate <=32 dual active sets for five objectives; no ridge.

    v = G.T @ (1/m + lambda); lambda >= 0.
    Dual KKT: H lambda + H(1/m) >= 0, complementary to lambda.
    Tolerances below are scale-relative solver verification, not loss gates.
    Rank decisions and tolerances are exported for review.
    """
    h = np.asarray(gram, dtype=np.float64)
    n = len(h)
    if h.shape != (n, n) or not n or not np.isfinite(h).all():
        raise ValueError('invalid Gram matrix')
    scale = float(np.max(abs(h)))
    a = np.ones(n) / n
    if scale == 0:
        return a, dict(status='all_estimates_zero', dual=[0.] * n,
                       residual=0., scale=0., norm_squared=0.)
    hs = h / scale
    if np.max(abs(hs-hs.T)) > 1e-12 or np.linalg.eigvalsh(hs).min() < -1e-10:
        raise ValueError('Gram matrix not symmetric positive semidefinite')
    b = hs @ a
    candidates = []
    for size in range(n + 1):
        for face in itertools.combinations(range(n), size):
            lam = np.zeros(n)
            if face:
                ix = list(face)
                sub = hs[np.ix_(ix, ix)]
                sol = np.linalg.lstsq(sub, -b[ix], rcond=1e-12)[0]
                if np.min(sol) < -1e-10:
                    continue
                lam[ix] = np.maximum(sol, 0)
            slack = hs @ lam + b
            residual = max(float(max(0, -slack.min())),
                           float(np.max(abs(slack[lam > 0]))) if np.any(lam > 0) else 0.)
            if residual > 1e-9:
                continue
            # Every verified KKT solution is optimal; choose smallest primal
            # correction, then smallest dual norm for redundant faces.
            candidates.append((float(lam @ hs @ lam), float(lam @ lam), lam, residual))
    if not candidates:
        raise RuntimeError('no verified cone projection; do not substitute zero')
    _, _, lam, residual = min(candidates, key=lambda x: (x[0], x[1]))
    coeff = a + lam
    return coeff, dict(status='verified', dual=lam.tolist(), residual=residual,
                       scale=scale, rank=int(np.linalg.matrix_rank(hs, tol=1e-12)),
                       norm_squared=float(max(0, coeff @ h @ coeff)),
                       estimated_inner_products=(h @ coeff).tolist(),
                       lstsq_rcond=1e-12, kkt_normalized_tolerance=1e-9)
