"""CPU-only soft degree-conditioned distribution score, not a model loss integration.

Reference-only soft neighborhoods; projected weighted CDF discrepancy retains
within-neighborhood distributions. Finite projections are not an injective code.
"""
import numpy as np
from quantile import directions


def membership(x, anchors, h2):
    dist = np.square(x[:, None] - anchors[None]).sum(-1)
    k = np.square(1. / (1. + dist / h2[None]))
    return k / k.sum(1, keepdims=True)


def neighborhoods(x):
    anchors = np.unique(x, axis=0)
    if len(anchors) == 1:
        return anchors, np.ones(1)
    dist = np.square(anchors[:, None] - anchors[None]).sum(-1)
    k = min(len(anchors)-1, int(np.ceil(np.sqrt(len(anchors)))))
    h2 = np.sort(dist, axis=1)[:, k]
    assert np.all(h2 > 0)
    return anchors, h2


def cdf_distance(x, y, wx, wy, prior, u):
    """Exact integral of squared projected empirical CDF difference.

    wx/wy: nodes x soft groups, each column sums to 1.
    Repeated locations contribute zero-length intervals, no tie-order dependence.
    """
    if x.shape[1] == 0:
        return 0.
    xp = np.concatenate([x, x @ u], axis=1)
    yp = np.concatenate([y, y @ u], axis=1)
    z = np.concatenate([xp, yp], axis=0).T
    signed = np.concatenate([wx, -wy], axis=0)
    order = np.argsort(z, axis=1, kind='stable')
    ordered = np.take_along_axis(z, order, axis=1)
    diff = np.cumsum(signed[order], axis=1)
    by_projection = (np.square(diff[:, :-1]) * np.diff(ordered, axis=1)[:, :, None]).sum(1) @ prior
    d = x.shape[1]
    return float(.5 * by_projection[:d].mean() + .5 * by_projection[d:].mean())


class ConditionalScore:
    def __init__(self, core):
        self.core = core
        self.cache = {}
        ref = core.features(core.reference).numpy()
        for t, ids in core.nodes.items():
            ids = ids.numpy()
            mask = core.masks[t].numpy()
            dc = np.flatnonzero(mask[:core.r])
            hc = np.flatnonzero(mask[core.r:])
            d, h = self.local(ref, ids)
            d = d[:, dc]; h = h[:, hc]
            ds = np.maximum(1., np.sqrt(np.square(d).mean(0))) * np.sqrt(max(1, len(dc)))
            hs = np.maximum(1., np.sqrt(np.square(h).mean(0)))
            blocks = [hc < core.base_width-core.r, hc >= core.base_width-core.r]
            active = sum(bool(b.any()) for b in blocks)
            for b in blocks:
                if b.any(): hs[b] *= np.sqrt(active * b.sum())
            d = d/ds; h = h/hs
            anchors, h2 = neighborhoods(d)
            q = membership(d, anchors, h2)
            prior = q.mean(0)
            self.cache[t] = dict(ids=ids, dc=dc, hc=hc, ds=ds, hs=hs,
                d=d, h=h, anchors=anchors, h2=h2, prior=prior, weights=q/q.sum(0),
                ud=directions(len(dc), t), uh=directions(len(hc), t))

    def local(self, raw, ids):
        c = self.core
        d = raw[ids, :c.r]
        p = raw[ids, c.base_width:].reshape(len(ids), c.r, c.r)
        den = d.sum(0)
        mu = np.divide(p.sum(0), den[:, None], out=np.zeros((c.r, c.r)), where=den[:, None] > 0)
        residual = p-d[:, :, None]*mu[None]
        h = np.concatenate([raw[ids, c.r:c.base_width], residual.reshape(len(ids), -1)], axis=1)
        return d, h

    def terms(self, raw):
        raw = np.asarray(raw)
        if raw.ndim == 2: raw = raw[None]
        degree = np.zeros(len(raw)); conditional = np.zeros(len(raw))
        for i, a in enumerate(raw):
            for c in self.cache.values():
                d, h = self.local(a, c['ids'])
                d=d[:, c['dc']]/c['ds']; h=h[:, c['hc']]/c['hs']
                q=membership(d, c['anchors'], c['h2']); w=q/q.sum(0)
                uniform=np.ones((len(d), 1))/len(d)
                degree[i] += cdf_distance(d, c['d'], uniform, uniform, np.ones(1), c['ud'])
                conditional[i] += cdf_distance(h, c['h'], w, c['weights'], c['prior'], c['uh'])
        degree/=len(self.cache); conditional/=len(self.cache)
        total=.5*(degree+conditional)
        if not np.isfinite(total).all() or (total < -1e-12).any():
            raise FloatingPointError('invalid conditional score')
        return dict(total=total, degree=degree, conditional=conditional)

    def configuration(self):
        return {str(t):dict(anchors=c['anchors'].tolist(), h2=c['h2'].tolist(),
            prior=c['prior'].tolist(), degree_scale=c['ds'].tolist(), structure_scale=c['hs'].tolist(),
            reference_effective_nodes=(1/np.square(c['weights']).sum(0)).tolist())
            for t,c in self.cache.items()}
