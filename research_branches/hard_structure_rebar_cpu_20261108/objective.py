"""CPU-only structural score and REBAR components; no model imports.

This is an estimator feasibility objective, NOT an adopted training loss.
Empirical per-type distributions of JOINT node relation degrees and typed
triangle participation are compared by a fixed Gaussian-kernel MMD V-statistic.
"""
import itertools
import numpy as np
import torch


class JointLocalScore:
    def __init__(self, types, query, families, reference, n_relations):
        self.types = np.asarray(types, dtype=np.int64)
        self.query = torch.as_tensor(np.asarray(query), dtype=torch.long)
        self.families = torch.as_tensor(np.asarray(families), dtype=torch.long)
        self.n = len(types)
        self.r = int(n_relations)
        self.type_values = sorted(set(self.types.tolist()))
        self.nodes = {t: torch.tensor(np.flatnonzero(self.types == t)) for t in self.type_values}
        self.triangle_pairs = list(itertools.combinations_with_replacement(self.type_values, 2))
        self.reference = torch.as_tensor(reference, dtype=torch.float64)
        assert self.reference.shape == (self.query.shape[1],)
        assert bool((self.query[0] != self.query[1]).all())
        assert len(set(map(tuple, np.sort(self.query.numpy(), axis=0).T))) == self.query.shape[1]
        with torch.no_grad():
            raw = self.features(self.reference)
            inventory = self.features(torch.ones_like(self.reference))
            self.masks = {t: inventory[ix].max(0).values > 0 for t, ix in self.nodes.items()}
            self.scales = {}
            self.targets = {}
            self.target_self = {}
            for t, ix in self.nodes.items():
                x = torch.log1p(raw[ix][:, self.masks[t]])
                self.scales[t] = torch.sqrt(1 + (x * x).mean(0))
                self.targets[t] = x / self.scales[t]
                self.target_self[t] = self.kernel(self.targets[t], self.targets[t]).mean()

    def features(self, weights):
        shape = weights.shape[:-1]
        flat = weights.new_zeros(*shape, self.n * self.n)
        i, j = self.query
        flat = flat.index_add(-1, i * self.n + j, weights)
        flat = flat.index_add(-1, j * self.n + i, weights)
        a = flat.reshape(*shape, self.n, self.n)
        degree = weights.new_zeros(*shape, self.n * self.r)
        degree = degree.index_add(-1, i * self.r + self.families, weights)
        degree = degree.index_add(-1, j * self.r + self.families, weights)
        degree = degree.reshape(*shape, self.n, self.r)
        triangles = []
        for ta, tb in self.triangle_pairs:
            ia, ib = self.nodes[ta], self.nodes[tb]
            left = a.index_select(-1, ia)
            middle = a.index_select(-2, ia).index_select(-1, ib)
            right = a.index_select(-1, ib)
            count = ((left @ middle) * right).sum(-1)
            triangles.append(count / (2 if ta == tb else 1))
        return torch.cat([degree, torch.stack(triangles, -1)], -1)

    @staticmethod
    def kernel(x, y):
        # Fixed bandwidth sqrt(number of included coordinates), no search.
        d = max(x.shape[-1], 1)
        distance = (x.square().sum(-1)[..., :, None] + y.square().sum(-1)[..., None, :]
                    - 2 * (x @ y.transpose(-1, -2))).clamp_min(0)
        return torch.exp(-distance / (2 * d))

    def __call__(self, weights):
        feature = self.features(weights)
        terms = []
        for t, ix in self.nodes.items():
            mask = self.masks[t]
            if not bool(mask.any()):
                terms.append(weights.sum(-1) * 0)
                continue
            x = torch.log1p(feature.index_select(-2, ix)[..., mask]) / self.scales[t]
            y = self.targets[t]
            terms.append(self.kernel(x, x).mean((-1, -2))
                         - 2 * self.kernel(x, y).mean((-1, -2)) + self.target_self[t])
        return torch.stack(terms, -1).mean(-1)


def conditional_logistic(logits, bits, v):
    """Exact reparameterization z|b, with b treated as fixed (REBAR eq.4).

    Stable algebra avoids subtracting 1-p when p is very near one.
    """
    pos = torch.logaddexp(torch.zeros_like(logits), v.log() + logits) - torch.log1p(-v)
    neg = v.log() - torch.logaddexp(torch.zeros_like(logits), torch.log1p(-v) - logits)
    return torch.where(bits.bool(), pos, neg)


def estimate_parts(score, probability, u, v, temperature=1.0):
    """Return RF, detached-baseline RF, zero-mean REBAR control, and hard bits.

    Fixed support is removed by caller. Every derivative is wrt free logits,
    NOT network parameters. No straight-through estimator is used.
    """
    logits = torch.logit(probability).detach().requires_grad_(True)
    p = logits.sigmoid()
    z = logits + u.log() - torch.log1p(-u)
    bits = (z.detach() >= 0).to(logits.dtype)
    z_cond = conditional_logistic(logits, bits, v)
    soft, soft_cond = (z / temperature).sigmoid(), (z_cond / temperature).sigmoid()
    with torch.no_grad():
        hard_value = score(bits)
        baseline = score(p.detach())
    soft_value, cond_value = score(soft), score(soft_cond)
    gs = torch.autograd.grad(soft_value.sum(), logits)[0]
    gc = torch.autograd.grad(cond_value.sum(), logits)[0]
    likelihood_score = bits - p.detach()
    rf = hard_value[..., None] * likelihood_score
    rf_base = (hard_value - baseline)[..., None] * likelihood_score
    control = -(cond_value.detach() - baseline)[..., None] * likelihood_score + gs - gc
    return dict(rf=rf.detach(), rf_base=rf_base.detach(), control=control.detach(),
                bits=bits, hard_value=hard_value, baseline=baseline,
                soft_value=soft_value.detach(), cond_value=cond_value.detach())


def edge_conditional(score, probability, bits, indices, chunk=16):
    """Integrate out each selected edge exactly, sample all OTHER edges.

    g_e = p_e(1-p_e)[D(B_-e,1)-D(B_-e,0)]. An unbiased Rao-Blackwellized
    coordinate estimate relative to raw score-function sampling. Does not
    claim every realization or comparison against REBAR has smaller error.
    """
    rows = []
    with torch.no_grad():
        for start in range(0, len(indices), chunk):
            ix = indices[start:start + chunk]
            lo = bits.expand(len(ix), -1).clone()
            hi = lo.clone()
            lo[torch.arange(len(ix)), ix] = 0
            hi[torch.arange(len(ix)), ix] = 1
            difference = score(hi) - score(lo)
            rows.append(probability[ix] * (1 - probability[ix]) * difference)
    return torch.cat(rows)
