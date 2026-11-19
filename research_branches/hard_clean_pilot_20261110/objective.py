"""CPU hard-graph costs and unbiased LOO margin gradients. No edge editing.

The five groups extend the finite experiment: relation counts, anonymous joint
relation degrees, reference-present and reference-absent typed triangles, and
component count. MGDA of noisy gradients is NOT an unbiased MGDA direction.
"""
import itertools
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


class HardScore:
    names = ['relation_count', 'joint_degree', 'present_triangles',
             'absent_triangles', 'components']

    def __init__(self, types, query, families, reference, n_relations):
        self.types = np.asarray(types, dtype=np.int64)
        self.q = np.asarray(query, dtype=np.int64)
        self.f = np.asarray(families, dtype=np.int64)
        self.n, self.r = len(types), int(n_relations)
        self.reference = np.asarray(reference, dtype=np.uint8)
        assert self.q.shape == (2, len(self.reference))
        assert np.all(self.q[0] < self.q[1])
        assert len(set(map(tuple, self.q.T))) == self.q.shape[1]
        self.nodes = [np.flatnonzero(self.types == t) for t in np.unique(types)]
        self.triples = list(itertools.combinations_with_replacement(np.unique(types).tolist(), 3))
        self.tindex = {t: i for i, t in enumerate(self.triples)}
        self.ref = self.features(self.reference)
        inventory = self.features(np.ones_like(self.reference))
        self.present = self.ref['triangles'] > 0
        self.absent = (inventory['triangles'] > 0) & ~self.present
        # Only mathematically impossible groups are omitted, never noisy zero gradients.
        self.active = [0, 1] + ([2] if self.present.any() else []) + ([3] if self.absent.any() else []) + [4]

    def features(self, bits):
        bits = np.asarray(bits)
        assert bits.shape == self.reference.shape and np.isin(bits, [0, 1]).all()
        i, j = self.q[:, bits.astype(bool)]
        f = self.f[bits.astype(bool)]
        a = np.zeros((self.n, self.n), dtype=bool)
        a[i, j] = True; a[j, i] = True
        degree = np.zeros((self.n, self.r), dtype=np.int64)
        np.add.at(degree, (i, f), 1); np.add.at(degree, (j, f), 1)
        tri = np.zeros(len(self.triples), dtype=np.int64)
        for u, v in zip(i, j):
            for w in np.flatnonzero(a[u] & a[v]):
                if w > v:
                    tri[self.tindex[tuple(sorted((int(self.types[u]), int(self.types[v]), int(self.types[w]))))]] += 1
        nc, component = connected_components(csr_matrix(a), directed=False)
        return dict(counts=np.bincount(f, minlength=self.r), degree=degree,
                    triangles=tri, components=int(nc),
                    lcc=int(np.bincount(component).max()), isolated=int((degree.sum(1) == 0).sum()),
                    max_degree=int(degree.sum(1).max()))

    def costs(self, bits):
        v = self.features(bits); ref = self.ref
        joint = []
        for ix in self.nodes:
            c = abs(v['degree'][ix, None, :] - ref['degree'][ix][None, :, :]).sum(-1)
            r, col = linear_sum_assignment(c)
            joint.append(float(c[r, col].mean()) / max(1, self.n - 1))
        terr = ((v['triangles'] - ref['triangles']) / np.maximum(1, ref['triangles'])) ** 2
        cost = np.array([np.mean(((v['counts'] - ref['counts']) / np.maximum(1, ref['counts'])) ** 2),
                         np.mean(joint), terr[self.present].mean() if self.present.any() else 0.,
                         terr[self.absent].mean() if self.absent.any() else 0.,
                         ((v['components'] - ref['components']) / max(1, self.n - 1)) ** 2])
        assert np.isfinite(cost).all()
        return cost, v

    def batch(self, bits):
        return np.stack([self.costs(b)[0] for b in bits])


def loo_gradient(probability, bits, costs):
    """K independent hard draws; no sample-std normalization or clipping."""
    p, b, c = map(np.asarray, (probability, bits, costs))
    k = len(b)
    assert k > 1 and b.shape[1] == len(p) and c.shape[0] == k
    advantage = (k * c - c.sum(0)) / (k - 1)
    return advantage.T @ (b - p) / k


def solve_gram(h):
    """Rank-aware simplex minimum norm; no ridge, no hidden gradient scaling."""
    h = np.asarray(h, dtype=np.float64)
    scale = max(float(abs(h).max()), 1e-30); hs = h / scale
    n = len(h); candidates = []
    for size in range(1, n + 1):
        for ix0 in itertools.combinations(range(n), size):
            ix = list(ix0); sub = hs[np.ix_(ix, ix)]
            a = np.block([[sub, np.ones((size, 1))], [np.ones((1, size)), np.zeros((1, 1))]])
            rhs = np.r_[np.zeros(size), 1.]
            sol = np.linalg.lstsq(a, rhs, rcond=1e-12)[0]
            if abs(a @ sol - rhs).max() > 1e-8 or sol[:-1].min() < -1e-10:
                continue
            lam = np.zeros(n); lam[ix] = np.maximum(sol[:-1], 0); lam /= lam.sum()
            candidates.append((float(lam @ hs @ lam), lam))
    if not candidates: raise RuntimeError('no feasible simplex face')
    _, lam = min(candidates, key=lambda x: x[0]); hv = hs @ lam; val = float(lam @ hv)
    residual = max(float(max(0, -np.min(hv - val))), float(abs((hv - val)[lam > 1e-9]).max()), abs(lam.sum() - 1))
    if residual > 1e-8: raise RuntimeError('MGDA KKT residual')
    return lam, residual
