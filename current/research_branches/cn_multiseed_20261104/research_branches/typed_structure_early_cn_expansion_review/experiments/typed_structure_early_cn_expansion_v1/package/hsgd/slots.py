"""Uniform random query blocks over typed, undirected legal slots.

No adjacency, BFS, communities, positive-edge-only selection or node ID feature.
IDs below are storage indices, not conditioning features. Strata are unordered
endpoint types with one binary relation per pair; parallel relations are rejected.
"""
from dataclasses import dataclass
import math
import numpy as np


@dataclass
class TypedSlots:
    types: np.ndarray
    pairs: tuple

    def __post_init__(self):
        self.types = np.asarray(self.types, dtype=np.int64).copy()
        if self.types.ndim != 1 or (self.types < 0).any():
            raise ValueError("types must be a nonnegative vector")
        self.pairs = tuple(tuple(int(v) for v in p) for p in self.pairs)
        if len(set(self.pairs)) != len(self.pairs) or any(a > b for a, b in self.pairs):
            raise ValueError("one unique unordered family per type pair required")
        self.nodes = {t: np.flatnonzero(self.types == t) for p in self.pairs for t in p}
        self.starts = {}
        counts = []
        for a, b in self.pairs:
            na, nb = len(self.nodes[a]), len(self.nodes[b])
            counts.append(na * (na - 1) // 2 if a == b else na * nb)
            if a == b:
                i = np.arange(na, dtype=np.int64)
                self.starts[a] = i * (2 * na - i - 1) // 2
        self.counts = np.asarray(counts, dtype=np.int64)
        self.offsets = np.r_[0, np.cumsum(self.counts)]
        self.total = int(self.offsets[-1])

    def decode(self, family, local):
        local = np.asarray(local, dtype=np.int64)
        if ((local < 0) | (local >= self.counts[family])).any():
            raise ValueError("slot outside family")
        a, b = self.pairs[family]
        if a == b:
            i = np.searchsorted(self.starts[a], local, side="right") - 1
            j = i + 1 + local - self.starts[a][i]
        else:
            i, j = np.divmod(local, len(self.nodes[b]))
        u, v = self.nodes[a][i], self.nodes[b][j]
        return np.stack([np.minimum(u, v), np.maximum(u, v)])

    def encode(self, edges):
        edges = np.asarray(edges, dtype=np.int64).reshape(2, -1)
        if ((edges < 0) | (edges >= len(self.types))).any() or (edges[0] == edges[1]).any():
            raise ValueError("illegal endpoints")
        positions = np.empty(len(self.types), dtype=np.int64)
        for nodes in self.nodes.values():
            positions[nodes] = np.arange(len(nodes))
        families = np.full(edges.shape[1], -1, dtype=np.int64)
        local = np.empty(edges.shape[1], dtype=np.int64)
        for f, (a, b) in enumerate(self.pairs):
            t0, t1 = self.types[edges[0]], self.types[edges[1]]
            mask = ((t0 == a) & (t1 == b)) | ((t0 == b) & (t1 == a))
            u, v = edges[:, mask]
            if a == b:
                i, j = np.minimum(positions[u], positions[v]), np.maximum(positions[u], positions[v])
                local[mask] = self.starts[a][i] + j - i - 1
            else:
                i = np.where(self.types[u] == a, positions[u], positions[v])
                j = np.where(self.types[v] == b, positions[v], positions[u])
                local[mask] = i * len(self.nodes[b]) + j
            families[mask] = f
        if (families < 0).any():
            raise ValueError("edge outside schema")
        return families, local


class UniformQueryPlan:
    """Family-stratified uniform permutations, followed by balanced query rounds.

    Every active legal slot occurs exactly once per sweep. Choose a training
    round uniformly: every ACTIVE slot has inclusion probability 1/R, including
    tiny families. Inactive families stay in schema but must have exact fixed
    support, and may be skipped using active_families (default: query all).
    Equal family quotas would be a different, biased full-slot CE estimator.
    Memory O(S) integer indices, sparse neural graph O(N+m+query_cap), NOT O(S)
    learned activations. No claim of subquadratic total reverse sweep work.
    """
    def __init__(self, space, seed, rounds=None, query_cap=None, active_families=None):
        self.space = space
        self.active = np.ones(len(space.pairs), dtype=bool) if active_families is None else np.asarray(active_families, dtype=bool).copy()
        if self.active.shape != space.counts.shape:
            raise ValueError("active family mask must match schema")
        self.query_counts = np.where(self.active, space.counts, 0)
        self.query_total = int(self.query_counts.sum())
        if (rounds is None) == (query_cap is None):
            raise ValueError("specify rounds OR query_cap")
        nonempty = int((self.query_counts > 0).sum())
        if query_cap is not None:
            if query_cap <= nonempty:
                raise ValueError("cap must exceed number of nonempty families")
            rounds = max(1, math.ceil(self.query_total / (query_cap - nonempty)))
        if int(rounds) != rounds or rounds < 1:
            raise ValueError("positive integer rounds required")
        self.rounds = int(rounds)
        seeds = np.random.SeedSequence(seed).spawn(len(space.pairs))
        self.permutations, self.cuts = [], []
        for size, child in zip(self.query_counts, seeds):
            rng = np.random.default_rng(child)
            perm = rng.permutation(int(size))
            quota = np.full(self.rounds, size // self.rounds, dtype=np.int64)
            quota[rng.permutation(self.rounds)[:int(size % self.rounds)]] += 1
            self.permutations.append(perm)
            self.cuts.append(np.r_[0, quota.cumsum()])
        self.sizes = sum((np.diff(c) for c in self.cuts), np.zeros(self.rounds, dtype=np.int64))
        if query_cap is not None and (self.sizes > query_cap).any():
            raise AssertionError("query cap violated")

    def block(self, round_id):
        if not 0 <= round_id < self.rounds:
            raise ValueError("round index outside plan")
        ids, families, edges = [], [], []
        for f, (perm, cut) in enumerate(zip(self.permutations, self.cuts)):
            local = perm[cut[round_id]:cut[round_id + 1]]
            ids.append(self.space.offsets[f] + local)
            families.append(np.full(len(local), f, dtype=np.int64))
            edges.append(self.space.decode(f, local))
        return np.concatenate(ids), np.concatenate(families), np.concatenate(edges, axis=1)

    @property
    def index_bytes(self):
        return sum(p.nbytes + c.nbytes for p, c in zip(self.permutations, self.cuts)) + self.sizes.nbytes

    def training_ce_weight(self):
        """(R/S)*sum(query losses), keeping the full-schema denominator.

        Excluded families must have exact fixed support and hence exactly zero
        clean CE; full_sweep verifies they are fixed. Removing nonzero-loss
        families would invalidate this estimator. Do not silently renormalize
        by query_total and change loss scaling when zero families are skipped.
        """
        return self.rounds / self.space.total if self.space.total else 0.0
