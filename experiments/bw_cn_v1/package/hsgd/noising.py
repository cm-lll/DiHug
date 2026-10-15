"""Global typed marginal forward sampling, retaining outside-query context.

Exact binary marginal law via Binomial absent-slot count and uniform complement
positions. Produces sparse training inputs, not block-local patched graphs.
"""
import numpy as np


def complement_positions(size, occupied, ranks):
    occupied = np.asarray(occupied, dtype=np.int64)
    ranks = np.asarray(ranks, dtype=np.int64)
    if occupied.ndim != 1 or not np.array_equal(np.unique(occupied), occupied) or ((occupied < 0) | (occupied >= size)).any():
        raise ValueError("occupied slots must be sorted, unique, within range")
    if ((ranks < 0) | (ranks >= size - len(occupied))).any():
        raise ValueError("complement rank outside support")
    offsets = np.searchsorted(occupied - np.arange(len(occupied)), ranks, side="right")
    return ranks + offsets


def global_noisy_edges(space, clean_edges, prior, retention, rng):
    prior = np.asarray(prior, dtype=np.float64)
    if prior.shape != space.counts.shape or not np.isfinite(prior).all() or ((prior < 0) | (prior > 1)).any() or not 0 <= retention <= 1:
        raise ValueError("invalid prior/retention")
    clean_family, clean_local = space.encode(clean_edges)
    if len(np.unique(space.offsets[clean_family] + clean_local)) != len(clean_local):
        raise ValueError("duplicate clean edges")
    all_edges, families, stats = [], [], []
    for f, size in enumerate(space.counts):
        occupied = np.sort(clean_local[clean_family == f])
        if prior[f] == 0 and len(occupied) or prior[f] == 1 and len(occupied) != size:
            raise ValueError("clean graph contradicts exact fixed support")
        kept = occupied[rng.random(len(occupied)) < retention + (1 - retention)*prior[f]]
        absent_probability = (1 - retention)*prior[f]
        births = int(rng.binomial(int(size - len(occupied)), absent_probability))
        ranks = rng.choice(int(size - len(occupied)), size=births, replace=False)
        born = complement_positions(size, occupied, ranks)
        local = np.r_[kept, born]
        edges = space.decode(f, local)
        all_edges.append(edges)
        families.append(np.full(len(local), f, dtype=np.int64))
        stats.append(dict(family=f, clean_edges=len(occupied), kept=len(kept), born=len(born), noisy_edges=len(local),
                          expected_edges=float(retention*len(occupied) + (1-retention)*prior[f]*size)))
    return np.concatenate(all_edges, axis=1), np.concatenate(families), stats
