"""CPU reference specification only; not imported by production workers.

Inputs are one complete hard Xt, never X0 labels or candidate query edges.
No model calls, learned parameters, RNG, graph edits, or dictionary fitting.
"""
from collections import Counter
from itertools import combinations_with_replacement
from math import log1p


def extract(types, edges, families, relation_pairs, n_types):
    types = tuple(types)
    edges, families = tuple(edges), tuple(families)
    n = len(types)
    if n < 1 or n_types < 1 or any(t not in range(n_types) for t in types):
        raise ValueError('invalid node inventory')
    pairs = tuple(tuple(p) for p in relation_pairs)
    if len(set(pairs)) != len(pairs) or any(a > b or a < 0 or b >= n_types for a, b in pairs):
        raise ValueError('requires unique unordered type-pair relations')
    if len(edges) != len(families) or len(set(edges)) != len(edges):
        raise ValueError('edge/family lengths or duplicate edges')
    adj = [set() for _ in types]
    for (u, v), r in zip(edges, families):
        if not 0 <= u < v < n or r not in range(len(pairs)):
            raise ValueError('canonical edge or family index')
        if tuple(sorted((types[u], types[v]))) != pairs[r]:
            raise ValueError('schema mismatch')
        adj[u].add(v)
        adj[v].add(u)

    channels = tuple(combinations_with_replacement(range(n_types), 2))
    channel_index = {p: j for j, p in enumerate(channels)}
    counts = [[0] * len(channels) for _ in types]
    # Independent reference implementation. Production reuse of CN wedges is
    # still pending; this does not claim GPU performance or cache integration.
    for u, v in edges:
        for w in adj[u].intersection(adj[v]):
            if w <= v:
                continue
            for center, a, b in ((u, v, w), (v, u, w), (w, u, v)):
                counts[center][channel_index[tuple(sorted((types[a], types[b])))]] += 1

    component = [-1] * n
    groups = []
    for root in range(n):
        if component[root] >= 0:
            continue
        cid, stack, nodes = len(groups), [root], []
        component[root] = cid
        while stack:
            u = stack.pop()
            nodes.append(u)
            for v in adj[u]:
                if component[v] < 0:
                    component[v] = cid
                    stack.append(v)
        groups.append(nodes)
    # Count each component once, not once per member (quadratic on a large
    # connected component). This leaves the exact feature values unchanged.
    compositions = [Counter(types[j] for j in group) for group in groups]
    inventory = Counter(types)
    features = []
    for i in range(n):
        remaining = inventory.copy()
        remaining[types[i]] -= 1
        row = []
        for j, (a, b) in enumerate(channels):
            capacity = remaining[a] * remaining[b] if a != b else remaining[a] * (remaining[a] - 1) // 2
            # Declared-schema inventory bound, not a bound tightened using
            # fixed-zero relation priors. This is not a closure rate.
            if any(tuple(sorted(pair)) not in pairs for pair in ((types[i], a), (types[i], b), (a, b))):
                capacity = 0
            if capacity == 0:
                if counts[i][j]:
                    raise ValueError('triangle outside declared capacity')
                row.append(0.0)
            else:
                row.append(log1p(counts[i][j]) / log1p(capacity))
        group = groups[component[i]]
        composition = compositions[component[i]]
        row += [len(group) / n] + [composition[t] / len(group) for t in range(n_types)]
        features.append(row)
    return dict(node_features=features, triangle_counts=counts,
                triangle_channels=channels, component_ids=component,
                component_sizes=[len(g) for g in groups],
                note='component IDs are internal only, never model inputs')


def same_component(result, queries):
    ids = result['component_ids']
    out = []
    for u, v in queries:
        if not 0 <= u < v < len(ids):
            raise ValueError('canonical query required')
        out.append(float(ids[u] == ids[v]))
    return out
