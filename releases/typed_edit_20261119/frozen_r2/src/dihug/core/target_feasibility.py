"""Exact diagnostics for simple, undirected, disjoint relation domains.

This validates a target; it does not project, resample, reject soft targets,
or construct a graph. No connectivity constraint is imposed.
"""
import numpy as np


def validate_role_target(graph, target_rows, family_names=None):
    names = tuple(sorted(graph.family_specs)) if family_names is None else tuple(family_names)
    rows = np.asarray(target_rows)
    if names != tuple(sorted(graph.family_specs)) or rows.shape != (graph.num_nodes, len(names)):
        raise ValueError('target shape/family column mapping disagrees with schema')
    try:
        graph.validate_disjoint_family_domains()
    except ValueError as exc:
        return {'status': 'UNKNOWN_UNSUPPORTED_SCHEMA', 'feasible': None, 'reason': str(exc)}
    if not np.isfinite(rows).all() or np.any(rows < 0) or np.any(rows != np.floor(rows)):
        return {'status': 'INFEASIBLE', 'feasible': False, 'reason': 'nonfinite, negative or noninteger target'}
    records = {}
    for i, name in enumerate(names):
        spec = graph.family_specs[name]
        col = rows[:, i].astype(np.int64)
        src, dst = list(spec.src_nodes), list(spec.dst_nodes)
        allowed = set(src) | set(dst)
        errors = []
        if any(col[n] != 0 for n in range(graph.num_nodes) if n not in allowed):
            errors.append('positive degree outside relation domain')
        m = len(graph.family_edges[name])
        left = sorted(col[src].tolist(), reverse=True)
        witness = None
        if spec.bipartite:
            right = sorted(col[dst].tolist(), reverse=True)
            if any(x > len(dst) for x in left) or any(x > len(src) for x in right):
                errors.append('endpoint capacity exceeded')
            if sum(left) != m or sum(right) != m:
                errors.append('endpoint degree sums disagree with fixed family edge count')
            for k in range(1, len(left)+1):
                lhs, rhs = sum(left[:k]), sum(min(k, d) for d in right)
                if lhs > rhs:
                    witness = {'criterion': 'Gale-Ryser', 'k': k, 'lhs': lhs, 'rhs': rhs}
                    errors.append('nongraphical bipartite sequence'); break
        else:
            if any(x > max(0, len(src)-1) for x in left):
                errors.append('endpoint capacity exceeded')
            if sum(left) != 2*m:
                errors.append('degree sum disagrees with twice fixed family edge count')
            for k in range(1, len(left)+1):
                lhs = sum(left[:k])
                rhs = k*(k-1) + sum(min(k, d) for d in left[k:])
                if lhs > rhs:
                    witness = {'criterion': 'Erdos-Gallai', 'k': k, 'lhs': lhs, 'rhs': rhs}
                    errors.append('nongraphical homogeneous sequence'); break
        records[name] = {'feasible': not errors, 'errors': errors, 'witness': witness,
                         'edge_count': m, 'src_degree_sum': sum(left)}
    feasible = all(v['feasible'] for v in records.values())
    return {'status': 'FEASIBLE' if feasible else 'INFEASIBLE', 'feasible': feasible,
            'families': records, 'contract': 'simple disjoint domains; connectivity not required'}
