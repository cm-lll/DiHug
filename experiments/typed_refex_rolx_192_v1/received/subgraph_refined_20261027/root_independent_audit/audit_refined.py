#!/usr/bin/env python3
"""Independent CPU audit. Reads all candidate/reference inputs; writes only --out.

Usage: python audit_refined.py --snapshot
       python audit_refined.py --baseline
       python audit_refined.py --case IMDB new.npz ids.npy before_seed \
           --case PubMed new.npz ids.npy before_seed
"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import argparse, hashlib, itertools, json
from pathlib import Path
from collections import Counter
import numpy as np
from scipy.sparse import csr_matrix

ROOT = Path('/workspace/shared/subgraph_preview_20261027')
OUT = Path('/workspace/shared/subgraph_refined_audit_20261027')
QUOTAS = {'IMDB': [48, 56, 78, 10], 'PubMed': [40, 74, 66, 12]}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')

def snapshot():
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(ROOT.rglob('*')) if p.is_file()}

def unchanged(old):
    now = snapshot()
    return {'all_prior_files_unchanged': all(now.get(k) == v for k, v in old.items()),
            'changed_or_missing': [k for k, v in old.items() if now.get(k) != v],
            'new_files': sorted(now.keys() - old.keys()), 'prior_file_count': len(old)}

def load(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}

def signature(types, schema):
    family = {tuple(pair): i for i, pair in enumerate(schema.tolist())}
    out = [int(types[0])]
    for a, b in zip(types[:-1], types[1:]):
        out.extend([family[tuple(sorted((int(a), int(b))))], int(b)])
    return min(tuple(out), tuple(reversed(out)))

def graph_metrics(g):
    """No import/execution of evaluator or sampler code.

    MP2 uses neighbor-type combinatorics. MP3 independently computes each
    central edge's endpoint-type outer product and removes common-neighbor
    diagonals. Relations are deterministic endpoint-type pairs in this schema.
    Sparse matrix identities independently cross-check all path totals.
    """
    t, e, r, schema = g['types'], g['edges'], g['relations'].astype(int) - 1, g['schema']
    n = len(t)
    adj = [set() for _ in range(n)]
    typed_degree = np.zeros((n, len(schema)), dtype=np.int64)
    neighbor_type_degree = np.zeros((n, 4), dtype=np.int64)
    for (u, v), family in zip(e, r):
        adj[u].add(int(v)); adj[v].add(int(u))
        typed_degree[u, family] += 1; typed_degree[v, family] += 1
        neighbor_type_degree[u, t[v]] += 1; neighbor_type_degree[v, t[u]] += 1
    degree = typed_degree.sum(axis=1)
    node_tri = np.zeros(n, dtype=np.int64)
    typed_tri = Counter({','.join(map(str, x)): 0 for x in itertools.combinations_with_replacement(range(4), 3)})
    mp2, mp3 = Counter(), Counter()
    for u, neighbors in enumerate(adj):
        counts = neighbor_type_degree[u]
        for a in range(4):
            for b in range(a, 4):
                c = int(counts[a] * counts[b] if a != b else counts[a] * (counts[a] - 1) // 2)
                if c:
                    mp2[signature((a, t[u], b), schema)] += c
        for v in neighbors:
            if v <= u:
                continue
            for w in neighbors.intersection(adj[v]):
                if w <= v:
                    continue
                node_tri[[u, v, w]] += 1
                typed_tri[','.join(map(str, sorted((int(t[u]), int(t[v]), int(t[w])))))] += 1
    for u, v in e:
        left = neighbor_type_degree[u].copy(); left[t[v]] -= 1
        right = neighbor_type_degree[v].copy(); right[t[u]] -= 1
        outer = np.outer(left, right)
        common = adj[u].intersection(adj[v])
        if common:
            counts = np.bincount(t[list(common)], minlength=4)
            outer[np.arange(4), np.arange(4)] -= counts
        for a, b in zip(*np.nonzero(outer)):
            mp3[signature((a, t[u], t[v], b), schema)] += int(outer[a, b])
    triangles = int(node_tri.sum() // 3)
    matrix = csr_matrix((np.ones(2 * len(e), dtype=np.int64),
                         (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), shape=(n, n))
    sparse_triangles = int((matrix @ matrix).multiply(matrix).sum() // 6)
    exact_mp2 = int((degree * (degree - 1) // 2).sum())
    exact_mp3 = int(((degree[e[:, 0]] - 1) * (degree[e[:, 1]] - 1)).sum() - 3 * triangles)
    assert triangles == sparse_triangles
    assert sum(mp2.values()) == exact_mp2
    assert sum(mp3.values()) == exact_mp3
    clustering = np.divide(2 * node_tri, degree * (degree - 1), out=np.zeros(n, dtype=float), where=degree > 1)
    seen, components = set(), []
    for u in range(n):
        if u in seen:
            continue
        todo, comp = [u], []
        seen.add(u)
        while todo:
            v = todo.pop(); comp.append(v)
            for w in adj[v] - seen:
                seen.add(w); todo.append(w)
        components.append(len(comp))
    families = np.bincount(r, minlength=len(schema))
    def paths_output(counter):
        total = sum(counter.values())
        return {'total': total, 'counts': {','.join(map(str, k)): int(v) for k, v in sorted(counter.items())},
                'proportions': {','.join(map(str, k)): v / total for k, v in sorted(counter.items())}}
    return {'nodes': n, 'edges': len(e), 'type_counts': np.bincount(t, minlength=4).tolist(),
            'family_counts': families.tolist(), 'family_proportions': (families / len(e)).tolist(),
            'degree_mean': float(degree.mean()), 'isolates': int((degree == 0).sum()),
            'component_sizes': sorted(components, reverse=True), 'triangles': triangles,
            'typed_triangles': dict(typed_tri), 'average_clustering': float(clustering.mean()),
            'MP2': paths_output(mp2), 'MP3': paths_output(mp3),
            'sparse_total_crosscheck': True}, typed_degree, degree

def tv(a, b):
    if not isinstance(a, dict):
        return float(np.abs(np.array(a) - np.array(b)).sum() / 2)
    return sum(abs(a.get(k, 0) - b.get(k, 0)) for k in a.keys() | b.keys()) / 2

def cdf_distance(original, candidate):
    """Integer-degree CDF F(k)=fraction degree<=k. No fitted scaling.

    KS=max |F_ref-F_candidate|; W1=sum_{k=0}^{K-1}|F_ref-F_candidate|;
    mean_CDF_gap=W1/K for K=max(max original,max candidate), or zero at K=0.
    """
    k = max(int(np.max(original, initial=0)), int(np.max(candidate, initial=0)))
    f = np.cumsum(np.bincount(original, minlength=k + 1)) / len(original)
    h = np.cumsum(np.bincount(candidate, minlength=k + 1)) / len(candidate)
    gap = np.abs(f - h)
    scale = float(np.log1p(np.max(original, initial=0)))
    log_integral = float(np.sum(gap[:-1] * np.diff(np.log1p(np.arange(k + 1)))))
    return {'KS': float(gap.max(initial=0)), 'W1': float(gap.sum()),
            'mean_CDF_gap': float(gap.sum() / k) if k else 0., 'support_max_K': k,
            'original_mean': float(original.mean()), 'candidate_mean': float(candidate.mean()),
            'logdegree_CDF_L1_reference_max_normalized': log_integral / scale if scale else 0.}

def compare(ref_g, ref_m, ref_d, ref_total, new_g, new_m, new_d, new_total):
    # Only structurally admissible node-type x active-family cells count in macros.
    typed = {}
    for fam, pair in enumerate(ref_g['schema']):
        if not ref_m['family_counts'][fam]:
            continue
        for typ in sorted(set(map(int, pair))):
            typed[f'type={typ},family={fam + 1}'] = cdf_distance(ref_d[ref_g['types'] == typ, fam], new_d[new_g['types'] == typ, fam])
    total = {str(typ): cdf_distance(ref_total[ref_g['types'] == typ], new_total[new_g['types'] == typ]) for typ in range(4)}
    ref_tri_total = sum(ref_m['typed_triangles'].values())
    new_tri_total = sum(new_m['typed_triangles'].values())
    result = {'family_proportions_TV': tv(ref_m['family_proportions'], new_m['family_proportions']),
            'typed_degree_CDF': typed, 'typed_degree_CDF_macro_KS': float(np.mean([x['KS'] for x in typed.values()])),
            'typed_degree_CDF_macro_mean_gap': float(np.mean([x['mean_CDF_gap'] for x in typed.values()])),
            'typed_degree_CDF_macro_W1': float(np.mean([x['W1'] for x in typed.values()])),
            'typed_logdegree_CDF': float(np.mean([x['logdegree_CDF_L1_reference_max_normalized'] for x in typed.values()])),
            'total_degree_CDF_by_type': total,
            'MP2_composition_TV': tv(ref_m['MP2']['proportions'], new_m['MP2']['proportions']),
            'MP3_composition_TV': tv(ref_m['MP3']['proportions'], new_m['MP3']['proportions']),
            'clustering_absolute_error': abs(ref_m['average_clustering'] - new_m['average_clustering']),
            'typed_triangle_TV': tv({k: v / max(ref_tri_total, 1) for k, v in ref_m['typed_triangles'].items()}, {k: v / max(new_tri_total, 1) for k, v in new_m['typed_triangles'].items()}),
            'missing_positive_triangle_fraction': float(np.mean([new_m['typed_triangles'][k] == 0 for k, v in ref_m['typed_triangles'].items() if v]))}
    result['frozen_objective_independent'] = .2 * sum(result[k] for k in ['family_proportions_TV', 'typed_logdegree_CDF', 'clustering_absolute_error', 'typed_triangle_TV', 'MP2_composition_TV']) + .05 * result['missing_positive_triangle_fraction']
    return result

def load_ids(path, graph):
    if path == '-':
        assert 'original_ids' in graph, 'Explicit IDs absent from graph'
        return graph['original_ids']
    p = Path(path)
    if p.suffix == '.npy':
        return np.load(p, allow_pickle=False)
    if p.suffix == '.npz':
        return load(p)['original_ids']
    data = json.loads(p.read_text())
    return np.asarray(data if isinstance(data, list) else data['original_ids'])

def audit(ds, ref, graph, ids, metadata=None):
    t, e, r = graph['types'], graph['edges'], graph['relations']
    checks = {'nodes_192': len(t) == 192, 'quotas_exact': np.array_equal(np.bincount(t, minlength=4), QUOTAS[ds]),
              'ids_integer': np.issubdtype(ids.dtype, np.integer), 'ids_1d_node_aligned': ids.shape == (len(t),),
              'ids_unique': len(np.unique(ids)) == len(ids),
              'ids_in_original_bounds': bool(((ids >= 0) & (ids < len(ref['types']))).all()),
              'schema_exact': np.array_equal(graph['schema'], ref['schema']),
              'edge_shape': e.ndim == 2 and e.shape[1] == 2,
              'relations_edge_aligned': r.shape == (len(e),),
              'edge_integer': np.issubdtype(e.dtype, np.integer),
              'relation_integer': np.issubdtype(r.dtype, np.integer),
              'edges_in_node_bounds': bool(((e >= 0) & (e < len(t))).all()),
              'relation_bounds': bool(((r >= 1) & (r <= len(ref['schema']))).all()),
              'no_self_loops': bool((e[:, 0] != e[:, 1]).all())}
    assert all(checks.values()), checks
    checks['types_match_original_ids'] = np.array_equal(t, ref['types'][ids])
    edges_original_ids = ids[e]
    actual = {(min(int(u), int(v)), max(int(u), int(v)), int(f)) for (u, v), f in zip(edges_original_ids, r)}
    actual_pairs = {(u, v) for u, v, _ in actual}
    selected = set(map(int, ids))
    expected = {(min(int(u), int(v)), max(int(u), int(v)), int(f)) for (u, v), f in zip(ref['edges'], ref['relations']) if int(u) in selected and int(v) in selected}
    checks['no_duplicate_edges_or_parallel_relations'] = len(actual) == len(actual_pairs) == len(e)
    checks['exact_induced_edge_relation_set'] = expected == actual
    checks['every_edge_relation_matches_endpoint_types'] = all(tuple(sorted((int(t[u]), int(t[v])))) == tuple(ref['schema'][int(f) - 1]) for (u, v), f in zip(e, r))
    original_families = set(map(int, ref['relations']))
    checks['all_original_active_families'] = set(map(int, r)) == original_families
    deg = np.bincount(e.ravel(), minlength=len(t))
    checks['no_isolates'] = bool((deg > 0).all())
    # Critical rare IMDB Actor-Actor-Movie triangle, independently checked directly.
    if ds == 'IMDB':
        neighbors = [set() for _ in t]
        for u, v in e:
            neighbors[u].add(int(v)); neighbors[v].add(int(u))
        checks['rare_AAM_triangle_retained'] = any(t[u] == t[v] == 1 and any(t[w] == 2 for w in neighbors[u] & neighbors[v]) for u, v in e)
    return {'passed': all(checks.values()), 'checks': {k: bool(v) for k, v in checks.items()},
            'missing_induced_edges': [list(v) for v in sorted(expected - actual)],
            'invented_or_changed_edges': [list(v) for v in sorted(actual - expected)],
            'original_ids_sha256': hashlib.sha256(np.asarray(ids, dtype='<i8').tobytes()).hexdigest()}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshot', action='store_true')
    parser.add_argument('--baseline', action='store_true')
    parser.add_argument('--output-name', default='refined_audit.json')
    parser.add_argument('--require-canonical', action='store_true')
    parser.add_argument('--require-connectivity-mp3', action='store_true')
    parser.add_argument('--case', action='append', nargs=4, metavar=('DATASET', 'GRAPH_NPZ', 'ORIGINAL_IDS', 'BEFORE_SEED'))
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    frozen = OUT / 'old_preview_snapshot.json'
    if args.snapshot:
        assert not frozen.exists(), 'Never overwrite an established old-preview snapshot'
        save(frozen, snapshot())
        print('Frozen', len(json.loads(frozen.read_text())), 'old preview files')
        return
    assert frozen.exists(), 'Run --snapshot before baseline or candidate audit'
    protocol = json.loads((ROOT / 'protocol.json').read_text())
    old_metrics = json.loads((ROOT / 'evaluation/metrics.json').read_text())
    evidence = {'source_inspected': ['evaluation/evaluate.py', 'evaluation/audited_paths_excerpt.py', 'evaluation/independent_check.py'],
                'source_hashes': {str(p.relative_to(ROOT)): sha(p) for p in [ROOT / 'evaluation/evaluate.py', ROOT / 'evaluation/audited_paths_excerpt.py', ROOT / 'evaluation/independent_check.py']},
                'method': 'Independent implementation, no evaluator imports, no generated edges, no optimizer execution. CPU only.',
                'CDF_definition': 'For each active admissible node-type/family cell, F(k)=fraction with degree <= k. KS=max absolute CDF gap. W1=sum gaps at integer k. Mean gap=W1/K, K=max degree across reference and candidate. Macro is unweighted mean of active admissible cells, including zero-degree nodes.',
                'MP_definition': 'Exact simple undirected paths; reversal identified; alternating node types and zero-based relation family. MP3 excludes repeated endpoint via common-neighbor type diagonals. Sparse identities cross-check totals.',
                'original': {}, 'baseline_validation': {}, 'cases': {}}
    for ds in ['IMDB', 'PubMed']:
        ref = load(ROOT / 'inputs/reference' / f'{ds}.npz')
        ref_m, ref_d, ref_total = graph_metrics(ref)
        assert sha(ROOT / 'inputs/reference' / f'{ds}.npz') == protocol['inputs'][ds]['hashes']['npz']
        for key, old_key in [('MP2', 1), ('MP3', 2)]:
            assert ref_m[key]['counts'] == old_metrics['original'][ds]['mp'][old_key]['counts']
        assert ref_m['typed_triangles'] == old_metrics['original'][ds]['typed_triangles']
        assert abs(ref_m['average_clustering'] - old_metrics['original'][ds]['average_clustering']) < 1e-14
        evidence['original'][ds] = ref_m
        if args.baseline:
            evidence['baseline_validation'][ds] = {}
            for p in sorted((ROOT / 'samples' / ds).iterdir()):
                g = load(p / 'graph.npz'); ids = load_ids(str(p / 'original_node_ids.npy'), g)
                m, d, total = graph_metrics(g)
                old = old_metrics['samples'][f'{ds}/{p.name}']['final_induced']
                assert m['MP2']['counts'] == old['mp'][1]['counts']
                assert m['MP3']['counts'] == old['mp'][2]['counts']
                assert m['typed_triangles'] == old['typed_triangles']
                evidence['baseline_validation'][ds][p.name] = {'metrics': m, 'comparison': compare(ref, ref_m, ref_d, ref_total, g, m, d, total), 'hard_constraints': audit(ds, ref, g, ids)}
        for cds, graph_path, ids_path, before_seed in (args.case or []):
            if cds != ds:
                continue
            g = load(graph_path); ids = load_ids(ids_path, g)
            hard = audit(ds, ref, g, ids)
            m, d, total = graph_metrics(g)
            bp = ROOT / 'samples' / ds / before_seed
            before = load(bp / 'graph.npz'); before_ids = load_ids(str(bp / 'original_node_ids.npy'), before)
            bm, bd, bt = graph_metrics(before)
            before_comp = compare(ref, ref_m, ref_d, ref_total, before, bm, bd, bt)
            after_comp = compare(ref, ref_m, ref_d, ref_total, g, m, d, total)
            hard['checks']['no_baseline_triangle_type_loss'] = all(m['typed_triangles'].get(k, 0) > 0 for k, v in bm['typed_triangles'].items() if v)
            hard['checks']['no_baseline_MP2_type_loss'] = all(m['MP2']['counts'].get(k, 0) > 0 for k, v in bm['MP2']['counts'].items() if v)
            if args.require_canonical:
                hard['checks']['all_edges_u_lt_v'] = bool((g['edges'][:, 0] < g['edges'][:, 1]).all())
                hard['checks']['edge_rows_lexicographically_sorted'] = bool(np.array_equal(np.lexsort((g['edges'][:, 1], g['edges'][:, 0])), np.arange(len(g['edges']))))
            if args.require_connectivity_mp3:
                hard['checks']['no_baseline_MP3_type_loss'] = all(m['MP3']['counts'].get(k, 0) > 0 for k, v in bm['MP3']['counts'].items() if v)
                hard['checks']['LCC_not_below_baseline'] = max(m['component_sizes']) >= max(bm['component_sizes'])
                hard['checks']['components_not_above_baseline'] = len(m['component_sizes']) <= len(bm['component_sizes'])
            hard['checks']['embedded_original_ids_match'] = bool(np.array_equal(g['original_ids'], ids)) if 'original_ids' in g else True
            md_path = Path(graph_path).parent / 'metadata.json'
            objective_check = None
            if md_path.exists():
                md = json.loads(md_path.read_text())
                objective_check = {'baseline_score_expected': md['baseline_score'], 'baseline_score_independent': before_comp['frozen_objective_independent'],
                                   'final_score_expected': md['final_score'], 'final_score_independent': after_comp['frozen_objective_independent'],
                                   'matches_1e_12': abs(md['baseline_score'] - before_comp['frozen_objective_independent']) < 1e-12 and abs(md['final_score'] - after_comp['frozen_objective_independent']) < 1e-12,
                                   'strictly_improved': after_comp['frozen_objective_independent'] < before_comp['frozen_objective_independent']}
                hard['checks']['metadata_original_ids_match'] = bool(np.array_equal(md['original_ids'], ids))
                hard['checks']['metadata_graph_hash_matches'] = md['graph_sha256'] == sha(graph_path)
            trace_path = Path(graph_path).parent / 'optimization_trace.json'
            trace_check = None
            if trace_path.exists():
                trace = json.loads(trace_path.read_text())
                replay_ids = before_ids.copy()
                last_score = before_comp['frozen_objective_independent']
                accepted = 0
                for i, item in enumerate(trace, start=1):
                    assert item['iteration'] == i
                    if item['accepted']:
                        old, new = item['removed_original_id'], item['added_original_id']
                        assert old in replay_ids and new not in replay_ids
                        assert 0 <= new < len(ref['types'])
                        assert ref['types'][old] == ref['types'][new] == item['type']
                        assert not item['hard_failures'] and item['score'] < last_score - 1e-12
                        replay_ids[replay_ids == old] = new
                        last_score = item['score']; accepted += 1
                    assert abs(item['retained_score'] - last_score) < 1e-12
                trace_check = {'iterations': len(trace), 'accepted_swaps': accepted, 'accepted_only_same_type_original_ID_swaps': True,
                               'final_IDs_equal_recorded_swap_replay_order': bool(np.array_equal(replay_ids, ids)),
                               'final_ID_set_equal_recorded_swap_replay': bool(np.array_equal(np.sort(replay_ids), np.sort(ids))), 'strictly_decreasing_accepted_scores': True,
                               'note': 'Replayed saved accepted swap IDs, not optimizer proposals or objective search.'}
                hard['checks']['swap_trace_matches_final_id_set'] = trace_check['final_ID_set_equal_recorded_swap_replay']
                hard['checks']['exact_2000_proposal_budget'] = len(trace) == 2000
            hard['passed'] = all(hard['checks'].values())
            evidence['cases'][ds] = {'graph_path': graph_path, 'graph_sha256': sha(graph_path), 'before_seed': before_seed,
                                     'hard_constraints': hard, 'before': bm, 'after': m,
                                     'before_comparison': before_comp, 'after_comparison': after_comp,
                                     'objective_check': objective_check, 'swap_trace_check': trace_check,
                                     'removed_original_ids': sorted(set(map(int, before_ids)) - set(map(int, ids))),
                                     'added_original_ids': sorted(set(map(int, ids)) - set(map(int, before_ids)))}
    evidence['old_preview_integrity'] = unchanged(json.loads(frozen.read_text()))
    evidence['all_candidate_constraints_passed'] = all(v['hard_constraints']['passed'] for v in evidence['cases'].values()) if evidence['cases'] else None
    dest = OUT / args.output_name
    save(dest, evidence)
    print(json.dumps({'saved': str(dest), 'old_preview_integrity': evidence['old_preview_integrity'],
                      'cases': {ds: {'passed': c['hard_constraints']['passed'], 'checks': c['hard_constraints']['checks'],
                                    'before': {k: v for k, v in c['before_comparison'].items() if isinstance(v, float)},
                                    'after': {k: v for k, v in c['after_comparison'].items() if isinstance(v, float)}} for ds, c in evidence['cases'].items()}}, indent=2))

if __name__ == '__main__':
    main()
