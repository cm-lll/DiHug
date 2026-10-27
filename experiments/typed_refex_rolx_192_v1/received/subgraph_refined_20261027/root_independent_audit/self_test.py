"""Negative fixtures and representation invariance; no input writes."""
import json
import numpy as np
import audit_refined as a

ref = a.load(a.ROOT / 'inputs/reference/IMDB.npz')
path = a.ROOT / 'samples/IMDB/2026102701'
g = a.load(path / 'graph.npz')
ids = np.load(path / 'original_node_ids.npy')
checks = {}

def reject(label, graph, mapping):
    rejected = False
    try:
        rejected = not a.audit('IMDB', ref, graph, mapping)['passed']
    except (AssertionError, IndexError):
        rejected = True
    assert rejected, f'Invalid fixture passed: {label}'
    checks[label] = True

def copygraph():
    return {k: v.copy() for k, v in g.items()}

bad_ids = ids.copy(); bad_ids[0] = bad_ids[1]
reject('duplicate_original_ID', g, bad_ids)
bad_ids = ids.copy(); bad_ids[0] = len(ref['types'])
reject('invented_out_of_bounds_ID', g, bad_ids)
bad = copygraph(); bad['relations'][0] = 1 if bad['relations'][0] != 1 else 2
reject('changed_relation', bad, ids)
bad = copygraph(); bad['edges'] = bad['edges'][1:]; bad['relations'] = bad['relations'][1:]
reject('missing_induced_edge', bad, ids)
bad = copygraph(); bad['edges'] = np.r_[bad['edges'], bad['edges'][:1]]; bad['relations'] = np.r_[bad['relations'], bad['relations'][:1]]
reject('duplicate_edge', bad, ids)
pairs = {tuple(e) for e in ref['edges']}
i, j = next((i, j) for i in range(len(ids)) for j in range(i + 1, len(ids)) if tuple(sorted((int(ids[i]), int(ids[j])))) not in pairs)
family = {tuple(p): k + 1 for k, p in enumerate(ref['schema'])}[tuple(sorted((g['types'][i], g['types'][j])))]
bad = copygraph(); bad['edges'] = np.r_[bad['edges'], [[i, j]]]; bad['relations'] = np.r_[bad['relations'], [family]]
reject('invented_valid_schema_edge', bad, ids)
i, j = next((i, j) for i in range(len(ids)) for j in range(i + 1, len(ids)) if g['types'][i] != g['types'][j])
bad = copygraph(); bad['types'][[i, j]] = bad['types'][[j, i]]
reject('changed_node_type_with_same_quota', bad, ids)

invariance = {}
for ds in ['IMDB', 'PubMed']:
    graph = a.load(f'/workspace/shared/subgraph_refined_20261027/samples/{ds}/graph.npz')
    expected, _, _ = a.graph_metrics(graph)
    rng = np.random.default_rng(934)
    # First reverse all edge endpoints and randomly reorder edge-relation rows.
    rows = rng.permutation(len(graph['edges']))
    variant = {k: v.copy() for k, v in graph.items()}
    variant['edges'] = graph['edges'][rows, ::-1]
    variant['relations'] = graph['relations'][rows]
    observed, _, _ = a.graph_metrics(variant)
    assert observed == expected
    # Then relabel all nodes, preserving original ID/type alignment.
    order = rng.permutation(len(graph['types']))
    inverse = np.argsort(order)
    variant['types'] = graph['types'][order]
    variant['edges'] = inverse[variant['edges']]
    observed, _, _ = a.graph_metrics(variant)
    for key in expected:
        if key == 'average_clustering':
            assert abs(observed[key] - expected[key]) < 1e-14
        else:
            assert observed[key] == expected[key], key
    invariance[ds] = {'endpoint_reversal_row_shuffle': True, 'node_relabeling': True, 'triangles': expected['triangles'], 'MP2': expected['MP2']['total'], 'MP3': expected['MP3']['total']}
a.save(a.OUT / 'audit_self_tests.json', {'invalid_fixtures_rejected': checks, 'representation_invariance': invariance})
print(json.dumps({'negative_tests_passed': len(checks), 'representation_invariance': invariance}, indent=2))
