"""Final hashes, frozen objectives, baseline selection and full evaluator parity."""
import json
from pathlib import Path
import numpy as np
import audit_refined as a

r = Path('/workspace/shared/subgraph_refined_20261027')
aud = json.loads((a.OUT / 'v2_final_audit.json').read_text())
receipt = json.loads((r / 'protocol_freeze_receipt.json').read_text())
protocol = json.loads((r / 'protocol.json').read_text())
selection = json.loads((r / 'selection_and_refinement.json').read_text())
evaluator = json.loads((r / 'evaluation/final_induced_metrics.json').read_text())
result = {'frozen_protocol_hash_matches': a.sha(r / 'protocol.json') == receipt['protocol_sha256'],
          'frozen_optimizer_code_hash_matches': a.sha(r / 'code/refine.py') == receipt['code_sha256'] == protocol['code_sha256'],
          'datasets': {}}
for ds in ['IMDB', 'PubMed']:
    c = aud['cases'][ds]
    m = c['after']; old_style = evaluator[ds]
    p = r / 'samples' / ds
    ids = np.load(p / 'original_node_ids.npy')
    baseline_seed = c['before_seed']
    baseline_copy = a.load(p / 'baseline_graph.npz')
    original_baseline = a.load(a.ROOT / 'samples' / ds / baseline_seed / 'graph.npz')
    all_candidate_scores = []
    for candidate in selection[ds]['baselines']:
        independent = aud['baseline_validation'][ds][candidate['seed']]['comparison']['frozen_objective_independent']
        assert abs(independent - candidate['score']) < 1e-12
        all_candidate_scores.append({'seed': candidate['seed'], 'score': independent, 'hard_failures': candidate['hard_failures']})
    winner = min((x for x in all_candidate_scores if not x['hard_failures']), key=lambda x: (x['score'], x['seed']))['seed']
    checks = {'graph_hash_still_matches_independent_audit': a.sha(p / 'graph.npz') == c['graph_sha256'],
              'original_graph_copy_hash_matches': a.sha(r / 'inputs/reference' / f'{ds}.npz') == a.sha(a.ROOT / 'inputs/reference' / f'{ds}.npz'),
              'baseline_graph_copy_core_arrays_exact': all(np.array_equal(baseline_copy[k], original_baseline[k]) for k in ('types', 'edges', 'relations', 'schema')),
              'baseline_ID_copy_exact': np.array_equal(np.load(p / 'baseline_original_node_ids.npy'), np.load(a.ROOT / 'samples' / ds / baseline_seed / 'original_node_ids.npy')),
              'original_IDs_canonical_sorted': bool((np.diff(ids) > 0).all()),
              'uniform_all_three_baseline_scores_match_independent': True,
              'selected_baseline_is_lowest_feasible': winner == baseline_seed,
              'frozen_objective_matches_and_improves': c['objective_check']['matches_1e_12'] and c['objective_check']['strictly_improved'],
              'evaluator_family_counts_exact': m['family_counts'] == old_style['family_edges'],
              'evaluator_triangle_total_and_types_exact': m['triangles'] == old_style['triangles'] and m['typed_triangles'] == old_style['typed_triangles'],
              'evaluator_clustering_matches': abs(m['average_clustering'] - old_style['average_clustering']) < 1e-14,
              'evaluator_MP2_counts_exact': m['MP2']['counts'] == old_style['mp'][1]['counts'],
              'evaluator_MP3_counts_exact': m['MP3']['counts'] == old_style['mp'][2]['counts'],
              'evaluator_components_exact': m['component_sizes'] == old_style['component_sizes']}
    result['datasets'][ds] = {'all_passed': bool(all(checks.values())), 'checks': {k: bool(v) for k, v in checks.items()}, 'baselines': all_candidate_scores, 'graph_sha256': c['graph_sha256']}
result['old_preview_integrity'] = a.unchanged(json.loads((a.OUT / 'old_preview_snapshot.json').read_text()))
result['all_passed'] = (result['frozen_protocol_hash_matches'] and result['frozen_optimizer_code_hash_matches'] and all(x['all_passed'] for x in result['datasets'].values()) and result['old_preview_integrity']['all_prior_files_unchanged'])
a.save(a.OUT / 'v2_delivery_checks.json', result)
print(json.dumps(result, indent=2))
assert result['all_passed']
