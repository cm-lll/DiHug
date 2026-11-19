"""Read-only arithmetic consistency audit; no new model or optimization call."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/structure_tradeoff_cpu_20261110'
def read(name):
    return json.loads((OUT / (name + '.json')).read_text())

rows, cross, lp, checks = map(read, ['gradients', 'cross_component_effects', 'common_descent', 'checks'])
errors = []
for r in rows:
    key = (r['seed'], r['step'], r['space'])
    parts = [x for x in cross if (x['seed'], x['step'], x['space']) == key and x['target'] == r['component']]
    assert len(parts) == 4
    errors.append(abs(sum(x['contribution_to_average_direction'] for x in parts) - r['derivative_under_average_descent']))
    if r['component'] == 'typed_triangles':
        sub = [x for x in rows if (x['seed'], x['step'], x['space']) == key and x['component'] in ['triangle001', 'triangle011']]
        assert len(sub) == 2
        for field in ['loss', 'derivative_under_average_descent']:
            errors.append(abs(sum(x[field] for x in sub) / 2 - r[field]))
assert max(errors) < 1e-12
bindings = read('bindings')
for path, sha in bindings.items():
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == sha, path
shared = [x for x in lp if x['space'] == 'shared_parameters']
assert len(rows) == 180 and len(cross) == 720 and len(lp) == 60 and len(shared) == 30
result = dict(status='passed', rows=len(rows), cross_terms=len(cross), LPs=len(lp),
              consistency_max_abs_error=max(errors),
              finite_difference_checks=len(checks),
              finite_difference_max_abs_error=max(x['finite_difference_error'] for x in checks),
              LP_primal_max_residual=max(x['residual'] for x in lp),
              shared_finite_common_improvements=sum(x['all_selected_finite_changes_negative'] for x in shared),
              shared_terminal_total_improvements=sum(x['terminal_total_change'] < 0 for x in shared),
              source_hashes_unchanged=True,
              limitations='Arithmetic audit of saved outputs; finite derivatives are from main script. No independent real-network replication.')
(OUT / 'output_audit.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result))
