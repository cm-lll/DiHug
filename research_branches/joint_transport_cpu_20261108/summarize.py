"""Read-only aggregation of the frozen stratified edge panel. No model imports."""
import csv, json, hashlib
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/joint_transport_cpu_20261108'
rows = list(csv.DictReader((OUT / 'graph_summary.csv').open()))
metrics = ['triangle_abs_error', 'total_degree_W1', 'joint_degree_TV']

def aggregate(keys):
    groups = defaultdict(list)
    for row in rows:
        row['dataset'] = row['condition'].split('_')[0]
        groups[tuple(row[k] for k in keys)].append(row)
    result = []
    for key, group in sorted(groups.items()):
        rec = dict(zip(keys, key))
        rec['graphs'] = len(group)
        den = sum(float(x['improving_HT_slots']) for x in group)
        mass = sum(float(x['improving_HT_gradient_mass']) for x in group)
        rec['improving_sample_slots'] = sum(int(x['improving_sample_slots']) for x in group)
        rec['improving_HT_slots'] = den
        for metric in metrics:
            bad = sum(float(x[metric + '_bad_HT_slots']) for x in group)
            bad_mass = sum(float(x['improving_HT_gradient_mass']) * float(x[metric + '_bad_gradient_fraction_HT'] or 0) for x in group)
            rec[metric + '_bad_sample_slots'] = sum(int(x[metric + '_bad_sample_slots']) for x in group)
            rec[metric + '_bad_fraction_HT'] = bad / den if den else None
            rec[metric + '_bad_gradient_fraction_HT'] = bad_mass / mass if mass else None
        rec['all_three_bad_sample_slots'] = sum(int(x['all_three_bad_sample_slots']) for x in group)
        result.append(rec)
    return result

def write_csv(name, records):
    with (OUT / name).open('w') as f:
        writer = csv.DictWriter(f, list(records[0]))
        writer.writeheader()
        writer.writerows(records)

for name, keys in [
    ('dataset_summary.csv', ['dataset', 'method']),
    ('time_summary.csv', ['dataset', 't', 'method']),
    ('parent_summary.csv', ['condition', 'arm', 'method']),
    ('triangle_state_summary.csv', ['dataset', 'triangle_state', 'method']),
]:
    write_csv(name, aggregate(keys))

cases = list(csv.DictReader((OUT / 'old_case_retest.csv').open()))
case_groups = defaultdict(list)
for row in cases:
    case_groups[(row['condition'].split('_')[0], row['kind'])].append(row)
case_summary = []
for (dataset, kind), group in sorted(case_groups.items()):
    values = [float(x['JOINT_W1_flip']) for x in group]
    case_summary.append(dict(dataset=dataset, kind=kind, cases=len(group),
                             still_improves=sum(v < -1e-12 for v in values),
                             worsens=sum(v > 1e-12 for v in values),
                             neutral=sum(abs(v) <= 1e-12 for v in values)))
write_csv('case_summary.csv', case_summary)

# Recheck bound assets/sources and the inherited production manifest without loading models.
checks = {}
for name in ['asset_binding.json', 'source_binding.json']:
    manifest = json.loads((OUT / name).read_text())
    checks[name] = {'files': len(manifest), 'changed': [p for p, h in manifest.items()
        if hashlib.sha256(Path(p).read_bytes()).hexdigest() != h]}
(OUT / 'summary_audit.json').write_text(json.dumps(checks, indent=2))
production_manifest = ROOT / 'results/joint_output_formal_projected_20261108T043000/source_binding.json'
manifest = json.loads(production_manifest.read_text())
production = {'manifest': str(production_manifest), 'files': len(manifest),
              'changed': [p for p, h in manifest.items()
                          if hashlib.sha256(Path(p).read_bytes()).hexdigest() != h]}
(OUT / 'production_audit.json').write_text(json.dumps(production, indent=2))
assert not production['changed']
assert all(not x['changed'] for x in checks.values())
print(json.dumps({'dataset': aggregate(['dataset', 'method']), 'time': aggregate(['dataset', 't', 'method']), 'cases': case_summary, 'bindings': checks}, indent=2))
