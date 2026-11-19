"""CPU-only aggregation of saved results; no model or scientific rerun."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/analytic_joint_risk_20261113'
rows = json.loads((OUT / 'local_probes.json').read_text())
summary = []
for dataset in ('IMDB', 'PubMed'):
    seeds = sorted({r['train_seed'] for r in rows if r['dataset'] == dataset})
    for seed in seeds:
        item = dict(dataset=dataset, train_seed=seed, times=6)
        for arm in ('risk', 'bias', 'plugin'):
            sub = [r for r in rows if r['dataset'] == dataset and r['train_seed'] == seed and r['arm'] == arm]
            item[arm + '_mean_risk_change_percent'] = 100 * sum(r['risk_relative_change'] for r in sub) / len(sub)
            if arm == 'risk':
                for metric in ('low_fraction_error', 'hub_fraction_error', 'joint_expected_degree_proxy'):
                    delta = [r[metric + '_after'] - r[metric + '_before'] for r in sub]
                    item[metric + '_improved'] = sum(d < 0 for d in delta)
                    item[metric + '_worsened'] = sum(d > 0 for d in delta)
                    item[metric + '_mean_delta'] = sum(delta) / len(delta)
                target = 45 if dataset == 'IMDB' else 1937
                delta = [abs(r['expected_triangles_after'] - target) - abs(r['expected_triangles_before'] - target) for r in sub]
                item['expected_triangle_error_improved'] = sum(d < 0 for d in delta)
                item['expected_triangle_error_worsened'] = sum(d > 0 for d in delta)
        summary.append(item)
(OUT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
with (OUT / 'summary.csv').open('w') as handle:
    writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
    writer.writeheader(); writer.writerows(summary)
# Preserve executed source and original output. Correct a presentation-only
# label in a derived table: the stored gradient is with respect to logits.
relations = json.loads((OUT / 'relations.json').read_text())
for row in relations:
    row['logit_gradient_sum'] = row.pop('probability_gradient_sum')
(OUT / 'relations_logit.json').write_text(json.dumps(relations, indent=2) + '\n')
with (OUT / 'relations_logit.csv').open('w') as handle:
    writer = csv.DictWriter(handle, fieldnames=list(relations[0]))
    writer.writeheader(); writer.writerows(relations)
(OUT / 'column_correction.json').write_text(json.dumps({
    'original_preserved': True,
    'source': 'relations.json', 'derived': 'relations_logit.json',
    'rename': {'probability_gradient_sum': 'logit_gradient_sum'},
    'reason': 'Stored values are sums of gradients with respect to Bernoulli logits; no numerical values changed.'
}, indent=2) + '\n')
print(json.dumps(summary, indent=2))
