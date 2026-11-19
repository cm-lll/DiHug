"""Summarize frozen stratified slots; retain HT estimates vs observed counts."""
import csv
import json
import argparse
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/role_score_allocation_audit_20261109'
NEW = ['node_triangle_W1_flip', 'typed_node_triangle_W1_flip', 'joint_role_TV_flip',
       'max_node_triangle_abs_error_flip', 'triangle_composition_TV_flip']
OLD = ['triangle_abs_error_flip', 'total_degree_W1_flip', 'joint_degree_TV_flip']
TOL = 1e-12


def value(row, key):
    return float(row[key]) if row[key] != '' else None


def main():
    rows = list(csv.DictReader((OUT / 'selected_edge_metrics.csv').open()))
    groups = defaultdict(list)
    for r in rows:
        for t in (r['t'], 'all'):
            groups[(r['dataset'], t)].append(r)
    summary = []
    for (ds, t), g in groups.items():
        for method in ['MMD', 'JOINT_W1']:
            for label, predicate in [
                ('rewarded', lambda r: True),
                ('rewarded_three_strictly_worse', lambda r: all(value(r, k) > TOL for k in OLD)),
                ('rewarded_triangle_worse', lambda r: value(r, OLD[0]) > TOL),
                ('rewarded_three_nonbetter', lambda r: all(value(r, k) >= -TOL for k in OLD) and any(value(r, k) > TOL for k in OLD))]:
                sub = [r for r in g if value(r, method + '_flip') < -TOL and predicate(r)]
                denom = sum(float(r['inverse_inclusion_probability']) for r in sub)
                row = dict(dataset=ds, t=t, method=method, subset=label,
                    selected_count=len(sub), HT_count=denom,
                    parent_graphs=len({(r['condition'], r['arm'], r['t'], r['draw']) for r in sub}))
                for k in NEW:
                    available = [r for r in sub if value(r, k) is not None]
                    den = sum(float(r['inverse_inclusion_probability']) for r in available)
                    good = [r for r in available if value(r, k) < -TOL]
                    row[k + '_available'] = len(available)
                    row[k + '_better_selected'] = len(good)
                    row[k + '_better_HT_fraction'] = sum(float(r['inverse_inclusion_probability']) for r in good) / den if den else None
                complete = [r for r in sub if all(value(r,k) is not None for k in NEW)]
                no_gain = [r for r in complete if all(value(r,k) >= -TOL for k in NEW)]
                cden = sum(float(r['inverse_inclusion_probability']) for r in complete)
                row['all_extra_available'] = len(complete)
                row['no_extra_gain_selected'] = len(no_gain)
                row['no_extra_gain_HT_fraction'] = sum(float(r['inverse_inclusion_probability']) for r in no_gain) / cden if cden else None
                summary.append(row)
    with (OUT / 'summary.csv').open('w') as f:
        w = csv.DictWriter(f, list(summary[0])); w.writeheader(); w.writerows(summary)
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    for row in summary:
        if row['t'] == 'all' and row['subset'] in ('rewarded_three_strictly_worse', 'rewarded_triangle_worse'):
            print(json.dumps({k:v for k,v in row.items() if k in ('dataset','method','subset','selected_count','parent_graphs','node_triangle_W1_flip_better_HT_fraction','joint_role_TV_flip_better_HT_fraction','no_extra_gain_selected','no_extra_gain_HT_fraction')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=OUT)
    OUT = parser.parse_args().output
    main()
