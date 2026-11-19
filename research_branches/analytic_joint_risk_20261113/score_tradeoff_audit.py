"""Read-only CPU diagnostic on frozen final graphs; no model or new graph calls.

Keep degree and organization errors as a vector. This is an audit comparison
rule, not a trainable replacement loss or an adoption gate.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[k] = '1'
import csv
import hashlib
import importlib.util
import json
import resource
import signal
import statistics
import sys
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'results/analytic_joint_generation_run_20261113T225800'
AUDIT = ROOT / 'results/analytic_joint_terminal_score_audit_20261114'
HEAT = ROOT / 'results/typed_heat_audit_20261111'
OUT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else ROOT / 'results/score_tradeoff_audit_20261116'
EPS = 1e-10  # Numerical comparison only; not a practical-quality tolerance.


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dump(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))


def table(name, rows):
    dump(name + '.json', rows)
    if rows:
        with (OUT / (name + '.csv')).open('w') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


def relation(a, b):
    d = np.asarray(a) - np.asarray(b)
    up, down = int((d > EPS).sum()), int((d < -EPS).sum())
    status = 'tradeoff' if up and down else 'worse' if up else 'better' if down else 'equal'
    return status, up, down


def main():
    start, cpu = time.time(), time.process_time()
    OUT.mkdir(exist_ok=False)
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('600s limit')))
    signal.alarm(600)
    bindings = {}

    def bind(p, expected=None):
        p = Path(p).resolve()
        h = sha(p)
        if expected is not None:
            assert h == expected, str(p)
        bindings[str(p)] = h
        return p

    def read(p):
        return json.loads(bind(p).read_text())

    bind(__file__)
    source_bindings = read(AUDIT / 'asset_bindings.json')
    inventory = read(SOURCE / 'inventory.json')
    protocol = read(SOURCE / 'protocol.json')
    old_groups = read(AUDIT / 'per_type.json')
    old_graphs = read(AUDIT / 'per_graph.json')
    assets = read(HEAT / 'assets.json')
    old_heat_rows = read(HEAT / 'per_graph.json')
    cache = np.load(bind(HEAT / 'signatures.npz'))
    old_heat_bindings = read(HEAT / 'protocol.json')['bindings']
    old_tasks = read(ROOT / 'results/real192_update_audit_20261111/protocol.json')['tasks']
    helper = ROOT / 'research_branches/neighbor_signal_audit_20261111/run.py'
    helper_bindings = read(ROOT / 'results/neighbor_signal_audit_20261111/protocol.json')['bindings']
    bind(helper, helper_bindings[str(helper)])
    spec = importlib.util.spec_from_file_location('existing_heat_reference', helper)
    heat = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(heat)
    assert len(inventory) == 80
    dump('protocol.json', dict(
        start_epoch=start, graphs=80, wall_cap_seconds=600, CPU_threads=1, RSS_cap_GiB=2,
        model_calls=0, GPU_hours=0, new_graphs=0,
        hypothesis='Separate type-specific degree error and existing heat-organization error reveals compensation without fitting a new weighted sum.',
        degree='Original frozen 32-frequency joint relation-degree CF error, unchanged.',
        organization='Reuse original 9-scale (0.1..100) typed node heat return/type-reach features and per-type reference Gaussian bandwidth; MMD2/2.',
        comparison='8 coordinates (4 degree + 4 heat). Pareto better only if none increases beyond numerical epsilon and at least one decreases. Mixed directions stay tradeoff.',
        epsilon=EPS, epsilon_meaning='Floating comparison only, not adoption threshold or significance.',
        scalar_diagnostics='Mean, RMS and max of original degree errors; reported together, no winner selected.',
        limitations=['All 80 graphs are previously viewed development evidence.', 'Not a new training loss or candidate acceptance gate.',
                     'Heat is not a pure cycle metric, may overlap degree and connectivity, does not preserve all relation path order.',
                     'Pareto comparison is incomplete; absence of domination is not proof of failure.',
                     'No independent confirmation and no claim of fixing forward-to-reverse distribution gap.'],
        bindings_at_freeze=bindings))
    checks = heat.engineering_checks()
    contexts = {}
    for parent in protocol['parents']:
        ds = parent['dataset']
        if ds in contexts:
            continue
        path = parent['data']
        z = np.load(bind(path, source_bindings[path]))
        if path in old_heat_bindings:
            assert sha(path) == old_heat_bindings[path]
        types, schema = z['types'], z['schema']
        e = z['edges']
        e = e if e.shape[0] == 2 else e.T
        a = np.zeros((len(types), len(types)))
        a[e[0], e[1]] = a[e[1], e[0]] = 1
        _, _, h, _ = heat.describe(a, types, np.zeros((len(types), len(schema))))
        refrow = next(r for r in old_heat_rows if r['dataset'] == ds and r['arm'] == 'reference')
        cached_h = cache[refrow['signature_key'] + '_features']
        difference = float(np.max(abs(h - cached_h)))
        assert difference < 1e-10
        # Bind reference graph identity independently of descriptor equality.
        # Historical heat cache uses active-support queries, not all N(N-1)/2 slots.
        task = next(t for t in old_tasks if t['parent']['dataset'] == ds)
        query_path = ROOT / 'results/real192_update_audit_20261111' / task['id'] / 'validation_u2000_raw_t1.npz'
        q = np.load(bind(query_path))['query']
        cached_bits = cache[refrow['signature_key'] + '_bits']
        assert q.shape == (2, len(cached_bits))
        assert np.all(q[0] < q[1]) and len(set(map(tuple, q.T))) == q.shape[1]
        cached_a = np.zeros_like(a)
        cached_a[q[0], q[1]] = cached_a[q[1], q[0]] = cached_bits
        np.testing.assert_array_equal(a, cached_a)
        asset = next(x for x in assets if x['dataset'] == ds)
        assert np.array_equal(schema, asset['schema'])
        bands = {int(k): v for k, v in asset['bandwidth_squared'].items()}
        contexts[ds] = dict(types=types, schema=schema, reference=h, bands=bands,
                            names=parent['config']['datasets'][ds]['type_names'])
        checks[ds + '_reference_heat_max_error'] = difference
        checks[ds + '_reference_adjacency_identical'] = True

    rows, groups = [], []
    for item in inventory:
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > 2 * 1024 * 1024:
            raise MemoryError('2GiB RSS cap')
        ident = {k: item[k] for k in ('model', 'dataset', 'train_seed', 'arm', 'readout', 'sample_seed')}
        old = next(x for x in old_graphs if all(x[k] == v for k, v in ident.items()))
        path = item['graph']
        z = np.load(bind(path, old['graph_sha256']))
        sc = contexts[item['dataset']]
        ty, e = sc['types'], z['edges']
        np.testing.assert_array_equal(z['types'], ty)
        np.testing.assert_array_equal(z['schema'], sc['schema'])
        assert e.shape[0] == 2 and np.all(e[0] < e[1])
        assert len(set(map(tuple, e.T))) == e.shape[1]
        a = np.zeros((len(ty), len(ty)))
        a[e[0], e[1]] = a[e[1], e[0]] = 1
        _, _, h, _ = heat.describe(a, ty, np.zeros((len(ty), len(sc['schema']))))
        dscores, hscores = [], []
        for t in sorted(np.unique(ty)):
            select = ty == t
            x, y, band = h[select], sc['reference'][select], sc['bands'][int(t)]
            value = heat.mmd(heat.kernel(x, x, band), heat.kernel(y, y, band), heat.kernel(x, y, band))
            assert value >= -1e-12
            d = next(v['hard_score'] for v in old_groups if v['type'] == t and all(v[k] == n for k, n in ident.items()))
            dscores.append(d)
            hscores.append(value)
            groups.append(dict(**ident, type=int(t), type_name=sc['names'][t], nodes=int(select.sum()), degree_score=d, heat_score=value))
        assert abs(np.mean(dscores) - old['hard_score']) < 1e-14
        rows.append(dict(**old, degree_RMS=float(np.sqrt(np.mean(np.square(dscores)))), degree_max=max(dscores),
                         heat_mean=float(np.mean(hscores)), **{f'degree_type{t}':v for t,v in enumerate(dscores)},
                         **{f'heat_type{t}':v for t,v in enumerate(hscores)}))

    coordinates = [f'{kind}_type{t}' for kind in ('degree', 'heat') for t in range(4)]
    pairs = []
    for a in rows:
        if a['arm'] == 'parent':
            continue
        b = next(x for x in rows if x['arm'] == 'parent' and x['model'] == a['model'] and x['sample_seed'] == a['sample_seed'])
        status, up, down = relation([a[k] for k in coordinates], [b[k] for k in coordinates])
        pairs.append(dict(**{k:a[k] for k in ident}, status=status, groups_worse=up, groups_better=down,
                          **{k+'_delta':a[k]-b[k] for k in coordinates + ['hard_score','degree_RMS','degree_max','heat_mean','triangle_abs_error','joint_TV','joint_relation_degree_L1','MP3_abs_error','LCC']}))
    summaries = []
    for key in sorted(set((r['model'],r['arm'],r['readout']) for r in rows)):
        selected = [r for r in rows if (r['model'],r['arm'],r['readout']) == key]
        assert len(selected) == 4
        summary = {k:selected[0][k] for k in ('model','dataset','train_seed','arm','readout')}
        summary['samples'] = 4
        for k in coordinates + ['hard_score','degree_RMS','degree_max','heat_mean','triangle_abs_error','joint_TV','joint_relation_degree_L1','MP3_abs_error','LCC']:
            summary[k] = statistics.mean(r[k] for r in selected)
            summary[k+'_sample_SD'] = statistics.stdev(r[k] for r in selected)
        summaries.append(summary)
    comparisons = []
    for a in summaries:
        if a['arm'] == 'parent':
            continue
        b = next(x for x in summaries if x['arm'] == 'parent' and x['model'] == a['model'])
        status, up, down = relation([a[k] for k in coordinates], [b[k] for k in coordinates])
        pp = [r for r in pairs if all(r[k] == a[k] for k in ('model','arm','readout'))]
        comparison = dict(**{k:a[k] for k in ('model','dataset','train_seed','arm','readout')}, status=status,
            degree_types_worse=sum(a[k]-b[k]>EPS for k in coordinates[:4]), heat_types_worse=sum(a[k]-b[k]>EPS for k in coordinates[4:]),
            sample_Pareto_better=sum(p['status']=='better' for p in pp),sample_tradeoff=sum(p['status']=='tradeoff' for p in pp))
        for k in coordinates + ['hard_score','degree_RMS','degree_max','heat_mean','triangle_abs_error','joint_TV','joint_relation_degree_L1','MP3_abs_error','LCC']:
            comparison[k+'_delta'] = a[k]-b[k]
            comparison[k+'_relative_percent'] = 100*(a[k]/b[k]-1) if b[k]!=0 else None
        comparisons.append(comparison)
    for name, values in [('per_graph',rows),('per_type',groups),('paired_samples',pairs),('model_summary',summaries),('comparisons',comparisons)]:
        table(name, values)
    dump('checks.json', checks)
    assert all(sha(path) == h for path,h in bindings.items())
    dump('asset_bindings.json', bindings)
    dump('completion.json', dict(status='complete', graphs=80, model_readouts=20, model_calls=0, new_graphs=0, GPU_hours=0,
        CPU_seconds=time.process_time()-cpu, wall_seconds=time.time()-start,
        RSS_peak_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024, original_assets_unchanged=True))
    for r in comparisons:
        if r['arm']=='risk':
            print(json.dumps({k:r[k] for k in ('dataset','train_seed','readout','status','degree_types_worse','heat_types_worse','hard_score_relative_percent','degree_RMS_relative_percent','degree_max_relative_percent','heat_mean_relative_percent','triangle_abs_error_delta')}))
    signal.alarm(0)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        if OUT.exists():
            dump('failure.json', dict(type=type(exc).__name__, message=str(exc)))
        raise
