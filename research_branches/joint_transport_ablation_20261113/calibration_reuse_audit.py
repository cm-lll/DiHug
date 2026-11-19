"""One CPU-only extension of an ALREADY FROZEN calibration; no fitting/sweep."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import csv
import hashlib
import itertools
import json
import resource
import signal
import sys
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from transport import TransportScore, ROOT, heatmod

OUT = ROOT / 'results/calibration_reuse_audit_20261113'
SRC = ROOT / 'results/joint_transport_ablation_20261113T011928'
CAL = ROOT / 'results/transport_sensitivity_20261113'
ROB = ROOT / 'results/robust_heat_transport_20261113'
hashes = {}

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def bind(p):
    hashes[str(p)] = sha(p)
    return p

def read(p):
    return json.loads(bind(p).read_text())

def dump(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))

def table(name, rows):
    with (OUT / name).open('w') as f:
        writer = csv.DictWriter(f, list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)

def bits(sc, z):
    assert np.array_equal(z['types'], sc.types)
    b = np.zeros_like(sc.labels)
    e = z['edges']; e = e.T if e.shape[0] == 2 else e
    for u, v in e:
        b[sc.lookup[tuple(sorted((int(u), int(v))))]] = 1
    assert b.sum() == len(e)
    return b

def load(ds, parent):
    z = np.load(bind(Path(parent['data'])))
    types, schema = z['types'], z['schema']
    lookup = {tuple(sorted(map(int, r))): i for i, r in enumerate(schema)}
    pairs = [(i, j) for i in range(len(types)) for j in range(i + 1, len(types))
             if tuple(sorted((int(types[i]), int(types[j])))) in lookup]
    q = np.array(pairs).T
    fam = np.array([lookup[tuple(sorted((int(types[i]), int(types[j]))))] for i, j in pairs])
    slots = {p: i for i, p in enumerate(pairs)}
    labels = np.zeros(len(pairs), np.uint8)
    e = z['edges']; e = e.T if e.shape[0] == 2 else e
    for u, v in e:
        labels[slots[tuple(sorted((int(u), int(v))))]] = 1
    prior = np.array([labels[fam == r].mean() for r in range(len(schema))])
    sc = TransportScore(ds, dict(space=SimpleNamespace(types=types), n_families=len(schema), prior=prior), q, fam, labels)
    sc.lookup = slots
    return sc

def evaluate(sc, b, coefficient):
    h, d = sc.features(b)
    parts, checks, matches = [], [], []
    for typ, (scale, db, hb) in sc.scales.items():
        ix = np.flatnonzero(sc.types == typ)
        gd = cdist(d[ix] / scale / np.sqrt(db), sc.rd[ix] / scale / np.sqrt(db), 'sqeuclidean')
        gh = cdist(h[ix] / np.sqrt(hb), sc.rh[ix] / np.sqrt(hb), 'sqeuclidean')
        block = []
        for c in (1., coefficient):
            ii, jj = linear_sum_assignment(gd + c * gh)
            assert len(set(jj)) == len(ix)
            dc, hc = float(gd[ii, jj].mean()), float(gh[ii, jj].mean())
            block.append((dc, hc, dc + c * hc, jj))
        old, new = block
        assert new[2] <= old[0] + coefficient * old[1] + 1e-9
        parts.append(dict(type=int(typ), old_degree=old[0], old_heat=old[1], old_score=old[2],
            new_degree=new[0], new_heat_unweighted=new[1], new_heat=coefficient * new[1], new_score=new[2],
            old_assignment_new_cost=old[0] + coefficient * old[1],
            changed_assignment_nodes=int(np.count_nonzero(old[3] != new[3]))))
        checks.append(float(np.max(abs(d[ix] - sc.rd[ix]))))
        matches.append(len(ix))
    row = {k: float(np.mean([p[k] for p in parts])) for k in parts[0] if k not in ('type', 'changed_assignment_nodes')}
    row['changed_assignment_nodes'] = sum(p['changed_assignment_nodes'] for p in parts)
    row['node_relation_degree_max_change'] = max(checks)
    row['edges'] = int(b.sum())
    row['isolates'] = int((d.sum(1) == 0).sum())
    row['old_heat_share'] = row['old_heat'] / row['old_score'] if row['old_score'] > 1e-12 else None
    row['new_heat_share'] = row['new_heat'] / row['new_score'] if row['new_score'] > 1e-12 else None
    return row, parts

def main():
    start = time.monotonic(); OUT.mkdir(exist_ok=False)
    resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('600s CPU wall bound')))
    signal.alarm(600)
    for p in (Path(__file__), Path(__file__).with_name('transport.py'), Path(heatmod.__file__)):
        bind(p)
    frozen = read(CAL / 'candidate_frozen_before_holdout.json')
    protocol = dict(scope='reuse frozen calibration; re-match 120 old perturbations, 156 old stress states, 64 old generated graphs',
        coefficients=frozen['weights'], fit_count=0, sweep=False, new_graphs=0, model_calls=0,
        rule='L=D/sD+H/sH, implemented as (D+cH)/sD; c=sD/sH fixed historically. All assignments recomputed.',
        generation_score_reporting='D+cH for comparability; global 1/sD also saved, not recommended optimizer gain.',
        calibration='64 historical development relocations; existing other 64 already viewed, no new independent confirmation',
        CPU_threads=1, memory_GiB=2, max_wall_seconds=600,
        missing='existing calibration lacks pure add/delete and exact degree-preserving swaps; these tested only, not refitted')
    dump('protocol.json', protocol)
    parents = read(SRC / 'protocol.json')['parents']
    cache = {ds: load(ds, next(v for v in parents.values() if v['dataset'] == ds)) for ds in ('IMDB', 'PubMed')}
    # Verify frozen medians by reading historical developer records, not re-fitting on new cases.
    dev = list(csv.DictReader(bind(CAL / 'original.csv').open()))
    for ds in cache:
        rr = [r for r in dev if r['dataset'] == ds and r['split'] == 'development']
        assert len(rr) == 32
        for key, col in [('degree_median', 'degree'), ('heat_median', 'heat')]:
            assert abs(float(np.median([float(r[col]) for r in rr])) - frozen['weights'][ds][key]) < 1e-12
    rows, types = [], []
    def process(ident, b, expected=None):
        ds = ident['dataset']; sc = cache[ds]; w = frozen['weights'][ds]
        v, pp = evaluate(sc, b, w['heat_coefficient'])
        if expected is not None:
            assert abs(v['old_score'] - expected) < 1e-8, (ident, v['old_score'], expected)
        v['new_fully_normalized_score'] = v['new_score'] / w['degree_median']
        rows.append({**ident, **v})
        types.extend({**ident, **p} for p in pp)
    oldgate = {r['state']: r['score'] for r in read(ROOT / 'results/joint_transport_score_gate_20261113/scores.json')}
    z = np.load(bind(ROOT / 'results/unified_organization_audit_20261113/cpu_perturbation_bits.npz'))
    for key in z.files:
        ds, route, seed, size = key.split('_')
        b = np.unpackbits(z[key])[:len(cache[ds].labels)]
        process(dict(dataset=ds, panel='legacy_perturbation', state=key, route=route, perturb_seed=seed, level=int(size)), b, oldgate[key])
        if route == 'swap':
            assert rows[-1]['node_relation_degree_max_change'] == 0
    manifest = read(ROB / 'panel_manifest.json'); z = np.load(bind(ROB / 'panel.npz'))
    for i, r in enumerate(manifest):
        b = z[str(i)]
        if 'state_hash' in r:
            assert hashlib.sha256(b.tobytes()).hexdigest() == r['state_hash']
        process(dict(dataset=r['dataset'], panel='legacy_' + r['kind'], state=str(i), route=r['route'],
                     perturb_seed=r['index'], level=r.get('level', 2)), b)
    for ep in sorted((SRC / 'evaluation').glob('*.json')):
        ev = read(ep); ds, seed, *rest = ep.stem.split('_'); arm = '_'.join(rest[:-1]); ss = rest[-1]
        gp = bind(Path(ev['graph_path'])); assert hashes[str(gp)] == ev['graph_sha256']
        process(dict(dataset=ds, panel='evaluation', state=ep.stem, route=arm, train_seed=int(seed), sample_seed=int(ss),
            triangles=ev['raw']['statistics']['triangles'], triangle_error=abs(ev['raw']['statistics']['triangles']-ev['reference']['triangles'])),
            bits(cache[ds], np.load(gp)), ev['unified_transport_score'][0])
    assert len(rows) == 340
    checks = []
    for ds, sc in cache.items():
        c = frozen['weights'][ds]['heat_coefficient']
        zero, _ = evaluate(sc, sc.labels, c); assert zero['new_score'] < 1e-12
        # Reuse deterministic type permutation for invariance only, not a new scored perturbation.
        rng = np.random.default_rng(2026111309); perm = np.arange(sc.n)
        for typ in sc.scales:
            ix = np.flatnonzero(sc.types == typ); perm[ix] = rng.permutation(ix)
        b = np.zeros_like(sc.labels)
        for u, v in sc.q[:, sc.labels.astype(bool)].T:
            b[sc.lookup[tuple(sorted((int(perm[u]), int(perm[v]))))]] = 1
        pv, _ = evaluate(sc, b, c); assert pv['new_score'] < 1e-10
        checks.append(dict(dataset=ds, self_cost=zero['new_score'], permutation_cost=pv['new_score']))
    table('states.csv', rows); table('types.csv', types)
    models, pairs = [], []
    evalrows = [r for r in rows if r['panel'] == 'evaluation']
    for ds, seed, arm in sorted({(r['dataset'], r['train_seed'], r['route']) for r in evalrows if not r['route'].endswith('_transport')}):
        before = sorted([r for r in evalrows if (r['dataset'], r['train_seed'], r['route']) == (ds, seed, arm)], key=lambda r:r['sample_seed'])
        after = sorted([r for r in evalrows if (r['dataset'], r['train_seed'], r['route']) == (ds, seed, arm+'_transport')], key=lambda r:r['sample_seed'])
        assert len(before) == len(after) == 4
        result = dict(dataset=ds, train_seed=seed, arm=arm, samples=4)
        for k in ('old_score', 'new_score', 'new_degree', 'new_heat', 'triangle_error'):
            bv, av = [r[k] for r in before], [r[k] for r in after]
            result[k+'_before'] = float(np.mean(bv)); result[k+'_after'] = float(np.mean(av))
            result[k+'_change_pct'] = float((np.mean(av)/np.mean(bv)-1)*100) if np.mean(bv)>1e-12 else None
        for b,a in zip(before,after):
            assert b['sample_seed'] == a['sample_seed']
            pairs.append(dict(dataset=ds, train_seed=seed, arm=arm, sample_seed=b['sample_seed'],
                old_score_delta=a['old_score']-b['old_score'], new_score_delta=a['new_score']-b['new_score'],
                triangle_error_delta=a['triangle_error']-b['triangle_error']))
        models.append(result)
    table('models.csv', models); table('pairs.csv', pairs)
    grouped=[]
    for ds,panel,route in sorted({(r['dataset'],r['panel'],r['route']) for r in rows if r['panel']!='evaluation'}):
        rr=[r for r in rows if (r['dataset'],r['panel'],r['route'])==(ds,panel,route)]
        item=dict(dataset=ds,panel=panel,route=route,n=len(rr))
        for k in ('old_score','new_score','old_heat_share','new_heat_share','changed_assignment_nodes'):
            vv=[r[k] for r in rr if r[k] is not None]
            item[k+'_mean']=float(np.mean(vv));item[k+'_min']=float(min(vv));item[k+'_max']=float(max(vv))
        grouped.append(item)
    table('groups.csv',grouped)
    assert 'torch' not in sys.modules
    assert all(sha(p)==v for p,v in hashes.items())
    dump('sources.json', hashes); dump('checks.json', checks)
    dump('receipt.json', dict(status='complete', seconds=time.monotonic()-start,
        peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024, states=340, invariant_permutations=2,
        original_models=8, graph_pairs=32, source_unchanged=True, model_calls=0, GPU_hours=0,
        new_model_graphs=0, new_scientific_perturbations=0, calibration_reused=True))
    print(json.dumps(dict(models=models, receipt=json.loads((OUT/'receipt.json').read_text())), indent=2))

if __name__ == '__main__':
    try:
        main()
    except BaseException:
        import traceback
        if OUT.exists(): dump('failure.json', dict(traceback=traceback.format_exc()))
        raise
