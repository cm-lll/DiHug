"""Frozen CPU-only comparison on existing graphs. Not an A-DOGE reproduction."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
import csv
import hashlib
import json
import resource
import time
import traceback
from pathlib import Path
import numpy as np
from scipy.linalg import eigh, expm
from scipy.spatial.distance import cdist, pdist
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / 'results/typed_heat_degree_audit_20261111'
HEAT = ROOT / 'results/typed_heat_audit_20261111'
OUT = ROOT / 'results/neighbor_signal_audit_20261111'
TIMES = np.logspace(-1, 2, 9)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, allow_nan=False))


def table(name, rows):
    save(name + '.json', rows)
    if rows:
        with (OUT / (name + '.csv')).open('w') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def median_band(x):
    d = pdist(x, 'sqeuclidean')
    d = d[d > 1e-24]
    return float(np.median(d)) if len(d) else 1.0


def kernel(x, y, scale):
    return np.exp(-cdist(x, y, 'sqeuclidean') / (2 * scale))


def mmd(a, b, c):
    return float((a.mean() + b.mean() - 2 * c.mean()) / 2)


def describe(a, types, degree_features):
    """Same normalized Laplacian/isolate convention as cached heat descriptors."""
    n = len(a)
    d = a.sum(1)
    inv = np.divide(1, np.sqrt(d), out=np.zeros(n), where=d > 0)
    lap = np.diag((d > 0).astype(float)) - inv[:, None] * a * inv[None, :]
    lam, vectors = eigh(lap, check_finite=True, driver='evr')
    indicators = np.column_stack([types == t for t in np.unique(types)]).astype(float)
    # Type-major then relation-major, including fixed-zero channels; no fitted roles.
    signals = (indicators[:, :, None] * degree_features[:, None, :]).reshape(n, -1)
    attrs = np.column_stack([indicators, signals])
    proj_rw = vectors.T @ (np.sqrt(d)[:, None] * signals)
    proj_type = vectors.T @ (np.sqrt(d)[:, None] * indicators)
    proj_attrs = vectors.T @ attrs
    propagated, couplings, returns, reach = [], [], [], []
    for s in TIMES:
        weights = np.exp(-s * lam)
        value = inv[:, None] * (vectors @ (weights[:, None] * proj_rw))
        value[d == 0] = signals[d == 0]
        propagated.append(value)
        tvalue = inv[:, None] * (vectors @ (weights[:, None] * proj_type))
        tvalue[d == 0] = indicators[d == 0]
        reach.append(tvalue)
        returns.append((vectors * vectors) @ weights)
        couplings.append(proj_attrs.T @ (weights[:, None] * proj_attrs) / n)
    propagated = np.stack(propagated, axis=1).reshape(n, -1)
    propagated /= np.sqrt(len(TIMES) * signals.shape[1])
    old_features = np.column_stack([
        np.stack(returns, axis=1) / np.sqrt(2 * len(TIMES)),
        np.stack(reach, axis=1).reshape(n, -1) / np.sqrt(2 * len(TIMES) * indicators.shape[1])])
    assert np.isfinite(propagated).all()
    return propagated, np.stack(couplings), old_features, (lap, signals, attrs)


def spectral_blocks(c, ntypes):
    return [c[:, :ntypes, :ntypes], c[:, :ntypes, ntypes:], c[:, ntypes:, ntypes:]]


def engineering_checks():
    a = np.zeros((6, 6))
    for u, v in [(0, 1), (1, 2), (2, 0), (2, 3), (3, 4)]:
        a[u, v] = a[v, u] = 1
    types = np.array([0, 1, 0, 1, 0, 1])
    df = np.column_stack([a[:, types == t].sum(1) for t in [0, 1]])
    z, c, h, (lap, signals, attrs) = describe(a, types, df)
    d = a.sum(1)
    inv = np.divide(1, np.sqrt(d), out=np.zeros(6), where=d > 0)
    err = 0.0
    for k, s in enumerate(TIMES):
        sym = expm(-s * lap)
        rw = inv[:, None] * sym * np.sqrt(d)[None, :]
        rw[d == 0, d == 0] = 1
        direct = rw @ signals / np.sqrt(len(TIMES) * signals.shape[1])
        err = max(err, float(np.max(abs(direct - z.reshape(6, 9, -1)[:, k]))))
        np.testing.assert_allclose(c[k], attrs.T @ sym @ attrs / 6, atol=1e-11)
    assert err < 1e-11
    perm = np.array([5, 2, 0, 4, 1, 3])
    zp, cp, hp, _ = describe(a[np.ix_(perm, perm)], types[perm], df[perm])
    np.testing.assert_allclose(zp, z[perm], atol=1e-11)
    np.testing.assert_allclose(cp, c, atol=1e-11)
    np.testing.assert_allclose(hp, h[perm], atol=1e-11)
    # Source location changes the received signal at different nodes.
    # Checks propagation semantics, not a scientific quality example.
    signal = np.zeros((6, 1)); signal[1] = 1
    sym = expm(-TIMES[0] * lap)
    assert abs((sym @ signal)[0, 0] - (sym @ signal)[4, 0]) > 1e-6
    return dict(expm_max_error=err, permutation_pass=True, isolated_pass=True,
                spectral_bilinear_identity_pass=True, neighbor_signal_semantics_pass=True)


def main():
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=False)
    oldrows = json.loads((OLD / 'per_graph.json').read_text())
    oldgroups = json.loads((OLD / 'per_type.json').read_text())
    scales = json.loads((OLD / 'reference_scaling.json').read_text())
    assets = json.loads((HEAT / 'assets.json').read_text())
    source_files = [OLD / p for p in ['per_graph.json', 'per_type.json', 'reference_scaling.json', 'degree_features.npz', 'protocol.json']]
    source_files += [HEAT / p for p in ['signatures.npz', 'assets.json', 'protocol.json']]
    bindings = {str(p): sha(p) for p in source_files + [Path(__file__).resolve()]}
    for p, digest in json.loads((HEAT / 'protocol.json').read_text())['bindings'].items():
        assert sha(p) == digest
        bindings[p] = digest
    save('protocol.json', dict(
        frozen_before_scores=True, track='EVALUATION supporting structural training-objective design',
        scope='Exactly 70 existing graphs; 48 generated historically, 18 existing perturbations, 4 references/empty controls',
        times=TIMES.tolist(), budget='single CPU thread, external timeout 600 seconds, GPU 0',
        signals='4 type indicators and 40 type-masked relation log-degree channels; scales and degree features reused unchanged',
        baseline='Heat-filtered attribute coupling inspired by A-DOGE: C_s=X.T exp(-s Lsym) X/n. NOT full A-DOGE histogram/filterbank replication.',
        baseline_distance='Type-type, type-degree, degree-degree blocks; each mean squared difference / max(reference block RMS,1/n)^2; equal mean of three blocks; scale fixed from reference only',
        extension='Per-node Hrw(s) applied to 40 masked degree signals, concatenated over 9 times and divided by sqrt(360); extra Gaussian kernel with per-type reference positive median squared distance (>1e-24, fallback1)',
        extension_distance='Original k_heat*k_degree multiplied by k_propagation, per-type empirical MMD2/2 then equal mean; no change to old bandwidths',
        decision='Descriptive incremental information, existing deletion/overdensity/tail and degree-preserving swaps. No universal rank claim across incompatible errors; compare ratios/ranks within each scoring method, not raw score size between methods.',
        limitations=['Existing development panel; not independent confirmation', 'Swaps preserve each node relation degrees, NOT individual relation spectra', 'No hard subtype labels', 'No gradient, AdamW, model or generation evidence', 'Not complete relation-order representation'],
        bindings=bindings))
    checks = engineering_checks()
    cache = np.load(HEAT / 'signatures.npz')
    dc = np.load(OLD / 'degree_features.npz')
    tasks = json.loads((ROOT / 'results/real192_update_audit_20261111/protocol.json').read_text())['tasks']
    meta = {}
    features = {}
    timings = []
    old_error = 0.0
    # References are processed first; all scale estimates use references only.
    ordered = sorted(oldrows, key=lambda row: row['arm'] != 'reference')
    for row in ordered:
        if time.time() - start > 590:
            raise TimeoutError('CPU limit approaching')
        dataset, key = row['dataset'], row['signature_key']
        if dataset not in meta:
            task = next(t for t in tasks if t['parent']['dataset'] == dataset)
            data = np.load(task['parent']['data'])
            validate = ROOT / 'results/real192_update_audit_20261111' / task['id'] / 'validation_u2000_raw_t1.npz'
            v = np.load(validate)
            meta[dataset] = dict(types=data['types'], query=v['query'], n=len(data['types']), ntypes=len(np.unique(data['types'])))
        m = meta[dataset]
        a = np.zeros((m['n'], m['n']))
        q = m['query'][:, cache[key + '_bits'].astype(bool)]
        a[q[0], q[1]] = 1; a[q[1], q[0]] = 1
        tick = time.perf_counter()
        z, c, h, _ = describe(a, m['types'], dc[key + '_degree_features'])
        timings.append(time.perf_counter() - tick)
        err = float(np.max(abs(h - cache[key + '_features'])))
        old_error = max(old_error, err)
        assert err < 1e-9
        features[key] = (z, c)
        if row['arm'] == 'reference':
            m['ref_key'] = key
            m['bands'] = {int(t): median_band(z[m['types'] == t]) for t in np.unique(m['types'])}
            m['spectral_scales'] = [max(float(np.sqrt(np.mean(b*b))), 1/m['n']) for b in spectral_blocks(c, m['ntypes'])]
    save('reference_scaling.json', {ds: dict(propagation_bandwidth_squared=m['bands'], spectral_block_RMS=m['spectral_scales']) for ds, m in meta.items()})
    rows, groups = [], []
    old_recompute = 0.0
    for row in oldrows:
        tick = time.perf_counter()
        ds, key = row['dataset'], row['signature_key']
        m = meta[ds]; rk = m['ref_key']; z, c = features[key]; zr, cr = features[rk]
        costs, oldcosts = [], []
        for t in np.unique(m['types']):
            sel = m['types'] == t
            parts = []
            for x, y, scale in [
                (cache[key+'_features'][sel], cache[rk+'_features'][sel], scales[ds]['heat_bandwidth_squared'][str(t)]),
                (dc[key+'_degree_features'][sel], dc[rk+'_degree_features'][sel], scales[ds]['degree_bandwidth_squared'][str(t)]),
                (z[sel], zr[sel], m['bands'][int(t)])]:
                parts.append((kernel(x,x,scale), kernel(y,y,scale), kernel(x,y,scale)))
            oldcost = mmd(*[parts[0][j]*parts[1][j] for j in range(3)])
            cost = mmd(*[parts[0][j]*parts[1][j]*parts[2][j] for j in range(3)])
            assert cost > -1e-10
            costs.append(cost); oldcosts.append(oldcost)
            base = next(v for v in oldgroups if v['signature_key'] == key and v['type'] == int(t))
            group = dict(base); group['neighbor_signal_cost'] = cost
            group['propagation_only_cost'] = mmd(*parts[2])
            groups.append(group)
        old_recompute = max(old_recompute, abs(np.mean(oldcosts)-row['joint_heat_degree_cost']))
        assert old_recompute < 1e-11
        blockcosts = [float(np.mean((b-br)**2)/s**2) for b, br, s in zip(spectral_blocks(c,m['ntypes']), spectral_blocks(cr,m['ntypes']), m['spectral_scales'])]
        rec = dict(row)
        rec.update(spectral_coupling_cost=float(np.mean(blockcosts)), neighbor_signal_cost=float(np.mean(costs)),
                   spectral_type_type=blockcosts[0], spectral_type_degree=blockcosts[1], spectral_degree_degree=blockcosts[2],
                   scoring_seconds=time.perf_counter()-tick)
        if row['arm'] == 'reference':
            assert max(abs(rec['spectral_coupling_cost']), abs(rec['neighbor_signal_cost'])) < 1e-12
        rows.append(rec)
    table('per_graph', rows); table('per_type', groups)
    metrics = ['joint_heat_degree_cost', 'spectral_coupling_cost', 'neighbor_signal_cost']
    summary = []
    for ds, task, arm in sorted({(r['dataset'],r['task'],r['arm']) for r in rows}):
        rr = [r for r in rows if (r['dataset'],r['task'],r['arm']) == (ds,task,arm)]
        rec = dict(dataset=ds, task=task, arm=arm, n=len(rr))
        for metric in metrics + ['edges','triangles','degree_W1']:
            rec[metric+'_mean'] = float(np.mean([r[metric] for r in rr]))
            rec[metric+'_sd'] = float(np.std([r[metric] for r in rr],ddof=1)) if len(rr)>1 else None
        summary.append(rec)
    table('summary', summary)
    correlations=[]
    for ds in meta:
        for panel in ['ordinary_CE_only','all_old_generated_graphs']:
            rr=[r for r in rows if r['dataset']==ds and r['path'] and (panel!='ordinary_CE_only' or r['arm']=='ordinary_CE')]
            for method in metrics:
                for target in ['relation_count_cost','joint_degree_cost','degree_W1','present_triangle_cost','components_cost']:
                    x=[r[method] for r in rr]; y=[r[target] for r in rr]
                    rho=float(spearmanr(x,y).statistic) if np.std(x)>0 and np.std(y)>0 else None
                    correlations.append(dict(dataset=ds,panel=panel,n=len(rr),method=method,target=target,spearman=rho))
    table('correlations',correlations)
    np.savez_compressed(OUT/'features.npz', **{k+suffix:v[j] for k,v in features.items() for j,suffix in enumerate(['_propagation','_coupling'])})
    checks.update(old_heat_feature_max_error=old_error, old_joint_score_max_error=old_recompute, references_zero=True)
    save('checks.json', checks)
    for p, digest in bindings.items():
        assert sha(p) == digest
    save('receipt.json', dict(status='complete', seconds=time.time()-start,
         peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
         cpu_threads=1, graphs_reused=len(rows), type_rows=len(groups),
         median_descriptor_seconds=float(np.median(timings)), median_scoring_seconds=float(np.median([r['scoring_seconds'] for r in rows])),
         gpu_hours=0, model_calls=0, training_updates=0, new_generated_graphs=0, source_assets_unchanged=True))
    print(json.dumps({'receipt':json.loads((OUT/'receipt.json').read_text()),'checks':checks,'summary':summary},indent=2))


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        if OUT.exists():
            save('failure.json', {'traceback':traceback.format_exc()})
        raise
