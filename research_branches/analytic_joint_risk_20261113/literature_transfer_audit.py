"""CPU attribution on already viewed graphs; no proposed training objective is fitted.

Checks what residual calibration can change, and what existing heat features see.
No torch/model imports, new graphs, random draws or optimization.
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import csv
import hashlib
import importlib.util
import json
import resource
import signal
import sys
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / 'results/score_tradeoff_audit_20261116_v2'
GEN = ROOT / 'results/analytic_joint_generation_run_20261113T225800'
OUT = Path(sys.argv[1]).resolve() if len(sys.argv)>1 else ROOT / 'results/literature_transfer_audit_20261116_v2'


def main():
    start, cpu = time.time(), time.process_time()
    OUT.mkdir(exist_ok=False)
    def alarm(*_):
        raise TimeoutError('300 second CPU audit window')
    signal.signal(signal.SIGALRM, alarm)
    signal.alarm(300)
    bindings = {}
    def bind(path, expected=None):
        p = Path(path).resolve()
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        if expected is not None:
            assert digest == expected, str(p)
        bindings[str(p)] = digest
        return p
    def read(path):
        return json.loads(bind(path).read_text())
    def save(name, value):
        (OUT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    def table(name, rows):
        save(name + '.json', rows)
        with (OUT / (name + '.csv')).open('w') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    bind(__file__)
    historical = read(OLD / 'asset_bindings.json')
    inventory = read(GEN / 'inventory.json')
    old_rows = read(OLD / 'per_graph.json')
    protocol = read(GEN / 'protocol.json')
    assets = read(ROOT / 'results/typed_heat_audit_20261111/assets.json')
    helper_path = ROOT / 'research_branches/neighbor_signal_audit_20261111/run.py'
    bind(helper_path, historical[str(helper_path)])
    spec = importlib.util.spec_from_file_location('frozen_heat', helper_path)
    helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
    save('protocol.json', dict(
        graphs=80, references=2, model_calls=0, GPU_hours=0, new_graphs=0,
        seconds_cap=300, threads=1, RSS_cap_GiB=2,
        scope='Existing development outputs only; no tuning, no trainable score selected.',
        heat='Original helper and reference bandwidth. Subset kernels keep old scales, no bandwidth refit.',
        blocks='return, type reach, nine individual diffusion times; subset MMD is NOT additive attribution of full MMD.',
        closure='Raw triangle count and tr(S^3)/6; weighted triangles, not a replacement quality metric.',
        calibration='Profile Gaussian illustration at positive MSE e: sigma^2=e, loss=.5+.5log(e). Not a GraphVAE-MM reproduction.',
        limits=['Old final graphs cannot identify the time of reverse-chain drift.',
                'All samples previously viewed; not independent confirmation.',
                'No network/gradient/optimization evidence or formal training authorization.']))
    contexts = {}
    checks = helper.engineering_checks()
    def describe(a, ty):
        _, _, h, _ = helper.describe(a, ty, np.zeros((len(ty), 1)))
        d = a.sum(1)
        inv = np.divide(1., np.sqrt(d), out=np.zeros_like(d), where=d > 0)
        s = inv[:, None] * a * inv[None, :]
        raw = np.diag(a @ a @ a) / 6
        weighted = np.diag(s @ s @ s) / 6
        # Independent common-neighbour expansion of degree-weighted triangles.
        brute = 0.
        for i, j in zip(*np.where(np.triu(a, 1))):
            k = np.flatnonzero((a[i] > 0) & (a[j] > 0) & (np.arange(len(a)) > j))
            if len(k):
                brute += np.sum(1. / (d[i] * d[j] * d[k]))
        np.testing.assert_allclose(weighted.sum(), brute, atol=1e-11, rtol=1e-11)
        return h, raw, weighted
    def load_graph(path):
        z = np.load(bind(path, historical[str(path)]))
        ty, e = z['types'], z['edges']
        e = e if e.shape[0] == 2 else e.T
        assert len(set(tuple(sorted(x)) for x in e.T)) == e.shape[1]
        assert np.all(e[0] != e[1])
        a = np.zeros((len(ty), len(ty)))
        a[e[0], e[1]] = a[e[1], e[0]] = 1.
        return a, ty, z['schema']
    for p in protocol['parents']:
        ds = p['dataset']
        if ds in contexts:
            continue
        a, ty, schema = load_graph(p['data'])
        h, raw, weighted = describe(a, ty)
        asset = next(x for x in assets if x['dataset'] == ds)
        contexts[ds] = dict(types=ty, schema=schema, h=h, raw=raw, weighted=weighted,
            bands={int(k): v for k,v in asset['bandwidth_squared'].items()})
    rows, group_rows = [], []
    id_keys = ('model','dataset','train_seed','arm','readout','sample_seed')
    for item in inventory:
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > 2*1024*1024:
            raise MemoryError('2GiB RSS cap')
        ident = {k:item[k] for k in id_keys}
        old = next(x for x in old_rows if all(x[k] == v for k,v in ident.items()))
        c = contexts[item['dataset']]
        a, ty, schema = load_graph(item['graph'])
        np.testing.assert_array_equal(ty, c['types'])
        np.testing.assert_array_equal(schema, c['schema'])
        h, raw, weighted = describe(a, ty)
        assert abs(abs(raw.sum()-c['raw'].sum())-old['triangle_abs_error']) < 1e-9
        T, K = len(helper.TIMES), len(np.unique(ty))
        blocks = {'full':np.arange(h.shape[1]), 'return':np.arange(T), 'reach':np.arange(T,h.shape[1])}
        for t in range(T):
            blocks[f'time{t}'] = np.r_[t, np.arange(T+t*K, T+(t+1)*K)]
        scores = {b:[] for b in blocks}
        for typ in np.unique(ty):
            mask = ty == typ
            g = dict(**ident, type=int(typ), nodes=int(mask.sum()))
            for b, cols in blocks.items():
                x, y = h[mask][:,cols], c['h'][mask][:,cols]
                band = c['bands'][int(typ)]
                value = helper.mmd(helper.kernel(x,x,band), helper.kernel(y,y,band), helper.kernel(x,y,band))
                scores[b].append(value); g['heat_'+b] = value
            g.update(raw_triangle_incidence_share=float(raw[mask].sum()),
                weighted_triangle_incidence_share=float(weighted[mask].sum()))
            group_rows.append(g)
        np.testing.assert_allclose(np.mean(scores['full']),old['heat_mean'],atol=1e-11)
        rows.append(dict(**ident, triangles=float(raw.sum()), triangle_abs_error=old['triangle_abs_error'],
            weighted_triangles=float(weighted.sum()), weighted_triangle_abs_error=float(abs(weighted.sum()-c['weighted'].sum())),
            **{'heat_'+b:float(np.mean(v)) for b,v in scores.items()}))
    summaries=[]
    metric_keys=[k for k in rows[0] if k not in id_keys]
    for key in sorted({tuple(r[k] for k in id_keys[:-1]) for r in rows}):
        selected=[r for r in rows if tuple(r[k] for k in id_keys[:-1]) == key]
        assert len(selected)==4
        summaries.append(dict(zip(id_keys[:-1],key),samples=4,
            **{m:float(np.mean([r[m] for r in selected])) for m in metric_keys}))
    comparisons=[]
    for r in summaries:
        if r['arm']=='parent': continue
        p=next(x for x in summaries if x['model']==r['model'] and x['arm']=='parent')
        comparisons.append(dict(**{k:r[k] for k in id_keys[:-1]},
            **{m+'_delta':r[m]-p[m] for m in metric_keys},
            **{m+'_relative_percent':100*(r[m]/p[m]-1) if p[m] else None for m in metric_keys}))
    # Algebraic audit: adaptive residual scaling still permits compensating errors.
    e=np.logspace(-5,3,50); step=1e-5
    objective=lambda x: .5+.5*np.log(x)
    numeric=(objective(e*(1+step))-objective(e*(1-step)))/(2*step*e)
    np.testing.assert_allclose(numeric,.5/e,rtol=1e-8)
    ratios=np.array([1.3,1.3,1.3,.1])
    checks.update(profile_gaussian_derivative_checked=True,
        illustrative_four_group_error_ratios=ratios.tolist(),
        mean_error_change_fraction=float(ratios.mean()-1),
        profile_loss_change=float(.5*np.log(ratios).mean()),
        illustration_is_not_project_experiment=True,
        normalized_triangle_identity_checked_graphs=len(rows)+len(contexts),
        original_full_heat_scores_reproduced=True)
    for name, values in [('per_graph',rows),('per_type',group_rows),('model_summary',summaries),('comparisons',comparisons)]:
        table(name,values)
    save('checks.json',checks)
    save('references.json',{ds:dict(triangles=float(c['raw'].sum()),weighted_triangles=float(c['weighted'].sum())) for ds,c in contexts.items()})
    for p,h in bindings.items():
        assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
    save('asset_bindings.json',bindings)
    save('completion.json',dict(status='complete',graphs=80,model_calls=0,new_graphs=0,GPU_hours=0,
        wall_seconds=time.time()-start,CPU_seconds=time.process_time()-cpu,
        RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,original_assets_unchanged=True))
    for r in comparisons:
        if r['arm']=='risk':
            print(json.dumps({k:r[k] for k in ('dataset','train_seed','readout','triangle_abs_error_relative_percent',
                'weighted_triangle_abs_error_relative_percent','heat_full_relative_percent','heat_return_relative_percent','heat_reach_relative_percent')}))
    signal.alarm(0)


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        if OUT.exists():
            (OUT/'failure.json').write_text(json.dumps(dict(type=type(exc).__name__,error=str(exc))))
        raise
