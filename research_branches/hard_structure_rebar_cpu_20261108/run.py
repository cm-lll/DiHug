import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[key] = '1'
import csv
import hashlib
import itertools
import json
import resource
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
from objective import JointLocalScore, estimate_parts, edge_conditional, conditional_logistic

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.set_default_dtype(torch.float64)
ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = ROOT / 'results/hard_structure_rebar_cpu_20261108'
OLD = ROOT / 'results/reproducible_objective_20261108T135622/attempt2'
CFG = json.loads((HERE / 'protocol.json').read_text())
START = time.time()
COUNTS = dict(model_calls=0, optimizer_updates=0, GPU_hours=0, diffusion_chains=0,
              real_auxiliary_hard_graph_samples=0, synthetic_hard_graph_samples=0,
              real_edge_counterfactual_score_evaluations=0, panels_complete=0)
ASSETS = {}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def dump(name, data):
    (OUT / name).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False))


def csvout(name, rows):
    with (OUT / name).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader()
        writer.writerows(rows)


def check_time():
    if time.time() - START >= CFG['worker_wall_seconds'] - 10:
        raise TimeoutError('CPU worker budget reached; no panel may silently shrink')


def graph_fixture(n, edges, types=None):
    types = np.zeros(n, int) if types is None else np.asarray(types)
    q = np.array(np.triu_indices(n, 1))
    pairs = sorted(set(tuple(sorted((int(types[i]), int(types[j])))) for i, j in q.T))
    families = np.array([pairs.index(tuple(sorted((int(types[i]), int(types[j]))))) for i, j in q.T])
    wanted = set(tuple(sorted(e)) for e in edges)
    y = np.array([tuple(x) in wanted for x in q.T], float)
    return JointLocalScore(types, q, families, y, len(pairs)), torch.tensor(y)


def tests():
    rows = []
    rng = np.random.default_rng(CFG['synthetic_seed'])
    for trial in range(20):
        n = 5 + trial % 4
        types = np.arange(n) % 3
        edges = [x for x in itertools.combinations(range(n), 2) if rng.random() < .35]
        score, y = graph_fixture(n, edges, types)
        feature = score.features(y).numpy()
        manual = np.zeros_like(feature)
        edge_set = set(edges)
        for e, (i, j) in enumerate(score.query.T.tolist()):
            if y[e]:
                r = int(score.families[e]); manual[i, r] += 1; manual[j, r] += 1
        for tri in itertools.combinations(range(n), 3):
            if all(tuple(sorted(e)) in edge_set for e in itertools.combinations(tri, 2)):
                for i in tri:
                    others = sorted(types[j] for j in tri if j != i)
                    manual[i, score.r + score.triangle_pairs.index(tuple(others))] += 1
        error = float(np.max(np.abs(feature - manual)))
        assert error == 0
        assert abs(float(score(y))) < 1e-12
        # A same-type permutation must not change the score against the reference.
        perm = np.arange(n)
        for t in set(types):
            ix = np.flatnonzero(types == t); perm[ix] = rng.permutation(ix)
        remapped = {tuple(sorted((perm[i], perm[j]))) for i, j in edges}
        moved = torch.tensor([float(tuple(x) in remapped) for x in score.query.T.tolist()])
        assert abs(float(score(moved))) < 1e-12
        rows.append(dict(test='typed_triangle_degree_and_permutation', trial=trial, max_error=error))
    cycle6 = [(i, (i + 1) % 6) for i in range(6)]
    two_triangles = [(0,1),(1,2),(0,2),(3,4),(4,5),(3,5)]
    score, _ = graph_fixture(6, cycle6)
    _, b = graph_fixture(6, two_triangles)
    separated = float(score(b)); assert separated > 0
    rows.append(dict(test='same_degree_different_triangles', score=separated, detected=True))
    cycle8 = [(i, (i + 1) % 8) for i in range(8)]
    two_cycles = [(0,1),(1,2),(2,3),(0,3),(4,5),(5,6),(6,7),(4,7)]
    score, _ = graph_fixture(8, cycle8); _, b = graph_fixture(8, two_cycles)
    blind = float(score(b)); assert abs(blind) < 1e-12
    rows.append(dict(test='same_local_features_different_connectivity', score=blind,
                     detected=False, limitation='Intentional: no connectivity guarantee'))
    score, _ = graph_fixture(5, [(0,1),(0,2),(0,3),(0,4)])
    _, b = graph_fixture(5, [(0,1),(1,2),(2,3),(3,4)])
    assert float(score(b)) > 0
    rows.append(dict(test='same_edge_count_different_allocation', score=float(score(b)), detected=True))
    # Conditional sampler sign/support and finite extreme-logit derivative check.
    logits = torch.tensor([-700., -40., 0., 40., 700.], requires_grad=True)
    v = torch.tensor([.1, .3, .5, .7, .9])
    for bit in [0., 1.]:
        z = conditional_logistic(logits, torch.full_like(logits, bit), v)
        assert bool(torch.isfinite(z).all())
        assert bool((z >= 0).all()) if bit else bool((z <= 0).all())
        grad = torch.autograd.grad(z.sum(), logits)[0]
        assert bool(torch.isfinite(grad).all())
    rows.append(dict(test='conditional_sampler_extreme_logits', passed=True))
    dump('structural_tests.json', rows)
    return rows


def exact_toy():
    score, _ = graph_fixture(4, [(0,1),(0,2),(1,2),(2,3)], [0,0,1,1])
    m = score.query.shape[1]
    states = torch.tensor(list(itertools.product([0., 1.], repeat=m)))
    values = score(states).detach()
    reports = []
    for profile, p in enumerate([torch.full((m,), .5), torch.linspace(.1,.9,m),
                                  torch.tensor([.01,.99,.05,.95,.2,.8])]):
        logits = torch.logit(p).requires_grad_(True)
        pp = logits.sigmoid()
        masses = torch.where(states.bool(), pp, 1-pp).prod(-1)
        expectation = (masses * values).sum()
        exact = torch.autograd.grad(expectation, logits)[0].detach()
        enumerated = ((states-p) * values[:,None] * masses.detach()[:,None]).sum(0)
        direct_all = torch.stack([edge_conditional(score,p,b,torch.arange(m)) for b in states])
        direct_exact = (direct_all * masses.detach()[:,None]).sum(0)
        assert torch.max(abs(exact-enumerated)) < 1e-12
        assert torch.max(abs(exact-direct_exact)) < 1e-12
        # Exact Rao-Blackwell coordinate variance inequality, with no MC uncertainty.
        rf_all = values[:,None]*(states-p)
        vr = ((rf_all-exact).square()*masses.detach()[:,None]).sum(0)
        vd = ((direct_all-exact).square()*masses.detach()[:,None]).sum(0)
        assert bool((vd <= vr+1e-14).all())
        generator = torch.Generator().manual_seed(CFG['synthetic_seed']+100+profile)
        estimates = []
        for j in range(0, CFG['synthetic_MC_samples'], 256):
            check_time()
            n = min(256, CFG['synthetic_MC_samples']-j)
            u=torch.rand((n,m), generator=generator); v=torch.rand((n,m), generator=generator)
            parts=estimate_parts(score,p.expand(n,-1),u,v)
            estimates.append((parts['rf_base']+parts['control']).numpy())
        array=np.concatenate(estimates); avg=array.mean(0); se=array.std(0,ddof=1)/len(array)**.5
        discrepancy=np.abs(avg-exact.numpy())
        max_z=float(np.max(discrepancy/np.maximum(se,1e-15)))
        assert np.isfinite(array).all() and max_z < 8, 'REBAR MC disagrees with exact toy gradient'
        np.savez_compressed(OUT/f'toy_{profile}.npz',exact=exact.numpy(),mean=avg,se=se,
                            raw_variance=vr.numpy(),conditional_variance=vd.numpy())
        reports.append(dict(profile=profile,probability=p.tolist(),states=len(states),
                            exact_loss=float(expectation),exact_gradient=exact.tolist(),
                            max_exact_derivative_error=float(torch.max(abs(exact-direct_exact))),
                            REBAR_mean_max_abs_error=float(discrepancy.max()),REBAR_max_SE_units=max_z,
                            REBAR_samples=len(array),conditional_variance_ratio=float(vd.sum()/vr.sum())))
        COUNTS['synthetic_hard_graph_samples'] += len(array)
    dump('exact_toy_tests.json', reports)
    return reports


def real_panels():
    source_bind=json.loads((ROOT/'results/local_allocation_design_20261108/asset_binding.json').read_text())
    for path, expected in source_bind.items():
        actual=sha(path); assert actual==expected, f'asset changed {path}'
        ASSETS[path]=actual
    rows=[]; coordinates=[]; metadata=[]
    for dataset_index, ds in enumerate(CFG['datasets']):
        data=next(Path(path) for path in ASSETS if path.endswith(f'/assets/{ds}.npz'))
        with np.load(data) as a:
            types=a['types']; schema=a['schema']; clean={tuple(sorted(x)) for x in a['edges'].tolist()}
        inputs=torch.load(OLD/f'{ds}_fixed_inputs.pt',map_location='cpu')['inputs']
        for t in CFG['noise_times']:
            inp=next(x for panel,tt,seed,x in inputs if panel=='new_diagnostic' and tt==t)
            q, families, labels=np.asarray(inp['query']),np.asarray(inp['f']),np.asarray(inp['labels'])
            assert np.array_equal(labels, np.array([tuple(sorted(x)) in clean for x in q.T],int))
            assert len(set(map(tuple,np.sort(q,axis=0).T)))==q.shape[1]
            for arm in CFG['parent_local_arms']:
                check_time()
                name=f'{ds}_{arm}_{t}'; panel_id=len(metadata)
                with np.load(OLD/f'{ds}_{arm}_32_raw_new_diagnostic_{t}.npz') as a:
                    fullp=a['probability'].astype(float)
                assert np.isfinite(fullp).all() and ((fullp>=0)&(fullp<=1)).all()
                mutable=(fullp>0)&(fullp<1); retained=fullp>0
                assert not np.any((~retained)&(labels==1)), 'Fixed-zero support contains true edge'
                retained_indices=np.flatnonzero(retained)
                retained_p=fullp[retained]; active=np.flatnonzero((retained_p>0)&(retained_p<1))
                core=JointLocalScore(types,q[:,retained],families[retained],labels[retained],len(schema))
                active_idx=torch.tensor(active)
                fixed=torch.tensor(retained_p)
                def score(x):
                    base=fixed.expand(*x.shape[:-1],len(fixed))
                    return core(base.index_copy(-1,active_idx,x))
                p=torch.tensor(retained_p[active])
                af=families[retained][active]; ay=labels[retained][active]
                select_rng=np.random.default_rng(CFG['selection_seed']+dataset_index)
                selected=[]
                for r in range(len(schema)):
                    for label in [0,1]:
                        choices=np.flatnonzero((af==r)&(ay==label))
                        if len(choices):
                            ix=int(select_rng.choice(choices)); selected.append(ix)
                            coordinates.append(dict(panel=name,coordinate=len(selected)-1,active_index=ix,
                                original_query_index=int(retained_indices[active[ix]]),relation=r,label=label,
                                stratum_support=len(choices),probability=float(p[ix])))
                selected=torch.tensor(selected); num=len(selected)
                rowmeta=dict(panel=name,dataset=ds,arm=arm,t=t,nodes=len(types),query_slots=len(fullp),
                    mutable_slots=int(mutable.sum()),fixed_zero=int((fullp==0).sum()),fixed_one=int((fullp==1).sum()),
                    selected_coordinates=num,reference_score=float(score(torch.tensor(ay,dtype=torch.float64))),
                    feature_dimensions={str(k):int(v.sum()) for k,v in core.masks.items()},
                    type_support={str(k):len(v) for k,v in core.nodes.items()},
                    probability_file=str(OLD/f'{ds}_{arm}_32_raw_new_diagnostic_{t}.npz'))
                assert abs(rowmeta['reference_score'])<1e-12
                all_parts={}; times={}; hard_counts={}
                for phase, count, seed in [('development',CFG['development_samples'],CFG['development_seed']),
                                           ('evaluation',CFG['evaluation_samples'],CFG['evaluation_seed'])]:
                    generator=torch.Generator().manual_seed(seed+panel_id)
                    arrays={k:[] for k in ['rf','rf_base','control','direct','hard_value']}
                    common_seconds=0.; direct_seconds=0.
                    for j in range(count):
                        check_time()
                        u=torch.rand(p.shape,generator=generator);v=torch.rand(p.shape,generator=generator)
                        tick=time.perf_counter(); parts=estimate_parts(score,p,u,v,CFG['temperature'])
                        common_seconds+=time.perf_counter()-tick
                        for k in ['rf','rf_base','control']:
                            arrays[k].append(parts[k][selected].numpy())
                        arrays['hard_value'].append(float(parts['hard_value']))
                        if phase=='evaluation':
                            tick=time.perf_counter(); direct=edge_conditional(score,p,parts['bits'],selected)
                            direct_seconds+=time.perf_counter()-tick; arrays['direct'].append(direct.numpy())
                            COUNTS['real_edge_counterfactual_score_evaluations']+=2*num
                        COUNTS['real_auxiliary_hard_graph_samples']+=1
                    all_parts[phase]={k:np.asarray(v) for k,v in arrays.items() if v}
                    times[phase]=dict(rebar_bundle_seconds=common_seconds,direct_selected_seconds=direct_seconds)
                    if phase=='development':
                        r=all_parts[phase]['rf_base'];c=all_parts[phase]['control']
                        rr=r-r.mean(0);cc=c-c.mean(0)
                        denom=float((cc*cc).sum())
                        eta= -float((rr*cc).sum())/denom if denom>0 else 0.
                        assert np.isfinite(eta)
                        # Freeze BEFORE reading evaluation draws; do not refit on evaluation.
                        dump(f'{name}_frozen_eta.json',dict(eta=eta,dev_samples=count,dev_seed=seed+panel_id,
                            eval_seed=CFG['evaluation_seed']+panel_id,temperature=CFG['temperature'],
                            baseline='detached score at probability mean',selected=selected.tolist()))
                ev=all_parts['evaluation'];estimates=dict(raw_RF=ev['rf'],baseline_RF=ev['rf_base'],
                    REBAR=ev['rf_base']+eta*ev['control'],edge_conditional=ev['direct'])
                # Cost comparison includes one additional RF-only pass on the SAME saved draws,
                # not additional random graph samples or a new evaluation set.
                generator=torch.Generator().manual_seed(CFG['evaluation_seed']+panel_id)
                tick=time.perf_counter()
                with torch.no_grad():
                    baseline=score(p)
                    for j in range(CFG['evaluation_samples']):
                        u=torch.rand(p.shape,generator=generator);torch.rand(p.shape,generator=generator)
                        bits=(torch.logit(p)+u.log()-torch.log1p(-u)>=0).double()
                        hard=score(bits); g=(hard-baseline)*(bits-p)
                        assert torch.allclose(g[selected],torch.tensor(ev['rf_base'][j]),atol=1e-12,rtol=1e-12)
                rf_seconds=time.perf_counter()-tick
                raw_var=float(estimates['raw_RF'].var(0,ddof=1).sum())
                for method, array in estimates.items():
                    assert np.isfinite(array).all()
                    variance=float(array.var(0,ddof=1).sum())
                    seconds=(times['evaluation']['direct_selected_seconds'] if method=='edge_conditional'
                             else times['evaluation']['rebar_bundle_seconds'] if method=='REBAR' else rf_seconds)
                    # Direct uses existing bits; include measured draw/reuse overhead only in bound.
                    gap=array-estimates['edge_conditional']
                    diff=gap.mean(0);se=gap.std(0,ddof=1)/len(gap)**.5
                    row=dict(panel=name,dataset=ds,arm=arm,t=t,method=method,selected=num,samples=len(array),
                        eta=eta,mean_loss=float(ev['hard_value'].mean()),loss_sd=float(ev['hard_value'].std(ddof=1)),
                        gradient_mean_L2=float(np.linalg.norm(array.mean(0))),coordinate_variance_sum=variance,
                        variance_ratio_to_raw_RF=variance/raw_var if raw_var else None,
                        evaluation_seconds=seconds,variance_times_seconds_per_sample=variance*seconds/len(array),
                        paired_mean_diff_L2_vs_conditional=float(np.linalg.norm(diff)),
                        paired_max_SE_units_vs_conditional=float(np.max(abs(diff)/np.maximum(se,1e-15))),
                        development_bundle_seconds=times['development']['rebar_bundle_seconds'])
                    rows.append(row)
                np.savez_compressed(OUT/f'{name}_samples.npz',**estimates,probabilities=p[selected].numpy(),
                                    selected=selected.numpy(),hard_value=ev['hard_value'])
                rowmeta.update(times=times,RF_only_eval_seconds=rf_seconds,eta=eta)
                metadata.append(rowmeta);COUNTS['panels_complete']+=1
                dump('results.json',rows);dump('panels.json',metadata);dump('progress.json',dict(counts=COUNTS,seconds=time.time()-START))
                print(json.dumps(dict(panel=name,eta=eta,seconds=time.time()-START,
                      variance_ratios={x['method']:x['variance_ratio_to_raw_RF'] for x in rows[-4:]})),flush=True)
    csvout('results.csv',rows);csvout('selected_edges.csv',coordinates);dump('selected_edges.json',coordinates)
    return rows


def main():
    OUT.mkdir(exist_ok=False)
    dump('protocol.json',CFG)
    dump('start.json',dict(start=datetime.now().astimezone().isoformat(),pid=os.getpid(),
        CPU_threads=1,worker_wall_seconds=CFG['worker_wall_seconds'],GPU_hours=0,
        python=sys.version,torch=torch.__version__,numpy=np.__version__,cwd=str(ROOT)))
    dump('source_binding.json',{str(p):sha(p) for p in HERE.glob('*') if p.is_file()})
    tests();exact_toy();real_panels()
    assert all(sha(p)==h for p,h in ASSETS.items())
    assert not torch.cuda.is_initialized()
    dump('asset_binding.json',ASSETS)
    dump('receipt.json',dict(status='complete',counts=COUNTS,seconds=time.time()-START,
         max_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
         cuda_initialized=False,assets_unchanged=True,finish=datetime.now().astimezone().isoformat()))


if __name__=='__main__':
    try:
        main()
    except BaseException as error:
        if OUT.exists():
            dump('failure.json',dict(error=repr(error),traceback=traceback.format_exc(),counts=COUNTS,
                 seconds=time.time()-START,max_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024))
        raise
