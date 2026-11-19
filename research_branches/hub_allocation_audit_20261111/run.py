"""CPU-only fixed-count allocation audit. No model or loss changes."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '1'
import csv
import hashlib
import importlib.util
import json
import resource
import time
import traceback
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import wasserstein_distance

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/hub_allocation_audit_20261111'


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT/path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


heat = load_module('allocation_heat', 'research_branches/typed_heat_audit_20261111/run.py')
joint = load_module('allocation_joint', 'research_branches/typed_heat_degree_audit_20261111/run.py')
SEEDS = [2026111191, 2026111192, 2026111193]
FRACTIONS = [.05, .10, .20]
MODES = ['spread', 'concentrate', 'random']


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(name, value):
    (OUT/name).write_text(json.dumps(value, indent=2, allow_nan=False))


def table(name, rows):
    save(name+'.json', rows)
    if rows:
        with (OUT/(name+'.csv')).open('w') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def propose(a, original, nodes, mode, rng):
    """Move one endpoint within a type; never remove newly-added edges or re-add originals.

    Select maximal degree-gap pairs for directional edits, random order for ties.
    No objective scores are inspected. Return None when no legal move exists.
    """
    d = a.sum(1).astype(int)
    pairs = [(int(u), int(v)) for u in nodes for v in nodes if u != v and d[u] > 0]
    if mode == 'spread':
        pairs = [(u,v) for u,v in pairs if d[u] > d[v]+1]
    elif mode == 'concentrate':
        pairs = [(u,v) for u,v in pairs if d[v] >= d[u]]
    rng.shuffle(pairs)
    if mode == 'spread':
        pairs.sort(key=lambda uv: d[uv[0]]-d[uv[1]], reverse=True)
    elif mode == 'concentrate':
        pairs.sort(key=lambda uv: d[uv[1]]-d[uv[0]], reverse=True)
    for u,v in pairs:
        eligible = np.flatnonzero(a[u] & original[u] & ~a[v] & ~original[v])
        eligible = eligible[(eligible != u) & (eligible != v)]
        if len(eligible):
            return u, v, int(rng.choice(eligible))
    return None


def apply_move(a, move):
    u,v,w = move
    assert a[u,w] and not a[v,w] and len({u,v,w}) == 3
    a[u,w] = a[w,u] = False
    a[v,w] = a[w,v] = True


def basic_checks():
    a = np.zeros((5,5), dtype=bool)
    for v in [1,2,3]: a[0,v] = a[v,0] = True
    original=a.copy()
    for mode in ['spread','concentrate','random']:
        b=a.copy(); move=propose(b,original,np.arange(5),mode,np.random.default_rng(910))
        if move is None:
            assert mode == 'concentrate'
            continue
        before=b.sum(1).astype(int); apply_move(b,move); after=b.sum(1).astype(int)
        assert b.sum()==a.sum() and not b.diagonal().any() and np.array_equal(b,b.T)
        if mode=='spread': assert np.sum(after**2)<np.sum(before**2)
        u,v,w=move; apply_move(b,(v,u,w)); assert np.array_equal(b,a)
    return dict(simple_undirected_move_pass=True, no_available_move_handled=True, reverse_edit_pass=True)


def main():
    start=time.time(); OUT.mkdir(parents=True,exist_ok=False)
    base=ROOT/'results/typed_heat_degree_audit_20261111'
    scaling=json.loads((base/'reference_scaling.json').read_text())
    tasks=json.loads((ROOT/'results/real192_update_audit_20261111/protocol.json').read_text())['tasks']
    bind={}
    for mod in [heat,joint,heat.old,heat.old.lib]: bind[mod.__file__]=sha(mod.__file__)
    for p in [Path(__file__).resolve(),base/'reference_scaling.json',base/'per_graph.json',base/'per_type.json',heat.OUT/'protocol.json',heat.OUT/'signatures.npz']:
        bind[str(p)]=sha(p)
    for p,h in json.loads((heat.OUT/'protocol.json').read_text())['bindings'].items():
        assert sha(p)==h; bind[p]=h
    save('protocol.json',dict(frozen_before_scoring=True,
        track='EVALUATION: existing output objective tail sensitivity',
        scope='Two references x four focal types x three CPU edit seeds x three directions x three checkpoints = at most 216 transformed snapshots, plus two references; no generated-model graphs',
        seeds=SEEDS, fractions=FRACTIONS, modes=MODES,
        move='Replace (u,w) with (v,w), u and v same focal type; legal original edge removed and original nonedge added, no loops/duplicates, total and all relation counts invariant',
        spread='Prefer largest current degree donor-recipient gap >1; strictly decrease focal sum of squared degrees',
        concentrate='Prefer largest recipient-donor gap >=0; strictly increase focal sum of squared degrees',
        random='Same endpoint-reassignment support without degree preference',
        exposure='Checkpoints after floor(fraction*reference edges incident to focal type), minimum1; equal requested move counts across directions. Keep partial blocked trajectories; no extra attempts or replacement seeds.',
        score='Unchanged per-type k_heat*k_degree empirical MMD2/2, then mean over all four types; old scales/bandwidths. Relation-count term is exactly zero for all constructed states for any coefficient. No new hub penalty or lambda selection.',
        evaluation='Reference-frozen low degree<=1, hub>=max(2,ceil(reference type q90)), remaining ordinary; distributions not ID matching. Also sorted top ceil(.1*n_type) degree mass and count support. All graph/node-degree and triangle/component changes retained.',
        interpretation='Nonzero discrimination and magnitude/monotonicity are descriptive. Increasing anonymous degree error with decreasing joint score is a possible tradeoff/blind spot, not automatically a bug. No numerical acceptance threshold or statistical significance.',
        boundary='Single CPU thread 600 seconds, no model/GPU/updates/reverse chain; no parameter sweep, no formal-loss modification', bindings=bind))
    checks=basic_checks(); rows=[]; groups=[]; operations=[]; trajectories=[]; stored={}; meta={}
    data_tasks={t['parent']['dataset']:t for t in tasks}
    for ds,task in data_tasks.items():
        data=np.load(task['parent']['data'])
        z=np.load(ROOT/'results/real192_update_audit_20261111'/task['id']/'validation_u2000_raw_t1.npz')
        core=heat.old.FastScore(data['types'],z['query'],z['families'],z['labels'],len(data['schema']))
        a=heat.adj(core.n,core.q[:,core.reference.astype(bool)].T).astype(bool)
        ref=core.features(core.reference); hr=heat.heat(a,core.types)['feat']; dr=np.zeros_like(ref['degree'],dtype=float)
        for t in np.unique(core.types):
            sel=core.types==t
            dr[sel]=np.log1p(ref['degree'][sel])/scaling[ds]['log_degree_scales'][str(t)]/np.sqrt(core.r)
        thresholds={int(t):scaling[ds]['hub_thresholds'][str(t)] for t in np.unique(core.types)}
        meta[ds]=dict(types=core.types.tolist(), reference_edges=int(core.reference.sum()),
                      reference_relation_counts=np.bincount(core.f,weights=core.reference,minlength=core.r).astype(int).tolist(),
                      hub_thresholds=thresholds, schema=data['schema'].tolist())

        def evaluate(graph, focal, mode, seed, fraction, count, requested, complete):
            if time.time()-start>590: raise TimeoutError('CPU budget approaching')
            bits=graph[core.q[0],core.q[1]].astype(np.uint8)
            # Check the query support really encodes all existing graph edges.
            assert bits.sum()*2==graph.sum() and not graph.diagonal().any()
            np.testing.assert_array_equal(np.bincount(core.f,weights=bits,minlength=core.r),meta[ds]['reference_relation_counts'])
            costs, f=core.costs(bits); assert costs[0]==0
            h=heat.heat(graph.astype(float),core.types)['feat']; d=np.zeros_like(dr); scores=[]; heat_scores=[]; degree_scores=[]
            key=f'graph_{len(rows):03d}'
            per=[]
            for typ in np.unique(core.types):
                typ=int(typ); sel=core.types==typ
                d[sel]=np.log1p(f['degree'][sel])/scaling[ds]['log_degree_scales'][str(typ)]/np.sqrt(core.r)
                hk=[joint.kernel(x,y,scaling[ds]['heat_bandwidth_squared'][str(typ)]) for x,y in [(h[sel],h[sel]),(hr[sel],hr[sel]),(h[sel],hr[sel])]]
                dk=[joint.kernel(x,y,scaling[ds]['degree_bandwidth_squared'][str(typ)]) for x,y in [(d[sel],d[sel]),(dr[sel],dr[sel]),(d[sel],dr[sel])]]
                value=joint.mmd(*[hk[j]*dk[j] for j in range(3)]); assert value>-1e-10
                scores.append(value); heat_scores.append(joint.mmd(*hk)); degree_scores.append(joint.mmd(*dk))
                total=f['degree'][sel].sum(1); rt=ref['degree'][sel].sum(1); top=max(1,int(np.ceil(.1*len(total))))
                mass=float(np.sort(total)[-top:].sum()/total.sum()) if total.sum() else None
                rmass=float(np.sort(rt)[-top:].sum()/rt.sum()) if rt.sum() else None
                cross=np.abs(f['degree'][sel,None,:]-ref['degree'][None,sel,:]).sum(-1)
                ii,jj=linear_sum_assignment(cross); joint_mae=float(cross[ii,jj].mean())
                group=dict(dataset=ds,key=key,focal_type=focal,mode=mode,seed=seed,fraction=fraction,type=typ,nodes=len(total),
                    joint_score=value,heat_only_score=heat_scores[-1],degree_only_score=degree_scores[-1],
                    degree_W1=float(wasserstein_distance(total,rt)),relation_joint_L1=joint_mae,
                    low_count=int((total<=1).sum()),ref_low_count=int((rt<=1).sum()),
                    ordinary_count=int(((total>1)&(total<thresholds[typ])).sum()),
                    ref_ordinary_count=int(((rt>1)&(rt<thresholds[typ])).sum()),
                    hub_count=int((total>=thresholds[typ]).sum()),ref_hub_count=int((rt>=thresholds[typ]).sum()),
                    hub_threshold=thresholds[typ],top10_mass=mass,ref_top10_mass=rmass,
                    top10_mass_error=abs(mass-rmass) if mass is not None and rmass is not None else None,
                    max_degree=int(total.max()),ref_max_degree=int(rt.max()),
                    second_moment=float(np.mean(total**2)),ref_second_moment=float(np.mean(rt**2)))
                groups.append(group); per.append(group)
            if focal is not None:
                other=core.types!=focal
                np.testing.assert_array_equal(f['degree'][other],ref['degree'][other])
            focus=per[focal] if focal is not None else None
            rows.append(dict(dataset=ds,key=key,focal_type=focal,mode=mode,seed=seed,fraction=fraction,
                moves=count,requested=requested,complete=complete,edges=int(bits.sum()),triangles=int(f['triangles'].sum()),
                components=int(f['components']),lcc=int(f['lcc']),relation_count_cost=float(costs[0]),
                joint_score=float(np.mean(scores)),heat_only_score=float(np.mean(heat_scores)),degree_only_score=float(np.mean(degree_scores)),
                focal_joint_score=focus['joint_score'] if focus else None,
                focal_degree_W1=focus['degree_W1'] if focus else None,
                focal_relation_joint_L1=focus['relation_joint_L1'] if focus else None,
                focal_top10_error=focus['top10_mass_error'] if focus else None,
                focal_second_moment_error=abs(focus['second_moment']-focus['ref_second_moment']) if focus else None,
                typed_triangle_cost=float(costs[2]+costs[3]), original_joint_degree_cost=float(costs[1])))
            stored[key+'_bits']=bits
            return rows[-1]

        evaluate(a,None,'reference',None,0,0,0,True)
        for focal in np.unique(core.types):
            focal=int(focal); nodes=np.flatnonzero(core.types==focal)
            incident=int(np.count_nonzero(core.reference.astype(bool)&((core.types[core.q[0]]==focal)|(core.types[core.q[1]]==focal))))
            for seed in SEEDS:
                for mode in MODES:
                    rng=np.random.default_rng(np.random.SeedSequence([seed,focal,0 if ds=='IMDB' else 1]))
                    graph=a.copy(); count=0; blocked=False; localrows=[]
                    for fraction in FRACTIONS:
                        target=max(1,int(np.floor(fraction*incident)))
                        while count<target and not blocked:
                            if time.time()-start>580: raise TimeoutError('CPU budget approaching')
                            move=propose(graph,a,nodes,mode,rng)
                            if move is None: blocked=True; break
                            before=graph.sum(1).astype(int); apply_move(graph,move); after=graph.sum(1).astype(int)
                            delta=int(np.sum(after[nodes]**2)-np.sum(before[nodes]**2))
                            assert (mode!='spread' or delta<0) and (mode!='concentrate' or delta>0)
                            count+=1
                            operations.append(dict(dataset=ds,focal_type=focal,seed=seed,mode=mode,index=count,
                                donor=move[0],recipient=move[1],neighbor=move[2],degree_square_delta=delta))
                        localrows.append(evaluate(graph,focal,mode,seed,fraction,count,target,count==target))
                    for previous,current in zip(localrows,localrows[1:]):
                        trajectories.append(dict(dataset=ds,focal_type=focal,seed=seed,mode=mode,
                            from_fraction=previous['fraction'],to_fraction=current['fraction'],new_moves=current['moves']-previous['moves'],
                            joint_score_change=current['joint_score']-previous['joint_score'],
                            focal_score_change=current['focal_joint_score']-previous['focal_joint_score'],
                            degree_W1_change=current['focal_degree_W1']-previous['focal_degree_W1'],
                            tail_mass_error_change=current['focal_top10_error']-previous['focal_top10_error'],
                            second_moment_error_change=current['focal_second_moment_error']-previous['focal_second_moment_error'],
                            both_complete=previous['complete'] and current['complete']))
    table('per_graph',rows); table('per_type',groups); table('operations',operations); table('trajectory_differences',trajectories)
    save('assets.json',meta); np.savez_compressed(OUT/'graphs.npz',**stored)
    summary=[]
    for ds,focal,mode,fraction in sorted({(r['dataset'],r['focal_type'],r['mode'],r['fraction']) for r in rows if r['focal_type'] is not None}):
        rr=[r for r in rows if (r['dataset'],r['focal_type'],r['mode'],r['fraction'])==(ds,focal,mode,fraction)]
        rec=dict(dataset=ds,focal_type=focal,mode=mode,fraction=fraction,n=len(rr),completed=sum(r['complete'] for r in rr))
        for name in ['joint_score','focal_joint_score','focal_degree_W1','focal_relation_joint_L1','focal_top10_error','triangles','components']:
            rec[name+'_mean']=float(np.mean([r[name] for r in rr]));rec[name+'_sd']=float(np.std([r[name] for r in rr],ddof=1))
        summary.append(rec)
    table('summary',summary)
    checks.update(relation_counts_preserved_all=True,nonfocal_relation_degrees_unchanged=True,
                  degree_second_moment_direction_verified=True,reference_scores_zero=all(abs(r['joint_score'])<1e-12 for r in rows if r['mode']=='reference'))
    save('checks.json',checks)
    for p,h in bind.items(): assert sha(p)==h
    save('receipt.json',dict(status='complete',seconds=time.time()-start,peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        cpu_threads=1,rows=len(rows),type_rows=len(groups),moves=len(operations),incomplete_checkpoints=sum(not r['complete'] for r in rows),
        new_CPU_transformed_snapshots=len(rows)-2,model_calls=0,GPU_hours=0,training_updates=0,reverse_chains=0,source_assets_unchanged=True))
    print(json.dumps(json.loads((OUT/'receipt.json').read_text()),indent=2))


if __name__=='__main__':
    try: main()
    except BaseException:
        if OUT.exists(): save('failure.json',dict(traceback=traceback.format_exc()))
        raise
