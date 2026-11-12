"""Frozen log1p hard-risk CPU audit; no torch/model imports or parameter updates."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[k] = '1'
import csv
import hashlib
import importlib.util
import itertools
import json
import resource
import time
import traceback
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/log_risk_cpu_20261111/run2'
OLD = ROOT / 'results/real192_update_audit_20261111'
SPEC = importlib.util.spec_from_file_location('frozen_hard_score', ROOT / 'research_branches/hard_clean_pilot_v2_20261110/objective.py')
lib = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(lib)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=2, allow_nan=False))


def table(name, rows):
    save(name + '.json', rows)
    if rows:
        with (OUT / (name + '.csv')).open('w') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


class FastScore(lib.HardScore):
    """Same hard costs; exact typed triangle counts through small matrix blocks."""
    def features(self, bits):
        bits = np.asarray(bits)
        assert bits.shape == self.reference.shape and np.isin(bits, [0, 1]).all()
        i, j = self.q[:, bits.astype(bool)];f = self.f[bits.astype(bool)]
        a = np.zeros((self.n, self.n), dtype=np.float64)
        a[i, j] = 1;a[j, i] = 1
        degree = np.zeros((self.n, self.r), dtype=np.int64)
        np.add.at(degree, (i, f), 1);np.add.at(degree, (j, f), 1)
        nodes = {t: np.flatnonzero(self.types == t) for t in np.unique(self.types)}
        tri = []
        for t, u, v in self.triples:
            ab = a[np.ix_(nodes[t], nodes[u])]
            bc = a[np.ix_(nodes[u], nodes[v])]
            ac = a[np.ix_(nodes[t], nodes[v])]
            divisor = 6 if t == v else (2 if t == u or u == v else 1)
            value = np.sum((ab @ bc) * ac) / divisor
            assert abs(value-round(value)) < 1e-9
            tri.append(round(value))
        nc, component = connected_components(csr_matrix(a), directed=False)
        return dict(counts=np.bincount(f, minlength=self.r), degree=degree,
                    triangles=np.asarray(tri), components=int(nc),
                    lcc=int(np.bincount(component).max()), isolated=int((degree.sum(1) == 0).sum()),
                    max_degree=int(degree.sum(1).max()))


def loo(p, b, cost):
    c = cost - cost[0]
    return ((len(b)*c-c.sum(0))/(len(b)-1)).T @ (b-p)/len(b)


def checks():
    types = np.array([0, 0, 1, 1])
    q = np.asarray(list(itertools.combinations(range(4), 2))).T
    f = np.asarray([int(types[a]+types[b]) for a, b in q.T])
    y = np.array([1, 1, 0, 1, 0, 1], dtype=np.uint8)
    core = FastScore(types, q, f, y, 3);old = lib.HardScore(types, q, f, y, 3)
    bits = np.asarray(list(itertools.product([0, 1], repeat=6)), dtype=np.uint8)
    costs = core.batch(bits)
    np.testing.assert_allclose(costs, old.batch(bits), rtol=0, atol=0)
    transformed = np.log1p(costs)
    rows = []
    for i, p in enumerate([np.linspace(.1, .8, 6), np.full(6, .5), np.array([.0001,.9999,.02,.9,.3,.6])]):
        probs = np.prod(np.where(bits, p, 1-p), axis=1)
        exact = (probs[:,None]*transformed).T @ (bits-p)
        z = np.log(p/(1-p));fd = np.zeros_like(exact);h=1e-5
        for j in range(6):
            vals=[]
            for sign in [-1,1]:
                zz=z.copy();zz[j]+=sign*h;pp=1/(1+np.exp(-zz))
                qq=np.prod(np.where(bits,pp,1-pp),axis=1);vals.append(qq@transformed)
            fd[:,j]=(vals[1]-vals[0])/(2*h)
        np.testing.assert_allclose(exact,fd,rtol=2e-6,atol=2e-10)
        # Exhaustive expectation of the K=2 leave-one-out estimator.
        pair = np.zeros_like(exact)
        for a in range(64):
            dc=transformed[a]-transformed
            pair += probs[a]*((probs[:,None]*dc).T @ (bits[a].astype(float)-bits))/2
        np.testing.assert_allclose(pair,exact,rtol=1e-11,atol=1e-13)
        rows.append(dict(case=i,finite_difference_max_abs=float(abs(exact-fd).max()),
                         exhaustive_LOO_max_abs=float(abs(exact-pair).max()),
                         expectation_transform_gap=(np.log1p(probs@costs)-probs@transformed).tolist()))
    const = np.ones((16,5));b=np.zeros((16,6));assert np.array_equal(loo(np.full(6,.2),b,const),np.zeros((5,6)))
    save('formula_checks.json', dict(passed=True, exhaustive_graphs=64, distributions=rows,
          identical_cost_zero_exact=True, note='Formula tests only, not toy training or quality evidence'))


def main():
    started=time.time();OUT.mkdir(exist_ok=False, parents=True)
    tasks=json.loads((OLD/'protocol.json').read_text())['tasks']
    panels=[];bindings={str(Path(__file__).resolve()):sha(__file__),str(Path(lib.__file__)):sha(lib.__file__)}
    for task in tasks:
        tid=task['id'];row=task['parent'];data=Path(row['data'])
        assert sha(data)==row['data_sha256'];bindings[str(data)]=sha(data)
        for phase in ['before','after']:
            for i in range(4):
                path=OLD/'same_input_step'/f'{tid}_{phase}_i{i}.npz'
                panels.append(dict(task=tid,dataset=row['dataset'],seed=row['seed'],kind='same_training_input_u2012',phase=phase,index=i,path=str(path)))
        for u in [2000,2050]:
            for t in [1,250,1000]:
                path=OLD/tid/f'validation_u{u}_raw_t{t}.npz'
                panels.append(dict(task=tid,dataset=row['dataset'],seed=row['seed'],kind='historical_fixed_validation',phase=str(u),index=t,path=str(path)))
    for x in panels:bindings[x['path']]=sha(x['path'])
    save('protocol.json',dict(objective='E[mean_j log1p(C_j(hard clean graph))]',
          risks=lib.HardScore.names, coefficients=[.2]*5, transform_scale=1,
          panel=panels,repeats=16,K=16,random_root=2026111151,
          compute='one CPU thread, 20-minute internal limit, no model/GPU/update/chain',
          comparisons='paired old/new all-MC logit gradients, NOT new MC vs old hybrid production optimizer',
          truth='No exact real-graph gradient; split-half agreement and empirical variance only',
          decision='Do not train from score reduction. Report rank tradeoffs, zero feedback, sampling uncertainty and any failure.',
          frozen_before_results=True, bindings=bindings))
    checks();cores={};fixed={};tests=[]
    for task in tasks:
        row=task['parent'];tid=task['id'];z=np.load(OLD/tid/'validation_u2000_raw_t1.npz')
        types=np.load(row['data'])['types'];r=len(np.load(row['data'])['schema'])
        core=FastScore(types,z['query'],z['families'],z['labels'],r)
        cores[tid]=core;fixed[tid]={k:z[k].copy() for k in ['ids','labels','families','query']}
        if row['dataset'] not in [v['dataset'] for v in tests]:
            legacy=lib.HardScore(types,z['query'],z['families'],z['labels'],r)
            rng=np.random.default_rng(2026111150)
            checkbits=[core.reference,np.zeros_like(core.reference),np.ones_like(core.reference)]
            checkbits += [(rng.random(len(core.reference))<p).astype(np.uint8) for p in [.02,.1,.5]]
            for bits in checkbits:
                a,aa=core.costs(bits);b,bb=legacy.costs(bits)
                np.testing.assert_allclose(a,b,rtol=0,atol=0)
                for k in aa:np.testing.assert_array_equal(aa[k],bb[k])
            tests.append(dict(dataset=row['dataset'],graphs=len(checkbits),passed=True))
    save('real_feature_equivalence.json',tests)
    rows=[];pairs=[];per_component=[]
    for index,item in enumerate(panels):
        if time.time()-started>1200:raise TimeoutError('CPU internal time bound')
        tick=time.time();tid=item['task'];c=cores[tid];z=np.load(item['path']);p=z['probability']
        for k in ['ids','labels','families']:np.testing.assert_array_equal(z[k],fixed[tid][k])
        assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
        all_cost=[];all_features=[];grad_old=[];grad_new=[];unique=[]
        for rep in range(16):
            # Pair before/after within an input, and u2000/u2050 within t.
            key=0 if item['kind']=='same_training_input_u2012' else 1
            rng=np.random.default_rng(np.random.SeedSequence([2026111151,item['seed'],key,item['index'],rep]))
            bits=(rng.random((16,len(p)))<p).astype(np.uint8)
            costs=[]
            for b in bits:
                cost,feat=c.costs(b);costs.append(cost)
                all_features.append([int(b.sum()),int(feat['triangles'].sum()),feat['components'],feat['lcc'],feat['max_degree']])
            costs=np.asarray(costs);new=np.log1p(costs)
            grad_old.append(loo(p,bits,costs.mean(1)[:,None])[0])
            grad_new.append(loo(p,bits,new.mean(1)[:,None])[0])
            all_cost.append(costs);unique.append(len(np.unique(bits,axis=0)))
        costs=np.concatenate(all_cost);new=np.log1p(costs);feats=np.asarray(all_features)
        base={k:v for k,v in item.items() if k!='path'}
        result=dict(**base,expected_edges=float(p.sum()),old_score=float(costs.mean()),new_score=float(new.mean()),
            old_dominant_share=float(costs.mean(0).max()/max(costs.mean(0).sum(),1e-30)),
            new_dominant_share=float(new.mean(0).max()/max(new.mean(0).sum(),1e-30)),
            empty_fraction=float((feats[:,0]==0).mean()),mean_unique_per_K16=float(np.mean(unique)),
            seconds=time.time()-tick)
        for name,grads in [('old',grad_old),('new',grad_new)]:
            gg=np.asarray(grads);gmean=gg.mean(0);var=float(((gg-gmean)**2).sum()/15)
            norm=float(np.linalg.norm(gmean));a=gg[:8].mean(0);b=gg[8:].mean(0)
            den=float(np.linalg.norm(a)*np.linalg.norm(b))
            result[name+'_mean_gradient_norm']=norm;result[name+'_variance_trace']=var
            result[name+'_mean_SE_over_norm']=float(np.sqrt(var/16)/norm) if norm>1e-30 else None
            result[name+'_split_half_cosine']=float(a@b/den) if den>1e-30 else None
            result[name+'_zero_gradient_repeats']=int((np.linalg.norm(gg,axis=1)==0).sum())
        for j,name in enumerate(c.names):
            per_component.append(dict(**base,component=name,old_mean=float(costs[:,j].mean()),
                 new_mean=float(new[:,j].mean()),new_SD=float(new[:,j].std(ddof=1)),
                 wrong_log_of_mean=float(np.log1p(costs[:,j].mean())),
                 jensen_gap=float(np.log1p(costs[:,j].mean())-new[:,j].mean())))
        np.savez_compressed(OUT/f'panel_{index:02d}.npz',old_costs=costs,new_costs=new,
              hard_features=feats,old_logit_gradients=np.asarray(grad_old),new_logit_gradients=np.asarray(grad_new))
        rows.append(result);table('panel_summary',rows);table('component_summary',per_component)
        print(json.dumps(dict(panel=index+1,total=len(panels),seconds=round(time.time()-started,2),task=tid,
                             phase=item['phase'],new_score=result['new_score'])),flush=True)
    # Re-score existing whole graphs, never generate a reverse trajectory.
    graphrows=[];graphbindings={}
    for task in tasks:
        tid=task['id'];c=cores[tid];row=task['parent'];lookup={tuple(edge):i for i,edge in enumerate(c.q.T)}
        sources=[]
        for B,K in [(1,64),(4,16)]:
            d=ROOT/'results/single_step_formal_resume_20261111'/f'{row["dataset"]}_{row["seed"]}_B{B}K{K}'
            sources.extend((f'failed_B{B}K{K}',p) for p in sorted(d.glob('sample_*/graph.npz')))
        sources.extend(('ordinary_CE',p) for p in sorted((Path(row['checkpoint']).parent/'ema_0.99').glob('sample_*/graph.npz')))
        for arm,path in sources:
            z=np.load(path);np.testing.assert_array_equal(z['types'],c.types)
            edges=z['edges'];edges=edges.T if edges.shape[0]==2 else edges
            bits=np.zeros(len(c.reference),dtype=np.uint8)
            for a,b in edges:bits[lookup[tuple(sorted((int(a),int(b))))]]=1
            cost,feat=c.costs(bits);graphbindings[str(path)]=sha(path)
            graphrows.append(dict(task=tid,arm=arm,path=str(path),old_score=float(cost.mean()),new_score=float(np.log1p(cost).mean()),
               edges=int(bits.sum()),triangles=int(feat['triangles'].sum()),components=feat['components'],lcc=feat['lcc']))
        for arm,bits in [('reference',c.reference),('empty_control',np.zeros_like(c.reference))]:
            cost,feat=c.costs(bits)
            graphrows.append(dict(task=tid,arm=arm,path=None,old_score=float(cost.mean()),new_score=float(np.log1p(cost).mean()),
                edges=int(bits.sum()),triangles=int(feat['triangles'].sum()),components=feat['components'],lcc=feat['lcc']))
    table('existing_graph_scores',graphrows);save('graph_bindings.json',graphbindings)
    for p,h in bindings.items():assert sha(p)==h
    save('receipt.json',dict(status='completed',seconds=time.time()-started,
          RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,threads=1,
          panels=len(panels),hard_conditional_draws=len(panels)*256,existing_graphs=len(graphbindings),
          GPU_hours=0,model_calls=0,updates=0,reverse_chains=0,bindings_unchanged=True))


if __name__=='__main__':
    try:main()
    except BaseException:
        if OUT.exists():save('failure.json',dict(traceback=traceback.format_exc()))
        raise
