"""CPU-only retrospective identity/structure audit. No model calls or fitting."""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import argparse
import csv
import hashlib
import itertools
import json
import resource
import sys
import time
from pathlib import Path
import numpy as np
import torch
torch.set_num_threads(1)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'research_branches/hard_structure_rebar_cpu_20261108'))
sys.path.insert(0, str(ROOT/'research_branches/direct_structure_cpu_20261108'))
from objective import JointLocalScore
from score import DirectScore
OLD = ROOT/'results/hard_structure_response_20261108T182154'
EDGE = ROOT/'results/hard_structure_edge_direction_cpu_20261108'
DIRECT = ROOT/'results/direct_structure_cpu_20261108'

def write_csv(path, rows):
    if rows:
        with path.open('w') as f:
            w = csv.DictWriter(f, list(rows[0])); w.writeheader(); w.writerows(rows)

def ce(p, y):
    return -np.mean(y*np.log(p)+(1-y)*np.log1p(-p))

def main(out):
    out.mkdir(parents=True, exist_ok=False)
    start, cpu = time.time(), time.process_time()
    bindings = {}
    def bind(path):
        path = Path(path); bindings[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest(); return path
    def read(path): return json.loads(bind(path).read_text())
    def save(name, obj): (out/name).write_text(json.dumps(obj, indent=2, allow_nan=False))
    def check():
        assert time.time()-start < 600, 'CPU time budget'
        assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024 < 2048, 'RSS budget'
    save('protocol.json', dict(scope='read-only sources and archived 96 conditional samples; finite synthetic enumeration',
        max_seconds=600, max_RSS_MiB=2048, CPU_threads=1, GPU=0, model_calls=0, training_updates=0,
        scores=['MMD', 'DIRECT'], score_sign_tolerance=1e-12,
        permutation_seeds=[2026110901], permutation_examples=['fixed random type-preserving', 'swap reference max/min degree within each type'],
        warning='posthoc descriptive directions, not parameter gradient or instability causation; DIRECT rescored on historical MMD panel, not DIRECT training results'))
    # Finite exact demonstration: two same-type centers and two same-type leaves.
    types = np.array([0,0,1,1]); q = np.array([[0,0,1,1],[2,3,2,3]])
    f = np.zeros(4, dtype=int); y = np.array([1.,1.,0.,0.]); other=1-y
    core = JointLocalScore(types,q,f,y,1)
    states = ((np.arange(16)[:,None] >> np.arange(4)) & 1).astype(float)
    direct = np.array([DirectScore(types,q,f,y,b,1).base for b in states])
    with torch.no_grad(): mmd=core(torch.from_numpy(states)).numpy()
    assert direct[3]==direct[12]==0 and abs(mmd[3])+abs(mmd[12])<1e-12
    toy=[]
    for s in [.05,.25,.5,.75,.95]:
        p=(1-s)*y+s*other; law=np.prod(np.where(states==1,p,1-p),axis=1)
        grad=[]
        for costs in [direct,mmd]:
            g=[]
            for e in range(4):
                lo=np.arange(16)[states[:,e]==0]; hi=lo+(1<<e)
                weights=np.prod(np.where(np.delete(states[lo],e,axis=1)==1,np.delete(p,e),1-np.delete(p,e)),axis=1)
                g.append(float(weights@(costs[hi]-costs[lo])))
            grad.append(float(np.array(g)@(other-y)))
        toy.append(dict(s=s,original_label_CE=ce(p,y),CE_derivative_wrt_s=1/(1-s),
            expected_DIRECT=float(law@direct),DIRECT_derivative_wrt_s=grad[0],
            expected_MMD=float(law@mmd),MMD_derivative_wrt_s=grad[1],
            probability_valid_star=float(law[3]+law[12]),
            joint_mixture_structure_score=0.,joint_mixture_valid_star_probability=1.))
    assert toy[3]['DIRECT_derivative_wrt_s']<0 and toy[3]['CE_derivative_wrt_s']>0
    assert abs(toy[2]['probability_valid_star']-.125)<1e-14
    write_csv(out/'toy_exact.csv',toy)
    save('toy_definition.json',dict(types=types.tolist(),query=q.tolist(),reference=y.tolist(),
        alternative_same_type_relabel=other.tolist(),states=16,loss_values_DIRECT=direct.tolist(),
        note='Joint mixture and independent product have identical marginals but different graph laws; not a simulation of production reverse chains.'))
    tasks=read(ROOT/'results/output_boundary_replay_20261108T081654/tasks.json')
    graphrows=list(csv.DictReader(bind(EDGE/'graph_summary.csv').open()))
    oldprobes=read(OLD/'probes.json')
    assets={}; permutations=[]
    for ds in ['IMDB','PubMed']:
        task=next(x for x in tasks if x['dataset']==ds)
        cfg=task['row']['config']['datasets'][ds]
        path=ROOT/'research_branches/typed_structure_early_cn_expansion_review/experiments/typed_structure_early_cn_expansion_v1/package'/cfg['file']
        bind(path); assert bindings[str(path)]==cfg['sha256']
        with np.load(path) as z: ty=z['types'].copy()
        fixed=torch.load(bind(OLD/f'{ds}_fixed_inputs.pt'),map_location='cpu',weights_only=False)['inputs']
        inputs={int(t): inp for t,seed,inp in fixed}; assets[ds]=(ty,inputs)
        for kind in ['random','hub_low_swap']:
            perm=np.arange(len(ty)); rng=np.random.default_rng(2026110901)
            inp=inputs[1]; qq,yy,ff=map(np.asarray,(inp['query'],inp['labels'],inp['f']))
            yy=yy.astype(np.float64)
            a=np.zeros((len(ty),len(ty)),dtype=int); a[qq[0],qq[1]]=yy; a+=a.T
            deg=a.sum(1)
            for t in np.unique(ty):
                ix=np.flatnonzero(ty==t)
                if kind=='random':perm[ix]=rng.permutation(ix)
                else:
                    u,v=ix[np.argmax(deg[ix])],ix[np.argmin(deg[ix])];perm[u],perm[v]=v,u
            alt=a[np.ix_(perm,perm)]; bb=alt[qq[0],qq[1]].astype(float)
            assert int(bb.sum())==int(yy.sum()), 'relabel violates query support'
            core=JointLocalScore(ty,qq,ff,yy,int(ff.max())+1)
            with torch.no_grad(): val=float(core(torch.from_numpy(bb)))
            dsval=DirectScore(ty,qq,ff,yy,bb,int(ff.max())+1).base
            assert abs(val)<1e-12 and dsval==0
            eps=1e-4; p=eps+(1-2*eps)*bb; baseline=eps+(1-2*eps)*yy
            for t,inp in inputs.items():
                q0,f0,y0=map(np.asarray,(inp['query'],inp['f'],inp['labels']))
                y0=y0.astype(np.float64)
                alt0=alt[q0[0],q0[1]]
                count=np.bincount(f0)
                # Absent fixed-zero relations are not queried. Never divide their
                # empty support; only used active relations enter likelihoods.
                prior=np.divide(np.bincount(f0,weights=y0),count,out=np.zeros(len(count),dtype=float),where=count>0)
                retention=np.cos((t/1000+.008)/1.008*np.pi/2)**2/np.cos(.008/1.008*np.pi/2)**2 if t<1000 else 0.
                xt=np.zeros_like(a); edges=np.asarray(inp['edges']);xt[edges[0],edges[1]]=1;xt[edges[1],edges[0]]=1
                observed=xt[q0[0],q0[1]]
                p_orig=retention*y0+(1-retention)*prior[f0]
                p_alt=retention*alt0+(1-retention)*prior[f0]
                assert np.all((p_orig>0)&(p_orig<1)) and np.all((p_alt>0)&(p_alt<1))
                logratio=float(np.where(observed==1,np.log(p_alt/p_orig),np.log1p(-p_alt)-np.log1p(-p_orig)).sum())
                permutations.append(dict(dataset=ds,kind=kind,t=t,changed_slots=int(np.sum(bb!=yy)),
                    reference_edges=int(yy.sum()),MMD=val,DIRECT=dsval,
                    illustrative_eps=eps,CE_reference=ce(baseline,yy),CE_permuted_prediction=ce(p,yy),
                    forward_log_likelihood_ratio_alternative_vs_reference=logratio,
                    note='CE here mean over active slots, artificial softened reference, not model output'))
    write_csv(out/'permutation_examples.csv',permutations)
    rows=[]; groups={}; examples=[]
    for row in graphrows:
        check(); condition,arm=row['condition'],row['arm']; t,draw=int(row['t']),int(row['draw'])
        ds=condition.split('_')[0]; ty,inputs=assets[ds]; inp=inputs[t]
        q,f,y=map(np.asarray,(inp['query'],inp['f'],inp['labels']))
        oldp=next(x for x in oldprobes if x['condition']==condition and x['arm']==arm and x['t']==t)
        with np.load(bind(OLD/f'{condition}_{arm}_t{t}.npz')) as z:p=z['probability'].copy()
        with np.load(bind(EDGE/f'{condition}_{arm}_t{t}_draw{draw}_flips.npz')) as z:
            sign=z['flip_sign'].copy(); m=z['score_flip'].copy()
            dsflip=(z['total_degree_W1_flip']/max(1,2*float(y.sum())/len(ty))+z['joint_degree_TV_flip']+z['triangle_abs_error_flip']/max(1,float(row['reference_triangles'])))/3
        with np.load(bind(DIRECT/f'{condition}_{arm}_t{t}_draw{draw}_direct.npz')) as z:
            assert np.max(abs(dsflip-z['score_flip']))<1e-12
        b=(1-sign)/2; reference_agreement=(b==y)
        gce=(p-y)/(len(ty)*(len(ty)-1)/2)
        for score,diff in [('MMD',m),('DIRECT',dsflip)]:
            directional=diff*sign; g=p*(1-p)*directional
            key=(condition,arm,t,score)
            if key not in groups:groups[key]=dict(p=p,y=y,f=f,g=[],gce=gce)
            groups[key]['g'].append(g)
            good=diff < -1e-12
            for rel in [-1]+list(np.unique(f)):
                mask=np.ones(len(f),bool) if rel==-1 else f==rel
                sel=mask&good; away=sel&reference_agreement
                active=mask&(abs(directional)>1e-12)&(p>0)&(p<1)
                opposed=active&(((y==0)&(directional<0))|((y==1)&(directional>0)))
                norm=float(abs(g[active]).sum())
                rows.append(dict(dataset=ds,condition=condition,seed=oldp['seed'],arm=arm,t=t,draw=draw,score=score,relation=int(rel),
                    slots=int(mask.sum()),structure_improving_flips=int(sel.sum()),away_from_reference=int(away.sum()),
                    away_fraction=float(away.sum()/sel.sum()) if sel.any() else None,
                    opposed_nonzero_logit_coordinates=int(opposed.sum()),nonzero_logit_coordinates=int(active.sum()),
                    opposed_gradient_abs_mass=float(abs(g[opposed]).sum()),gradient_abs_mass=norm,
                    conflict_mass_fraction=float(abs(g[opposed]).sum()/norm) if norm else None,
                    positive_opposed_mass=float(abs(g[opposed&(y==1)]).sum()),
                    negative_opposed_mass=float(abs(g[opposed&(y==0)]).sum()),
                    logit_gradient_dot_CE=float(g[mask]@gce[mask])))
            # First qualifying slot only: illustrative, not representative.
            ix=np.flatnonzero(good&reference_agreement)
            if len(ix):
                e=int(ix[0]);examples.append(dict(dataset=ds,condition=condition,arm=arm,t=t,draw=draw,score=score,
                    slot=e,node_u=int(q[0,e]),node_v=int(q[1,e]),relation=int(f[e]),reference=int(y[e]),sampled=int(b[e]),
                    probability=float(p[e]),score_flip=float(diff[e]),CE_logit_gradient=float(gce[e]),structure_logit_estimate=float(g[e])))
    write_csv(out/'per_graph_relation.csv',rows);write_csv(out/'illustrative_conflicts.csv',examples)
    panel=[]
    for key,v in groups.items():
        c,a,t,s=key; g=np.mean(v['g'],axis=0); gce=v['gce']; product=g*gce
        denom=np.linalg.norm(g)*np.linalg.norm(gce)
        panel.append(dict(condition=c,arm=a,t=t,score=s,draws=len(v['g']),
            cosine_with_CE=float(g@gce/denom) if denom else None,
            conflict_mass_fraction=float(abs(g[product<0]).sum()/abs(g).sum()) if abs(g).sum() else None,
            warning='four correlated coordinate-vector estimates; descriptive only, no parameter gradient/no significance'))
    write_csv(out/'four_draw_panel.csv',panel)
    summary=[]
    for ds,t,s in itertools.product(['IMDB','PubMed'],[1,500,1000],['MMD','DIRECT']):
        sub=[r for r in rows if r['dataset']==ds and r['t']==t and r['score']==s and r['relation']==-1]
        better=sum(r['structure_improving_flips'] for r in sub); away=sum(r['away_from_reference'] for r in sub)
        mass=sum(r['gradient_abs_mass'] for r in sub); oppose=sum(r['opposed_gradient_abs_mass'] for r in sub)
        summary.append(dict(dataset=ds,t=t,score=s,graphs=len(sub),improving_flips=better,away_flips=away,
            away_fraction=away/better if better else None,conflict_mass_fraction=oppose/mass if mass else None))
    write_csv(out/'summary.csv',summary)
    for p in [Path(__file__), ROOT/'research_branches/hard_structure_rebar_cpu_20261108/objective.py',
              ROOT/'research_branches/direct_structure_cpu_20261108/score.py',ROOT/'research_branches/hard_structure_edge_direction_cpu_20261108/metrics.py',
              ROOT/'research_branches/direct_medium_run_20261108/run.py',
              ROOT/'research_branches/typed_structure_early_cn_expansion_review/experiments/typed_structure_early_cn_expansion_v1/package/hsgd/bw_preflight/common.py',
              ROOT/'research_branches/typed_structure_early_cn_expansion_review/experiments/typed_structure_early_cn_expansion_v1/package/hsgd/losses.py']:
        bind(p)
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in bindings.items())
    assert not torch.cuda.is_initialized()
    save('asset_source_binding.json',bindings)
    save('receipt.json',dict(status='complete',seconds=time.time()-start,cpu_seconds=time.process_time()-cpu,
        peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,graphs=96,unique_flip_operations=1509888,
        sources_unchanged=True,model_calls=0,GPU_hours=0,training_updates=0,new_model_graphs=0,
        counts_are_not_independent_training_repeats=True))
    print(json.dumps(summary,indent=2)); print((out/'receipt.json').read_text())

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    main(args.output)
