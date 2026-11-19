"""CPU exact small-graph proof checks and 24 cached real192 probability panels."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import sys,time,json,csv,hashlib,itertools,signal,traceback,resource
from pathlib import Path
import numpy as np
from scipy.special import expit,logit
import torch
from core import risks,hard_losses,probability_metrics
ROOT=Path(__file__).resolve().parents[2]
LEGACY=ROOT/'research_branches/sparsediff_hetero_restart/sparse_restart/role_degree_design_v1'
sys.path.insert(0,str(LEGACY))
from role_statistics import joint_profile
from role_guidance import Layout,freeze_target,RoleGuide
OUT=ROOT/'results/analytic_joint_risk_20261113'
OLD=ROOT/'results/increment_update_check_20261113'
PLAN=ROOT/'results/fullchain_reliability_plan_20261113/protocol.json'
START=time.time();bindings={}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):bindings[str(Path(p).resolve())]=sha(p);return Path(p)
def write(name,x):(OUT/name).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False))
def table(name,rows):
    write(name+'.json',rows)
    if rows:
        with (OUT/(name+'.csv')).open('w') as f:
            w=csv.DictWriter(f,list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
def check():
    if time.time()-START>1100:raise TimeoutError('bounded 20-minute CPU ticket; delivery reserve')
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20>4:raise MemoryError('4GiB RSS bound')
def cs(a,b):
    d=np.linalg.norm(a)*np.linalg.norm(b);return float(a@b/d) if d>1e-25 else None
def fixture(types,q,f,pairs,ref):
    capacity=np.bincount(f,minlength=len(pairs));prior=np.bincount(f,weights=ref,minlength=len(pairs))/capacity
    layout=Layout(types,q,f,pairs,prior)
    profile=joint_profile(types,q[:,ref.astype(bool)],f[ref.astype(bool)],pairs)
    frozen=freeze_target(profile,layout);return layout,frozen
def checks():
    types=np.array([0,0,1,1]);q=np.array(list(itertools.combinations(range(4),2))).T
    pairs=np.array([[0,0],[0,1],[1,1]]);lookup={tuple(x):i for i,x in enumerate(pairs)}
    f=np.array([lookup[tuple(types[e])] for e in q.T]);ref=np.array([1,1,0,0,1,1],float)
    layout,frozen=fixture(types,q,f,pairs,ref)
    bits=np.array(list(itertools.product([0.,1.],repeat=6)))
    hard=hard_losses(bits,types,q,f,frozen)
    rows=[]
    for name,p in [('interior',np.array([.07,.2,.42,.65,.83,.96])),('boundary',np.array([0.,.2,.42,.65,.83,1.]))]:
        prob=np.prod(np.where(bits,p,1-p),axis=1);expected=float(prob@hard)
        tensor=torch.tensor(p,dtype=torch.float64,requires_grad=True);vals,parts=risks(tensor,types,q,f,frozen)
        gradient=torch.autograd.grad(vals['risk'],tensor)[0].numpy()*p*(1-p)
        exact=((prob*hard)[:,None]*(bits-p)).sum(0)
        np.testing.assert_allclose(float(vals['risk'].detach()),expected,atol=1e-13,rtol=1e-11)
        np.testing.assert_allclose(gradient,exact,atol=1e-13,rtol=1e-10)
        rows.append(dict(case=name,risk=expected,bias=float(vals['bias'].detach()),variance=float(vals['variance'].detach()),
                         covariance=float(vals['covariance'].detach()),max_gradient_error=float(abs(gradient-exact).max())))
    assert abs(rows[0]['covariance'])>1e-8
    # Deterministic reference and a within-type relabeling have zero risk.
    perm=np.array([1,0,3,2]);refset={tuple(sorted(perm[e])) for e in q[:,ref.astype(bool)].T}
    permuted=np.array([tuple(e) in refset for e in q.T],float)
    for b in [ref,permuted]:
        v,_=risks(torch.tensor(b),types,q,f,frozen);assert abs(float(v['risk']))<1e-13
    # Existing method's mean-distribution formula agrees, in its valid support.
    valid=np.array([1.,.2,.42,.65,.83,1.]);guide=RoleGuide(layout,frozen)
    old,og,_=guide.statistic(valid,True);t=torch.tensor(valid,requires_grad=True);vv,_=risks(t,types,q,f,frozen)
    gg=torch.autograd.grad(vv['bias'],t)[0].numpy()*valid*(1-valid)
    np.testing.assert_allclose(float(vv['bias'].detach()),old,atol=1e-13);np.testing.assert_allclose(gg,og,atol=1e-13)
    return dict(exhaustive_graphs=64,probability_cases=rows,reference_and_type_permutation_zero=True,old_bias_loss_gradient_reproduced=True)
def main():
    OUT.mkdir(exist_ok=False);signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('1200s CPU stop')));signal.alarm(1200)
    torch.set_num_threads(1)
    for p in [Path(__file__),Path(__file__).with_name('core.py'),LEGACY/'role_statistics.py',LEGACY/'role_guidance.py']:bind(p)
    parents=json.loads(bind(PLAN).read_text())['models'];times=[1,75,250,550,800,1000]
    write('protocol.json',dict(scope='CPU exact expected finite-frequency joint-degree loss; four cached A/EMA3000 parents x6 fixed forward-noised inputs',
        counts=dict(conditions=24,conditional_independent_draws_per_condition=256,full_reverse_chains=0,model_calls=0,model_updates=0,GPU_hours=0),
        candidate='E[mean_type mean_frequency |empirical joint-degree CF(G)-reference CF|^2], exact shared-edge covariance included',
        controls=['historical squared error of expected empirical CF','CF of expected degree plug-in'],
        legacy=str(LEGACY),frequency='unchanged legacy K32, seed2026101603, bound.5, scale1+reference RMS; no frequency/weight search',
        validation='small exhaustive risk/logit gradient equality; cached192 conditional Monte Carlo only as independent numerical check',
        probability_space_probe='single fixed step: logit_new=logit_old-.05*g/max(abs(g)); three objectives, no backtracking/selection; boundary p0/1 unchanged; not a model update',
        random_root=202611133101,CPU_threads=1,CPU_wall_limit_seconds=1200,RSS_GiB_limit=4,start_epoch=START,
        limitations=['new objective; not an analytic version of heat/Hungarian score','finite frequencies do not identify every joint law or topology',
        'exact conditional product law only, not terminal diffusion law','expected hard loss can reward reduced structural randomness, not proven diversity preservation',
        'cached inputs already viewed; no independent final validation; free-logit direction is not shared-parameter/AdamW update']))
    test=checks();write('checks.json',test)
    panel=[];directions=[];relations=[];sources=[];saved={};cache={}
    for model in parents:
        check();ds=model['dataset'];seed=model['train_seed'];z=np.load(bind(OLD/f'{model["id"]}_baseline_probabilities.npz'))
        history=json.loads(bind(OLD/f'{model["id"]}.json').read_text())
        data=np.load(bind(model['data']));types=data['types'];pairs=data['schema'];ee=data['edges'];ee=ee if ee.shape[0]==2 else ee.T
        if ds not in cache:
            allq=np.array(list(itertools.combinations(range(len(types)),2))).T
            lookup={tuple(r):i for i,r in enumerate(pairs)};fs=np.array([lookup[tuple(sorted(types[e]))] for e in allq.T])
            refset={tuple(sorted(e)) for e in ee.T};ref=np.array([tuple(e) in refset for e in allq.T],float)
            layout,frozen=fixture(types,allq,fs,pairs,ref);write(f'{ds}_frozen_target.json',frozen)
            cache[ds]=(allq,fs,ref,layout,frozen)
        q,f,ref,layout,frozen=cache[ds];keymap={tuple(e):j for j,e in enumerate(q.T)}
        np.testing.assert_array_equal(z['types'],types)
        for t in times:
            check();tick=time.time();meta=dict(dataset=ds,train_seed=seed,t=t)
            original=z[f'p_{t}'];queries=z[f'query_{t}'];ix=np.array([keymap[tuple(sorted(e))] for e in queries.T])
            assert len(set(ix))==len(ix);np.testing.assert_array_equal(f[ix],z[f'family_{t}']);np.testing.assert_array_equal(ref[ix],z[f'label_{t}'])
            p=layout.prior[f].copy();p[ix]=original;missing=np.ones(len(p),bool);missing[ix]=False
            assert np.all((layout.prior[f[missing]]==0)|(layout.prior[f[missing]]==1));layout.probability(p)
            src=next(x for x in history['baseline'] if x['t']==t)
            sources.append(dict(**meta,probability_file=str(OLD/f'{model["id"]}_baseline_probabilities.npz'),input_hash=src['input_hash'],checkpoint=model['readout'],weight_hash=model['model_hash']))
            tp=torch.tensor(p,dtype=torch.float64,requires_grad=True);val,part=risks(tp,types,q,f,frozen)
            gs={}
            for name in ('risk','bias','plugin'):
                gs[name]=torch.autograd.grad(val[name],tp,retain_graph=True)[0].detach().numpy()*p*(1-p)
            scalar={k:float(v.detach()) for k,v in val.items()};assert scalar['variance']>=-1e-12
            old,og,diag=RoleGuide(layout,frozen).statistic(p,True)
            np.testing.assert_allclose(scalar['bias'],old,atol=2e-13);np.testing.assert_allclose(gs['bias'],og,atol=2e-12,rtol=1e-8)
            for a in part:
                assert float(a['variance'].detach())>=-1e-12
            rng=np.random.default_rng(np.random.SeedSequence([202611133101,seed,t]))
            bits=(rng.random((256,len(p)))<p).astype(np.uint8);mc=hard_losses(bits,types,q,f,frozen)
            se=float(mc.std(ddof=1)/16)
            bm=probability_metrics(p,types,q,f,ref,len(pairs))
            row=dict(**meta,**scalar,variance_share=scalar['variance']/scalar['risk'] if scalar['risk'] else None,
                MC_mean=float(mc.mean()),MC_SE=se,MC_error_in_SE=(float(mc.mean())-scalar['risk'])/se if se else None,
                risk_bias_gradient_cosine=cs(gs['risk'],gs['bias']),risk_plugin_gradient_cosine=cs(gs['risk'],gs['plugin']),
                covariance_share=scalar['covariance']/scalar['variance'] if scalar['variance'] else None,
                gradient_norm=float(np.linalg.norm(gs['risk'])),expected_edges=bm['expected_edges'],expected_triangles=bm['expected_triangles'])
            row['CPU_seconds']=time.time()-tick;panel.append(row)
            for arm,g in gs.items():
                changed=p.copy();movable=(p>0)&(p<1);scale=float(abs(g).max())
                if scale:changed[movable]=expit(logit(p[movable])-.05*g[movable]/scale)
                with torch.no_grad():after,_=risks(torch.tensor(changed),types,q,f,frozen)
                am=probability_metrics(changed,types,q,f,ref,len(pairs))
                entry=dict(**meta,arm=arm,logit_step_max=.05 if scale else 0.,gradient_norm=float(np.linalg.norm(g)),
                    p_change_L1=float(abs(changed-p).sum()),risk_before=scalar['risk'],risk_after=float(after['risk']),
                    risk_relative_change=float(after['risk'])/scalar['risk']-1 if scalar['risk'] else None)
                for metric in ('expected_edges','expected_triangles','low_fraction_error','hub_fraction_error','joint_expected_degree_proxy'):
                    entry[metric+'_before']=bm[metric];entry[metric+'_after']=am[metric]
                directions.append(entry)
                for r in range(len(pairs)):
                    mask=f==r;pos=mask&(ref==1);neg=mask&(ref==0)
                    relations.append(dict(**meta,arm=arm,relation=r,positive_slots=int(pos.sum()),negative_slots=int(neg.sum()),
                        probability_gradient_sum=float(g[mask].sum()),missed_mass_before=float((1-p[pos]).sum()),missed_mass_after=float((1-changed[pos]).sum()),
                        false_mass_before=float(p[neg].sum()),false_mass_after=float(changed[neg].sum()),
                        expected_edges_before=float(p[mask].sum()),expected_edges_after=float(changed[mask].sum()),reference_edges=int(pos.sum())))
                write(f'{ds}_{seed}_t{t}_{arm}_structure.json',dict(before=bm,after=am))
            saved[f'{ds}_{seed}_{t}_risk']=gs['risk'];saved[f'{ds}_{seed}_{t}_bias']=gs['bias'];saved[f'{ds}_{seed}_{t}_plugin']=gs['plugin']
            table('panel',panel);table('local_probes',directions);table('relations',relations)
            del tp,val,part,bits
    np.savez_compressed(OUT/'logit_gradients.npz',**saved)
    write('sources.json',sources);write('bindings.json',bindings)
    assert all(sha(path)==h for path,h in bindings.items());assert not torch.cuda.is_initialized()
    write('receipt.json',dict(status='complete',seconds=time.time()-START,peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        CPU_threads=1,GPU_hours=0,model_calls=0,model_updates=0,new_reverse_chains=0,conditional_CPU_draws=24*256,conditions=len(panel),
        local_probability_space_probes=len(directions),original_assets_unchanged=True))
if __name__=='__main__':
    try:main()
    except BaseException:
        if OUT.exists():write('failure.json',dict(epoch=time.time(),traceback=traceback.format_exc()))
        raise
