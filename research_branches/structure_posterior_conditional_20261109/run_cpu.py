"""Conditional averaging on archived finite MCMC paths; no new random draws."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import csv,hashlib,json,resource,signal,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
OLD=ROOT/'results/structure_posterior_mcmc_20261109'
UNIV=ROOT/'results/structure_without_edge_ce_theory_20261109/finite_universe.npz'
OUT=ROOT/'results/structure_posterior_conditional_20261109'
START=time.monotonic()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,obj):
    (OUT/(name+'.json')).write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    if isinstance(obj,list) and obj:
        with (OUT/(name+'.csv')).open('w',newline='') as f:
            w=csv.DictWriter(f,list(obj[0]));w.writeheader();w.writerows(obj)
def budget():
    if time.monotonic()-START>540:raise TimeoutError('science time limit')
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>2048*1024:raise MemoryError('RSS limit')
def cond(codes,energy,families,obs,alpha,bits):
    masks=1<<np.arange(10);zero=codes[:,None]&~masks;one=zero|masks
    pi=np.array([.25,.4,.35])[families]
    p0=(1-alpha)*pi;p1=alpha+(1-alpha)*pi
    lratio=np.log(np.where(bits[obs],p1,1-p1))-np.log(np.where(bits[obs],p0,1-p0))
    logit=energy[zero]-energy[one]+lratio
    return 1/(1+np.exp(-logit)),zero,one
def main():
    OUT.mkdir(parents=True,exist_ok=False);signal.alarm(600)
    resource.setrlimit(resource.RLIMIT_AS,(2*1024**3,2*1024**3))
    files=[UNIV,OLD/'protocol.json',OLD/'completion.json',OLD/'trajectories.npz',OLD/'exact_posteriors.npz',OLD/'chains.json',OLD/'structure_expectations.json']
    hashes={str(p):sha(p) for p in files}
    save('protocol',dict(start_utc=datetime.now(timezone.utc).isoformat(),
        scope='Read all archived 108 toy chains; no new trajectories, RNG draws, networks or production graphs',
        estimator='h_e(g_-e,x)=rho(g_e=1,g_-e|x)/(rho(g_e=0,g_-e|x)+rho(g_e=1,g_-e|x)); average over saved states',
        unchanged='same target, E, tau, likelihood, 500 burn-in, all 12 conditions and starts/repeats',
        prefixes=[100,500,1000,3500],primary_prefix=3500,
        structure='coordinate-averaged E[F(G)|G_-e,x]; never multiply edge marginals to estimate triangles',
        comparisons='paired raw occupancy vs conditional average; all chains/edges/families/prefixes retained',
        uncertainty='finite empirical errors and repeat spread, no significance or universal MCMC variance reduction claim',
        cost='cached and on-demand conditional arithmetic measured separately; both use cached energy, not production scoring',
        limits=dict(threads=1,seconds=600,science_seconds=540,memory_MiB=2048,GPU=0)))
    save('binding',dict(source=str(Path(__file__)),source_sha256=sha(Path(__file__)),assets=hashes))
    completion=json.loads((OLD/'completion.json').read_text())
    for f,h in completion['output_sha256'].items():assert sha(ROOT/f)==h
    data=np.load(UNIV);bits=data['bits'];energy=data['energy'];fam=data['families']
    paths=np.load(OLD/'trajectories.npz');targets=np.load(OLD/'exact_posteriors.npz')
    oldrows=json.loads((OLD/'chains.json').read_text())
    # Reconstruct observables independently of the previous chain runner.
    query=data['query'].T;lookup={tuple(e):i for i,e in enumerate(query)}
    import itertools
    triangle=np.zeros(1024)
    for i,j,k in itertools.combinations(range(5),3):
        triangle+=bits[:,[lookup[(i,j)],lookup[(i,k)],lookup[(j,k)]]].prod(1)
    component=[]
    for b in bits:
        groups=[{i} for i in range(5)]
        for present,(u,v) in zip(b,query):
            if not present:continue
            a=next(g for g in groups if u in g);c=next(g for g in groups if v in g)
            if a is not c:a.update(c);groups.remove(c)
        component.append(len(groups))
    names=['edges_rel0','edges_rel1','edges_rel2','triangle_count','component_count','energy',
           'number_error','joint_allocation_error','closure_error','connection_error']
    obsvalues=np.column_stack([*(bits[:,fam==r].sum(1) for r in range(3)),triangle,np.array(component),energy,*data['energy_parts'].T])
    checks=[];rows=[];edge_rows=[];struct=[];timings=[];tables={};max_old_diff=0.
    starts=[0,1023,659]
    for ci in range(12):
        budget();cr=[r for r in oldrows if r['condition']==ci];alpha=cr[0]['retention'];obs=cr[0]['observation'];post=targets[str(ci)]
        tick=time.perf_counter();h,zero,one=cond(np.arange(1024),energy,fam,obs,alpha,bits)
        setup=time.perf_counter()-tick;tables[str(ci)]=h
        explicit=post[one]/(post[zero]+post[one])
        conditional_error=float(abs(h-explicit).max());identity_error=float(abs(post@h-post@bits).max())
        assert conditional_error<1e-13 and identity_error<1e-13
        # Single independent-draw total variance identity, not an MCMC asymptotic assertion.
        mean=post@bits;v_raw=mean*(1-mean);v_cond=post@((h-mean)**2);residual=post@(h*(1-h))
        variance_error=float(abs(v_raw-v_cond-residual).max());assert variance_error<1e-13
        assert np.max(abs(h-h[zero,np.arange(10)]))<1e-14
        # Each coordinate's conditional expectation of a graph observable, averaged over coordinates.
        f_cond=((1-h[:,:,None])*obsvalues[zero]+h[:,:,None]*obsvalues[one]).mean(1)
        structure_identity_error=float(abs(post@f_cond-post@obsvalues).max());assert structure_identity_error<1e-12
        checks.append(dict(condition=ci,retention=alpha,observation=obs,conditional_formula_error=conditional_error,
            expectation_identity_error=identity_error,variance_identity_error=variance_error,
            structure_identity_error=structure_identity_error,IID_single_draw_variance_ratio_mean=float(np.mean(v_cond/v_raw))))
        cache_sec=direct_sec=raw_sec=0.;eval_count=0
        for r in cr:
            path=paths[f"c{ci}_s{starts.index(r['start'])}_r{r['repeat']}"][500:]
            tick=time.perf_counter();raw=bits[path];raw_sec+=time.perf_counter()-tick
            tick=time.perf_counter();soft=h[path];cache_sec+=time.perf_counter()-tick
            tick=time.perf_counter();direct,_,_=cond(path.astype(int),energy,fam,obs,alpha,bits);direct_sec+=time.perf_counter()-tick
            assert abs(soft-direct).max()<1e-14;eval_count+=len(path)*10
            key=dict(condition=ci,retention=alpha,observation=obs,start=r['start'],repeat=r['repeat'],seed=r['seed'])
            for length in [100,500,1000,3500]:
                a=raw[:length].mean(0);b=soft[:length].mean(0);truth=post@bits
                olderr=abs(a-truth);newerr=abs(b-truth)
                rows.append(dict(**key,length=length,raw_MAE=float(olderr.mean()),conditional_MAE=float(newerr.mean()),
                    raw_MSE=float(np.mean((a-truth)**2)),conditional_MSE=float(np.mean((b-truth)**2)),
                    raw_max_error=float(olderr.max()),conditional_max_error=float(newerr.max()),
                    edges_better=int(np.sum(newerr<olderr-1e-12)),edges_worse=int(np.sum(newerr>olderr+1e-12))))
                if length==3500:
                    max_old_diff=max(max_old_diff,abs(float(olderr.mean())-r['edge_MAE']))
                    for e in range(10):
                        edge_rows.append(dict(**key,edge=e,family=int(fam[e]),exact=float(truth[e]),raw=float(a[e]),conditional=float(b[e]),
                            raw_error=float(a[e]-truth[e]),conditional_error=float(b[e]-truth[e])))
                    for k,name in enumerate(names):
                        true=float(post@obsvalues[:,k]);ar=float(obsvalues[path,k].mean());br=float(f_cond[path,k].mean())
                        struct.append(dict(**key,observable=name,exact=true,raw=ar,conditional=br,raw_error=ar-true,conditional_error=br-true))
        timings.append(dict(condition=ci,table_seconds=setup,raw_gather_seconds=raw_sec,cached_conditional_gather_seconds=cache_sec,
            on_demand_conditional_seconds=direct_sec,conditional_queries=eval_count,
            table_state_edge_pairs=10240,energy_lookups_if_uncached_probability=2*eval_count,
            real_energy_evaluations=0,limitation='energy lookup only; not real graph scoring time'))
    assert max_old_diff<1e-12
    save('checks',checks);save('chains_by_length',rows);save('edges',edge_rows);save('structure',struct);save('cost',timings)
    summary=[]
    for a in [0.,.5,.95]:
        for length in [100,500,1000,3500]:
            rs=[r for r in rows if r['retention']==a and r['length']==length]
            summary.append(dict(retention=a,length=length,chains=len(rs),
                raw_MAE=float(np.mean([r['raw_MAE'] for r in rs])),conditional_MAE=float(np.mean([r['conditional_MAE'] for r in rs])),
                raw_RMSE=float(np.sqrt(np.mean([r['raw_MSE'] for r in rs]))),conditional_RMSE=float(np.sqrt(np.mean([r['conditional_MSE'] for r in rs]))),
                chains_improved=sum(r['conditional_MAE']<r['raw_MAE']-1e-12 for r in rs),
                chains_worse=sum(r['conditional_MAE']>r['raw_MAE']+1e-12 for r in rs),
                worst_edge_error=max(r['conditional_max_error'] for r in rs)))
    save('summary',summary)
    spread=[]
    for ci in range(12):
        for start in starts:
            for e in range(10):
                rs=[r for r in edge_rows if r['condition']==ci and r['start']==start and r['edge']==e]
                spread.append(dict(condition=ci,start=start,edge=e,repeats=3,
                    raw_sample_SD=float(np.std([r['raw'] for r in rs],ddof=1)),
                    conditional_sample_SD=float(np.std([r['conditional'] for r in rs],ddof=1))))
    save('repeat_spread',spread);np.savez_compressed(OUT/'conditional_tables.npz',**tables)
    assert all(sha(Path(f))==h for f,h in hashes.items())
    save('receipt',dict(status='complete_stopped',seconds=time.monotonic()-START,RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        old_occupancy_MAE_max_difference=max_old_diff,unchanged_assets=True,chains_reused=108,states_reused=378000,
        new_random_draws=0,model_calls=0,training_updates=0,GPU_hours=0,new_production_graphs=0,formal_training_authorized=False))
if __name__=='__main__':
    try:main()
    except BaseException as e:
        if OUT.exists():save('failure',dict(error=repr(e),seconds=time.monotonic()-START))
        raise
