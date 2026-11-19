"""Frozen-background paired estimator coverage study; CPU only, no model calls."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,json,csv,time,hashlib,signal,resource,traceback,importlib.util,itertools
from pathlib import Path
import numpy as np
import torch
torch.set_num_threads(1)
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('pair_cpu_helpers',ROOT/'research_branches/pair_allocation_signal_20261109/run_cpu.py')
H=importlib.util.module_from_spec(spec);spec.loader.exec_module(H)
OUT=ROOT/'results/pair_allocation_coverage_20261109';PREV=ROOT/'results/pair_allocation_signal_20261109'
START=time.monotonic();BIND={};KS=[16,64,256];REPS=3;EPS=1e-12
def bind(p):
    p=Path(p);BIND[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest();return p
def read(p):return json.loads(bind(p).read_text())
def save(n,x):
    (OUT/(n+'.json')).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    if isinstance(x,list) and x and isinstance(x[0],dict):
        with (OUT/(n+'.csv')).open('w',newline='') as h:
            w=csv.DictWriter(h,list(x[0]));w.writeheader();w.writerows(x)
def limits():
    if time.monotonic()-START>=540:raise TimeoutError('540s scientific cap')
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024>=2048:raise MemoryError('2GiB RSS cap')
def stats(rows):
    good=[r for r in rows if r['score_change']<-EPS];mass=sum(abs(r['h']) for r in good)
    out=dict(pairs=len(rows),rewarded=len(good),rewarded_feedback_mass=mass)
    for name in ['triangle_error','typed_triangle_L1','joint_distance','hub','low','isolates_error','LCC_error']:
        cost=[r for r in good if r[name+'_change']>EPS]
        out[name+'_cost']=len(cost)
        out[name+'_cost_share']=len(cost)/len(good) if good else None
        out[name+'_cost_feedback_share']=sum(abs(r['h']) for r in cost)/mass if mass else None
    return out
def check_variance():
    x=np.array([[1.,-1.,0.],[0.,2.,-2.],[-1.,0.,1.],[1.,0.,-1.]])
    mu=x.mean(0);actual=[];estimated=[]
    for ids in itertools.combinations(range(4),2):
        y=x[list(ids)];avg=y.mean(0);actual.append(float(np.square(avg-mu).sum()))
        s2=float(np.square(y-avg).sum())
        estimated.append((1-2/4)*s2/2)
    assert abs(np.mean(actual)-np.mean(estimated))<1e-12
    return dict(subsets=6,actual_MSE=float(np.mean(actual)),mean_variance_estimator=float(np.mean(estimated)))

def main():
    OUT.mkdir(parents=True,exist_ok=False);signal.alarm(600)
    save('protocol',dict(track='CORE paired-feedback sampling feasibility',states='same4 PubMed0422 t1000 archived conditional hard graphs, CE/PS parents; not4 trained models',
        counts_per_relation=KS,replicates=REPS,population='all live x absent slots within one relation of frozen hard graph',
        sampling='three independent SHA seeded uniform without-replacement random ordered samples per state/relation; 16/64/256 nested prefixes. Census if population smaller.',
        weights='10 eligible relations equal, m=min(requested,N) pairs equal; no effect selection or coefficient tuning',
        separation='within-background independent pair samples vs between-background replicate means; background comparison still includes pair uncertainty',
        variance='unbiased finite-population estimated logit MSE: sum_r (1-m_r/N_r) trace(sample covariance_r)/m_r / R^2; not parameter variance',
        limits=dict(threads=1,RSS_MiB=2048,science_seconds=540,hard_seconds=600,GPU=0),
        decision='report convergence and costs, no training/gradient/model call; no auto choice of best K or pair sample'))
    save('finite_variance_test',check_variance())
    binding=read(PREV/'binding.json');completion=read(H.FULL/'completion.json')
    def checked(p,bb=binding):
        p=bind(p);assert bb[str(p)]==BIND[str(p)],str(p);return p
    with np.load(checked(ROOT/'results/rooted_connection_20261109T015142/PubMed_cpu_cost_panel.npz')) as z:types=z['types'].copy();fq=z['query'].copy();ff=z['families'].copy();fy=z['reference'].copy()
    inp=next(v for t,seed,v in torch.load(checked(H.OLD/'PubMed_fixed_inputs.pt'),map_location='cpu',weights_only=False)['inputs'] if t==1000)
    q,f,y=map(np.asarray,(inp['query'],inp['f'],inp['labels']));n=len(types)
    i=np.argsort(q[0]*n+q[1]);j=np.argsort(fq[0]*n+fq[1]);assert np.array_equal(q[:,i],fq[:,j]) and np.array_equal(f[i],ff[j]) and np.array_equal(y[i],fy[j])
    plan=[];states=[];requested=0
    for parent in ['CE','PS']:
        for draw in [0,1]:
            tag=f'PubMed_{parent}_CE_t1000_draw{draw}'
            with np.load(checked(H.FULL/(tag+'.npz'),completion['output_hashes'])) as z:data={k:z[k].copy() for k in ['probability','bits','reference','relation']}
            assert np.array_equal(data['reference'],y) and np.array_equal(data['relation'],f)
            assert np.all((data['probability']>0)&(data['probability']<1))
            b=data['bits'];parts=[]
            for rep in range(REPS):
                for r in range(10):
                    live=np.flatnonzero((f==r)&(b==1));absent=np.flatnonzero((f==r)&(b==0));N=len(live)*len(absent)
                    assert N>0 # all ten eligible verified before freeze
                    seed=int.from_bytes(hashlib.sha256(f'pair_coverage_v1|{parent}|{draw}|{rep}|{r}'.encode()).digest()[:8],'little')
                    rng=np.random.default_rng(seed);m=min(256,N);ix=rng.choice(N,m,replace=False);ix=rng.permutation(ix)
                    pairs=[[int(live[x//len(absent)]),int(absent[x%len(absent)])] for x in ix]
                    parts.append(dict(rep=rep,relation=r,population=N,seed=seed,pairs=pairs));requested+=m
            plan.append(dict(state=tag,parts=parts));states.append((tag,data,parts))
    save('frozen_pairs',plan);save('frozen_count',dict(requested_pair_occurrences=requested,maximum_without_caps=30720))
    core=H.RootedScore(types,q,f,y,10);sc=H.ExplicitScore(core);jd=H.jdmod.JointDegree(dict(types=types,query=q,families=f,reference=y))
    channel=np.full((4,4),-1,int)
    for k,(a,b) in enumerate(core.triangle_pairs):channel[a,b]=channel[b,a]=10+k
    refconn=H.connectivity(H.adjacency(q,y,n));oldrows=read(PREV/'pairs.json')
    oldcache={(r['state'],r['remove'],r['add']):r for r in oldrows}
    for mod in list(sys.modules.values()):
        path=getattr(mod,'__file__',None)
        if path and str(path).startswith(str(ROOT)) and str(path).endswith('.py'):bind(path)
    bind(Path(__file__));save('initial_binding',BIND)
    summaries=[];relations=[];allrows=[];tests=[];vectors={};computed=0;reused=0;previous_matches=0;durations=[];state_done=0
    for tag,data,parts in states:
        limits();tick=time.monotonic();p=data['probability'];bits=data['bits'];logits=np.log(p)-np.log1p(-p)
        with torch.no_grad():raw=core.features(torch.from_numpy(bits)).numpy()[:,:core.base_width]
        adj=H.adjacency(q,bits,n);base=H.describe(sc,jd,raw,adj);cache={};local={}
        for part in parts:
            rep,r,N=part['rep'],part['relation'],part['population'];rr=[]
            for pos,(e,add) in enumerate(part['pairs']):
                limits();key=(e,add)
                if key not in cache:
                    ar=raw.copy();aa=adj.copy();H.flip(ar,aa,q,types,channel,e,r);H.flip(ar,aa,q,types,channel,add,r)
                    assert np.array_equal(ar[:,:10].sum(0),raw[:,:10].sum(0))
                    after=H.describe(sc,jd,ar,aa);d=after['score']-base['score'];cond=float(H.sigmoid(logits[add]-logits[e]));h=d*cond*(1-cond)
                    row=dict(remove=e,add=add,relation=r,label_remove=int(y[e]),label_add=int(y[add]),score_change=float(d),h=float(h),q=cond,
                        **{k+'_change':float(after[k]-base[k]) for k in ['triangle_error','typed_triangle_L1','joint_distance','hub','low']},
                        **{k+'_error_change':float(abs(after[k]-refconn[k])-abs(base[k]-refconn[k])) for k in ['isolates','LCC']})
                    cache[key]=row;computed+=1
                    if (tag,e,add) in oldcache:
                        o=oldcache[(tag,e,add)];assert abs(d-o['score_change'])<1e-10 and abs(h-o['pair_g_add'])<1e-12;previous_matches+=1
                    if pos==0 and rep==0:
                        bb=bits.copy();bb[e]=0;bb[add]=1
                        with torch.no_grad():full=core.features(torch.from_numpy(bb)).numpy()[:,:core.base_width]
                        err=float(abs(full-ar).max());assert err==0;tests.append(dict(state=tag,relation=r,max_feature_error=err))
                else:reused+=1
                rr.append(cache[key]);allrows.append(dict(state=tag,rep=rep,position=pos,**cache[key]))
            local[(rep,r)]=rr
            if r in [4,9]:
                save('progress',dict(state=tag,rep=rep,relation=r,computed=computed,reused=reused,elapsed_seconds=time.monotonic()-START))
            if tag==states[0][0] and rep==0 and r==9:
                seconds=time.monotonic()-tick;estimate=seconds/2560*requested
                save('calibration',dict(first_rep_seconds=seconds,remaining_linear_estimate=estimate-seconds,safety_factor=1.5,remaining_budget=540-(time.monotonic()-START)))
                if (estimate-seconds)*1.5>540-(time.monotonic()-START):raise TimeoutError('Full frozen panel does not fit; no scope reduction')
        for rep in range(REPS):
            for k in KS:
                g=np.zeros(len(p));var=0.;included=[];counts=[]
                for r in range(10):
                    selected=local[(rep,r)][:k];m=len(selected);N=next(x['population'] for x in parts if x['rep']==rep and x['relation']==r)
                    sg=np.zeros(len(p));ss=0.
                    for row in selected:
                        h=row['h'];sg[row['remove']]-=h;sg[row['add']]+=h;ss+=2*h*h
                    avg=sg/m;s2=(ss-m*float(avg@avg))/(m-1) if m>1 else 0.
                    assert s2>=-1e-18
                    vv=(1-m/N)*max(0.,s2)/m;var+=vv/100;g+=avg/10
                    relations.append(dict(state=tag,rep=rep,requested_per_relation=k,relation=r,population=N,actual_pairs=m,census=m==N,
                        gradient_norm=float(np.linalg.norm(avg)),estimated_logit_variance=vv,**stats(selected)))
                    included.extend(selected);counts.append(m)
                norm2=float(g@g);vectors[(tag,rep,k)]=g
                summaries.append(dict(state=tag,rep=rep,requested_per_relation=k,actual_pairs=sum(counts),gradient_norm=norm2**.5,
                    estimated_logit_MSE=var,estimated_mean_squared_norm=norm2-var,
                    estimated_noise_RMS_over_observed_norm=(var/norm2)**.5 if norm2 else None,
                    max_relation_gradient_sum=max(abs(float(g[f==r].sum())) for r in range(10)),**stats(included)))
        np.savez_compressed(OUT/(tag+'_vectors.npz'),**{f'rep{rep}_k{k}':vectors[(tag,rep,k)] for rep in range(REPS) for k in KS})
        save('summary',summaries);save('relations',relations);save('pairs',allrows);save('tests',tests)
        state_done+=1;durations.append(dict(state=tag,seconds=time.monotonic()-tick,unique_pairs=len(cache)))
        print(json.dumps(durations[-1]),flush=True)
    within=[];between=[];means={};mean_variances={}
    for tag,_,_ in states:
        for k in KS:
            gs=np.array([vectors[(tag,rep,k)] for rep in range(REPS)]);mean=gs.mean(0);means[(tag,k)]=mean
            variance=sum(r['estimated_logit_MSE'] for r in summaries if r['state']==tag and r['requested_per_relation']==k)/REPS**2
            mean_variances[(tag,k)]=variance;cs=[H.cos(gs[a],gs[b]) for a,b in itertools.combinations(range(REPS),2)]
            within.append(dict(state=tag,requested_per_relation=k,replicates=REPS,cosine_mean=float(np.mean(cs)),cosine_min=min(cs),cosine_max=max(cs),
                mean_gradient_norm=float(np.linalg.norm(mean)),empirical_variance_trace=float(np.square(gs-mean).sum()/(REPS-1)),
                mean_estimated_noise_MSE=variance,mean_squared_norm_debiased=float(mean@mean)-variance))
    for parent in ['CE','PS']:
        for k in KS:
            t0=f'PubMed_{parent}_CE_t1000_draw0';t1=f'PubMed_{parent}_CE_t1000_draw1';a=means[(t0,k)];b=means[(t1,k)]
            observed=float(np.square(a-b).sum());noise=mean_variances[(t0,k)]+mean_variances[(t1,k)]
            between.append(dict(parent=parent,requested_per_relation=k,mean_cosine=H.cos(a,b),observed_squared_difference=observed,
                estimated_pair_noise_contribution=noise,debiased_background_difference_squared=observed-noise,
                note='two archived hard backgrounds only; negative debiased estimates retained; no parameter or training variance claim'))
    save('within_background',within);save('between_background',between);save('durations',durations)
    assert len(allrows)==requested and computed+reused==requested and len(summaries)==36 and len(within)==12
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in BIND.items());assert not torch.cuda.is_initialized();limits()
    save('binding',BIND);save('receipt',dict(status='complete',seconds=time.monotonic()-START,RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        pair_occurrences=requested,new_score_calculations=computed,cache_reuses=reused,prior_matches=previous_matches,states=4,replicates=3,
        GPU_hours=0,models=0,updates=0,new_random_graphs=0,assets_unchanged=True))
    print((OUT/'receipt.json').read_text())
if __name__=='__main__':
    try:main()
    except BaseException as e:
        if OUT.exists():save('failure',dict(error=repr(e),traceback=traceback.format_exc(),elapsed=time.monotonic()-START))
        raise
