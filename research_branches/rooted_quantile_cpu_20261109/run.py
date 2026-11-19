"""One frozen CPU prototype, no search and no graph-generating/model calls."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,time,json,csv,hashlib,resource,itertools,traceback,datetime
from pathlib import Path
import numpy as np
import torch
torch.set_num_threads(1)
from quantile import QuantileScore,directions
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'research_branches/rooted_connection_20261109'))
from score import RootedScore,RootedIncremental
from feedback_audit_cpu import metrics
RUN=ROOT/'results/rooted_connection_20261109T015142';AUDIT=ROOT/'results/rooted_feedback_audit_20261109'
OLD=ROOT/'results/hard_structure_response_20261108T182154';EDGE=ROOT/'results/hard_structure_edge_direction_cpu_20261108'
OUT=ROOT/'results/rooted_quantile_cpu_20261109'
START=time.time();BIND={}
def bind(p):
    p=Path(p);BIND[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest();return p
def read(p):return json.loads(bind(p).read_text())
def save(n,x):
    (OUT/(n+'.json')).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False))
    if isinstance(x,list) and x:
        with (OUT/(n+'.csv')).open('w') as f:
            w=csv.DictWriter(f,list(x[0]));w.writeheader();w.writerows(x)
def check():
    assert time.time()-START<600,'compute cap'
    assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024<2048,'RSS cap'
def core_for(ds):
    with np.load(bind(RUN/f'{ds}_cpu_cost_panel.npz')) as z:return RootedScore(z['types'],z['query'],z['families'],z['reference'],10)
def geometry(x,y,u):
    return .5*(np.square(np.sort(x,axis=0)-np.sort(y,axis=0)).mean()+np.square(np.sort(x@u,axis=0)-np.sort(y@u,axis=0)).mean())
def finite_tests():
    y=np.array([[-2.,1.],[-1.,-1.],[1.,-2.],[2.,2.]])+3;u=directions(2,0)
    c=np.array([.5,1.]);C=.5*np.eye(2)/2+.5*(u@u.T)/32
    rows=[]
    for a in [0.,.5,1.,2.,4.]:
        val=geometry(y+a*c,y,u);expected=a*a*float(c@C@c)
        assert abs(val-expected)<1e-12
        rows.append(dict(test='translation',factor=a,value=val,expected=expected))
    mu=y.mean(0);cov=(y-mu).T@(y-mu)/len(y)
    for s in [0.,.5,1.,2.,4.]:
        val=geometry(mu+s*(y-mu),y,u);expected=(s-1)**2*float(np.trace(C@cov))
        assert abs(val-expected)<1e-12
        rows.append(dict(test='spread',factor=s,value=val,expected=expected))
    correlated=np.array([[0.,0.],[1.,1.]]);crossed=np.array([[0.,1.],[1.,0.]])
    assert np.array_equal(np.sort(correlated,axis=0),np.sort(crossed,axis=0))
    assert geometry(correlated,crossed,u)>1e-6
    rows.append(dict(test='same_marginals_different_joint',factor=None,value=geometry(correlated,crossed,u),expected=None))
    ty=np.array([0,0,1,1]);q=np.array(list(itertools.combinations(range(4),2))).T;f=ty[q[0]]+ty[q[1]];ref=np.array([1.,1.,0.,1.,0.,1.])
    core=RootedScore(ty,q,f,ref,3);score=QuantileScore(core);states=((np.arange(64)[:,None]>>np.arange(6))&1).astype(float)
    values=score.raw_score(core.features(torch.from_numpy(states)).numpy())[0]
    max_error=0;diffs=[]
    for i,b in enumerate(states):
        inc=RootedIncremental(core,b);diff=score.differences(inc);diffs.append(diff)
        for e in range(6):
            j=i^(1<<e);err=abs(values[j]-values[i]-(1-2*b[e])*diff[e]);max_error=max(max_error,err)
            assert err<1e-12
    assert min(values)>-1e-12 and abs(score.raw_score(core.features(core.reference).numpy())[0])<1e-14
    p=np.array([.1,.25,.45,.6,.75,.9]);law=np.prod(np.where(states==1,p,1-p),axis=1)
    exact=(law[:,None]*values[:,None]*(states-p)).sum(0)
    rb=(law[:,None]*np.array(diffs)*p*(1-p)).sum(0)
    assert np.max(abs(exact-rb))<1e-12
    a=np.zeros((4,4));a[q[0],q[1]]=ref;a+=a.T;perm=np.array([1,0,3,2]);relabeled=a[np.ix_(perm,perm)][q[0],q[1]]
    assert abs(score.raw_score(core.features(torch.from_numpy(relabeled)).numpy())[0])<1e-14
    save('finite_checks',dict(cases=rows,micro_graphs=64,edge_checks=384,max_difference_error=max_error,expected_gradient_error=float(np.max(abs(exact-rb))),type_permutation=True,passed=True))

def main():
    OUT.mkdir(exist_ok=False)
    save('protocol',dict(start_epoch=START,start_BJT=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        candidate='one fixed coordinate/mixed projected quantile squared discrepancy',
        formula='type mean of 0.5*coordinate sorted squared error mean +0.5*32 mixed-unit-direction sorted squared error mean',
        projections_seed=[2026110913,'type_id','active_dimension'],representation='unchanged RootedScore log1p/scales/masks, all three blocks',
        sample_counts_fixed=True,loss_weight='not selected; no model integration',
        final_panel='all64 saved graphs and exactly4096 old selected flips',
        conditional_panel='all24 prior saved hard graphs, each relation/presence stratum min(8,N) fixed random slots, seed2026110914; plus all6333 old-three-metric-worse operations explicitly posthoc',
        full_cost='predeclared IMDB/PubMed CE parent CE t500 draw0; execute iff 64-slot cost predicts <40s each and remaining >2*prediction+60s',
        evaluation='old and new on same slots; full favorable/adverse metric changes, preserve Pareto improvements; no performance-based tuning',
        CPU_seconds=600,RSS_MiB=2048,threads=1,GPU=0,models=0,updates=0,new_graph_samples=0,
        interpretation='development screen only; finite projection blind spots; old conditional panel not current RC training states; no automatic GPU/long training'))
    for p in [Path(__file__),Path(__file__).parent/'quantile.py',ROOT/'research_branches/rooted_connection_20261109/score.py',ROOT/'research_branches/rooted_connection_20261109/feedback_audit_cpu.py']:bind(p)
    finite_tests();cores={ds:core_for(ds) for ds in ['IMDB','PubMed']};all_edges=[];graphrows=[]
    old_edges=read(AUDIT/'final_edge_metrics.json')
    for job in read(RUN/'evaluation_jobs.json'):
        check();ident={k:job[k] for k in ('dataset','train_seed','arm','sample_seed')};core=cores[job['dataset']];score=QuantileScore(core)
        path=Path(job['folder'])/'graph.npz';b=read(path.parent/'binding.json');bind(path);assert BIND[str(path)]==b['graph_sha256']
        with np.load(path) as z:
            assert np.array_equal(z['types'],core.types);a=np.zeros((core.n,core.n));a[z['edges'][0],z['edges'][1]]=1;a+=a.T
        q=core.query.numpy();bits=a[q[0],q[1]];inc=RootedIncremental(core,bits)
        rows=[x for x in old_edges if all(x[k]==v for k,v in ident.items())];ix=np.array([x['slot'] for x in rows]);delta=score.differences(inc,ix)*(1-2*bits[ix])
        value,axis,mixed=score.raw_score(inc.raw);graphrows.append(dict(**ident,old_score=inc.base_score,new_score=value,axis_term=float(axis[0]),mixed_term=float(mixed[0])))
        for x,d in zip(rows,delta):all_edges.append(dict(**x,new_score_change=float(d)))
    save('final_graph_scores',graphrows);save('final_edge_comparison',all_edges);print('final64 complete',flush=True)
    prior_bad=read(AUDIT/'conditional_tradeoffs.json');cond=[];old_bad=[];costs=[];rng=np.random.default_rng(2026110914)
    history=read(AUDIT/'bindings.json')
    def historical(p):
        p=bind(p);assert str(p) in history and BIND[str(p)]==history[str(p)];return p
    for ds in ['IMDB','PubMed']:
        fixed=torch.load(historical(OLD/f'{ds}_fixed_inputs.pt'),map_location='cpu',weights_only=False)['inputs']
        for t,seed,inp in fixed:
            q,f,y=map(np.asarray,(inp['query'],inp['f'],inp['labels']));core=RootedScore(cores[ds].types,q,f,y,10);score=QuantileScore(core);ref=core.features(core.reference).numpy()
            for parent in ['CE','PS']:
                tag=f'{ds}_{parent}_CE_t{t}'
                for draw in [0,1]:
                    check()
                    with np.load(historical(EDGE/f'{tag}_draw{draw}_flips.npz')) as z:bits=(1-z['flip_sign'])/2
                    with np.load(bind(AUDIT/f'{tag}_draw{draw}_full.npz')) as z:old=z['conditional_difference']*(1-2*bits)
                    inc=RootedIncremental(core,bits);sel=[];weights={}
                    for rel in np.unique(f):
                        for presence in [0,1]:
                            pool=np.flatnonzero((f==rel)&(bits==presence));ids=rng.choice(pool,min(8,len(pool)),replace=False)
                            sel.extend(ids.tolist())
                            for e in ids:weights[int(e)]=len(pool)/len(ids)
                    ix=np.array(sorted(sel));save('conditional_selection_'+tag+f'_draw{draw}',dict(slots=ix.tolist(),weights=weights))
                    tick=time.time();delta=score.differences(inc,ix)*(1-2*bits[ix]);cost=time.time()-tick
                    raw,_=inc.raw_flips(ix);base=metrics(core,inc.raw,ref);mm=metrics(core,raw,ref)
                    for k,e in enumerate(ix):
                        cond.append(dict(dataset=ds,parent_history=parent,t=int(t),draw=draw,slot=int(e),relation=int(f[e]),present=int(bits[e]),stratum_weight=float(weights[int(e)]),old_score_change=float(old[e]),new_score_change=float(delta[k]),**{n+'_change':float(v[k]-base[n][0]) for n,v in mm.items()}))
                    known=[x for x in prior_bad if x['dataset']==ds and x['parent_history']==parent and x['t']==t and x['draw']==draw]
                    if known:
                        ids=np.array([x['slot'] for x in known]);dd=score.differences(inc,ids)*(1-2*bits[ids])
                        for x,d in zip(known,dd):old_bad.append(dict(**x,new_score_change=float(d)))
                    if t==500 and parent=='CE' and draw==0:
                        tick=time.time();small=score.differences(inc,np.arange(64));small_seconds=time.time()-tick;prediction=small_seconds*len(bits)/64
                        full_seconds=None;full_slots=0
                        if prediction<40 and 600-(time.time()-START)>2*prediction+60:
                            tick=time.time();full=score.differences(inc);full_seconds=time.time()-tick;full_slots=len(bits)
                            assert np.max(abs(full[:64]-small))<1e-12
                            np.savez_compressed(OUT/f'{ds}_full_reference.npz',conditional_difference=full)
                        costs.append(dict(dataset=ds,selected_slots=len(ix),selected_seconds=cost,projected_full_seconds=prediction,full_seconds=full_seconds,full_slots=full_slots))
                    save('conditional_edge_comparison',cond);save('old_bad_comparison',old_bad);save('cost',costs)
                    print(tag,draw,'sample',len(ix),'old_bad',len(known),flush=True)
    examples=read(AUDIT/'score_decomposition_examples.json');example_results=[]
    for x in examples:
        match=next(r for r in all_edges if all(x[k]==r[k] for k in ('dataset','train_seed','arm','sample_seed','slot')))
        example_results.append(match)
    save('known_examples',example_results)
    def summarise(rows,keys,weighted):
        grouped={}
        for r in rows:grouped.setdefault(tuple(r[k] for k in keys),[]).append(r)
        out=[]
        for key,rs in grouped.items():
            for method,col in [('RC','score_change' if 'score_change' in rs[0] else 'old_score_change'),('QUANTILE','new_score_change')]:
                counts={k:0. for k in ['total','rewarded','triangle_worse','three_worse','no_checked_gain','pareto_better_total','pareto_better_rewarded']}
                for r in rs:
                    w=r.get('stratum_weight',1.) if weighted else 1.;vals={k:v for k,v in r.items() if k.endswith('_change') and k not in ('score_change','old_score_change','new_score_change')}
                    good=r[col]<-1e-12;pareto=all(v<=1e-12 for v in vals.values()) and any(v<-1e-12 for v in vals.values())
                    counts['total']+=w;counts['rewarded']+=w*good;counts['triangle_worse']+=w*(good and r['triangle_error_change']>1e-12)
                    counts['three_worse']+=w*(good and all(r[k+'_change']>1e-12 for k in ('triangle_error','total_degree_W1','joint_degree_TV')))
                    counts['no_checked_gain']+=w*(good and all(v>=-1e-12 for v in vals.values()) and any(v>1e-12 for v in vals.values()))
                    counts['pareto_better_total']+=w*pareto;counts['pareto_better_rewarded']+=w*(pareto and good)
                out.append(dict(**dict(zip(keys,key)),method=method,**counts))
        return out
    save('final_summary',summarise(all_edges,['dataset','arm'],False));save('conditional_summary',summarise(cond,['dataset','t'],True))
    assert not torch.cuda.is_initialized()
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in BIND.items())
    save('binding',BIND);save('receipt',dict(status='complete',wall_seconds=time.time()-START,peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        final_graphs=len(graphrows),final_flips=len(all_edges),conditional_flips=len(cond),old_bad_flips=len(old_bad),GPU_hours=0,model_calls=0,updates=0,new_graph_samples=0,sources_unchanged=True))
    print((OUT/'receipt.json').read_text(),flush=True)
if __name__=='__main__':
    try:main()
    except BaseException as exc:
        if OUT.exists():save('failure',dict(error=repr(exc),traceback=traceback.format_exc()))
        raise
