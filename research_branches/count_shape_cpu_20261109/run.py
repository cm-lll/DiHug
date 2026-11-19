import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,json,time,hashlib,csv,resource,itertools,datetime
from pathlib import Path
import numpy as np
import torch
torch.set_num_threads(1)
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'research_branches/rooted_connection_20261109'))
sys.path.insert(0,str(ROOT/'research_branches/rooted_quantile_cpu_20261109'))
from score import RootedScore,RootedIncremental
from score_candidate import CountShapeScore
PREV=ROOT/'results/rooted_quantile_cpu_20261109';RUN=ROOT/'results/rooted_connection_20261109T015142'
OLD=ROOT/'results/hard_structure_response_20261108T182154';EDGE=ROOT/'results/hard_structure_edge_direction_cpu_20261108'
OUT=ROOT/'results/count_shape_cpu_20261109';START=time.time();BIND={};EPS=1e-12
def bind(p):
    p=Path(p);BIND[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest();return p
def read(p):return json.loads(bind(p).read_text())
def save(n,x):
    (OUT/(n+'.json')).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False))
    if isinstance(x,list) and x:
        with (OUT/(n+'.csv')).open('w') as f:
            w=csv.DictWriter(f,list(x[0]));w.writeheader();w.writerows(x)
def check():
    assert time.time()-START<600 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024<2048
def verify(core,s,bits,ids):
    inc=RootedIncremental(core,bits);raw,_=inc.raw_flips(ids);base=s.terms(inc.raw);new=s.terms(raw)
    mass,_=s.relation_mass(raw)
    assert np.max(abs(mass-new['degree_mass']))<1e-9
    return inc,raw,{k:new[k]-base[k][0] for k in new}
def tests():
    ty=np.array([0,0,1,1]);q=np.array(list(itertools.combinations(range(4),2))).T;f=ty[q[0]]+ty[q[1]];y=np.array([1.,1.,0.,1.,0.,1.])
    core=RootedScore(ty,q,f,y,3);s=CountShapeScore(core);states=((np.arange(64)[:,None]>>np.arange(6))&1).astype(float)
    vals=s.raw_score(core.features(torch.from_numpy(states)).numpy());diff=[];err=0.
    for k,b in enumerate(states):
        inc,raw,dd=verify(core,s,b,np.arange(6));diff.append(dd['total']*(1-2*b))
        for e in range(6):err=max(err,abs(dd['total'][e]-vals[k^(1<<e)]+vals[k]))
    assert err<1e-11 and abs(s.raw_score(core.features(core.reference).numpy())[0])<1e-12
    p=np.array([.1,.25,.45,.6,.75,.9]);law=np.prod(np.where(states==1,p,1-p),axis=1)
    exact=(law[:,None]*vals[:,None]*(states-p)).sum(0);rb=(law[:,None]*np.array(diff)*p*(1-p)).sum(0)
    assert np.max(abs(exact-rb))<1e-11
    # Translation changes means, not centered shapes, on the same fixed scale.
    raw=core.features(core.reference).numpy();before=s.terms(raw);after=s.terms(raw+2)
    assert after['mass'][0]>0 and abs(after['shape'][0]-before['shape'][0])<1e-11
    adj=np.zeros((4,4));adj[q[0],q[1]]=y;adj+=adj.T;perm=np.array([1,0,3,2])
    relabeled=adj[np.ix_(perm,perm)][q[0],q[1]]
    assert abs(s.raw_score(core.features(torch.from_numpy(relabeled)).numpy())[0])<1e-12
    save('finite_tests',dict(graphs=64,flips=384,flip_error=err,gradient_error=float(np.max(abs(exact-rb))),type_relabel=True,mean_shape_translation=True,passed=True))
def summary(rows,keys):
    groups={}
    for r in rows:groups.setdefault(tuple(r[k] for k in keys),[]).append(r)
    out=[]
    for key,rs in groups.items():
        for method,col in [('previous_quantile','new_score_change'),('count_shape','candidate_change')]:
            acc={k:0. for k in ['total','rewarded','triangle_worse','three_worse','no_aggregate_gain','pareto_better','pareto_rewarded','mass_worse_when_rewarded']}
            for r in rs:
                w=r.get('stratum_weight',1.);good=r[col]<-EPS
                vv=[r[k+'_change'] for k in ['total_degree_W1','relation_degree_W1','joint_degree_TV','node_triangle_W1','root_path_W1','hub_error','triangle_error','edge_error']]
                pareto=all(v<=EPS for v in vv) and any(v<-EPS for v in vv)
                acc['total']+=w;acc['rewarded']+=w*good;acc['triangle_worse']+=w*(good and r['triangle_error_change']>EPS)
                acc['three_worse']+=w*(good and all(r[k+'_change']>EPS for k in ['total_degree_W1','joint_degree_TV','triangle_error']))
                acc['no_aggregate_gain']+=w*(good and all(v>=-EPS for v in vv) and any(v>EPS for v in vv))
                acc['pareto_better']+=w*pareto;acc['pareto_rewarded']+=w*(good and pareto)
                acc['mass_worse_when_rewarded']+=w*(good and r['mass_change']>EPS)
            out.append(dict(**dict(zip(keys,key)),method=method,**acc))
    return out
def main():
    OUT.mkdir(exist_ok=False)
    save('protocol',dict(start_BJT=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        candidate='one raw-count mean plus centered joint quantile score',
        scales='s^2=max(1,reference coordinate mean)*active_blocks*block_dimensions/active_dimensions; no log',
        formula='type mean [mean_j(mean_z-mean_ref)^2 + .5 coordinate centered W2^2 + .5 mean32 centered projected W2^2]',
        directions='reuse previous fixed directions seed2026110913',loss_strength='not set; no training integration',
        prior_differences='not an isolated centering ablation: raw transform, scales, mean metric changed; old JOINT_W1 used other blocks/L1 assignment',
        data='64 archived graphs/4096 exact same slots;24 old conditional graphs/2765 exact same slots;172 predeclared old exchange operations',
        decision='report all directions and counterexamples; no automatic GPU/long training regardless of CPU result; no scale or weight search',
        limits=dict(worker_seconds=600,threads=1,RSS_MiB=2048,GPU=0,models=0,updates=0,new_random_graphs=0)))
    for p in [Path(__file__),Path(__file__).parent/'score_candidate.py',ROOT/'research_branches/rooted_connection_20261109/score.py',ROOT/'research_branches/rooted_quantile_cpu_20261109/quantile.py']:bind(p)
    tests();prior_bind=read(PREV/'binding.json');final_old=read(PREV/'final_edge_comparison.json');cond_old=read(PREV/'conditional_edge_comparison.json')
    def historical(p):
        p=bind(p);assert str(p) in prior_bind and BIND[str(p)]==prior_bind[str(p)];return p
    cores={};scales=[]
    for ds in ['IMDB','PubMed']:
        with np.load(historical(RUN/f'{ds}_cpu_cost_panel.npz')) as z:cores[ds]=RootedScore(z['types'],z['query'],z['families'],z['reference'],10)
        s=CountShapeScore(cores[ds])
        for t,c in s.cache.items():
            for k,col in enumerate(c['cols']):scales.append(dict(dataset=ds,type=int(t),feature=int(col),reference_mean=float(c['reference_mean'][k]),scale=float(c['scale'][k])))
    save('frozen_scales',scales)
    final=[];graphrows=[];exchange=[]
    for j in read(RUN/'evaluation_jobs.json'):
        check();ident={k:j[k] for k in ['dataset','train_seed','arm','sample_seed']};core=cores[j['dataset']];s=CountShapeScore(core)
        p=historical(Path(j['folder'])/'graph.npz');b=read(p.parent/'binding.json');assert BIND[str(p)]==b['graph_sha256']
        with np.load(p) as z:
            assert np.array_equal(z['types'],core.types);a=np.zeros((core.n,core.n));a[z['edges'][0],z['edges'][1]]=1;a+=a.T
        q=core.query.numpy();bits=a[q[0],q[1]];rs=[r for r in final_old if all(r[k]==v for k,v in ident.items())]
        inc,raw,dd=verify(core,s,bits,np.array([r['slot'] for r in rs]));graphrows.append(dict(**ident,**{k:float(v[0]) for k,v in s.terms(inc.raw).items()}))
        for k,r in enumerate(rs):final.append(dict(**r,candidate_change=float(dd['total'][k]),mass_change=float(dd['mass'][k]),shape_change=float(dd['shape'][k]),degree_mass_change=float(dd['degree_mass'][k])))
        if ident==dict(dataset='PubMed',train_seed=2026110421,arm='A',sample_seed=2026110232):
            er=read(ROOT/'results/type_price_audit_20261109/exchange_all.json');bb=bits.copy();bb[3489]=1;ai=RootedIncremental(core,bb);raw,_=ai.raw_flips([r['removed_slot'] for r in er]);tt=s.terms(raw);bt=s.terms(inc.raw)
            for k,r in enumerate(er):exchange.append(dict(**r,candidate_change=float(tt['total'][k]-bt['total'][0]),mass_change=float(tt['mass'][k]-bt['mass'][0]),shape_change=float(tt['shape'][k]-bt['shape'][0])))
    save('final_edges',final);save('graph_scores',graphrows);save('exchanges',exchange);save('final_summary',summary(final,['dataset']));save('final_summary_by_arm',summary(final,['dataset','arm']))
    print('64final graphs complete',flush=True)
    cond=[]
    for ds in cores:
        fixed=torch.load(historical(OLD/f'{ds}_fixed_inputs.pt'),map_location='cpu',weights_only=False)['inputs']
        for t,seed,inp in fixed:
            core=RootedScore(cores[ds].types,np.asarray(inp['query']),np.asarray(inp['f']),np.asarray(inp['labels']),10);s=CountShapeScore(core)
            for parent in ['CE','PS']:
                for draw in [0,1]:
                    check();tag=f'{ds}_{parent}_CE_t{t}_draw{draw}'
                    with np.load(historical(EDGE/f'{tag}_flips.npz')) as z:bits=(1-z['flip_sign'])/2
                    rs=[r for r in cond_old if r['dataset']==ds and r['t']==t and r['parent_history']==parent and r['draw']==draw]
                    _,_,dd=verify(core,s,bits,np.array([r['slot'] for r in rs]))
                    for k,r in enumerate(rs):cond.append(dict(**r,candidate_change=float(dd['total'][k]),mass_change=float(dd['mass'][k]),shape_change=float(dd['shape'][k]),degree_mass_change=float(dd['degree_mass'][k])))
    save('conditional_edges',cond);save('conditional_summary',summary(cond,['dataset','t']))
    examples=read(PREV/'known_examples.json');save('known_examples',[next(r for r in final if all(r[k]==e[k] for k in ['dataset','train_seed','arm','sample_seed','slot'])) for e in examples])
    assert not torch.cuda.is_initialized();assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in BIND.items())
    save('binding',BIND);save('receipt',dict(status='complete',wall_seconds=time.time()-START,peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        final_graphs=len(graphrows),final_edges=len(final),conditional_edges=len(cond),exchanges=len(exchange),assets_unchanged=True,GPU_hours=0,model_calls=0,updates=0,new_random_graphs=0))
    print((OUT/'final_summary.json').read_text());print((OUT/'known_examples.json').read_text());print((OUT/'receipt.json').read_text())
if __name__=='__main__':main()
