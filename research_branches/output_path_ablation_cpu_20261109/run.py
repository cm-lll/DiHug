import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,json,csv,time,resource,hashlib,datetime,itertools
from pathlib import Path
import numpy as np
import torch
torch.set_num_threads(1)
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'research_branches/rooted_connection_20261109'))
sys.path.insert(0,str(ROOT/'research_branches/rooted_quantile_cpu_20261109'))
sys.path.insert(0,str(ROOT/'research_branches/count_shape_cpu_20261109'))
from score import RootedScore,RootedIncremental
from score_candidate import CountShapeScore
from path_ablation import WithoutPaths
RUN=ROOT/'results/rooted_connection_20261109T015142';PREV=ROOT/'results/count_shape_cpu_20261109'
OLD=ROOT/'results/hard_structure_response_20261108T182154';EDGE=ROOT/'results/hard_structure_edge_direction_cpu_20261108'
OUT=ROOT/'results/output_path_ablation_cpu_20261109';START=time.time();BIND={};EPS=1e-12
METRICS=['total_degree_W1','relation_degree_W1','joint_degree_TV','node_triangle_W1','root_path_W1','hub_error','triangle_error','edge_error']
def bind(p):
    p=Path(p);BIND[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest();return p
def read(p):return json.loads(bind(p).read_text())
def save(n,x):
    (OUT/(n+'.json')).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False))
    if isinstance(x,list) and x:
        with (OUT/(n+'.csv')).open('w') as f:
            w=csv.DictWriter(f,list(x[0]));w.writeheader();w.writerows(x)
def limits():assert time.time()-START<600 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024<2048
def isolation_check(core,raw):
    full=CountShapeScore(core);small=WithoutPaths(core)
    for t,c in full.cache.items():
        z=small.cache[t]
        for key in ['ids','mask','cols','scale','u','reference_mean']:assert np.array_equal(c[key],z[key]),key
        keep=c['cols']<core.base_width
        assert np.array_equal(c['mu'][keep],z['mu'][keep])
        assert np.array_equal(c['target_axis'][:,keep],z['target_axis'][:,keep])
    augmented=raw.copy();augmented[...,core.base_width:]+=17
    assert np.array_equal(small.terms(raw)['total'],small.terms(augmented)['total'])
    # Independent evaluation after zeroing path values in both candidate and reference.
    arr=raw[None] if raw.ndim==2 else raw;ref=core.features(core.reference).numpy();ans=np.zeros(len(arr))
    for c in full.cache.values():
        keep=c['cols']<core.base_width;x=arr[:,c['ids']][:,:,c['mask']]/c['scale'];y=ref[c['ids']][:,c['mask']]/c['scale']
        x=x*keep;y=y*keep;mx=x.mean(1);my=y.mean(0);xc=x-mx[:,None];yc=y-my
        ans+=np.square(mx-my).mean(1)+.5*np.square(np.sort(xc,axis=1)-np.sort(yc,axis=0)).mean((1,2))
        ans+=.5*np.square(np.sort(xc@c['u'],axis=1)-np.sort(yc@c['u'],axis=0)).mean((1,2))
    ans/=len(core.nodes)
    assert np.max(abs(ans-small.terms(raw)['total']))<1e-11
    return full,small
def tests():
    ty=np.array([0,0,1,1]);q=np.array(list(itertools.combinations(range(4),2))).T;f=ty[q[0]]+ty[q[1]];ref=np.array([1.,1.,0.,1.,0.,1.])
    core=RootedScore(ty,q,f,ref,3);states=((np.arange(64)[:,None]>>np.arange(6))&1).astype(float);raw=core.features(torch.from_numpy(states)).numpy()
    _,s=isolation_check(core,raw);values=s.terms(raw)['total'];diff=[];err=0.
    for i,b in enumerate(states):
        inc=RootedIncremental(core,b);new,_=inc.raw_flips(np.arange(6));d=s.terms(new)['total']-values[i]
        err=max(err,float(np.max(abs(d-(values[i^(1<<np.arange(6))]-values[i])))));diff.append(d*(1-2*b))
    p=np.array([.1,.25,.45,.6,.75,.9]);law=np.prod(np.where(states==1,p,1-p),axis=1)
    exact=(law[:,None]*values[:,None]*(states-p)).sum(0);rb=(law[:,None]*np.array(diff)*p*(1-p)).sum(0)
    assert err<1e-11 and np.max(abs(exact-rb))<1e-11
    adj=np.zeros((4,4));adj[q[0],q[1]]=ref;adj+=adj.T;perm=np.array([1,0,3,2])
    assert abs(s.terms(core.features(torch.from_numpy(adj[np.ix_(perm,perm)][q[0],q[1]])).numpy())['total'][0])<1e-12
    save('finite_checks',dict(graphs=64,flips=384,flip_error=err,expected_gradient_error=float(np.max(abs(exact-rb))),scales_directions_and_width_unchanged=True,path_only_perturbation_invariant=True,independent_zeroed_formula=True,type_permutation=True))
def summarise(rows,keys):
    grouped={}
    for r in rows:grouped.setdefault(tuple(r[k] for k in keys),[]).append(r)
    out=[]
    for key,rs in grouped.items():
        for method,col in [('with_paths','candidate_change'),('without_paths','ablation_change')]:
            acc={k:0. for k in ['total','rewarded','three_worse','no_aggregate_gain','pareto_better','pareto_rewarded']+[m+'_worse' for m in METRICS]}
            for r in rs:
                w=r.get('stratum_weight',1.);good=r[col]<-EPS;v=[r[k+'_change'] for k in METRICS];pareto=all(z<=EPS for z in v) and any(z<-EPS for z in v)
                acc['total']+=w;acc['rewarded']+=w*good
                for m in METRICS:acc[m+'_worse']+=w*(good and r[m+'_change']>EPS)
                acc['three_worse']+=w*(good and all(r[k+'_change']>EPS for k in ['total_degree_W1','joint_degree_TV','triangle_error']))
                acc['no_aggregate_gain']+=w*(good and all(z>=-EPS for z in v) and any(z>EPS for z in v))
                acc['pareto_better']+=w*pareto;acc['pareto_rewarded']+=w*(pareto and good)
            out.append(dict(**dict(zip(keys,key)),method=method,**acc))
    return out
def panel(core,bits,rs,full,small):
    inc=RootedIncremental(core,bits);old=full.terms(inc.raw);base=small.terms(inc.raw);result=[]
    for start in range(0,len(rs),32):
        limits();rr=rs[start:start+32];raw,_=inc.raw_flips([r['slot'] for r in rr]);a=full.terms(raw);b=small.terms(raw)
        for k,r in enumerate(rr):
            assert abs(a['total'][k]-old['total'][0]-r['candidate_change'])<1e-9*max(1.,abs(r['candidate_change']))
            assert abs(a['degree_mass'][k]-b['degree_mass'][k])<1e-11
            result.append(dict(**r,ablation_change=float(b['total'][k]-base['total'][0]),ablation_mass_change=float(b['mass'][k]-base['mass'][0]),ablation_shape_change=float(b['shape'][k]-base['shape'][0])))
    return result,inc
def main():
    OUT.mkdir(exist_ok=False)
    save('protocol',dict(start_BJT=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        question='single output-path ablation against frozen count_shape; no exact-assignment replacement',
        rule='zero both candidate/reference path coordinates; retain original width, masks, scales, means denominators, direction matrix and .5/.5 weights; do not renormalize',
        effect='removes path main effects AND interactions; cannot attribute separately; changes objective not just strength',
        panel='64 old graphs4096 same slots,24 old conditional graphs2765 same slots,172 fixed old exchanges; all old/new adverse cases retained',
        limits=dict(worker_seconds=600,threads=1,RSS_MiB=2048,GPU=0),
        decision='complete directional benefits/costs, no score tuning or automatic GPU/training/generation; no claim of generation benefit',
        evaluation='same8 metrics including held-out-from-loss rooted pathW1; pertype/hub costs reuse previous exact ledgers; selected slots not independent samples',
        seed_policy='no new RNG or model seeds'))
    for p in [Path(__file__),Path(__file__).parent/'path_ablation.py',ROOT/'research_branches/count_shape_cpu_20261109/score_candidate.py',ROOT/'research_branches/rooted_connection_20261109/score.py',ROOT/'research_branches/rooted_quantile_cpu_20261109/quantile.py']:bind(p)
    tests();prev=read(PREV/'binding.json');edges=read(PREV/'final_edges.json');cond=read(PREV/'conditional_edges.json')
    def historical(p):
        p=bind(p);assert str(p) in prev and BIND[str(p)]==prev[str(p)];return p
    cores={};scores={}
    for ds in ['IMDB','PubMed']:
        with np.load(historical(RUN/f'{ds}_cpu_cost_panel.npz')) as z:cores[ds]=RootedScore(z['types'],z['query'],z['families'],z['reference'],10)
        raw=cores[ds].features(cores[ds].reference).numpy();scores[ds]=isolation_check(cores[ds],raw)
    final=[];graphs=[];exchanges=[]
    for job in read(RUN/'evaluation_jobs.json'):
        limits();ident={k:job[k] for k in ['dataset','train_seed','arm','sample_seed']};core=cores[job['dataset']];full,small=scores[job['dataset']]
        p=historical(Path(job['folder'])/'graph.npz');meta=read(p.parent/'binding.json');assert BIND[str(p)]==meta['graph_sha256']
        with np.load(p) as z:
            assert np.array_equal(z['types'],core.types);adj=np.zeros((core.n,core.n));adj[z['edges'][0],z['edges'][1]]=1;adj+=adj.T
        q=core.query.numpy();bits=adj[q[0],q[1]];rr=[r for r in edges if all(r[k]==v for k,v in ident.items())];out,inc=panel(core,bits,rr,full,small);final+=out
        graphs.append(dict(**ident,with_paths=float(full.terms(inc.raw)['total'][0]),without_paths=float(small.terms(inc.raw)['total'][0])))
        if ident==dict(dataset='PubMed',train_seed=2026110421,arm='A',sample_seed=2026110232):
            ers=read(PREV/'exchanges.json');bb=bits.copy();bb[3489]=1;ai=RootedIncremental(core,bb);raw,_=ai.raw_flips([r['removed_slot'] for r in ers]);vals=small.terms(raw);base=small.terms(inc.raw)
            for k,r in enumerate(ers):exchanges.append(dict(**r,ablation_change=float(vals['total'][k]-base['total'][0])))
    save('final_edges',final);save('graph_scores',graphs);save('exchanges',exchanges);save('final_summary',summarise(final,['dataset']));save('final_by_seed',summarise(final,['dataset','train_seed']));save('final_by_arm_seed',summarise(final,['dataset','arm','train_seed']))
    print('64 final graphs complete',flush=True)
    conditional=[]
    for ds in cores:
        fixed=torch.load(historical(OLD/f'{ds}_fixed_inputs.pt'),map_location='cpu',weights_only=False)['inputs']
        for t,seed,inp in fixed:
            core=RootedScore(cores[ds].types,np.asarray(inp['query']),np.asarray(inp['f']),np.asarray(inp['labels']),10);full,small=isolation_check(core,core.features(core.reference).numpy())
            for parent in ['CE','PS']:
                for draw in [0,1]:
                    limits()
                    with np.load(historical(EDGE/f'{ds}_{parent}_CE_t{t}_draw{draw}_flips.npz')) as z:bits=(1-z['flip_sign'])/2
                    rr=[r for r in cond if r['dataset']==ds and r['t']==t and r['parent_history']==parent and r['draw']==draw]
                    out,_=panel(core,bits,rr,full,small);conditional+=out
    save('conditional_edges',conditional);save('conditional_summary',summarise(conditional,['dataset','t']))
    old=read(PREV/'new_adverse_operations.json');cases=[next(r for r in final if all(r[k]==o[k] for k in ['dataset','train_seed','arm','sample_seed','slot'])) for o in old]
    save('all24_adverse_cases',cases);save('case_summary',[dict(dataset=ds,cases=len(rr),still_rewarded=sum(r['ablation_change']<-EPS for r in rr)) for ds in cores for rr in [[r for r in cases if r['dataset']==ds]]])
    typed=read(ROOT/'results/type_price_audit_20261109/per_type_edges.json');key=lambda r:tuple(r[k] for k in ['dataset','train_seed','arm','sample_seed','slot'])
    grouped={}
    for r in typed:grouped.setdefault(key(r),[]).append(r)
    groups=[]
    for ds in cores:
        for method,col in [('with_paths','candidate_change'),('without_paths','ablation_change')]:
            for t in cores[ds].nodes:
                rr=[next(z for z in grouped[key(r)] if z['type']==t) for r in final if r['dataset']==ds and r[col]<-EPS]
                groups.append(dict(dataset=ds,method=method,type=int(t),rewarded_flips=len(rr),**{m+'_'+sgn:sum(z[m+'_change']*direction>EPS for z in rr) for m in ['total_degree_W1','joint_degree_TV','hub_error','lower_half_degree_error'] for sgn,direction in [('worse',1),('better',-1)]}))
    save('type_costs',groups)
    assert not torch.cuda.is_initialized();assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in BIND.items())
    save('binding',BIND);save('receipt',dict(status='complete',wall_seconds=time.time()-START,peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        final_graphs=len(graphs),final_slots=len(final),conditional_slots=len(conditional),exchanges=len(exchanges),GPU_hours=0,models=0,updates=0,new_random_graphs=0,assets_unchanged=True))
    print((OUT/'final_summary.json').read_text());print((OUT/'case_summary.json').read_text());print((OUT/'receipt.json').read_text())
if __name__=='__main__':
    try:main()
    except BaseException as e:
        if OUT.exists():save('failure',dict(error=repr(e),wall_seconds=time.time()-START))
        raise
