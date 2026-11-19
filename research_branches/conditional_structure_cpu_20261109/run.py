import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']: os.environ[key]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,time,json,csv,hashlib,datetime,resource,itertools
from pathlib import Path
import numpy as np
import torch
torch.set_num_threads(1)
ROOT=Path(__file__).resolve().parents[2]
for sub in ['rooted_connection_20261109','rooted_quantile_cpu_20261109','count_shape_cpu_20261109']:
    sys.path.insert(0,str(ROOT/'research_branches'/sub))
from score import RootedScore,RootedIncremental
from quantile import directions
from conditional_score import ConditionalScore,cdf_distance,membership,neighborhoods
OUT=ROOT/'results/conditional_structure_cpu_20261109'
PREV=ROOT/'results/output_path_ablation_cpu_20261109'
RUN=ROOT/'results/rooted_connection_20261109T015142'
OLD=ROOT/'results/hard_structure_response_20261108T182154'
EDGE=ROOT/'results/hard_structure_edge_direction_cpu_20261108'
START=time.time(); BIND={}; EPS=1e-12
METRICS=['total_degree_W1','relation_degree_W1','joint_degree_TV','node_triangle_W1','root_path_W1','hub_error','triangle_error','edge_error']
def bind(p):
    p=Path(p);BIND[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest();return p
def read(p):return json.loads(bind(p).read_text())
def save(n,x):
    (OUT/(n+'.json')).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False))
    if isinstance(x,list) and x:
        keys=list(dict.fromkeys(k for row in x for k in row))
        with (OUT/(n+'.csv')).open('w') as f:
            w=csv.DictWriter(f,keys);w.writeheader();w.writerows(x)
def limits():
    if time.time()-START>900:raise TimeoutError('900 second CPU budget')
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024>2048:raise MemoryError('2GiB CPU budget')
def summarise(rows,keys):
    groups={}
    for r in rows:groups.setdefault(tuple(r[k] for k in keys),[]).append(r)
    ans=[]
    for key,rr in groups.items():
        for method,col in [('count_shape','candidate_change'),('no_paths','ablation_change'),('conditional','conditional_change')]:
            a={k:0. for k in ['total','rewarded','three_worse','no_aggregate_gain','pareto_better','pareto_rewarded']+[m+'_worse' for m in METRICS]}
            for r in rr:
                w=r.get('stratum_weight',1.); good=r[col]<-EPS; v=[r[m+'_change'] for m in METRICS]
                pareto=all(z<=EPS for z in v) and any(z<-EPS for z in v)
                a['total']+=w;a['rewarded']+=w*good
                a['three_worse']+=w*(good and all(r[k+'_change']>EPS for k in ['total_degree_W1','joint_degree_TV','triangle_error']))
                a['no_aggregate_gain']+=w*(good and all(z>=-EPS for z in v) and any(z>EPS for z in v))
                a['pareto_better']+=w*pareto;a['pareto_rewarded']+=w*(good and pareto)
                for m in METRICS:a[m+'_worse']+=w*(good and r[m+'_change']>EPS)
            ans.append(dict(**dict(zip(keys,key)),method=method,**a))
    return ans
def tests():
    # Independent interval sum checks weighted empirical CDF integral.
    x=np.array([[0.],[0.],[2.]]);y=np.array([[1.],[3.]])
    wx=np.array([[.1,.4],[.3,.1],[.6,.5]]);wy=np.array([[.8,.2],[.2,.8]]);pi=np.array([.3,.7])
    points=np.unique(np.r_[x[:,0],y[:,0]]);direct=0.
    for l,r in zip(points[:-1],points[1:]):
        f=wx[x[:,0]<=l].sum(0)-wy[y[:,0]<=l].sum(0)
        direct+=(r-l)*np.dot(f*f,pi)
    actual=cdf_distance(x,y,wx,wy,pi,np.ones((1,32)))
    assert abs(actual-direct)<1e-12
    # Equal means do not erase distributions; swapped D/H association detected.
    uniform=np.ones((4,1))/4
    y=np.array([[-1.],[-1.],[1.],[1.]])
    collapsed=np.zeros((4,1))
    collapse=cdf_distance(collapsed,y,uniform,uniform,np.ones(1),directions(1,0))
    assert collapse>0
    d=np.array([[0.],[0.],[4.],[4.]])
    anchors,h2=neighborhoods(d);q=membership(d,anchors,h2);w=q/q.sum(0)
    swapped=cdf_distance(-y,y,w,w,q.mean(0),directions(1,0))
    assert swapped>0
    a=np.array([[10.,0.],[0.,10.]]);b=np.array([[10.,10.],[0.,0.]])
    u=np.ones((2,1))/2
    joint=cdf_distance(a,b,u,u,np.ones(1),directions(2,0));assert joint>0
    # Full finite graph check: exact hard-flip difference and free-logit gradient.
    ty=np.array([0,0,1,1]);q=np.array(list(itertools.combinations(range(4),2))).T
    fam=ty[q[0]]+ty[q[1]];ref=np.array([1.,1.,0.,1.,0.,1.])
    core=RootedScore(ty,q,fam,ref,3);score=ConditionalScore(core)
    states=((np.arange(64)[:,None]>>np.arange(6))&1).astype(float)
    raw=core.features(torch.from_numpy(states)).numpy();val=score.terms(raw)['total']
    error=0.;diff=[]
    for i,bits in enumerate(states):
        inc=RootedIncremental(core,bits);new,_=inc.raw_flips(np.arange(6));delta=score.terms(new)['total']-val[i]
        error=max(error,float(np.max(abs(delta-(val[i^(1<<np.arange(6))]-val[i])))))
        diff.append(delta*(1-2*bits))
    p=np.array([.1,.25,.45,.6,.75,.9]);law=np.prod(np.where(states==1,p,1-p),axis=1)
    exact=(law[:,None]*val[:,None]*(states-p)).sum(0)
    rb=(law[:,None]*np.array(diff)*p*(1-p)).sum(0)
    graderr=float(np.max(abs(exact-rb)));assert error<1e-10 and graderr<1e-10
    adj=np.zeros((4,4));adj[q[0],q[1]]=ref;adj+=adj.T;perm=np.array([1,0,3,2])
    permraw=core.features(torch.from_numpy(adj[np.ix_(perm,perm)][q[0],q[1]])).numpy()
    assert abs(score.terms(permraw)['total'][0])<1e-10
    save('tests',dict(cdf_independent_error=abs(actual-direct),collapsed_mean_score=collapse,
        swapped_degree_structure_score=swapped,same_marginal_different_joint_score=joint,
        graphs=64,flips=384,flip_error=error,expected_gradient_error=graderr,type_permutation=True))
def panel(core,bits,rr,sc):
    inc=RootedIncremental(core,bits);base=sc.terms(inc.raw);out=[]
    for start in range(0,len(rr),8):
        limits();subset=rr[start:start+8];raw,_=inc.raw_flips([r['slot'] for r in subset]);v=sc.terms(raw)
        for k,r in enumerate(subset):
            out.append(dict(**r,conditional_change=float(v['total'][k]-base['total'][0]),
                degree_term_change=float(.5*(v['degree'][k]-base['degree'][0])),
                conditional_term_change=float(.5*(v['conditional'][k]-base['conditional'][0]))))
    return out,inc,base
def main():
    OUT.mkdir(exist_ok=False)
    save('protocol',dict(start_BJT=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        limits=dict(seconds=900,threads=1,memory_MiB=2048,GPU=0),
        rule='equal degree-distribution and soft degree-conditioned structure-distribution terms; no effect-driven tuning',
        degree='joint relation degree, reference RMS floor1, divided by sqrt(active dimensions)',
        structure='typed triangle plus degree-conditioned rooted path residual, reference RMS floor1 and equal block geometry',
        residual='N_v_rs-d_v_r*sum_type(N_rs)/sum_type(d_r); zero support coefficient=0',
        neighborhoods='all distinct reference degree vectors; kth distinct squared distance with k=ceil(sqrt(number anchors)), capped m-1; one anchor h2=1; rational kernel (1+distance/h2)^-2, per-node partition',
        discrepancy='exact integral squared empirical CDF differences; half coordinate mean plus half 32 frozen projection mean; conditional group weights from reference soft membership',
        scale_source='reference only; candidate residual coefficients describe candidate graph, not refitted learned parameters',
        panel='64 saved graphs4096 fixed flips,24 saved hard graphs2765 fixed flips,172 old exchanges',
        scope='CPU development only. Conditional smoothing and finite projections approximate conditional matching; multiple objective/scale changes versus old score, not single factor.',
        decision='retain all gains/costs; assess both datasets and seeds, low-degree/hub/path costs; no automatic training even if score improves'))
    for p in [Path(__file__),Path(__file__).parent/'conditional_score.py',ROOT/'research_branches/rooted_connection_20261109/score.py',ROOT/'research_branches/rooted_quantile_cpu_20261109/quantile.py']:bind(p)
    tests();previous=read(PREV/'binding.json');edges=read(PREV/'final_edges.json');cond=read(PREV/'conditional_edges.json')
    def historical(p):
        p=bind(p);assert str(p) in previous and BIND[str(p)]==previous[str(p)];return p
    cores={};scores={};config={}
    for ds in ['IMDB','PubMed']:
        with np.load(historical(RUN/f'{ds}_cpu_cost_panel.npz')) as z:
            cores[ds]=RootedScore(z['types'],z['query'],z['families'],z['reference'],10)
        scores[ds]=ConditionalScore(cores[ds]);config[ds]=scores[ds].configuration()
        assert abs(scores[ds].terms(cores[ds].features(cores[ds].reference).numpy())['total'][0])<1e-10
    save('reference_configuration',config)
    final=[];graphs=[];exchanges=[];jobs=read(RUN/'evaluation_jobs.json')
    first=time.time()
    for ji,job in enumerate(jobs):
        limits();ident={k:job[k] for k in ['dataset','train_seed','arm','sample_seed']};core=cores[job['dataset']];sc=scores[job['dataset']]
        p=historical(Path(job['folder'])/'graph.npz');meta=read(p.parent/'binding.json');assert BIND[str(p)]==meta['graph_sha256']
        with np.load(p) as z:
            assert np.array_equal(z['types'],core.types);adj=np.zeros((core.n,core.n));adj[z['edges'][0],z['edges'][1]]=1;adj+=adj.T
        q=core.query.numpy();bits=adj[q[0],q[1]];rr=[r for r in edges if all(r[k]==v for k,v in ident.items())]
        out,inc,base=panel(core,bits,rr,sc);final+=out
        graphs.append(dict(**ident,**{k:float(v[0]) for k,v in base.items()}))
        if ji==0:
            elapsed=time.time()-first;save('calibration',dict(first_graph_64flips_seconds=elapsed,remaining_rough_seconds=elapsed*110,
                note='cost-only calibration, full panel is fixed; no score-driven changes'))
            print('first graph seconds',elapsed,flush=True)
            if elapsed*110>900-(time.time()-START):
                save('partial_final_edges',final);raise TimeoutError('first measured cost cannot cover full fixed panel')
        if ident==dict(dataset='PubMed',train_seed=2026110421,arm='A',sample_seed=2026110232):
            ers=read(PREV/'exchanges.json');bb=bits.copy();bb[3489]=1;ai=RootedIncremental(core,bb)
            for st in range(0,len(ers),8):
                limits();batch=ers[st:st+8];raw,_=ai.raw_flips([r['removed_slot'] for r in batch]);v=sc.terms(raw)
                exchanges.extend(dict(**r,conditional_change=float(v['total'][k]-base['total'][0])) for k,r in enumerate(batch))
        if (ji+1)%16==0:
            save('partial_final_edges',final);print('graphs',ji+1,'elapsed',time.time()-START,flush=True)
    save('final_edges',final);save('graph_scores',graphs);save('exchanges',exchanges)
    for name,keys in [('final_summary',['dataset']),('final_by_seed',['dataset','train_seed']),('final_by_arm_seed',['dataset','arm','train_seed'])]:save(name,summarise(final,keys))
    conditional=[]
    for ds in cores:
        fixed=torch.load(historical(OLD/f'{ds}_fixed_inputs.pt'),map_location='cpu',weights_only=False)['inputs']
        for t,seed,inp in fixed:
            core=RootedScore(cores[ds].types,np.asarray(inp['query']),np.asarray(inp['f']),np.asarray(inp['labels']),10);sc=ConditionalScore(core)
            for parent in ['CE','PS']:
                for draw in [0,1]:
                    limits()
                    with np.load(historical(EDGE/f'{ds}_{parent}_CE_t{t}_draw{draw}_flips.npz')) as z:bits=(1-z['flip_sign'])/2
                    rr=[r for r in cond if r['dataset']==ds and r['t']==t and r['parent_history']==parent and r['draw']==draw]
                    out,_,_=panel(core,bits,rr,sc);conditional+=out
            save('partial_conditional_edges',conditional);print(ds,t,'elapsed',time.time()-START,flush=True)
    save('conditional_edges',conditional);save('conditional_summary',summarise(conditional,['dataset','t']))
    old=read(PREV/'all24_adverse_cases.json');key=lambda r:tuple(r[k] for k in ['dataset','train_seed','arm','sample_seed','slot'])
    lookup={key(r):r for r in final};save('all24_adverse_cases',[lookup[key(r)] for r in old])
    typed=read(ROOT/'results/type_price_audit_20261109/per_type_edges.json');grouped={}
    for r in typed:grouped.setdefault(key(r),[]).append(r)
    groups=[]
    for ds in cores:
        for method,col in [('count_shape','candidate_change'),('no_paths','ablation_change'),('conditional','conditional_change')]:
            for t in cores[ds].nodes:
                rr=[next(z for z in grouped[key(r)] if z['type']==t) for r in final if r['dataset']==ds and r[col]<-EPS]
                groups.append(dict(dataset=ds,method=method,type=int(t),rewarded_flips=len(rr),**{
                    m+'_'+sgn:sum(z[m+'_change']*direction>EPS for z in rr)
                    for m in ['total_degree_W1','joint_degree_TV','hub_error','lower_half_degree_error']
                    for sgn,direction in [('worse',1),('better',-1)]}))
    save('type_costs',groups)
    assert not torch.cuda.is_initialized()
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in BIND.items())
    save('binding',BIND);save('receipt',dict(status='complete',wall_seconds=time.time()-START,
        peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        final_graphs=len(graphs),final_slots=len(final),conditional_slots=len(conditional),exchanges=len(exchanges),
        GPU_hours=0,models=0,updates=0,new_graphs=0,assets_unchanged=True))
    print((OUT/'final_summary.json').read_text());print((OUT/'receipt.json').read_text())
if __name__=='__main__':
    try:main()
    except BaseException as e:
        if OUT.exists():save('failure',dict(error=repr(e),wall_seconds=time.time()-START))
        raise
