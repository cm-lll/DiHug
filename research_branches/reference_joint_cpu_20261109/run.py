import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,json,csv,time,datetime,hashlib,resource,itertools
from pathlib import Path
import numpy as np
import torch
torch.set_num_threads(1)
ROOT=Path(__file__).resolve().parents[2]
for sub in ['rooted_connection_20261109','rooted_quantile_cpu_20261109','conditional_structure_cpu_20261109']:
    sys.path.insert(0,str(ROOT/'research_branches'/sub))
from score import RootedScore,RootedIncremental
from quantile import directions
from conditional_score import ConditionalScore,cdf_distance,membership
from fixed_score import FixedScore,joint_discrepancy
OUT=ROOT/'results/reference_joint_cpu_20261109'
PREV=ROOT/'results/conditional_structure_cpu_20261109'
RUN=ROOT/'results/rooted_connection_20261109T015142'
OLD=ROOT/'results/hard_structure_response_20261108T182154'
EDGE=ROOT/'results/hard_structure_edge_direction_cpu_20261108'
START=time.time();BIND={};EPS=1e-12
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
    if time.time()-START>900:raise TimeoutError('900s CPU bound')
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024>2048:raise MemoryError('2GiB CPU bound')
def summaries(rows,keys):
    groups={}
    for r in rows:groups.setdefault(tuple(r[k] for k in keys),[]).append(r)
    out=[]
    for key,rr in groups.items():
        for method,col in [('count_shape','candidate_change'),('no_paths','ablation_change'),('conditional','conditional_change'),('reference_joint','fixed_change')]:
            a={k:0. for k in ['total','rewarded','three_worse','no_aggregate_gain','pareto_better','pareto_rewarded']+[m+'_worse' for m in METRICS]}
            for r in rr:
                w=r.get('stratum_weight',1.);good=r[col]<-EPS;v=[r[m+'_change'] for m in METRICS]
                pareto=all(z<=EPS for z in v) and any(z<-EPS for z in v)
                a['total']+=w;a['rewarded']+=w*good
                a['three_worse']+=w*(good and all(r[m+'_change']>EPS for m in ['total_degree_W1','joint_degree_TV','triangle_error']))
                a['no_aggregate_gain']+=w*(good and all(z>=-EPS for z in v) and any(z>EPS for z in v))
                a['pareto_better']+=w*pareto;a['pareto_rewarded']+=w*(good and pareto)
                for m in METRICS:a[m+'_worse']+=w*(good and r[m+'_change']>EPS)
            out.append(dict(**dict(zip(keys,key)),method=method,**a))
    return out
def tests():
    # Group masses are NOT normalized away, even if all H values coincide.
    h=np.zeros((4,1));qr=np.tile([.5,.5],(4,1));q=np.tile([.75,.25],(4,1));u=directions(1,0)
    occupied=joint_discrepancy(h,h,q,qr,qr.mean(0),u)[0];assert abs(occupied-.25)<1e-12
    # Equal mass: new score exactly equals old conditional score at fixed H.
    x=np.array([[-2.],[-1.],[1.],[3.]]);y=np.array([[-1.],[0.],[1.],[2.]])
    q=np.array([[.9,.1],[.7,.3],[.3,.7],[.1,.9]])
    v=joint_discrepancy(x,y,q,q,q.mean(0),u)[0]
    old=cdf_distance(x,y,q/q.sum(0),q/q.sum(0),q.mean(0),u)
    assert abs(v-old)<1e-12
    # Fixed-feature subdistribution is affine in node mixtures.
    z=np.array([-.5,.5,2.5]);f=lambda a,w:np.array([(w*(a[:,None]<=cut)).sum(0)/len(a) for cut in z])
    mixerr=np.max(abs(f(x[:,0],q)-.5*(f(x[:2,0],q[:2])+f(x[2:,0],q[2:]))));assert mixerr<1e-12
    spread=np.array([[-1.],[-1.],[1.],[1.]])
    collapse=joint_discrepancy(h,spread,qr,qr,qr.mean(0),u)[0];assert collapse>0
    ty=np.array([0,0,1,1]);query=np.array(list(itertools.combinations(range(4),2))).T
    fam=ty[query[0]]+ty[query[1]];ref=np.array([1.,1.,0.,1.,0.,1.])
    core=RootedScore(ty,query,fam,ref,3);sc=FixedScore(core);prior=ConditionalScore(core)
    for t,c in sc.cache.items():
        for k in ['ds','hs','anchors','h2','prior','ud','uh','d','h','weights']:
            assert np.array_equal(c[k],prior.cache[t][k]),k
    states=((np.arange(64)[:,None]>>np.arange(6))&1).astype(float)
    raw=core.features(torch.from_numpy(states)).numpy();values=sc.terms(raw)['total'];diff=[];err=0.
    for i,b in enumerate(states):
        limits();inc=RootedIncremental(core,b);new,_=inc.raw_flips(np.arange(6));delta=sc.terms(new)['total']-values[i]
        err=max(err,float(np.max(abs(delta-(values[i^(1<<np.arange(6))]-values[i])))))
        diff.append(delta*(1-2*b))
    p=np.array([.1,.25,.45,.6,.75,.9]);law=np.prod(np.where(states==1,p,1-p),axis=1)
    exact=(law[:,None]*values[:,None]*(states-p)).sum(0);rb=(law[:,None]*np.array(diff)*p*(1-p)).sum(0)
    graderr=float(np.max(abs(exact-rb)));assert err<1e-10 and graderr<1e-10
    # A node's encoding cannot change if its raw D/H values do not change.
    refraw=core.features(core.reference).numpy();altered=refraw.copy();altered[0,core.base_width:]+=3
    ids=np.arange(4);a=sc.local(refraw,ids)[1];b=sc.local(altered,ids)[1]
    assert np.array_equal(a[1:],b[1:])
    # Auxiliary all-type access above creates a separate key; does not alter cached type keys.
    adj=np.zeros((4,4));adj[query[0],query[1]]=ref;adj+=adj.T;perm=np.array([1,0,3,2])
    permraw=core.features(torch.from_numpy(adj[np.ix_(perm,perm)][query[0],query[1]])).numpy()
    assert abs(sc.terms(permraw)['total'][0])<1e-10
    save('tests',dict(occupation_difference=occupied,equal_mass_identity_error=abs(v-old),mixture_linearity_error=float(mixerr),
        collapse_score=collapse,graphs=64,flips=384,flip_error=err,exact_gradient_error=graderr,
        reference_scales_anchors_projections_unchanged=True,unchanged_node_encoding=True,type_permutation=True))
def panel(core,bits,rr,sc):
    inc=RootedIncremental(core,bits);base=sc.terms(inc.raw);out=[]
    for start in range(0,len(rr),8):
        limits();batch=rr[start:start+8];raw,_=inc.raw_flips([r['slot'] for r in batch]);v=sc.terms(raw)
        for k,r in enumerate(batch):
            out.append(dict(**r,fixed_change=float(v['total'][k]-base['total'][0]),
                fixed_degree_change=float(.5*(v['degree'][k]-base['degree'][0])),
                fixed_shape_change=float(.5*(v['shape'][k]-base['shape'][0])),
                fixed_occupation_change=float(.5*(v['occupation'][k]-base['occupation'][0]))))
    return out,inc,base
def main():
    OUT.mkdir(exist_ok=False)
    save('protocol',dict(start_BJT=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        limits=dict(worker_seconds=900,threads=1,memory_MiB=2048,GPU=0),
        rule='one reference-fixed mass-preserving joint soft-group score; no effect-driven rule changes',
        fixed='reference residual mu, reference scaling, anchors, widths, directions, prior masses',
        changes='replace current-graph mu by reference mu; replace per-group candidate normalization by full node mass and dummy atom; add group occupation discrepancy',
        formula='0.5*degree_CDF + 0.5*sum_a [CDF(P_a,Q_a)+(p_a-pi_a)^2]/pi_a; P_a=(1/n)sum_v q_a(D_v)delta_Hv+(1-p_a)delta_0',
        note='q varies with individual D through a fixed map. Does not freeze node identity; no candidate group division. Occupation coefficient1 is a design choice, not an optimality theorem.',
        panel='64 old graphs4096 flips;24 old conditional graphs2765 flips;172 old exchanges; old189 adverse cases retained',
        decision='all8 aggregate and pertype/seed costs; no GPU/training/generation admission; no alternate candidates or tuning'))
    for p in [Path(__file__),Path(__file__).parent/'fixed_score.py',ROOT/'research_branches/conditional_structure_cpu_20261109/conditional_score.py',ROOT/'research_branches/rooted_connection_20261109/score.py',ROOT/'research_branches/rooted_quantile_cpu_20261109/quantile.py']:bind(p)
    tests();previous=read(PREV/'binding.json');edges=read(PREV/'final_edges.json');cond=read(PREV/'conditional_edges.json')
    def historical(p):
        p=bind(p);assert str(p) in previous and BIND[str(p)]==previous[str(p)];return p
    cores={};scores={}
    for ds in ['IMDB','PubMed']:
        with np.load(historical(RUN/f'{ds}_cpu_cost_panel.npz')) as z:cores[ds]=RootedScore(z['types'],z['query'],z['families'],z['reference'],10)
        scores[ds]=FixedScore(cores[ds]);assert abs(scores[ds].terms(cores[ds].features(cores[ds].reference).numpy())['total'][0])<1e-10
    save('reference_configuration',{ds:sc.configuration() for ds,sc in scores.items()})
    final=[];graphs=[];exchanges=[];first=time.time()
    for ji,job in enumerate(read(RUN/'evaluation_jobs.json')):
        ident={k:job[k] for k in ['dataset','train_seed','arm','sample_seed']};core=cores[job['dataset']];sc=scores[job['dataset']]
        p=historical(Path(job['folder'])/'graph.npz');meta=read(p.parent/'binding.json');assert BIND[str(p)]==meta['graph_sha256']
        with np.load(p) as z:
            assert np.array_equal(z['types'],core.types);a=np.zeros((core.n,core.n));a[z['edges'][0],z['edges'][1]]=1;a+=a.T
        q=core.query.numpy();bits=a[q[0],q[1]];rr=[r for r in edges if all(r[k]==v for k,v in ident.items())]
        out,inc,base=panel(core,bits,rr,sc);final+=out
        graphs.append(dict(**ident,**{k:float(v[0]) for k,v in base.items()}))
        if ji==0:
            seconds=time.time()-first;save('calibration',dict(first_graph_seconds=seconds,rough_full_seconds=seconds*120))
            print('first graph',seconds,flush=True)
            if seconds*120>900-(time.time()-START):save('partial_final_edges',final);raise TimeoutError('full panel estimated outside CPU budget')
        if ident==dict(dataset='PubMed',train_seed=2026110421,arm='A',sample_seed=2026110232):
            ers=read(PREV/'exchanges.json');b=bits.copy();b[3489]=1;ai=RootedIncremental(core,b)
            for st in range(0,len(ers),8):
                limits();batch=ers[st:st+8];raw,_=ai.raw_flips([r['removed_slot'] for r in batch]);v=sc.terms(raw)
                exchanges.extend(dict(**r,fixed_change=float(v['total'][k]-base['total'][0])) for k,r in enumerate(batch))
        if (ji+1)%16==0:save('partial_final_edges',final);print('graphs',ji+1,'elapsed',time.time()-START,flush=True)
    save('final_edges',final);save('graph_scores',graphs);save('exchanges',exchanges)
    for name,keys in [('final_summary',['dataset']),('final_by_seed',['dataset','train_seed']),('final_by_arm_seed',['dataset','arm','train_seed'])]:save(name,summaries(final,keys))
    conditional=[]
    for ds in cores:
        fixed=torch.load(historical(OLD/f'{ds}_fixed_inputs.pt'),map_location='cpu',weights_only=False)['inputs']
        for t,seed,inp in fixed:
            core=RootedScore(cores[ds].types,np.asarray(inp['query']),np.asarray(inp['f']),np.asarray(inp['labels']),10);sc=FixedScore(core)
            for parent in ['CE','PS']:
                for draw in [0,1]:
                    with np.load(historical(EDGE/f'{ds}_{parent}_CE_t{t}_draw{draw}_flips.npz')) as z:bits=(1-z['flip_sign'])/2
                    rr=[r for r in cond if r['dataset']==ds and r['t']==t and r['parent_history']==parent and r['draw']==draw]
                    out,_,_=panel(core,bits,rr,sc);conditional+=out
            save('partial_conditional_edges',conditional);print(ds,t,'elapsed',time.time()-START,flush=True)
    save('conditional_edges',conditional);save('conditional_summary',summaries(conditional,['dataset','t']))
    key=lambda r:tuple(r[k] for k in ['dataset','train_seed','arm','sample_seed','slot'])
    old=read(PREV/'adverse_attribution.json');lookup={key(r):r for r in final}
    cases=[dict(**lookup[key(r)],old_geometry_flipped=r['becomes_rewarded_after_geometry_moves']) for r in old]
    save('old189_cases',cases)
    save('old_case_summary',[dict(dataset=ds,cases=len(rr),still_rewarded=sum(r['fixed_change']<-EPS for r in rr),
        old_geometry_flipped=sum(r['old_geometry_flipped'] for r in rr),
        geometry_subset_still_rewarded=sum(r['old_geometry_flipped'] and r['fixed_change']<-EPS for r in rr))
        for ds in cores for rr in [[r for r in cases if r['dataset']==ds]]])
    typed=read(ROOT/'results/type_price_audit_20261109/per_type_edges.json');grouped={}
    for r in typed:grouped.setdefault(key(r),[]).append(r)
    groups=[]
    for ds in cores:
        for method,col in [('count_shape','candidate_change'),('conditional','conditional_change'),('reference_joint','fixed_change')]:
            for t in cores[ds].nodes:
                rr=[next(z for z in grouped[key(r)] if z['type']==t) for r in final if r['dataset']==ds and r[col]<-EPS]
                groups.append(dict(dataset=ds,method=method,type=int(t),rewarded_flips=len(rr),**{
                    m+'_'+sgn:sum(z[m+'_change']*direction>EPS for z in rr)
                    for m in ['total_degree_W1','joint_degree_TV','hub_error','lower_half_degree_error']
                    for sgn,direction in [('worse',1),('better',-1)]}))
    save('type_costs',groups)
    assert not torch.cuda.is_initialized()
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in BIND.items())
    save('binding',BIND);save('receipt',dict(status='complete',wall_seconds=time.time()-START,peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        final_graphs=len(graphs),final_slots=len(final),conditional_slots=len(conditional),exchanges=len(exchanges),GPU_hours=0,models=0,updates=0,new_graphs=0,assets_unchanged=True))
    print((OUT/'final_summary.json').read_text());print((OUT/'old_case_summary.json').read_text());print((OUT/'receipt.json').read_text())
if __name__=='__main__':
    try:main()
    except BaseException as e:
        if OUT.exists():save('failure',dict(error=repr(e),wall_seconds=time.time()-START))
        raise
