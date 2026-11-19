import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,importlib.util,time,json,itertools,datetime,hashlib,resource
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
for sub in ['reference_joint_cpu_20261109','conditional_structure_cpu_20261109','rooted_connection_20261109','rooted_quantile_cpu_20261109']:
    sys.path.insert(0,str(ROOT/'research_branches'/sub))
spec=importlib.util.spec_from_file_location('previous_cpu_helpers',ROOT/'research_branches/reference_joint_cpu_20261109/run.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
from explicit_score import ExplicitScore,connectivity,connectivity_changes
RootedScore=base.RootedScore;RootedIncremental=base.RootedIncremental
OUT=ROOT/'results/explicit_structure_cpu_20261109';PREV=ROOT/'results/reference_joint_cpu_20261109'
base.OUT=OUT;base.START=time.time();base.BIND={}
save,read,bind,limits=base.save,base.read,base.bind,base.limits
RUN=base.RUN;OLD=base.OLD;EDGE=base.EDGE;EPS=1e-12
METRICS=base.METRICS+['LCC_error','components_error','isolates_error']
def summaries(rows,keys):
    groups={}
    for r in rows:groups.setdefault(tuple(r[k] for k in keys),[]).append(r)
    result=[]
    for key,rr in groups.items():
        for method,col in [('count_shape','candidate_change'),('reference_joint','fixed_change'),('old_DIRECT','old_DIRECT_change'),('explicit','explicit_change')]:
            a={k:0. for k in ['total','rewarded','three_worse','no_eight_gain','no_eleven_gain','pareto_better','pareto_rewarded']+[m+'_worse' for m in METRICS]}
            for r in rr:
                w=r.get('stratum_weight',1.);good=r[col]<-EPS;v=[r[m+'_change'] for m in METRICS];v8=v[:8]
                pareto=all(z<=EPS for z in v) and any(z<-EPS for z in v)
                a['total']+=w;a['rewarded']+=w*good
                a['three_worse']+=w*(good and all(r[m+'_change']>EPS for m in ['total_degree_W1','joint_degree_TV','triangle_error']))
                a['no_eight_gain']+=w*(good and all(z>=-EPS for z in v8) and any(z>EPS for z in v8))
                a['no_eleven_gain']+=w*(good and all(z>=-EPS for z in v) and any(z>EPS for z in v))
                a['pareto_better']+=w*pareto;a['pareto_rewarded']+=w*(good and pareto)
                for m in METRICS:a[m+'_worse']+=w*(good and r[m+'_change']>EPS)
            result.append(dict(**dict(zip(keys,key)),method=method,**a))
    return result
def tests():
    ty=np.array([0,0,1,1]);q=np.array(list(itertools.combinations(range(4),2))).T
    fam=ty[q[0]]+ty[q[1]];ref=np.array([1.,1.,0.,1.,0.,1.]);core=RootedScore(ty,q,fam,ref,3);sc=ExplicitScore(core)
    states=((np.arange(64)[:,None]>>np.arange(6))&1).astype(float);raw=core.features(torch.from_numpy(states)).numpy();v=sc.terms(raw);deltas=[];error=0.
    for i,b in enumerate(states):
        a=np.zeros((4,4),dtype=int);a[q[0],q[1]]=b;a+=a.T;expected=np.zeros(len(sc.classes))
        for u,w,z in itertools.combinations(range(4),3):
            if a[u,w] and a[w,z] and a[u,z]:expected[sc.class_index[tuple(sorted(ty[[u,w,z]]))]]+=1
        assert np.allclose(sc.triangle_counts(raw[i])[0],expected,rtol=0,atol=1e-12)
        inc=RootedIncremental(core,b);new,_=inc.raw_flips(np.arange(6));nv=sc.terms(new);d=nv['total']-v['total'][i]
        error=max(error,float(np.max(abs(d-(v['total'][i^(1<<np.arange(6))]-v['total'][i])))))
        block=np.stack([nv[k]-v[k][i] for k in ['joint','closure','low','hub']],axis=1)
        assert np.max(abs(block.mean(1)-d))<1e-12
        assert not np.any((block.min(1)>=-EPS)&(d<-EPS))
        deltas.append(d*(1-2*b))
    p=np.array([.1,.25,.45,.6,.75,.9]);law=np.prod(np.where(states==1,p,1-p),axis=1)
    exact=(law[:,None]*v['total'][:,None]*(states-p)).sum(0);rb=(law[:,None]*np.array(deltas)*p*(1-p)).sum(0)
    ge=float(np.max(abs(exact-rb)));assert error<1e-10 and ge<1e-10
    a=np.zeros((4,4));a[q[0],q[1]]=ref;a+=a.T;perm=np.array([1,0,3,2])
    assert abs(sc.terms(core.features(torch.from_numpy(a[np.ix_(perm,perm)][q[0],q[1]])).numpy())['total'][0])<1e-12
    ty8=np.zeros(8,dtype=int);q8=np.array(list(itertools.combinations(range(8),2))).T
    a=np.zeros((8,8),dtype=bool);b=np.zeros_like(a)
    for i in range(8):a[i,(i+1)%8]=a[(i+1)%8,i]=True
    for start in [0,4]:
        for i in range(4):u=start+i;w=start+(i+1)%4;b[u,w]=b[w,u]=True
    c8=RootedScore(ty8,q8,np.zeros(len(q8.T),dtype=int),a[q8[0],q8[1]].astype(float),1);s8=ExplicitScore(c8)
    split=s8.terms(c8.features(torch.from_numpy(b[q8[0],q8[1]].astype(float))).numpy())['total'][0]
    assert abs(split)<1e-12 and connectivity(a)['components']==1 and connectivity(b)['components']==2
    save('tests',dict(graphs=64,flips=384,flip_error=error,exact_gradient_error=ge,
        typed_triangle_independent_enumeration=True,type_permutation=True,
        componentwise_monotonicity='algebraic property, not independent empirical success',
        connectivity_limitation=dict(reference='cycle8',other='two cycle4',score=float(split),reference_components=1,other_components=2)))
def panel(core,bits,rr,sc):
    inc=RootedIncremental(core,bits);old=sc.terms(inc.raw);out=[]
    conn,basestats,refstats=connectivity_changes(core,bits,[r['slot'] for r in rr])
    for st in range(0,len(rr),16):
        limits();batch=rr[st:st+16];raw,_=inc.raw_flips([r['slot'] for r in batch]);nv=sc.terms(raw)
        for k,r in enumerate(batch):
            row=dict(**r,**conn[st+k],explicit_change=float(nv['total'][k]-old['total'][0]),old_DIRECT_change=float(nv['old_DIRECT'][k]-old['old_DIRECT'][0]))
            for name in ['joint','closure','low','hub','joint_TV','relation_W1','node_triangle_W1','typed_triangle_L1']:
                row['explicit_'+name+'_change']=float(nv[name][k]-old[name][0])
            check=(r['total_degree_W1_change']/sc.old_degree_scale+r['joint_degree_TV_change']+r['triangle_error_change']/sc.triangle_scale)/3
            assert abs(check-row['old_DIRECT_change'])<1e-10*max(1.,abs(check))
            assert abs(row['explicit_change']-.25*sum(row['explicit_'+name+'_change'] for name in ['joint','closure','low','hub']))<1e-10
            out.append(row)
    return out,inc,old,basestats,refstats
def main():
    OUT.mkdir(exist_ok=False)
    save('protocol',dict(start_BJT=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        limit=dict(worker_seconds=900,threads=1,memory_MiB=2048,GPU=0),
        formula='mean(J,C,Low,Hub); J=.5(type-mean joint degree TV + type/relation mean normalized degree W1); C=.5(typed triangle count L1/total ref triangles + type/channel mean normalized participation W1)',
        tails='within-type sorted total-degree bottom floor(n/2), at least1, and top min(2,n); mean absolute rank deviation; no node-ID target',
        scales='each reference mean with floor1; triangle-class L1 uses total reference triangles floor1; fixed all datasets; equal four block coefficients',
        legacy='old DIRECT total-degree W1/global ref mean degree + joint TV + triangle total AE/ref triangles, divided3; failed historical short updates remain evidence',
        objective_scope='penalty on hard sampled graph, not hard constraint, not soft-probability plug-in; CE retained conceptually, lambda not chosen',
        panel='64 old graphs4096 same slots,24 old hard states2765 slots,172 old exchanges',
        held_out='root paths and connectivity not included; no newly learned sample diversity claim',
        decision='target improvement is partly by construction; examine magnitude and unoptimized costs; no automatic train or alternate coefficient search'))
    for p in [Path(__file__),Path(__file__).parent/'explicit_score.py',ROOT/'research_branches/reference_joint_cpu_20261109/run.py',ROOT/'research_branches/direct_structure_cpu_20261108/score.py',ROOT/'research_branches/rooted_connection_20261109/score.py']:bind(p)
    tests();bindings=read(PREV/'binding.json');edges=read(PREV/'final_edges.json');conditional_old=read(PREV/'conditional_edges.json')
    def historical(p):
        p=bind(p);assert str(p) in bindings and base.BIND[str(p)]==bindings[str(p)];return p
    cores={};scores={}
    for ds in ['IMDB','PubMed']:
        with np.load(historical(RUN/f'{ds}_cpu_cost_panel.npz')) as z:cores[ds]=RootedScore(z['types'],z['query'],z['families'],z['reference'],10)
        scores[ds]=ExplicitScore(cores[ds]);assert scores[ds].terms(scores[ds].ref)['total'][0]==0
    save('reference_configuration',{ds:sc.configuration() for ds,sc in scores.items()})
    final=[];graphs=[];exchange=[];start=time.time()
    for ji,job in enumerate(read(RUN/'evaluation_jobs.json')):
        ident={k:job[k] for k in ['dataset','train_seed','arm','sample_seed']};core=cores[job['dataset']];sc=scores[job['dataset']]
        path=historical(Path(job['folder'])/'graph.npz');meta=read(path.parent/'binding.json');assert base.BIND[str(path)]==meta['graph_sha256']
        with np.load(path) as z:
            assert np.array_equal(z['types'],core.types);a=np.zeros((core.n,core.n));a[z['edges'][0],z['edges'][1]]=1;a+=a.T
        q=core.query.numpy();bits=a[q[0],q[1]];rr=[r for r in edges if all(r[k]==v for k,v in ident.items())]
        out,inc,old,stats,refstats=panel(core,bits,rr,sc);final+=out
        graphs.append(dict(**ident,**{k:float(v[0]) for k,v in old.items()},**stats,**{'reference_'+k:v for k,v in refstats.items()}))
        if ji==0:
            s=time.time()-start;save('calibration',dict(first_graph_64slots_seconds=s,rough_full_seconds=s*120))
            if s*120>900-(time.time()-base.START):save('partial_final_edges',final);raise TimeoutError('panel cannot fit')
        if ident==dict(dataset='PubMed',train_seed=2026110421,arm='A',sample_seed=2026110232):
            ers=read(PREV/'exchanges.json');b=bits.copy();b[3489]=1;ai=RootedIncremental(core,b)
            for st in range(0,len(ers),16):
                limits();batch=ers[st:st+16];raw,_=ai.raw_flips([r['removed_slot'] for r in batch]);nv=sc.terms(raw)
                exchange.extend(dict(**r,explicit_change=float(nv['total'][k]-old['total'][0]),old_DIRECT_change=float(nv['old_DIRECT'][k]-old['old_DIRECT'][0])) for k,r in enumerate(batch))
        if (ji+1)%16==0:save('partial_final_edges',final);print('graphs',ji+1,'elapsed',time.time()-base.START,flush=True)
    save('final_edges',final);save('graph_scores',graphs);save('exchanges',exchange)
    for name,keys in [('final_summary',['dataset']),('final_by_seed',['dataset','train_seed']),('final_by_arm_seed',['dataset','arm','train_seed'])]:save(name,summaries(final,keys))
    conditional=[]
    for ds in cores:
        fixed=torch.load(historical(OLD/f'{ds}_fixed_inputs.pt'),map_location='cpu',weights_only=False)['inputs']
        for t,seed,inp in fixed:
            core=RootedScore(cores[ds].types,np.asarray(inp['query']),np.asarray(inp['f']),np.asarray(inp['labels']),10);sc=ExplicitScore(core)
            for parent in ['CE','PS']:
                for draw in [0,1]:
                    with np.load(historical(EDGE/f'{ds}_{parent}_CE_t{t}_draw{draw}_flips.npz')) as z:bits=(1-z['flip_sign'])/2
                    rr=[r for r in conditional_old if r['dataset']==ds and r['t']==t and r['parent_history']==parent and r['draw']==draw]
                    out,_,_,_,_=panel(core,bits,rr,sc);conditional+=out
            save('partial_conditional_edges',conditional);print(ds,t,'elapsed',time.time()-base.START,flush=True)
    save('conditional_edges',conditional);save('conditional_summary',summaries(conditional,['dataset','t']))
    typed=read(ROOT/'results/type_price_audit_20261109/per_type_edges.json');key=lambda r:tuple(r[k] for k in ['dataset','train_seed','arm','sample_seed','slot']);grouped={}
    for r in typed:grouped.setdefault(key(r),[]).append(r)
    typeout=[];mag=[]
    for ds in cores:
        for method,col in [('old_DIRECT','old_DIRECT_change'),('explicit','explicit_change')]:
            good=[r for r in final if r['dataset']==ds and r[col]<-EPS]
            for t in cores[ds].nodes:
                rr=[next(z for z in grouped[key(r)] if z['type']==t) for r in good]
                typeout.append(dict(dataset=ds,method=method,type=int(t),rewarded_flips=len(rr),**{m+'_'+sgn:sum(z[m+'_change']*sg>EPS for z in rr) for m in ['total_degree_W1','joint_degree_TV','hub_error','lower_half_degree_error'] for sgn,sg in [('worse',1),('better',-1)]}))
            for m in METRICS:
                v=np.array([r[m+'_change'] for r in good]);pos=v[v>EPS];neg=-v[v<-EPS]
                mag.append(dict(dataset=ds,method=method,metric=m,rewarded_flips=len(v),worse_count=len(pos),better_count=len(neg),
                    summed_cost=float(pos.sum()),summed_gain=float(neg.sum()),max_cost=float(pos.max()) if len(pos) else 0.,
                    limitation='fixed overlapping edits, not a trajectory or population improvement'))
    save('type_costs',typeout);save('cost_magnitudes',mag)
    assert not torch.cuda.is_initialized()
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in base.BIND.items())
    save('binding',base.BIND);save('receipt',dict(status='complete',wall_seconds=time.time()-base.START,peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        graphs=len(graphs),final_slots=len(final),conditional_slots=len(conditional),exchanges=len(exchange),GPU_hours=0,models=0,updates=0,new_random_graphs=0,assets_unchanged=True))
    print((OUT/'final_summary.json').read_text());print((OUT/'receipt.json').read_text())
if __name__=='__main__':
    try:main()
    except BaseException as e:
        if OUT.exists():save('failure',dict(error=repr(e),wall_seconds=time.time()-base.START))
        raise
