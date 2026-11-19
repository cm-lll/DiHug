"""Exact joint correspondence as a diagnostic of the frozen projected score.
No model imports/calls, no loss fitting and no new graph sampling.
"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,json,csv,time,resource,hashlib,datetime,itertools
from pathlib import Path
import numpy as np
import torch
import scipy
from scipy.optimize import linear_sum_assignment
torch.set_num_threads(1)
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'research_branches/rooted_connection_20261109'))
sys.path.insert(0,str(ROOT/'research_branches/rooted_quantile_cpu_20261109'))
sys.path.insert(0,str(ROOT/'research_branches/count_shape_cpu_20261109'))
from score import RootedScore,RootedIncremental
from score_candidate import CountShapeScore
RUN=ROOT/'results/rooted_connection_20261109T015142';PREV=ROOT/'results/count_shape_cpu_20261109'
OLD=ROOT/'results/hard_structure_response_20261108T182154';EDGE=ROOT/'results/hard_structure_edge_direction_cpu_20261108'
OUT=ROOT/'results/allocation_matching_audit_20261109';START=time.time();BIND={};EPS=1e-12
METRICS=['total_degree_W1','relation_degree_W1','joint_degree_TV','node_triangle_W1','root_path_W1','hub_error','triangle_error','edge_error']
def bind(p):
    p=Path(p);BIND[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest();return p
def read(p):return json.loads(bind(p).read_text())
def save(n,x):
    (OUT/(n+'.json')).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False))
    if isinstance(x,list) and x:
        with (OUT/(n+'.csv')).open('w') as f:
            w=csv.DictWriter(f,list(x[0]));w.writeheader();w.writerows(x)
def limits():
    assert time.time()-START<600,'CPU window exceeded'
    assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024<2048,'RSS exceeded'
def assignment(x,y):
    cost=((x[:,None,:]-y[None,:,:])**2).sum(-1)
    r,c=linear_sum_assignment(cost)
    return float(cost[r,c].mean())

class JointAudit:
    def __init__(self,core):
        self.base=CountShapeScore(core);self.cache={}
        for t,c in self.base.cache.items():
            d=len(c['cols']);w=np.concatenate([np.eye(d)*np.sqrt(.5/d),c['u']*np.sqrt(.5/c['u'].shape[1])],axis=1)
            ref=core.features(core.reference).numpy()[c['ids']][:,c['mask']]/c['scale']
            centered=ref-ref.mean(0)
            self.cache[t]=dict(**c,w=w,target_full=centered@w)
    def terms(self,raw):
        raw=raw[None] if raw.ndim==2 else raw;total=np.zeros(len(raw));typ={}
        for t,c in self.cache.items():
            z=raw[:,c['ids']][:,:,c['mask']]/c['scale'];mu=z.mean(1);center=z-mu[:,None]
            val=np.array([assignment(x@c['w'],c['target_full']) for x in center])
            typ[t]=val/len(self.base.core.nodes);total+=typ[t]
        old=self.base.terms(raw)
        gap=total-old['shape'];tol=1e-9*np.maximum(1.,total)
        assert np.all(gap>=-tol),'joint matching must bound separately matched projections'
        return dict(total=old['mass']+total,mass=old['mass'],shape=total,gap=gap),typ

def finite():
    # Exhaustive independent assignment check for 4 rows, not a model experiment.
    x=np.array([[0.,1.,3.],[1.,3.,0.],[4.,0.,1.],[2.,1.,2.]])
    y=np.array([[1.,0.,2.],[2.,2.,0.],[3.,0.,2.],[0.,3.,1.]])
    brute=min(float(((x-y[list(p)])**2).sum(1).mean()) for p in itertools.permutations(range(4)))
    assert abs(assignment(x,y)-brute)<1e-12
    ty=np.array([0,0,1,1]);q=np.array(list(itertools.combinations(range(4),2))).T;f=ty[q[0]]+ty[q[1]];ref=np.array([1.,1.,0.,1.,0.,1.])
    core=RootedScore(ty,q,f,ref,3);audit=JointAudit(core);states=((np.arange(64)[:,None]>>np.arange(6))&1).astype(float)
    raw=core.features(torch.from_numpy(states)).numpy();vals=audit.terms(raw)[0]['total'];diff=[]
    for i,b in enumerate(states):
        inc=RootedIncremental(core,b);raws,_=inc.raw_flips(np.arange(6));dd=audit.terms(raws)[0]['total']-vals[i]
        assert np.max(abs(dd-(vals[i^(1<<np.arange(6))]-vals[i])))<1e-11
        diff.append(dd*(1-2*b))
    p=np.array([.1,.25,.45,.6,.75,.9]);law=np.prod(np.where(states==1,p,1-p),axis=1)
    a=(law[:,None]*vals[:,None]*(states-p)).sum(0);b=(law[:,None]*np.array(diff)*p*(1-p)).sum(0)
    assert np.max(abs(a-b))<1e-11
    save('finite_checks',dict(exhaustive_permutations=24,assignment_matches_brute=True,micrographs=64,flips=384,gradient_error=float(np.max(abs(a-b))),lower_bound_all=True))

def summarise(rs,keys):
    groups={}
    for r in rs:groups.setdefault(tuple(r[k] for k in keys),[]).append(r)
    out=[]
    for key,rows in groups.items():
        for method,col in [('projected','candidate_change'),('joint_reference','joint_change')]:
            acc={k:0. for k in ['total','rewarded','triangle_worse','three_worse','no_aggregate_gain','pareto_better','pareto_rewarded']}
            for r in rows:
                w=r.get('stratum_weight',1.);good=r[col]<-EPS;v=[r[k+'_change'] for k in METRICS];pareto=all(z<=EPS for z in v) and any(z<-EPS for z in v)
                acc['total']+=w;acc['rewarded']+=w*good;acc['triangle_worse']+=w*(good and r['triangle_error_change']>EPS)
                acc['three_worse']+=w*(good and all(r[k+'_change']>EPS for k in ['total_degree_W1','joint_degree_TV','triangle_error']))
                acc['no_aggregate_gain']+=w*(good and all(z>=-EPS for z in v) and any(z>EPS for z in v))
                acc['pareto_better']+=w*pareto;acc['pareto_rewarded']+=w*(good and pareto)
            out.append(dict(**dict(zip(keys,key)),method=method,**acc))
    return out
def panel(core,bits,rows,audit):
    inc=RootedIncremental(core,bits);base,_=audit.terms(inc.raw);old=audit.base.terms(inc.raw);ix=np.array([r['slot'] for r in rows]);out=[]
    for start in range(0,len(ix),16):
        limits();ids=ix[start:start+16];raw,_=inc.raw_flips(ids);new,_=audit.terms(raw);proj=audit.base.terms(raw)
        for k,r in enumerate(rows[start:start+16]):
            assert abs(proj['total'][k]-old['total'][0]-r['candidate_change'])<1e-8*max(1.,abs(r['candidate_change']))
            change=float(new['total'][k]-base['total'][0]);gap_delta=float(new['gap'][k]-base['gap'][0])
            assert abs(change-r['candidate_change']-gap_delta)<1e-8*max(1.,abs(change))
            out.append(dict(**r,joint_change=change,projection_gap_change=gap_delta,projection_gap_before=float(base['gap'][0]),projection_gap_after=float(new['gap'][k])))
    return out,base
def main():
    OUT.mkdir(exist_ok=False)
    save('protocol',dict(start_BJT=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        task='diagnose independent projected matchings against exact shared joint correspondence',
        invariant='unchanged raw features, centered transform, reference scales, means, .5 axis/.5 32 mixed direction metric; exact linear assignment used only as diagnostic reference',
        fixed_panel='all64 final graphs4096 slots,24 prior conditional graphs2765slots; same selections and comparisons; no resampling',
        cap=dict(worker_seconds=600,threads=1,RSS_MiB=2048,GPU=0),
        calibration='first complete64-slot final graph included; complete panel only if linear cost forecast plus180sec <remaining; otherwise report incomplete no cherry-picked reductions',
        interpretation='shared anonymous correspondence is not original node ID; no graph edit generation/architecture swap; not proof all metrics should improve',
        decision='if exact joint still has adverse rewards, no projection-count search or automatic training; report both improvements and losses',
        scipy=scipy.__version__,sign_tolerance=EPS))
    for p in [Path(__file__),ROOT/'research_branches/count_shape_cpu_20261109/score_candidate.py',ROOT/'research_branches/rooted_connection_20261109/score.py']:bind(p)
    finite();prev=read(PREV/'binding.json');edges=read(PREV/'final_edges.json');conds=read(PREV/'conditional_edges.json')
    def historical(p):
        p=bind(p);assert str(p) in prev and BIND[str(p)]==prev[str(p)];return p
    cores={};audits={}
    for ds in ['IMDB','PubMed']:
        with np.load(historical(RUN/f'{ds}_cpu_cost_panel.npz')) as z:cores[ds]=RootedScore(z['types'],z['query'],z['families'],z['reference'],10)
        audits[ds]=JointAudit(cores[ds])
    final=[];graphrows=[]
    for ji,j in enumerate(read(RUN/'evaluation_jobs.json')):
        limits();ident={k:j[k] for k in ['dataset','train_seed','arm','sample_seed']};core=cores[j['dataset']]
        p=historical(Path(j['folder'])/'graph.npz');meta=read(p.parent/'binding.json');assert BIND[str(p)]==meta['graph_sha256']
        with np.load(p) as z:
            assert np.array_equal(z['types'],core.types);adj=np.zeros((core.n,core.n));adj[z['edges'][0],z['edges'][1]]=1;adj+=adj.T
        q=core.query.numpy();rs=[r for r in edges if all(r[k]==v for k,v in ident.items())];tick=time.time()
        out,b=panel(core,adj[q[0],q[1]],rs,audits[j['dataset']]);elapsed=time.time()-tick;final+=out
        graphrows.append(dict(**ident,slots=len(rs),CPU_seconds=elapsed,joint_score=float(b['total'][0]),projection_gap=float(b['gap'][0])))
        if ji==0:
            predicted=elapsed/len(rs)*(4096+2765);remaining=600-(time.time()-START)
            save('cost_calibration',dict(first_graph_seconds=elapsed,first_slots=len(rs),full_slot_linear_prediction_seconds=predicted,additional_margin_seconds=180,remaining_seconds=remaining))
            assert predicted+180<remaining,'full panel budget not supported'
    save('final_edges',final);save('graph_costs',graphrows);save('final_summary',summarise(final,['dataset']));save('final_by_seed',summarise(final,['dataset','train_seed']))
    print('final64 complete',flush=True)
    conditional=[]
    for ds in cores:
        fixed=torch.load(historical(OLD/f'{ds}_fixed_inputs.pt'),map_location='cpu',weights_only=False)['inputs']
        for t,seed,inp in fixed:
            core=RootedScore(cores[ds].types,np.asarray(inp['query']),np.asarray(inp['f']),np.asarray(inp['labels']),10);audit=JointAudit(core)
            for parent in ['CE','PS']:
                for draw in [0,1]:
                    limits();p=historical(EDGE/f'{ds}_{parent}_CE_t{t}_draw{draw}_flips.npz')
                    with np.load(p) as z:bits=(1-z['flip_sign'])/2
                    rs=[r for r in conds if r['dataset']==ds and r['t']==t and r['parent_history']==parent and r['draw']==draw]
                    out,_=panel(core,bits,rs,audit);conditional+=out
    save('conditional_edges',conditional);save('conditional_summary',summarise(conditional,['dataset','t']))
    selected=read(PREV/'new_adverse_operations.json');chosen=[next(r for r in final if all(r[k]==s[k] for k in ['dataset','train_seed','arm','sample_seed','slot'])) for s in selected]
    save('all24_previous_adverse_cases',chosen)
    save('adverse_case_summary',[dict(dataset=ds,cases=len(rr),still_rewarded=sum(r['joint_change']<-EPS for r in rr),no_longer_rewarded=sum(r['joint_change']>=-EPS for r in rr)) for ds in cores for rr in [[r for r in chosen if r['dataset']==ds]]])
    assert not torch.cuda.is_initialized();assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in BIND.items())
    save('binding',BIND);save('receipt',dict(status='complete',wall_seconds=time.time()-START,peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        final_graphs=len(graphrows),final_slots=len(final),conditional_slots=len(conditional),GPU_hours=0,models=0,updates=0,new_random_graphs=0,assets_unchanged=True))
    print((OUT/'final_summary.json').read_text());print((OUT/'adverse_case_summary.json').read_text());print((OUT/'receipt.json').read_text())
if __name__=='__main__':
    try:main()
    except BaseException as e:
        if OUT.exists():save('failure',dict(error=repr(e),wall_seconds=time.time()-START))
        raise
