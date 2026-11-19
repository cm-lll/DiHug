"""Archived PubMed exact scoring cost/support audit; no denoising model."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,json,csv,time,resource,signal,hashlib,importlib.util
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/structure_posterior_real_cost_20261109'
START=time.monotonic();BIND={}
def save(name,obj):
    (OUT/(name+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    if isinstance(obj,list) and obj:
        with (OUT/(name+'.csv')).open('w',newline='') as f:
            w=csv.DictWriter(f,list(obj[0]));w.writeheader();w.writerows(obj)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p,expected=None):
    p=Path(p);h=sha(p)
    if expected is not None:assert expected[str(p)]==h,str(p)
    BIND[str(p)]=h;return p
def limit():
    if time.monotonic()-START>540:raise TimeoutError('540 second science limit')
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024>2048:raise MemoryError('2GiB RSS limit')
def load(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

class Cache:
    """Exact independent flips from one frozen base; not a sequential chain cache."""
    def __init__(self,sc,jd,raw,adj,q,f,types):
        self.sc=sc;self.jd=jd;self.raw=raw;self.a=adj;self.q=q;self.f=f;self.types=types
        self.channels=np.full((4,4),-1,int)
        for k,(a,b) in enumerate(sc.core.triangle_pairs):self.channels[a,b]=self.channels[b,a]=10+k
        self.costs={};self.degree={};self.triangle={}
        for t,c in jd.cache.items():
            x=raw[c['ids']][:,c['cols']]/c['scale']
            self.costs[t]=cdist(x,c['z'],metric='cityblock')/x.shape[1]
            self.degree[t]=self.degree_terms(t,raw,self.costs[t]);self.triangle[t]=self.tri_term(t,raw)
        self.tc=sc.triangle_counts(raw)[0];self.base=self.assemble(self.degree,self.triangle,self.tc)
    def degree_terms(self,t,raw,cost):
        c=self.sc.cache[t];d=raw[c['ids'],:10];sd=np.sort(d.sum(1));diff=abs(sd-c['sorted_degree'])
        i,j=linear_sum_assignment(cost);joint=float(cost[i,j].mean())
        marginal=float((abs(np.sort(d[:,c['dm']],axis=0)-c['rd'])/c['relation_scale']).mean())
        low=float(diff[:c['k']].mean()/c['low_scale']);hub=float(diff[-min(2,len(sd)):].mean()/c['hub_scale'])
        return joint/8+marginal/8+low/4+hub/4
    def tri_term(self,t,raw):
        c=self.sc.cache[t];tr=raw[c['ids'],10:self.sc.core.base_width][:,c['tm']]
        return float((abs(np.sort(tr,axis=0)-c['rt'])/c['tri_scale']).mean())/8 if tr.shape[1] else 0.
    def assemble(self,degree,tri,tc):
        return (sum(degree.values())+sum(tri.values()))/len(degree)+float(abs(tc-self.sc.reftri).sum())/self.sc.triangle_scale/8
    def flip(self,slot):
        tick=time.perf_counter();u,v=map(int,self.q[:,slot]);r=int(self.f[slot]);sign=-1 if self.a[u,v] else 1
        common=np.flatnonzero(self.a[u]&self.a[v]);raw=self.raw.copy();raw[u,r]+=sign;raw[v,r]+=sign
        for w in common:
            raw[u,self.channels[self.types[v],self.types[w]]]+=sign
            raw[v,self.channels[self.types[u],self.types[w]]]+=sign
            raw[w,self.channels[self.types[u],self.types[v]]]+=sign
        feature_sec=time.perf_counter()-tick;degree=dict(self.degree);tri=dict(self.triangle);tc=self.tc.copy()
        cost_sec=solve_sec=0.
        for t in set(map(int,self.types[[u,v]])):
            c=self.jd.cache[t];affected=np.flatnonzero(np.isin(c['ids'],[u,v]));cost=self.costs[t].copy()
            tick=time.perf_counter();new=raw[c['ids'][affected]][:,c['cols']]/c['scale']
            cost[affected]=cdist(new,c['z'],metric='cityblock')/new.shape[1];cost_sec+=time.perf_counter()-tick
            tick=time.perf_counter();degree[t]=self.degree_terms(t,raw,cost);solve_sec+=time.perf_counter()-tick
        tick=time.perf_counter();changed_types=set(map(int,self.types[np.r_[u,v,common]])) if len(common) else set()
        for t in changed_types:tri[t]=self.tri_term(t,raw)
        for w in common:tc[self.sc.class_index[tuple(sorted(map(int,self.types[[u,v,w]])))]]+=sign
        value=self.assemble(degree,tri,tc);other_sec=time.perf_counter()-tick
        assert raw.min()>=0 and tc.min()>=0
        return value,raw,dict(feature_seconds=feature_sec,cost_rows_seconds=cost_sec,solve_and_degree_seconds=solve_sec,
            triangle_and_reduce_seconds=other_sec,degree_types=len(set(self.types[[u,v]])),triangle_types=len(changed_types),common_neighbors=len(common))

def main():
    OUT.mkdir(parents=True,exist_ok=False);signal.alarm(600)
    save('protocol',dict(start_utc=datetime.now(timezone.utc).isoformat(),
        scope='4 archived PubMed0422 raw2040 t1000 conditional hard backgrounds, CE/PS parents draws0/1; no posterior chains or network',
        score='same existing six-term score: joint_transport/8 + relation_W1/8 + low/4 + hub/4 + typed_triangle_L1/8 + node_triangle_W1/8',
        distinction='Real score cost benchmark, not previous toy Gibbs energy or adoption of a production posterior. Connectivity evaluation only, no new penalty.',
        selection='each background/mutable relation: up to8 present+8 absent slots uniform without replacement; fixed SHA seed; no score selection',
        seed_rule='SHA256 real_posterior_cost_v1|state|relation|present first8bytes little-endian',
        checks='all selected flips full feature+score versus incremental and old saved deltas;3 timing passes with alternating order',
        limits=dict(threads=1,science_seconds=540,hard_seconds=600,RSS_MiB=2048,GPU=0),
        preparation_note='First apply_patch rejected malformed patch before file creation; zero scientific work, original sources unaffected'))
    H=load('helpers_real_cost',ROOT/'research_branches/pair_allocation_signal_20261109/run_cpu.py')
    torch=H.torch;torch.set_num_threads(1)
    prev=json.loads((ROOT/'results/pair_allocation_signal_20261109/binding.json').read_text())
    done=json.loads((H.FULL/'completion.json').read_text())
    with np.load(bind(ROOT/'results/rooted_connection_20261109T015142/PubMed_cpu_cost_panel.npz',prev)) as z:
        types=z['types'].copy();fq=z['query'].copy();ff=z['families'].copy();fy=z['reference'].copy()
    saved=torch.load(bind(H.OLD/'PubMed_fixed_inputs.pt',prev),map_location='cpu',weights_only=False)
    inp=next(x for t,s,x in saved['inputs'] if t==1000);q,f,y=map(np.asarray,(inp['query'],inp['f'],inp['labels']))
    n=len(types);assert n==192 and len(y)==18336 and y.sum()==926
    o=np.argsort(q[0]*n+q[1]);fo=np.argsort(fq[0]*n+fq[1])
    assert np.array_equal(q[:,o],fq[:,fo]) and np.array_equal(f[o],ff[fo]) and np.array_equal(y[o],fy[fo])
    package=ROOT/'research_branches/cn_multiseed_20261104/research_branches/typed_structure_early_cn_expansion_review/experiments/typed_structure_early_cn_expansion_v1/package'
    cfg=json.loads(bind(package/'config.json').read_text());asset=bind(package/'assets/PubMed.npz')
    assert sha(asset)==cfg['datasets']['PubMed']['sha256']
    with np.load(asset) as z:
        assert np.array_equal(z['types'],types);schema=z['schema'].copy();clean=z['edges'].copy()
    lookup={tuple(map(int,p)):r for r,p in enumerate(schema)}
    expected=np.array([lookup.get(tuple(sorted(map(int,types[[u,v]]))),-1) for u,v in q.T])
    assert np.array_equal(f,expected) and np.all(q[0]<q[1]) and len(set(map(tuple,q.T)))==len(y)
    target=np.zeros((n,n),bool);target[clean[:,0],clean[:,1]]=1;target|=target.T
    assert np.array_equal(y,target[q[0],q[1]])
    counts=np.bincount(f,weights=y,minlength=10);slots=np.bincount(f,minlength=10);prior=counts/slots
    for p in [package/'hsgd/bw_preflight/common.py',package/'hsgd/kernel.py']:bind(p)
    support=[dict(relation=r,endpoint_types=schema[r].tolist(),name='-'.join(cfg['datasets']['PubMed']['type_names'][t] for t in schema[r]),
        slots=int(slots[r]),reference_edges=int(counts[r]),prior=float(prior[r]),fixed_zero=bool(prior[r]==0),fixed_one=bool(prior[r]==1)) for r in range(10)]
    save('support',support);states=[];plans=[]
    for parent in ['CE','PS']:
        for draw in [0,1]:
            tag=f'PubMed_{parent}_CE_t1000_draw{draw}'
            with np.load(bind(H.FULL/(tag+'.npz'),done['output_hashes'])) as z:
                b=z['bits'].copy();old=z['new_delta'].copy();assert np.array_equal(z['reference'],y) and np.array_equal(z['relation'],f)
            assert np.isin(b,[0,1]).all() and np.all(b[prior[f]==0]==0) and np.all(b[prior[f]==1]==1)
            selected=[]
            for r in range(10):
                if prior[r] in (0,1):continue
                for present in [0,1]:
                    candidates=np.flatnonzero((f==r)&(b==present));seed=int.from_bytes(hashlib.sha256(f'real_posterior_cost_v1|{tag}|{r}|{present}'.encode()).digest()[:8],'little')
                    ix=np.random.default_rng(seed).choice(candidates,size=min(8,len(candidates)),replace=False)
                    selected.extend(map(int,ix));plans.append(dict(state=tag,relation=r,present=present,seed=seed,population=len(candidates),slots=list(map(int,ix))))
            states.append((tag,b,old,selected))
    save('frozen_slots',plans)
    core=H.RootedScore(types,q,f,y,10);sc=H.ExplicitScore(core);jd=H.jdmod.JointDegree(dict(types=types,query=q,families=f,reference=y))
    def full_score(raw):
        a=sc.terms(raw);j,_=jd.terms(raw[:,:10]);return float(a['total'][0]+(j['joint']-a['joint_TV'][0])/8)
    for mod in list(sys.modules.values()):
        p=getattr(mod,'__file__',None)
        if p and str(p).startswith(str(ROOT)) and str(p).endswith('.py'):bind(p)
    bind(__file__);save('binding',BIND);save('score_configuration',sc.configuration())
    checks=[];bench=[];base_rows=[]
    for tag,b,old,selected in states:
        limit();tick=time.perf_counter()
        with torch.no_grad():raw=core.features(torch.from_numpy(b)).numpy()[:,:core.base_width]
        adj=H.adjacency(q,b,n);cache=Cache(sc,jd,raw,adj,q,f,types);setup=time.perf_counter()-tick
        base=full_score(raw);assert abs(base-cache.base)<1e-12
        base_rows.append(dict(state=tag,edges=int(b.sum()),score=base,triangles=float(sc.triangle_counts(raw).sum()),cache_setup_seconds=setup,slots_selected=len(selected)))
        for ix in selected:
            limit();value,rr,detail=cache.flip(ix);bb=b.copy();bb[ix]=1-bb[ix]
            with torch.no_grad():oracle=core.features(torch.from_numpy(bb)).numpy()[:,:core.base_width]
            expected=full_score(oracle);fe=float(abs(rr-oracle).max());se=abs(value-expected);oe=abs(value-base-old[ix])
            assert fe==0 and se<1e-11 and oe<1e-10
            checks.append(dict(state=tag,slot=ix,relation=int(f[ix]),before=int(b[ix]),score_change=value-base,
                full_feature_max_difference=fe,full_score_difference=se,old_saved_delta_difference=oe,
                changed_node_type_count=detail['degree_types'],common_neighbors=detail['common_neighbors']))
        for rep in range(3):
            for mode in (['full','incremental'] if rep%2==0 else ['incremental','full']):
                limit();tick=time.perf_counter();values=[];details=[]
                for ix in selected:
                    if mode=='full':
                        bb=b.copy();bb[ix]=1-bb[ix]
                        with torch.no_grad():rr=core.features(torch.from_numpy(bb)).numpy()[:,:core.base_width]
                        values.append(full_score(rr))
                    else:
                        value,rr,detail=cache.flip(ix);values.append(value);details.append(detail)
                elapsed=time.perf_counter()-tick;assert np.allclose(np.array(values)-base,old[selected],atol=1e-10,rtol=0)
                bench.append(dict(state=tag,replicate=rep,mode=mode,flips=len(selected),seconds=elapsed,seconds_per_flip=elapsed/len(selected),
                    feature_seconds=sum(d['feature_seconds'] for d in details) if details else None,
                    cost_rows_seconds=sum(d['cost_rows_seconds'] for d in details) if details else None,
                    solve_and_degree_seconds=sum(d['solve_and_degree_seconds'] for d in details) if details else None,
                    triangle_and_reduce_seconds=sum(d['triangle_and_reduce_seconds'] for d in details) if details else None))
        save('checks',checks);save('benchmarks',bench);save('bases',base_rows)
    kernel=load('production_kernel_only',package/'hsgd/kernel.py')
    k=kernel.MarginalKernel(np.array([0.,1.,.2]),np.array([1.,.5,0.]))
    valid=k.reverse(np.array([0.,1.,.3]),np.array([0,1,0]),np.arange(3),0,1);assert valid[0]==0 and valid[1]==1
    rejected=0
    for p,x in [(np.array([.1,1.,.3]),np.array([0,1,0])),(np.array([0.,1.,.3]),np.array([1,1,0]))]:
        try:k.reverse(p,x,np.arange(3),0,1)
        except ValueError:rejected+=1
    assert rejected==2
    save('support_tests',dict(production_kernel_valid_fixed_states=True,invalid_cases_rejected=rejected,
        actual_PubMed_fixed_zero=int(np.sum(prior==0)),actual_PubMed_fixed_one=int(np.sum(prior==1)),
        prior_definition='reference relation counts / legal slot counts; verified common.make_case',
        teacher_boundary='fixed support forces clean bit, not an epsilon; impossible observed support must fail',
        t_terminal='alpha=0 likelihood independent of clean bit; structural prior/base measure determines clean posterior, not corruption prior by itself'))
    for p,h in BIND.items():assert sha(p)==h
    fast=[r for r in bench if r['mode']=='incremental'];slow=[r for r in bench if r['mode']=='full']
    seconds=sum(r['seconds'] for r in fast)/sum(r['flips'] for r in fast)
    save('budget_extrapolation',dict(incremental_seconds_per_flip=seconds,full_seconds_per_flip=sum(r['seconds'] for r in slow)/sum(r['flips'] for r in slow),
        full_support_slots=len(y),one_background_all_slot_seconds=seconds*len(y),hundred_background_all_slot_hours=seconds*len(y)*100/3600,
        teacher_3000_inputs_hundred_background_CPU_hours=seconds*len(y)*100*3000/3600,
        limitations='Arithmetic extrapolation, balanced slots not population-unbiased timings; excludes mixing, proposals, sequential cache commits, labels, model, I/O. Frozen base reused.',
        critical='No production teacher/temperature selected. This score has NO connectivity penalty. Exact assignment still globally resolved within changed types.'))
    save('receipt',dict(status='complete_stopped',seconds=time.monotonic()-START,RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        states=4,selected_flips=len(checks),timed_flips=sum(r['flips'] for r in bench),max_feature_difference=max(r['full_feature_max_difference'] for r in checks),
        max_score_difference=max(r['full_score_difference'] for r in checks),max_old_saved_delta_difference=max(r['old_saved_delta_difference'] for r in checks),
        CPU_threads=torch.get_num_threads(),GPU_hours=0,model_calls=0,training_updates=0,new_posterior_chain_steps=0,new_production_graphs=0,
        formal_training_authorized=False,unchanged_assets=True))
if __name__=='__main__':
    try:main()
    except BaseException as e:
        if OUT.exists():save('failure',dict(error=repr(e),seconds=time.monotonic()-START))
        raise
