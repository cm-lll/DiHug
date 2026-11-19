"""CPU-only fixed perturbation audit of the current typed organization score.

Raw joint-degree assignment is a diagnostic ruler, not a newly adopted reward.
No torch, model, optimizer, generative chain or production mutation.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import json,csv,time,hashlib,resource,importlib.util,traceback
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from scipy.sparse.csgraph import connected_components
from scipy.stats import wasserstein_distance
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/unified_organization_audit_20261113'
PANEL=ROOT/'results/paper_ablation_20261112T190000'
spec=importlib.util.spec_from_file_location('heat_only',ROOT/'research_branches/one_step_vs_terminal_plan_20261111/heat_solver_candidate.py')
heatlib=importlib.util.module_from_spec(spec);spec.loader.exec_module(heatlib)

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(name,obj):(OUT/name).write_text(json.dumps(obj,indent=2,allow_nan=False))
def table(name,rows):
    with (OUT/name).open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
def kernel(x,y,s):return np.exp(-cdist(x,y,'sqeuclidean')/(2*s))
def mmd(a,b,c):return float((a.mean()+b.mean()-2*c.mean())/2)

class Score:
    def __init__(self,ds,path,scales):
        z=np.load(path);self.types=z['types'];self.n=len(self.types);self.schema=[tuple(map(int,p)) for p in z['schema']];self.r=len(self.schema)
        self.q=np.array(np.triu_indices(self.n,1));lookup={p:i for i,p in enumerate(self.schema)}
        self.f=np.array([lookup[tuple(sorted((int(self.types[u]),int(self.types[v]))))] for u,v in self.q.T])
        self.lookup={tuple(p):i for i,p in enumerate(self.q.T)}
        self.ref=self.bits(z['edges']);self.count=np.bincount(self.f,weights=self.ref,minlength=self.r)
        self.active=self.count[self.f]>0;self.s=scales[ds];self.rd=self.degrees(self.ref)
        self.rh,self.rf=self.features(self.ref)
        self.rawscale={int(t):np.maximum(1,self.rd[self.types==t].mean(0)) for t in np.unique(self.types)}
    def bits(self,edges):
        e=np.asarray(edges);e=e.T if e.shape[0]!=2 else e
        b=np.zeros(len(self.f),dtype=bool)
        for u,v in e.T:
            assert u!=v
            i=self.lookup[tuple(sorted((int(u),int(v))))];assert not b[i];b[i]=1
        return b
    def degrees(self,b):
        d=np.zeros((self.n,self.r),dtype=int)
        np.add.at(d,(self.q[0,b],self.f[b]),1);np.add.at(d,(self.q[1,b],self.f[b]),1)
        assert d.sum()==2*b.sum();return d
    def adjacency(self,b):
        a=np.zeros((self.n,self.n));a[self.q[0,b],self.q[1,b]]=1;a+=a.T;return a
    def features(self,b):
        h=heatlib.heat(self.adjacency(b),self.types)['feat'];d=self.degrees(b).astype(float)
        for t in np.unique(self.types):
            sel=self.types==t;d[sel]=np.log1p(d[sel])/np.array(self.s['log_degree_scales'][str(t)])/np.sqrt(self.r)
        return h,d
    def score(self,b):
        assert not np.any(b&~self.active)
        h,d=self.features(b);raw=self.degrees(b);H=[];HD=[];HH=[];W=[];LB=[];low=[];hub=[]
        for t in np.unique(self.types):
            sel=self.types==t;hb=self.s['heat_bandwidth_squared'][str(t)];db=self.s['degree_bandwidth_squared'][str(t)]
            khh=kernel(h[sel],h[sel],hb);krr=kernel(self.rh[sel],self.rh[sel],hb);khr=kernel(h[sel],self.rh[sel],hb)
            kdd=kernel(d[sel],d[sel],db);kdr=kernel(d[sel],self.rf[sel],db);krrd=kernel(self.rf[sel],self.rf[sel],db)
            H.append(mmd(khh*kdd,krr*krrd,khr*kdr));HD.append(mmd(kdd,krrd,kdr));HH.append(mmd(khh,krr,khr))
            x=raw[sel]/self.rawscale[int(t)];y=self.rd[sel]/self.rawscale[int(t)]
            cost=cdist(x,y,'cityblock');ii,jj=linear_sum_assignment(cost)
            W.append(float(cost[ii,jj].mean()));LB.append(float(abs(x.mean(0)-y.mean(0)).sum()))
            assert W[-1]+1e-10>=LB[-1]
            deg=raw[sel].sum(1);ref=self.rd[sel].sum(1)
            low.append(abs(float((deg<=1).mean()-(ref<=1).mean())))
            n=max(1,int(np.ceil(.1*len(deg))));hub.append(abs(float(np.sort(deg)[-n:].sum()-np.sort(ref)[-n:].sum()))/max(1,float(ref.sum())))
        count=np.bincount(self.f,weights=b,minlength=self.r);a=self.adjacency(b);nc,lab=connected_components(a,directed=False)
        return dict(H=float(np.mean(H)),degree_log_MMD=float(np.mean(HD)),heat_only_MMD=float(np.mean(HH)),
            N=float(np.mean((count-self.count)**2/np.maximum(1,self.count)**2)),
            raw_joint_transport=float(np.mean(W)),count_lower_bound=float(np.mean(LB)),
            edges=int(b.sum()),relation_count_L1=int(abs(count-self.count).sum()),
            degree_W1=float(np.mean([wasserstein_distance(raw[:,r],self.rd[:,r]) for r in range(self.r)])),
            low_fraction_error=float(np.mean(low)),hub_mass_error=float(np.mean(hub)),
            triangles=int(round(np.trace(a@a@a)/6)),components=int(nc),LCC=int(np.bincount(lab).max()))

def mutate(sc,kind,seed,targets):
    rng=np.random.default_rng(seed);b=sc.ref.copy();states=[];events=[];accepted=0;attempts=0
    if kind in ('add','delete'):
        eligible=np.flatnonzero((~b&sc.active) if kind=='add' else b);order=rng.permutation(eligible)
        for n in targets:
            c=b.copy();c[order[:n]]=kind=='add';states.append((n,min(n,len(order)),c))
        return states,dict(order=order[:max(targets)].tolist())
    d=sc.degrees(b)
    while accepted<max(targets) and attempts<max(targets)*1000:
        attempts+=1
        if kind=='concentrate':
            old=int(rng.choice(np.flatnonzero(b)));u,v=map(int,sc.q[:,old])
            if rng.random()<.5:u,v=v,u
            r=int(sc.f[old]);possible=np.flatnonzero((sc.types==sc.types[u])&(d[:,r]>d[u,r]))
            if not len(possible):continue
            dest=int(rng.choice(possible));new=sc.lookup.get(tuple(sorted((dest,v))))
            if new is None or b[new] or sc.f[new]!=r:continue
            b[old]=0;b[new]=1;d[u,r]-=1;d[dest,r]+=1;events.append([old,new])
        else:
            r=int(rng.choice(np.flatnonzero(sc.count>=2)));pool=np.flatnonzero(b&(sc.f==r))
            if len(pool)<2:continue
            old1,old2=map(int,rng.choice(pool,2,replace=False));u,v=map(int,sc.q[:,old1]);w,x=map(int,sc.q[:,old2])
            if sc.types[u]>sc.types[v]:u,v=v,u
            if sc.types[w]>sc.types[x]:w,x=x,w
            if sc.types[u]==sc.types[v] and rng.random()<.5:w,x=x,w
            if len({u,v,w,x})<4:continue
            n1=sc.lookup.get(tuple(sorted((u,x))));n2=sc.lookup.get(tuple(sorted((w,v))))
            if n1 is None or n2 is None or b[n1] or b[n2] or sc.f[n1]!=r or sc.f[n2]!=r:continue
            b[[old1,old2]]=0;b[[n1,n2]]=1;events.append([old1,old2,n1,n2])
        accepted+=1
        if accepted in targets:states.append((accepted,accepted,b.copy()))
    if kind=='swap':np.testing.assert_array_equal(sc.degrees(b),sc.rd)
    np.testing.assert_array_equal(np.bincount(sc.f,weights=b,minlength=sc.r),sc.count)
    for n in targets:
        if n>accepted:states.append((n,accepted,b.copy()))
    return states,dict(events=events,accepted=accepted,attempts=attempts)

def main():
    start=time.time();OUT.mkdir(exist_ok=False)
    p=json.loads((PANEL/'protocol.json').read_text());scales=json.loads((ROOT/'results/typed_heat_degree_audit_20261111/reference_scaling.json').read_text())
    seedlist=[2026111301,2026111302,2026111303];fractions=[.01,.03,.05,.10,.25]
    bindings={str(Path(__file__).resolve()):sha(__file__),str(Path(heatlib.__file__)):sha(heatlib.__file__),str(ROOT/'results/typed_heat_degree_audit_20261111/reference_scaling.json'):sha(ROOT/'results/typed_heat_degree_audit_20261111/reference_scaling.json')}
    save('protocol.json',dict(frozen_before_scoring=True,datasets=['IMDB','PubMed'],seeds=seedlist,fractions=fractions,
        kinds=['add','delete','concentrate','swap'],selection='random prefix; concentration moves a relation stub to a currently higher same-type relation-degree node; swaps preserve every node relation degree',
        scores='production H/N unchanged; raw unlogged joint-degree optimal assignment is ONLY a diagnostic ruler, no new training reward',
        checks='self/permutation invariant; matching lower bound; exact degree/edge-count conservation',
        interpretation='Synthetic reference perturbations and already-viewed 32 A/A-typed graphs, not independent final validation; no assumed monotonicity for heat or swaps',
        GPU=0,model_calls=0,bindings=bindings))
    rows=[];hist=[];events={};store={};checks={};bound_rows=[]
    with (PANEL/'per_graph.csv').open() as f:old=list(csv.DictReader(f))
    for ds,seed in [('IMDB',2026110411),('PubMed',2026110421)]:
        path=p['parents'][f'{ds}_{seed}_A']['data'];bindings[path]=sha(path);assert bindings[path]==p['bindings'][path]
        sc=Score(ds,path,scales);base=sc.score(sc.ref);assert abs(base['H'])<1e-12 and base['raw_joint_transport']==0
        rows.append(dict(dataset=ds,kind='reference',seed=0,requested=0,accepted=0,**base))
        # Verify the production statistic is anonymous within each declared type.
        rng=np.random.default_rng(2026111399);perm=np.arange(sc.n)
        for t in np.unique(sc.types):sel=np.flatnonzero(sc.types==t);perm[sel]=rng.permutation(sel)
        pp=sc.bits(perm[sc.q[:,sc.ref]]);pr=sc.score(pp)
        assert abs(pr['H'])<1e-10 and pr['raw_joint_transport']==0
        checks[ds]=dict(self_H=base['H'],permuted_H=pr['H'],permuted_raw_transport=pr['raw_joint_transport'])
        targets=[max(1,int(round(float(sc.ref.sum())*f))) for f in fractions]
        for sd in seedlist:
            for kind in ['add','delete','concentrate','swap']:
                states,ev=mutate(sc,kind,sd,targets);events[f'{ds}_{kind}_{sd}']=ev
                for requested,accepted,b in states:
                    key=f'{ds}_{kind}_{sd}_{requested}';store[key]=np.packbits(b)
                    rows.append(dict(dataset=ds,kind=kind,seed=sd,requested=requested,accepted=accepted,**sc.score(b)))
        entries=[v for v in p['reused_graphs'] if v['dataset']==ds and v['arm']=='A']
        for job in p['jobs']:
            if job['dataset']==ds and job['architecture']=='A' and job['reward']=='typed':
                for ss in p['sample_seeds']:entries.append(dict(dataset=ds,train_seed=job['train_seed'],arm='A_typed',sample_seed=ss,path=str(PANEL/job['id']/f'{ds}_T_eager'/f'final_sample_{ss}'/'graph.npz')))
        for ent in entries:
            bindings[ent['path']]=sha(ent['path']);z=np.load(ent['path']);assert np.array_equal(z['types'],sc.types)
            val=sc.score(sc.bits(z['edges']));match=[v for v in old if v['dataset']==ds and v['arm']==ent['arm'] and v['view']=='raw' and int(v['train_seed'])==ent['train_seed'] and int(v['sample_seed'])==ent['sample_seed']]
            assert len(match)==1;diff=abs(float(match[0]['typed_joint_score'])-val['H']);assert diff<1e-10
            hist.append(dict(dataset=ds,train_seed=ent['train_seed'],arm=ent['arm'],sample_seed=ent['sample_seed'],production_H_difference=diff,path=ent['path'],**val))
        # Per-type mean relation degree is a deterministic function of counts.
        for t in np.unique(sc.types):
            n=int((sc.types==t).sum())
            for r,(a,b) in enumerate(sc.schema):
                factor=2 if a==b==t else int(t==a or t==b)
                assert abs(sc.rd[sc.types==t,r].mean()-factor*sc.count[r]/n)<1e-12
                bound_rows.append(dict(dataset=ds,type=int(t),relation=r,nodes=n,incidence_factor=factor,scale=float(sc.rawscale[int(t)][r])))
    table('perturbations.csv',rows);table('existing_graphs.csv',hist);table('count_bound_constants.csv',bound_rows)
    save('events.json',events);save('checks.json',checks);save('bindings.json',bindings);np.savez_compressed(OUT/'cpu_perturbation_bits.npz',**store)
    for path,h in bindings.items():assert sha(path)==h
    save('receipt.json',dict(status='complete',seconds=time.time()-start,perturbations=len(store),reference_graphs=2,existing_graphs=len(hist),model_calls=0,GPU_hours=0,optimizer_steps=0,new_model_generated_graphs=0,CPU_constructed_states=len(store),threads=1,peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024))
    print((OUT/'receipt.json').read_text())

if __name__=='__main__':
    try:main()
    except BaseException:
        if OUT.exists():save('failure.json',dict(traceback=traceback.format_exc()))
        raise
