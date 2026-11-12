"""Bounded independent hard-flip audit; no network, teacher chain or target fitting."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import csv, hashlib, itertools, json, platform, resource, signal, time
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/component_target_cpu_20261109/verified'
BIND={}; START=time.monotonic()
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p, expected=None):
    p=Path(p); h=sha(p)
    if expected: assert h==expected,str(p)
    BIND[str(p)]=h; return p
def save(name, obj):
    (OUT/(name+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    if isinstance(obj,list) and obj and isinstance(obj[0],dict):
        with (OUT/(name+'.csv')).open('w',newline='') as f:
            w=csv.DictWriter(f,list(obj[0])); w.writeheader();w.writerows(obj)
def load(p):
    with np.load(p) as z:
        t=z['types'].copy();s=z['schema'].copy();e=z['edges'].copy()
        # Reference NPZ relation labels reserve 0 for no-edge; generated families are zero-based.
        ef=z['relations'].copy()-1 if 'relations' in z else z['families'].copy()
    if e.shape[0]!=2:e=e.T
    pairs=[tuple(sorted(map(int,p))) for p in e.T]
    assert len(set(pairs))==len(pairs) and all(u!=v for u,v in pairs)
    for (u,v),r in zip(pairs,ef):assert tuple(sorted((int(t[u]),int(t[v]))))==tuple(s[r])
    a=np.zeros((len(t),len(t)),bool);a[e[0],e[1]]=True;a|=a.T
    return t,s,a
def bfs(a):
    lab=np.full(len(a),-1);cid=0
    for u in range(len(a)):
        if lab[u]>=0:continue
        stack=[u];lab[u]=cid
        while stack:
            v=stack.pop();new=np.flatnonzero(a[v]&(lab<0));lab[new]=cid;stack.extend(new)
        cid+=1
    return lab
def union(a):
    parent=list(range(len(a)))
    def root(u):
        while parent[u]!=u:parent[u]=parent[parent[u]];u=parent[u]
        return u
    for u,v in zip(*np.nonzero(np.triu(a,1))):
        x,y=root(int(u)),root(int(v));parent[x]=y
    return np.array([root(u) for u in range(len(a))])
def descriptor(t,lab):
    inventory=np.bincount(t);x=np.empty((len(t),len(inventory)))
    for c in np.unique(lab):
        ids=np.flatnonzero(lab==c);x[ids]=np.bincount(t[ids],minlength=len(inventory))/inventory
    return x
def dist(x,y,t):
    vals=[]
    for k in np.unique(t):
        ix=t==k;cost=cdist(x[ix],y[ix],metric='cityblock')/x.shape[1]
        i,j=linear_sum_assignment(cost);vals.append(cost[i,j].mean())
    return float(np.mean(vals))
def changed_labels(a,lab,u,v,present):
    out=lab.copy()
    if not present:
        if lab[u]!=lab[v]:out[lab==lab[v]]=lab[u]
        return out
    seen={u};stack=[u]
    while stack:
        n=stack.pop()
        for m in np.flatnonzero(a[n]):
            m=int(m)
            if {n,m}=={u,v} or m in seen:continue
            seen.add(m);stack.append(m)
    if v not in seen:out[list(seen)]=int(lab.max())+1
    return out
PAIRS=list(itertools.combinations_with_replacement(range(4),2))
TRIPLES=list(itertools.combinations_with_replacement(range(4),3))
PI={p:i for i,p in enumerate(PAIRS)};TI={p:i for i,p in enumerate(TRIPLES)}
def features(a,t,s):
    d=np.zeros((len(t),len(s)));tr=np.zeros((len(t),len(PAIRS)));tc=np.zeros(len(TRIPLES))
    rm={tuple(p):r for r,p in enumerate(s)}
    for u,v in zip(*np.nonzero(np.triu(a,1))):
        r=rm[tuple(sorted((t[u],t[v])))];d[u,r]+=1;d[v,r]+=1
        for w in np.flatnonzero(a[u]&a[v]&(np.arange(len(t))>v)):
            tc[TI[tuple(sorted((t[u],t[v],t[w])))]]+=1
            for n,i,j in ((u,v,w),(v,u,w),(w,u,v)):tr[n,PI[tuple(sorted((t[i],t[j])))]]+=1
    return d,tr,tc
def terms(f,ref,t,s):
    d,tr,tc=f;rd,rt,rtc=ref
    out={k:0. for k in ('joint','relation_W1','low','hub','node_triangle_W1')}
    for k in np.unique(t):
        ids=np.flatnonzero(t==k);cols=np.flatnonzero((s==k).any(1))
        x=d[ids][:,cols];r=rd[ids][:,cols];scale=np.maximum(1.,r.mean(0))
        cost=cdist(x/scale,r/scale,metric='cityblock')/len(cols);i,j=linear_sum_assignment(cost)
        out['joint']+=cost[i,j].mean()/4
        out['relation_W1']+=(abs(np.sort(x,axis=0)-np.sort(r,axis=0))/scale).mean()/4
        sr=np.sort(rd[ids].sum(1));diff=abs(np.sort(d[ids].sum(1))-sr);n=max(1,len(ids)//2)
        out['low']+=diff[:n].mean()/max(1.,sr[:n].mean())/4
        out['hub']+=diff[-2:].mean()/max(1.,sr[-2:].mean())/4
        out['node_triangle_W1']+=(abs(np.sort(tr[ids],axis=0)-np.sort(rt[ids],axis=0))/np.maximum(1.,rt[ids].mean(0))).mean()/4
    out['typed_triangle_L1']=float(abs(tc-rtc).sum()/max(1.,rtc.sum()))
    out['old_F']=(out['joint']+out['relation_W1']+out['typed_triangle_L1']+out['node_triangle_W1'])/8+(out['low']+out['hub'])/4
    return {k:float(v) for k,v in out.items()}
def main():
    OUT.mkdir(parents=True,exist_ok=False);signal.alarm(600);resource.setrlimit(resource.RLIMIT_AS,(2*1024**3,2*1024**3))
    save('protocol',dict(start_utc=datetime.now(timezone.utc).isoformat(),CPU_threads=1,memory_GiB=2,GPU=0,seconds=600,
        slot_order='SHA256(component_target_v1|dataset|background|relation|present|u|v), first four, no replacement',
        independence='Each flip starts from its archived background. No sampling chain or new final graph.',
        metric='equal-type assignment W1 of component type inventory fractions, L1/T',
        score='existing six terms retained separately; no new combined reward or fitted thresholds',
        numerical_check_tolerance=1e-10))
    plan=json.loads(bind(ROOT/'results/anonymous_teacher_protocol_20261109/next_cpu_proposal.json').read_text())
    prior=json.loads(bind(ROOT/'results/real_structure_envelope_20261109/binding.json').read_text())['assets']
    refs={}; backgrounds=[];slots=[];counts=[]
    pkg=ROOT/'research_branches/typed_structure_early_cn_expansion_review/experiments/typed_structure_early_cn_expansion_v1/package/assets'
    for ds in ['IMDB','PubMed']:
        p=pkg/(ds+'.npz');t,s,a=load(bind(p,prior[str(p)]));refs[ds]=(t,s,a)
        backgrounds.append((ds,'reference',0,0,p,a))
    for row in plan['archived_graphs']:
        p=bind(row['file'],row['expected_sha']);t,s,a=load(p);rt,rs,ra=refs[row['dataset']]
        assert np.array_equal(t,rt) and np.array_equal(s,rs)
        backgrounds.append((row['dataset'],'A_'+str(row['train_seed']),row['train_seed'],row['sample_seed'],p,a))
    for ds,tag,ts,ss,p,a in backgrounds:
        t,s,ref=refs[ds];rm={tuple(x):r for r,x in enumerate(s)};q=[[] for _ in s]
        for u in range(len(t)):
            for v in range(u+1,len(t)):
                key=tuple(sorted((t[u],t[v])))
                if key in rm:q[rm[key]].append((u,v))
                else:assert not a[u,v]
        for r,pairs in enumerate(q):
            nr=sum(int(ref[u,v]) for u,v in pairs);fixed=nr in (0,len(pairs))
            if fixed:assert all(a[u,v]==ref[u,v] for u,v in pairs)
            for present in (0,1):
                pool=[(u,v) for u,v in pairs if int(a[u,v])==present]
                pool.sort(key=lambda uv:hashlib.sha256(f'component_target_v1|{ds}|{tag}|{r}|{present}|{uv[0]}|{uv[1]}'.encode()).hexdigest())
                selected=[] if fixed else pool[:4]
                counts.append(dict(dataset=ds,background=tag,relation=r,present=present,available=len(pool),fixed_support=fixed,selected=len(selected)))
                for u,v in selected:slots.append(dict(dataset=ds,background=tag,train_seed=ts,sample_seed=ss,relation=r,u=u,v=v,present=present))
    assert len(slots)<=480
    save('binding',BIND);save('frozen_slots',slots);save('selection_counts',counts)
    rows=[];base_rows=[];maxerr=0.;checks=0;comp_s=old_s=oracle_s=0.
    for ds,tag,ts,ss,p,a in backgrounds:
        t,s,ref=refs[ds];lab=bfs(a);rx=descriptor(t,bfs(ref));x=descriptor(t,lab);dc=dist(x,rx,t)
        rf=features(ref,t,s);f=features(a,t,s);b=terms(f,rf,t,s)
        perm=np.arange(len(t))
        for k in np.unique(t):ids=np.flatnonzero(t==k);perm[ids]=ids[::-1]
        assert abs(dist(descriptor(t,bfs(a[np.ix_(perm,perm)])),rx,t)-dc)<1e-10
        base_rows.append(dict(dataset=ds,background=tag,train_seed=ts,sample_seed=ss,edges=int(a.sum()//2),triangles=int(f[2].sum()),components=len(np.unique(lab)),LCC=int(np.bincount(lab).max()),component_distance=dc,**b))
        for sl in [z for z in slots if z['dataset']==ds and z['background']==tag]:
            u,v,r,pr=[sl[k] for k in ('u','v','relation','present')];sign=1-2*pr
            tick=time.perf_counter();cl=changed_labels(a,lab,u,v,pr);cv=dist(descriptor(t,cl),rx,t);comp_s+=time.perf_counter()-tick
            tick=time.perf_counter();df=[v.copy() for v in f];df[0][u,r]+=sign;df[0][v,r]+=sign
            for w in np.flatnonzero(a[u]&a[v]):
                df[2][TI[tuple(sorted((t[u],t[v],t[w])))]]+=sign
                for n,i,j in ((u,v,w),(v,u,w),(w,u,v)):df[1][n,PI[tuple(sorted((t[i],t[j])))]]+=sign
            aft=terms(df,rf,t,s);old_s+=time.perf_counter()-tick
            tick=time.perf_counter();aa=a.copy();aa[u,v]=aa[v,u]=not pr;full=bfs(aa);ind=union(aa)
            assert np.array_equal(full[:,None]==full[None,:],ind[:,None]==ind[None,:])
            assert np.array_equal(cl[:,None]==cl[None,:],full[:,None]==full[None,:])
            err=abs(cv-dist(descriptor(t,full),rx,t));maxerr=max(maxerr,err);assert err<1e-10
            ff=features(aa,t,s);assert all(np.array_equal(q,z) for q,z in zip(df,ff))
            assert abs(np.trace(np.linalg.matrix_power(aa.astype(np.int64),3))/6-ff[2].sum())<1e-10
            if ds=='PubMed':
                def closed(l):return 1-sum(float(np.mean(np.bincount(t[l==c],minlength=4)/np.bincount(t)))**2 for c in np.unique(l))
                err=abs((cv-dc)-(closed(full)-closed(lab)));maxerr=max(maxerr,err);assert err<1e-10
            oracle_s+=time.perf_counter()-tick;checks+=1
            change=len(np.unique(cl))-len(np.unique(lab))
            row=dict(**sl,operation='delete' if pr else 'add',component_event='split' if change>0 else 'merge' if change<0 else 'neutral',component_delta=float(cv-dc),triangles_delta=int(df[2].sum()-f[2].sum()),**{k+'_delta':float(aft[k]-b[k]) for k in b})
            rows.append(row)
        save('flips',rows);save('backgrounds',base_rows)
    summary=[]
    for br in base_rows:
        rr=[r for r in rows if (r['dataset'],r['background'])==(br['dataset'],br['background'])];imp=[r for r in rr if r['component_delta'] < -1e-10]
        summary.append(dict(dataset=br['dataset'],background=br['background'],flips=len(rr),component_before=br['component_distance'],
            merges=sum(r['component_event']=='merge' for r in rr),splits=sum(r['component_event']=='split' for r in rr),neutral=sum(r['component_event']=='neutral' for r in rr),
            comp_improved=len(imp),comp_worsened=sum(r['component_delta']>1e-10 for r in rr),
            improved_with_joint_cost=sum(r['joint_delta']>1e-10 for r in imp),improved_with_triangle_cost=sum(r['typed_triangle_L1_delta']>1e-10 for r in imp),
            improved_with_low_cost=sum(r['low_delta']>1e-10 for r in imp),improved_with_hub_cost=sum(r['hub_delta']>1e-10 for r in imp),
            old_F_improved_comp_worse=sum(r['old_F_delta'] < -1e-10 and r['component_delta']>1e-10 for r in rr)))
    save('summary',summary)
    for p,h in BIND.items():assert sha(p)==h,p
    save('completion',dict(status='passed',end_utc=datetime.now(timezone.utc).isoformat(),seconds=time.monotonic()-START,
        RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,GPU_hours=0,flips=len(rows),oracle_checks=checks,max_component_error=maxerr,
        timing_seconds=dict(component=comp_s,old_terms=old_s,oracle=oracle_s),assets_unchanged=True,
        source_sha256=sha(__file__),python=platform.python_version(),numpy=np.__version__,threads=1,
        scope='No teacher posterior, tolerance/weight selection, network, training, diffusion generation or saved edited graph. Temporary flips only.'))
    print(json.dumps(dict(summary=summary,completion=json.loads((OUT/'completion.json').read_text())),ensure_ascii=False))
if __name__=='__main__':
    try:main()
    except BaseException as e:
        if OUT.exists():save('failure',dict(error=repr(e),seconds=time.monotonic()-START))
        raise
