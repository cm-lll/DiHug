"""Frozen bounded CPU-only endpoint-relocation sensitivity/calibration study."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import json,time,signal,resource,csv,hashlib,traceback
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from transport import TransportScore,ROOT
OUT=ROOT/'results/transport_sensitivity_20261113'
SRC=ROOT/'results/joint_transport_ablation_20261113T011928'
P=json.loads((OUT/'protocol.json').read_text())
def check():
    if time.time()>P['deadline_epoch']-30:raise TimeoutError('CPU science reserve')
def dump(name,obj):(OUT/name).write_text(json.dumps(obj,indent=2,allow_nan=False))
def table(name,rows):
    if rows:
        with (OUT/name).open('w') as f:
            w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def adjacency(sc,b):
    a=np.zeros((sc.n,sc.n));sel=b.astype(bool);a[sc.q[0,sel],sc.q[1,sel]]=1;return a+a.T
def evalscore(sc,b,c=1.):
    h,d=sc.features(b);cost=dc=hc=0.;times=np.zeros(9);nt=len(sc.scales)
    for typ,(scale,db,hb) in sc.scales.items():
        ix=sc.types==typ;x=np.concatenate([d[ix]/scale/np.sqrt(db),np.sqrt(c)*h[ix]/np.sqrt(hb)],1);y=sc.reference[typ].copy();y[:,sc.r:]*=np.sqrt(c)
        ii,jj=linear_sum_assignment(cdist(x,y,'sqeuclidean'));delta=(x[ii]-y[jj])**2
        cd=delta[:,:sc.r].sum(1).mean()/nt;ch=delta[:,sc.r:].sum(1).mean()/nt;cost+=cd+ch;dc+=cd;hc+=ch
        times+=(delta[:,sc.r:sc.r+9].mean(0)+delta[:,sc.r+9:].reshape(-1,9,nt).sum(2).mean(0))/nt
    return dict(cost=float(cost),degree=float(dc),heat=float(hc),**{f'time{i}':float(v) for i,v in enumerate(times)})
def load(ds,parent):
    z=np.load(parent['data']);t=z['types'];schema=z['schema'];q=np.array(np.triu_indices(len(t),1));rd={tuple(map(int,v)):i for i,v in enumerate(schema)}
    fam=np.array([rd[tuple(sorted((int(t[u]),int(t[v]))))] for u,v in q.T]);lookup={tuple(v):i for i,v in enumerate(q.T)};b=np.zeros(q.shape[1],np.uint8);e=z['edges'];e=e.T if e.shape[0]==2 else e
    for u,v in e:b[lookup[tuple(sorted((int(u),int(v))))]]=1
    prior=np.array([b[fam==i].mean() for i in range(len(schema))]);case=dict(space=SimpleNamespace(types=t),n_families=len(schema),prior=prior)
    sc=TransportScore(ds,case,q,fam,b);sc.lookup=lookup;sc.hub={int(k):float(np.quantile(sc.rd[t==k].sum(1),.9)) for k in np.unique(t)};return sc
def choose_move(sc,b,route,rng,used):
    a=adjacency(sc,b);d=a.sum(1);cn=a@a;by={}
    for old in np.flatnonzero(b):
        if int(old) in used:continue
        u0,v0=map(int,sc.q[:,old]);r=int(sc.f[old])
        for u,v in ((u0,v0),(v0,u0)):
            if (route=='isolation' and d[u]!=1) or (route!='isolation' and d[u]<2):continue
            dest=np.flatnonzero((sc.types==sc.types[u])&(d>0));dest=dest[(dest!=u)&(dest!=v)&(a[dest,v]==0)]
            if route=='hub':dest=dest[(d[dest]>=sc.hub[int(sc.types[u])])&(d[dest]>d[u])]
            elif route in ('ordinary','isolation'):dest=dest[d[dest]<sc.hub[int(sc.types[u])]]
            for w in dest:
                new=sc.lookup[tuple(sorted((int(w),v)))];assert sc.f[new]==r
                if new in used:continue
                # Removing u-v also removes u as possible common neighbour of w,v.
                tri_delta=cn[w,v]-a[w,u]-cn[u,v]
                if route=='closure' and tri_delta<=0:continue
                by.setdefault(r,[]).append((int(old),int(new),u,v,int(w),int(tri_delta)))
    if not by:return None
    r=int(rng.choice(sorted(by)));return by[r][int(rng.integers(len(by[r])))]
def main():
    start=time.time();resource.setrlimit(resource.RLIMIT_AS,(2*1024**3,2*1024**3))
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('hard wall')));signal.setitimer(signal.ITIMER_REAL,max(1,P['deadline_epoch']-time.time()-20))
    parents=json.loads((SRC/'protocol.json').read_text())['parents'];states=[];rows=[];fail=[];hashes={};cache={};seen=set()
    for path in [__file__,Path(__file__).with_name('transport.py'),ROOT/'research_branches/one_step_vs_terminal_plan_20261111/heat_solver_candidate.py',OUT/'protocol.json',SRC/'protocol.json']:hashes[str(path)]=digest(path)
    # Materialise and hash the complete panel BEFORE reading any score.
    for di,ds in enumerate(P['datasets']):
        parent=next(v for v in parents.values() if v['dataset']==ds);hashes[parent['data']]=digest(parent['data']);sc=load(ds,parent);cache[ds]=sc
        for ri,route in enumerate(P['routes']):
            rng=np.random.default_rng(np.random.SeedSequence([P['seed_base'],di,ri]));count=0;attempts=0
            while count<16 and attempts<2000:
                check();attempts+=1;b=sc.labels.copy();moves=[];used=set()
                for step in range(2):
                    move=choose_move(sc,b,route,rng,used)
                    if move is None:break
                    old,new,*_=move;b[old]=0;b[new]=1;used.update((old,new));moves.append(move)
                if len(moves)!=2:continue
                h=hashlib.sha256(b.tobytes()).hexdigest()
                if (ds,h) in seen:continue
                a=adjacency(sc,b);d=a.sum(1);tri=int(round(np.trace(a@a@a)/6));refa=adjacency(sc,sc.labels);reft=int(round(np.trace(refa@refa@refa)/6))
                assert np.count_nonzero(b!=sc.labels)==4 and b.sum()==sc.labels.sum()
                assert np.array_equal(np.bincount(sc.f,weights=b,minlength=sc.r),sc.target)
                assert (d==0).sum()==(2 if route=='isolation' else 0)
                assert sum(v[-1] for v in moves)==tri-reft
                seen.add((ds,h));item=dict(dataset=ds,route=route,index=count,split='development' if count<8 else 'heldout',state_hash=h,moves=moves,triangles=tri,triangle_delta=tri-reft,isolates=int((d==0).sum()),max_degree=float(d.max()))
                states.append((item,b));count+=1
            if count<16:fail.append(dict(dataset=ds,route=route,completed=count,attempts=attempts))
    dump('panel_manifest.json',[x for x,b in states]);np.savez_compressed(OUT/'panel_bits.npz',**{str(i):b for i,(x,b) in enumerate(states)})
    dump('panel_freeze.json',dict(at_epoch=time.time(),manifest_sha256=digest(OUT/'panel_manifest.json'),bits_sha256=digest(OUT/'panel_bits.npz'),failures=fail))
    assert not fail,fail
    for item,b in states:
        if item['split']!='development':continue
        check();v=evalscore(cache[item['dataset']],b);rows.append({**{k:v for k,v in item.items() if k!='moves'},**v})
    weights={}
    for ds in cache:
        vals=[x for x in rows if x['dataset']==ds];dg=float(np.median([x['degree'] for x in vals]));hg=float(np.median([x['heat'] for x in vals]));assert dg>1e-12 and hg>1e-12;weights[ds]=dict(degree_median=dg,heat_median=hg,heat_coefficient=dg/hg)
    dump('candidate_frozen_before_holdout.json',dict(at_epoch=time.time(),weights=weights,rule=P['candidate'],no_sweep=True))
    candidates=[]
    for item,b in states:
        check();sc=cache[item['dataset']]
        if item['split']=='heldout':rows.append({**{k:v for k,v in item.items() if k!='moves'},**evalscore(sc,b)})
        candidates.append({**{k:v for k,v in item.items() if k!='moves'},**evalscore(sc,b,weights[item['dataset']]['heat_coefficient'])})
    table('original.csv',rows);table('candidate.csv',candidates)
    checks=[]
    for ds,sc in cache.items():
        c=weights[ds]['heat_coefficient'];zero=evalscore(sc,sc.labels,c)['cost'];assert zero<1e-12
        rng=np.random.default_rng(P['seed_base']+99);perm=np.arange(sc.n)
        for typ in sc.scales:
            ix=np.flatnonzero(sc.types==typ);perm[ix]=rng.permutation(ix)
        b=np.zeros_like(sc.labels)
        for u,v in sc.q[:,sc.labels.astype(bool)].T:b[sc.lookup[tuple(sorted((int(perm[u]),int(perm[v]))))]]=1
        pv=evalscore(sc,b,c)['cost'];assert pv<1e-10
        vv=np.array([[x[f'time{i}'] for i in range(9)] for x in rows if x['dataset']==ds and x['split']=='heldout']);corr=np.corrcoef(vv,rowvar=False);dump(ds+'_heldout_time_correlation.json',corr.tolist())
        checks.append(dict(dataset=ds,self=zero,permutation=pv,late_adjacent_correlation=[float(corr[i,i+1]) for i in range(5,8)]))
    dump('checks.json',checks);dump('source_hashes.json',hashes)
    dump('receipt.json',dict(status='complete',states=len(states),failures=fail,seconds=time.time()-start,wall_minutes=(time.time()-P['start_epoch'])/60,peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,CPU_threads=1,GPU_hours=0,model_calls=0,new_model_graphs=0,source_unchanged=all(digest(k)==v for k,v in hashes.items()),formal_score_modified=False))
    print((OUT/'receipt.json').read_text())
if __name__=='__main__':
    try:main()
    except BaseException:
        dump('failure.json',dict(error=traceback.format_exc(),at_epoch=time.time()));raise
