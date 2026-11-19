"""Single-thread, bounded CPU check of one frozen robust heat cost."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import json,time,csv,resource,signal,traceback,itertools,hashlib
from pathlib import Path
import numpy as np
from scipy.special import pseudo_huber
from scipy.optimize import linear_sum_assignment
from scipy.sparse.csgraph import connected_components
import sensitivity_cpu as s
from robust_heat_v1 import robust_heat_squared,evaluate
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/robust_heat_transport_20261113';SRC=s.SRC;P=json.loads((OUT/'protocol.json').read_text())
def dump(n,x):(OUT/n).write_text(json.dumps(x,indent=2,allow_nan=False))
def check():
    if time.time()>P['deadline_epoch']-30:raise TimeoutError('deadline')
def table(name,rows):
    if rows:
        with (OUT/name).open('w') as f:
            w=csv.DictWriter(f,list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
def stats(sc,b):
    a=s.adjacency(sc,b);d=a.sum(1);nc,lab=connected_components(a,directed=False)
    return dict(edges=int(b.sum()),isolates=int((d==0).sum()),LCC=int(np.bincount(lab).max()),components=int(nc),triangles=int(round(np.trace(a@a@a)/6)),max_degree=float(d.max()))
def bits(sc,z):
    b=np.zeros_like(sc.labels);e=z['edges'];e=e.T if e.shape[0]==2 else e
    for u,v in e:b[sc.lookup[tuple(sorted((int(u),int(v))))]]=1
    return b
def main():
    start=time.time();resource.setrlimit(resource.RLIMIT_AS,(2*1024**3,2*1024**3));signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('hard deadline')));signal.setitimer(signal.ITIMER_REAL,max(1,P['deadline_epoch']-time.time()-20))
    hashes={str(p):s.digest(p) for p in [__file__,Path(__file__).with_name('robust_heat_v1.py'),Path(s.__file__),Path(__file__).with_name('transport.py'),ROOT/'research_branches/one_step_vs_terminal_plan_20261111/heat_solver_candidate.py',OUT/'protocol.json',SRC/'protocol.json']}
    q=np.logspace(-12,12,1000);rho=robust_heat_squared(q);assert np.allclose(rho,2*pseudo_huber(1,np.sqrt(q)),rtol=1e-10,atol=1e-12)
    assert (np.diff(rho)>0).all() and (rho<=q+1e-12).all();assert abs(rho[0]/q[0]-1)<1e-10
    rng=np.random.default_rng(901);x=rng.normal(size=7);eps=1e-6;numeric=[]
    for j in range(7):
        unit=np.eye(7)[j]*eps;numeric.append((robust_heat_squared(np.sum((x+unit)**2))-robust_heat_squared(np.sum((x-unit)**2)))/(2*eps))
    analytic=2*x/np.sqrt(1+sum(x*x));assert np.max(abs(np.array(numeric)-analytic))<1e-7;assert np.linalg.norm(analytic)<2
    cost=rng.random((4,4))+robust_heat_squared(rng.random((4,4))*20);ii,jj=linear_sum_assignment(cost);brute=min(sum(cost[i,p[i]] for i in range(4)) for p in itertools.permutations(range(4)));assert abs(cost[ii,jj].sum()-brute)<1e-12
    parents=json.loads((SRC/'protocol.json').read_text())['parents'];cache={};panel=[];fail=[];seen={(r['dataset'],r['state_hash']) for r in json.loads((ROOT/'results/transport_sensitivity_20261113/panel_manifest.json').read_text())}
    for di,ds in enumerate(('IMDB','PubMed')):
        par=next(v for v in parents.values() if v['dataset']==ds);hashes[par['data']]=s.digest(par['data']);sc=s.load(ds,par);cache[ds]=sc
        for ri,route in enumerate(P['fresh_panel']['routes']):
            rng=np.random.default_rng(np.random.SeedSequence([P['fresh_panel']['seed'],di,ri]));count=0;attempt=0
            while count<16 and attempt<2000:
                check();attempt+=1;b=sc.labels.copy();moves=[];used=set()
                for _ in range(2):
                    move=s.choose_move(sc,b,route,rng,used)
                    if move is None:break
                    old,new,*_=move;b[old]=0;b[new]=1;moves.append(move);used.update((old,new))
                if len(moves)!=2:continue
                sh=hashlib.sha256(b.tobytes()).hexdigest()
                if (ds,sh) in seen:continue
                seen.add((ds,sh));assert np.count_nonzero(sc.labels!=b)==4
                assert np.array_equal(np.bincount(sc.f,weights=b,minlength=sc.r),sc.target)
                st=stats(sc,b);assert st['isolates']==(2 if route=='isolation' else 0)
                panel.append((dict(dataset=ds,kind='fresh',route=route,index=count,state_hash=sh,moves=moves,**st),b));count+=1
            if count<16:fail.append(dict(dataset=ds,route=route,count=count,reason='construction capacity'))
        for seed in P['disconnect_checks']['isolation_path_seeds']:
            rng=np.random.default_rng(np.random.SeedSequence([seed,di]));b=sc.labels.copy();used=set();moves=[]
            for step in range(1,9):
                move=s.choose_move(sc,b,'isolation',rng,used)
                if move is None:fail.append(dict(dataset=ds,route='isolation_path',seed=seed,step=step,reason='no move'));break
                old,new,*_=move;b[old]=0;b[new]=1;used.update((old,new));moves.append(move)
                if step in P['disconnect_checks']['levels']:
                    st=stats(sc,b);assert st['isolates']==step;panel.append((dict(dataset=ds,kind='severity',route='isolation_path',index=seed,level=step,moves=list(moves),**st),b.copy()))
        for seed in P['disconnect_checks']['separate_two_halves_seeds']:
            rng=np.random.default_rng(np.random.SeedSequence([seed,di]));side=np.zeros(sc.n,int)
            for typ in sc.scales:
                ix=rng.permutation(np.flatnonzero(sc.types==typ));side[ix[:len(ix)//2]]=1
            cross=side[sc.q[0]]!=side[sc.q[1]];b=sc.labels.copy();removed=[];added=[];ok=True
            for fam in range(sc.r):
                old=np.flatnonzero((sc.f==fam)&cross&(b==1));new=np.flatnonzero((sc.f==fam)&~cross&(b==0))
                if len(new)<len(old):ok=False;break
                select=rng.choice(new,len(old),replace=False);b[old]=0;b[select]=1;removed+=old.tolist();added+=select.tolist()
            if not ok:fail.append(dict(dataset=ds,route='two_halves',seed=seed,reason='not enough internal slots'));continue
            assert not np.any(b[cross]);assert np.array_equal(np.bincount(sc.f,weights=b,minlength=sc.r),sc.target)
            st=stats(sc,b);assert st['LCC']<=max(int((side==0).sum()),int((side==1).sum()))
            panel.append((dict(dataset=ds,kind='severity',route='two_halves',index=seed,removed=removed,added=added,**st),b))
    dump('panel_manifest.json',[r for r,b in panel]);np.savez_compressed(OUT/'panel.npz',**{str(i):b for i,(r,b) in enumerate(panel)});dump('panel_frozen.json',dict(at_epoch=time.time(),sha256=s.digest(OUT/'panel_manifest.json'),failures=fail))
    scores=[]
    for row,b in panel:
        check();v,_=evaluate(cache[row['dataset']],b);out={k:z for k,z in row.items() if k not in ('moves','removed','added')}
        for name,p in v.items():
            for k in ('total','degree','heat'):out[name+'_'+k]=p[k]
        assert out['robust_total']<=out['original_total']+1e-9 and out['robust_total']>0;scores.append(out)
    table('fresh_and_severity.csv',scores);oldrows=[]
    for ep in sorted((SRC/'evaluation').glob('*.json')):
        check();ev=json.loads(ep.read_text());ds,seed,*rest=ep.stem.split('_');ss=int(rest[-1]);arm='_'.join(rest[:-1]);sc=cache[ds];gp=Path(ev['graph_path']);hashes[str(gp)]=s.digest(gp);assert hashes[str(gp)]==ev['graph_sha256'];hashes[str(ep)]=s.digest(ep);v,nodes=evaluate(sc,bits(sc,np.load(gp)))
        assert abs(v['original']['total']-ev['unified_transport_score'][0])<1e-8
        out=dict(dataset=ds,train_seed=int(seed),arm=arm,sample_seed=ss,original=v['original']['total'],robust=v['robust']['total'],robust_degree=v['robust']['degree'],robust_heat=v['robust']['heat'],robust_isolate_contribution=sum(x['contribution'] for x in nodes if x['degree']==0),triangles=ev['raw']['statistics']['triangles']);oldrows.append(out)
    table('old_graphs_rescored.csv',oldrows)
    invariants=[]
    for ds,sc in cache.items():
        zero,_=evaluate(sc,sc.labels);assert zero['robust']['total']<1e-12
        rng=np.random.default_rng(918);perm=np.arange(sc.n)
        for typ in sc.scales:
            ix=np.flatnonzero(sc.types==typ);perm[ix]=rng.permutation(ix)
        b=np.zeros_like(sc.labels)
        for u,v in sc.q[:,sc.labels.astype(bool)].T:b[sc.lookup[tuple(sorted((int(perm[u]),int(perm[v]))))]]=1
        pv,_=evaluate(sc,b);assert pv['robust']['total']<1e-10;invariants.append(dict(dataset=ds,self=zero['robust']['total'],permutation=pv['robust']['total']))
    dump('invariants.json',dict(scipy_formula=True,monotonic_residual=True,finite_difference_gradient=True,brute_assignment=True,invariants=invariants));dump('source_hashes.json',hashes)
    dump('receipt.json',dict(status='complete' if not fail else 'partial',panel=len(panel),old_graphs=len(oldrows),failures=fail,CPU_seconds=time.time()-start,peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,GPU_hours=0,model_calls=0,source_unchanged=all(s.digest(k)==v for k,v in hashes.items()),formal_score_modified=False));print((OUT/'receipt.json').read_text())
if __name__=='__main__':
    try:main()
    except BaseException:dump('failure.json',dict(error=traceback.format_exc(),at_epoch=time.time()));raise
