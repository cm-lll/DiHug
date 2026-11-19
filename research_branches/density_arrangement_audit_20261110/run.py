"""CPU decomposition of saved probabilities. No network, gradient or sampling."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import time,json,hashlib,csv,importlib.util,itertools,resource,traceback
from pathlib import Path
import numpy as np
from scipy.special import expit
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/density_arrangement_audit_20261110';PREV=ROOT/'results/displacement_response_20261110';OLD=ROOT/'results/exact_feedback_updates_20261110'
def mod(n,p):
    s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
HardScore=mod('objective',ROOT/'research_branches/hard_clean_pilot_v2_20261110/objective.py').HardScore
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for x in iter(lambda:f.read(1024*1024),b''):h.update(x)
    return h.hexdigest()
def save(n,x):(OUT/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False))
def write(n,rows):
    if not rows:return
    with (OUT/n).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
class Moments:
    def __init__(self,c):
        self.c=c;self.ix=[];lookup=np.full((c.n,c.n),-1,dtype=int);lookup[c.q[0],c.q[1]]=np.arange(len(c.f));lookup[c.q[1],c.q[0]]=np.arange(len(c.f));groups=[[] for _ in c.triples]
        for u in range(c.n):
            for v in np.flatnonzero(lookup[u]>=0):
                if v<=u:continue
                for w in np.flatnonzero((lookup[u]>=0)&(lookup[v]>=0)&(np.arange(c.n)>v)):
                    j=c.tindex[tuple(sorted((int(c.types[u]),int(c.types[v]),int(c.types[w]))))];groups[j].append((lookup[u,v],lookup[u,w],lookup[v,w]))
        self.ix=[np.asarray(x,dtype=int).reshape(-1,3) for x in groups]
    def score(self,p):
        c=self.c;counts=[];tr=[]
        for f in range(c.r):
            x=p[c.f==f];mu=x.sum();var=(x*(1-x)).sum();target=int(c.ref['counts'][f]);counts.append(dict(family=f,mean=float(mu),variance=float(var),risk=float(((mu-target)**2+var)/max(1,target)**2)))
        for j,ix in enumerate(self.ix):
            mu=var=0.
            if len(ix):
                ed=p[ix];q=ed.prod(1);mu=q.sum();s=np.zeros(len(p));ss=np.zeros(len(p))
                for k in range(3):
                    v=ed[:,(k+1)%3]*ed[:,(k+2)%3];s+=np.bincount(ix[:,k],weights=v,minlength=len(p));ss+=np.bincount(ix[:,k],weights=v*v,minlength=len(p))
                var=(q-q*q).sum()+(p*(1-p)*(s*s-ss)).sum();assert var>=-1e-8
            target=int(c.ref['triangles'][j]);tr.append(dict(triangle_type='-'.join(map(str,c.triples[j])),target=target,mean=float(mu),variance=float(var),risk=float(((mu-target)**2+var)/max(1,target)**2)))
        v=np.array([np.mean([x['risk'] for x in counts]),np.mean([tr[j]['risk'] for j in np.flatnonzero(c.present)]) if c.present.any() else 0.,np.mean([tr[j]['risk'] for j in np.flatnonzero(c.absent)]) if c.absent.any() else 0.])
        assert np.isfinite(v).all();return v,counts,tr
def shift(p,f,targets):
    out=p.copy();details=[]
    for family,target in enumerate(targets):
        ix=np.flatnonzero((f==family)&(p>0)&(p<1));ones=int(((f==family)&(p==1)).sum());desired=float(target)-ones
        if not len(ix):
            assert abs(desired)<1e-10;details.append(dict(family=family,shift=0.,residual=abs(desired)));continue
        if not 0<desired<len(ix):raise ValueError('target requires changing endpoint support')
        logits=np.log(p[ix])-np.log1p(-p[ix]);lo=-2048.;hi=2048.
        for _ in range(80):
            mid=(lo+hi)/2
            if expit(logits+mid).sum()<desired:lo=mid
            else:hi=mid
        delta=(lo+hi)/2;out[ix]=expit(logits+delta)
        if ((out[ix]==0)|(out[ix]==1)).any():raise ValueError('floating endpoint saturation after shift')
        error=abs(out[f==family].sum()-target);assert error<1e-9*max(1,float(target))
        details.append(dict(family=family,shift=float(delta),residual=float(error)))
    return out,details
def tests():
    p=np.array([0.,.1,.3,.7,1.,.2,.8]);f=np.array([0,0,0,0,0,1,1]);targets=[2.4,.7];q,d=shift(p,f,targets)
    assert q[0]==0 and q[4]==1
    for k in (0,1):
        ix=np.flatnonzero((f==k)&(p>0)&(p<1));a=np.log(p[ix])-np.log1p(-p[ix]);b=np.log(q[ix])-np.log1p(-q[ix]);assert np.ptp(b-a)<1e-12
    back,_=shift(q,f,[p[f==k].sum() for k in (0,1)]);np.testing.assert_allclose(back,p,atol=1e-14)
    # Enumerate a 4-node six-slot independent law to check exact risk/variance.
    query=np.array(list(itertools.combinations(range(4),2))).T;c=HardScore(np.array([0,0,1,1]),query,np.array([0,1,1,1,1,2]),np.array([1,1,0,0,1,1]),3);m=Moments(c);p=np.array([.1,.2,.3,.4,.7,.8]);v,_,tr=m.score(p);e=np.zeros(3)
    for bits in itertools.product((0,1),repeat=6):
        b=np.array(bits,dtype=np.uint8);prob=np.prod(np.where(b,p,1-p));cost=c.costs(b)[0];e+=prob*cost[[0,2,3]]
    np.testing.assert_allclose(v,e,rtol=1e-12,atol=1e-12)
    return dict(passed=True,enumerated_graphs=64,max_exact_error=float(abs(v-e).max()),shift_preserves_support_and_log_odds_differences=True,round_trip=True)
def main():
    start=time.time();OUT.mkdir(exist_ok=False);receipt=dict(status='running',GPU_hours=0,model_calls=0,updates=0,generated_graphs=0)
    try:
        protocol=json.loads((PREV/'protocol.json').read_text());tasks=protocol['tasks'];sources={str(Path(__file__).resolve()):sha(__file__)}
        for p in [ROOT/'research_branches/hard_clean_pilot_v2_20261110/objective.py',PREV/'protocol.json',PREV/'evaluation_summary.csv',ROOT/'results/fixed_count_diagnostic_20261110/REPORT_ZH.md']:sources[str(p)]=sha(p)
        panels=['same','t1','t75','t250','t550','t800','t1000'];alphas=[.25,.5,1.]
        records=[]
        for row in tasks:
            name=f"{row['dataset']}_{row['seed']}";src=Path(row['response_source']);sources[row['data']]=sha(row['data']);assert sources[row['data']]==row['data_sha256']
            for label in panels:
                bp=src/'same_before.npz' if label=='same' else OLD/name/f'eval_before_{label}.npz'
                for alpha in alphas:
                    ap=src/f'{label}_after.npz' if alpha==1 else PREV/name/f'a{alpha}_{label}.npz'
                    for p in (bp,ap):sources[str(p)]=sha(p)
                    records.append(dict(dataset=row['dataset'],seed=row['seed'],data=row['data'],panel=label,t=row['diagnostic_t'] if label=='same' else int(label[1:]),alpha=alpha,before=str(bp),after=str(ap)))
        save('protocol.json',dict(start_epoch=start,deadline_epoch=start+600,CPU_threads=1,conditions=84,crossed_distributions=168,alphas=alphas,panels=panels,parameter_or_model_calls=0,definition='Per relation common log-odds shift to swap expected counts, preserving relative log-odds; symmetric two-order decomposition. Not fixed hard counts or a training rule.',support_policy='Preserve exact zero/one entries; changed support or unreachable targets recorded and skipped, never clipped',names=['relation_count','present_triangles','absent_triangles'],conditions_manifest=records));save('bindings.json',sources);save('cpu_tests.json',tests())
        cached=list(csv.DictReader((PREV/'evaluation_summary.csv').open()));scores=[];decomp=[];relations=[];triangles=[];shifts=[];failures=[];maxcache=0.;maxsum=0.;completed=0
        for row in tasks:
            data=np.load(row['data']);subset=[r for r in records if r['dataset']==row['dataset'] and r['seed']==row['seed']];z=np.load(subset[0]['before']);core=HardScore(data['types'],z['query'],z['families'],z['labels'],len(data['schema']));moment=Moments(core)
            for rec in subset:
                if time.time()-start>540:raise TimeoutError('CPU science deadline')
                b=np.load(rec['before']);a=np.load(rec['after']);p0=b['probability'];p1=a['probability'];meta={k:rec[k] for k in ('dataset','seed','panel','t','alpha')}
                for k in ('ids','query','labels','families'):assert np.array_equal(a[k],b[k])
                if not np.array_equal(p0==0,p1==0) or not np.array_equal(p0==1,p1==1):failures.append(dict(**meta,reason='exact endpoint support changed; no odds decomposition'));continue
                n0=[p0[core.f==f].sum() for f in range(core.r)];n1=[p1[core.f==f].sum() for f in range(core.r)]
                try:q10,d10=shift(p0,core.f,n1);q01,d01=shift(p1,core.f,n0)
                except ValueError as exc:failures.append(dict(**meta,reason=str(exc)));continue
                values={};tri_values={}
                for arm,p in [('00',p0),('10',q10),('01',q01),('11',p1)]:
                    v,rr,tt=moment.score(p);values[arm]=v;tri_values[arm]=tt
                    scores.append(dict(**meta,arm=arm,relation_count=float(v[0]),present_triangles=float(v[1]),absent_triangles=float(v[2]),expected_edges=float(p.sum()),expected_triangles=sum(x['mean'] for x in tt)))
                    relations.extend(dict(**meta,arm=arm,**x) for x in rr);triangles.extend(dict(**meta,arm=arm,**x) for x in tt)
                    if arm in ('00','11'):
                        old=next(x for x in cached if x['dataset']==rec['dataset'] and int(x['seed'])==rec['seed'] and x['panel']==rec['panel'] and float(x['alpha'])==(0 if arm=='00' else rec['alpha']))
                        vv=np.array([float(old[k]) for k in ('relation_count','present_triangles','absent_triangles')]);maxcache=max(maxcache,float(abs(v-vv).max()));np.testing.assert_allclose(v,vv,rtol=1e-9,atol=1e-11)
                for arm,ds in [('10',d10),('01',d01)]:shifts.extend(dict(**meta,arm=arm,**d) for d in ds)
                for k,name in enumerate(('relation_count','present_triangles','absent_triangles')):
                    v00,v10,v01,v11=[values[x][k] for x in ('00','10','01','11')];density=.5*((v10-v00)+(v11-v01));arrange=.5*((v01-v00)+(v11-v10));total=v11-v00;maxsum=max(maxsum,abs(density+arrange-total))
                    decomp.append(dict(**meta,metric=name,before=float(v00),after=float(v11),total_change=float(total),density_part=float(density),arrangement_part=float(arrange),interaction=float(v11-v10-v01+v00),arrangement_at_before_count=float(v01-v00),arrangement_at_after_count=float(v11-v10)))
                completed+=1;save('progress.json',dict(completed=completed,total=84,seconds=time.time()-start))
        for p,h in sources.items():assert sha(p)==h,p
        write('scores.csv',scores);write('decomposition.csv',decomp);write('relations.csv',relations);write('typed_triangles.csv',triangles);write('shifts.csv',shifts);save('failures.json',failures)
        save('audit.json',dict(passed=True,completed=completed,skipped=len(failures),cached_endpoint_max_error=maxcache,decomposition_additivity_error=maxsum,assets_unchanged=True,CPU_scientific_seconds=time.time()-start))
        receipt.update(status='completed',completed=completed,skipped=len(failures))
    except BaseException as exc:
        receipt.update(status='failed',error=repr(exc));(OUT/'failure.txt').write_text(traceback.format_exc());raise
    finally:
        receipt.update(seconds=time.time()-start,peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024);save('receipt.json',receipt)
if __name__=='__main__':main()
