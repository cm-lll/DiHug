"""One frozen TV substitution on archived graphs; numpy/scipy CPU only."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import json,csv,time,signal,resource,hashlib,itertools
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
import scipy
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/joint_degree_distance_20261109'
SOURCE=ROOT/'results/structure_allocation_audit_20261109/complete'
RUN=ROOT/'results/rooted_connection_20261109T015142'
START=time.monotonic();BIND={};EPS=1e-12
METRICS=['total_degree_W1','relation_degree_W1','joint_degree_TV','node_triangle_W1','root_path_W1','hub_error','triangle_error','edge_error']

def bind(p):
    p=Path(p);BIND[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest();return p
def read(p):return json.loads(bind(p).read_text())
def serial(v):
    if isinstance(v,np.generic):return v.item()
    if isinstance(v,np.ndarray):return v.tolist()
    raise TypeError(type(v).__name__)
def save(name,x):
    (OUT/(name+'.json')).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False,default=serial)+'\n')
    if isinstance(x,list) and x:
        with (OUT/(name+'.csv')).open('w',newline='') as f:
            w=csv.DictWriter(f,list(x[0]));w.writeheader();w.writerows(x)
def limits():
    assert time.monotonic()-START<600,'600s cap'
    assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024<2048,'2GiB RSS cap'
def match(x,y):
    cost=cdist(x,y,metric='cityblock')/x.shape[1]
    i,j=linear_sum_assignment(cost)
    assert len(np.unique(j))==len(x)
    return float(cost[i,j].mean())
def tv(x,y):
    a=Counter(map(tuple,x));b=Counter(map(tuple,y))
    return .5*sum(abs(a.get(k,0)-b.get(k,0)) for k in a.keys()|b.keys())/len(x)
def degree(q,f,bits,n):
    out=np.zeros((n,10),dtype=float)
    np.add.at(out,(q[0],f),bits);np.add.at(out,(q[1],f),bits)
    return out

class JointDegree:
    def __init__(self,z):
        self.q=z['query'].copy();self.f=z['families'].copy();self.types=z['types'].copy();self.refbits=z['reference'].copy()
        self.ref=degree(self.q,self.f,self.refbits,len(self.types));self.cache={}
        for t in np.unique(self.types):
            ids=np.flatnonzero(self.types==t)
            cols=np.unique(self.f[np.isin(self.q[0],ids)|np.isin(self.q[1],ids)])
            target=self.ref[ids][:,cols];scale=np.maximum(1.,target.mean(0))
            self.cache[int(t)]=dict(ids=ids,cols=cols,target=target,scale=scale,z=target/scale)
    def terms(self,d):
        out=[]
        for t,c in self.cache.items():
            raw=d[c['ids']][:,c['cols']];x=raw/c['scale'];y=c['z']
            joint=match(x,y);marg=float(abs(np.sort(x,axis=0)-np.sort(y,axis=0)).mean())
            assert joint>=marg-1e-10
            out.append(dict(type=t,joint=joint,marginal=marg,TV=tv(raw,c['target']),coupling_gap=joint-marg))
        return {k:float(np.mean([r[k] for r in out])) for k in ['joint','marginal','TV','coupling_gap']},out
    def flip_degree(self,d,slot,present):
        new=d.copy();u,v=self.q[:,slot];r=self.f[slot];s=1-2*present
        new[u,r]+=s;new[v,r]+=s;assert new.min()>=0
        return new

def tests():
    x=np.array([[0.,2.],[2.,0.],[1.,5.],[4.,1.]])
    y=np.array([[1.,0.],[0.,4.],[3.,1.],[1.,2.]])
    expected=min(float(abs(x-y[list(p)]).mean()) for p in itertools.permutations(range(4)))
    assert abs(match(x,y)-expected)<1e-12
    target=np.array([[3.,1.],[0.,0.]])
    near=np.array([[3.,2.],[0.,0.]]);far=np.array([[15.,0.],[0.,0.]])
    assert tv(near,target)==tv(far,target) and match(near,target)<match(far,target)
    a=np.array([[0.,0.],[4.,4.]]);b=np.array([[0.,4.],[4.,0.]]);collapsed=np.full((2,2),2.)
    assert np.array_equal(np.sort(a,axis=0),np.sort(b,axis=0)) and match(a,b)>0
    assert match(collapsed,a)>0 and match(a,a[::-1])==0
    save('tests',dict(brute_permutations=24,brute_error=abs(match(x,y)-expected),
        near_TV=tv(near,target),far_TV=tv(far,target),near_distance=match(near,target),far_distance=match(far,target),
        same_marginal_joint_distance=match(a,b),collapsed_distance=match(collapsed,a),permutation_invariance=True,
        scope='finite vector checks; not graph generation, gradient validation, or model training'))

def augment(row,old,new):
    r=dict(row)
    # Exact frozen six-term substitution; TV coefficient is 1/8.
    dj=new['joint']-old['joint'];dt=new['TV']-old['TV'];dm=new['marginal']-old['marginal']
    assert abs(dt-r['joint_degree_TV_change'])<1e-10
    if 'explicit_relation_W1_change' in r:assert abs(dm-r['explicit_relation_W1_change'])<1e-10
    r.update(joint_distance_change=dj,marginal_distance_change=dm,coupling_gap_change=dj-dm,
        geometric_total_change=r['explicit_change']+(dj-dt)/8,
        joint_distance_before=old['joint'],TV_before=old['TV'])
    return r

def summarize(rows,keys):
    groups=defaultdict(list)
    for r in rows:groups[tuple(r[k] for k in keys)].append(r)
    out=[]
    for key,rr in sorted(groups.items()):
        for method,col in [('explicit_TV','explicit_change'),('explicit_geometric','geometric_total_change'),('TV_only','joint_degree_TV_change'),('joint_distance_only','joint_distance_change')]:
            good=[r for r in rr if r[col]<-EPS]
            out.append(dict(zip(keys,key),method=method,total=len(rr),rewarded=len(good),
                neutral=sum(abs(r[col])<=EPS for r in rr),
                **{m+'_worse':sum(r[m+'_change']>EPS for r in good) for m in METRICS},
                no_eight_gain=sum(all(r[m+'_change']>=-EPS for m in METRICS) and any(r[m+'_change']>EPS for m in METRICS) for r in good),
                joint_distance_worse=sum(r['joint_distance_change']>EPS for r in good),
                triangle_cost_sum=float(sum(max(0.,r['triangle_error_change']) for r in good)),
                hub_cost_sum=float(sum(max(0.,r['hub_error_change']) for r in good))))
    return out

def main():
    OUT.mkdir(exist_ok=False);signal.alarm(600)
    save('protocol',dict(rule='Typewise exact W1 of full relation-degree vectors, normalized L1 ground cost. Replace TV/8 only; other five terms unchanged.',
        scale='max(1, reference mean) per legal incident relation; same as previous marginal relation W1; types equal; no fitting or effect-based selection',
        scope='172 previously selected PubMed GD exchanges first, then same64 archived graphs4096 flips; PubMed primary, IMDB regression',
        limits=dict(seconds=600,threads=1,RSS_MiB=2048,GPU=0),scipy=scipy.__version__,
        limitations='TV bounded by1, transport unbounded: semantic and scale change, not matching solver-only ablation. Not earlier full degree+triangle transport or centered mixed quadratic matching.',
        decision='Retain all metrics and seeds; no training/weight search/extra graphs regardless of local improvements'))
    bind(Path(__file__))
    for p in ['research_branches/joint_transport_cpu_20261108/transport.py','research_branches/allocation_matching_audit_20261109/run.py','research_branches/explicit_structure_cpu_20261109/explicit_score.py']:bind(ROOT/p)
    tests();prev=read(SOURCE/'binding.json');rows=read(SOURCE/'final_edges.json');ex=read(SOURCE/'exchanges.json');jobs=read(RUN/'evaluation_jobs.json')
    def historical(p):
        p=bind(p);assert prev[str(p)]==BIND[str(p)],str(p);return p
    models={}
    for ds in ['PubMed','IMDB']:
        with np.load(historical(RUN/f'{ds}_cpu_cost_panel.npz')) as z:models[ds]=JointDegree(z)
        val,_=models[ds].terms(models[ds].ref);assert val['joint']==0 and val['TV']==0
    final=[];exout=[];gr=[];type_rows=[]
    ordered=sorted(jobs,key=lambda j:(j['dataset']!='PubMed',not(j['train_seed']==2026110421 and j['arm']=='A' and j['sample_seed']==2026110232)))
    for ji,job in enumerate(ordered):
        limits();start=time.monotonic();ident={k:job[k] for k in ['dataset','train_seed','arm','sample_seed']};model=models[job['dataset']]
        path=historical(Path(job['folder'])/'graph.npz');binding=read(path.parent/'binding.json');assert BIND[str(path)]==binding['graph_sha256']
        with np.load(path) as z:
            assert np.array_equal(z['types'],model.types)
            a=np.zeros((len(model.types),len(model.types)),dtype=bool);a[z['edges'][0],z['edges'][1]]=True;a|=a.T
        bits=a[model.q[0],model.q[1]].astype(float);d=degree(model.q,model.f,bits,len(model.types));old,oldtypes=model.terms(d)
        if ji==0:
            assert ident==dict(dataset='PubMed',train_seed=2026110421,arm='A',sample_seed=2026110232)
            added=model.flip_degree(d,3489,0);assert bits[3489]==0
            for r in ex:
                limits();e=r['removed_slot'];assert bits[e]==1 and model.f[e]==model.f[3489]
                nd=model.flip_degree(added,e,1);assert np.array_equal(nd.sum(0),d.sum(0))
                new,nt=model.terms(nd);exout.append(augment(r,old,new))
                for t,base in zip(nt,oldtypes):type_rows.append(dict(kind='exchange',removed_slot=e,**ident,type=t['type'],joint_change=t['joint']-base['joint'],TV_change=t['TV']-base['TV'],marginal_change=t['marginal']-base['marginal']))
            save('exchanges',exout);save('exchange_summary',summarize(exout,[]))
        chosen=[r for r in rows if all(r[k]==v for k,v in ident.items())]
        assert len(chosen)==64
        tick=time.monotonic()
        for r in chosen:
            nd=model.flip_degree(d,r['slot'],r['present']);new,nt=model.terms(nd);final.append(augment(r,old,new))
        elapsed=time.monotonic()-tick
        if ji==0:
            estimate=elapsed*64
            save('calibration',dict(first_graph_64_seconds=elapsed,full_remaining_estimate=estimate,remaining_seconds=600-(time.monotonic()-START)))
            if estimate+60>600-(time.monotonic()-START):raise TimeoutError('Complete panel does not fit; no subset selection')
        gr.append(dict(**ident,**old,seconds=time.monotonic()-start))
    save('final_edges',final);save('graph_scores',gr);save('exchange_types',type_rows)
    for keys,name in [(['dataset'],'dataset'),(['dataset','train_seed'],'seed'),(['dataset','arm'],'arm'),(['dataset','operation'],'operation')]:save(name,summarize(final,keys))
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in BIND.items())
    save('binding',BIND);save('receipt',dict(status='complete',seconds=time.monotonic()-START,RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        exchanges=len(exout),graphs=len(gr),flips=len(final),GPU_hours=0,models=0,updates=0,new_graph_samples=0,assets_unchanged=True))
    print((OUT/'exchange_summary.json').read_text());print((OUT/'dataset.json').read_text());print((OUT/'receipt.json').read_text())
if __name__=='__main__':
    try:main()
    except BaseException as e:
        if OUT.exists():save('failure',dict(error=repr(e),seconds=time.monotonic()-START))
        raise
