"""Explain fixed archived exchanges, not a new optimizer or generation rule."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import json,csv,time,resource,signal,hashlib,itertools
from pathlib import Path
from collections import Counter
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
SRC=ROOT/'results/joint_degree_distance_20261109'
OUT=ROOT/'results/exchange_closure_accounting_20261109/verified'
START=time.monotonic();BIND={};EPS=1e-12
PARTS=['joint_distance_change','marginal_distance_change','explicit_low_change','explicit_hub_change','explicit_typed_triangle_L1_change','explicit_node_triangle_W1_change']
WEIGHTS=np.array([.125,.125,.25,.25,.125,.125])
METRICS=['total_degree_W1','relation_degree_W1','joint_degree_TV','node_triangle_W1','root_path_W1','hub_error','triangle_error','edge_error']

def bind(p):
    p=Path(p);BIND[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest();return p
def read(p):return json.loads(bind(p).read_text())
def serial(v):
    if isinstance(v,np.generic):return v.item()
    if isinstance(v,np.ndarray):return v.tolist()
    raise TypeError(str(type(v)))
def save(name,obj):
    (OUT/(name+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False,default=serial)+'\n')
    if isinstance(obj,list) and obj and isinstance(obj[0],dict):
        with (OUT/(name+'.csv')).open('w',newline='') as f:
            w=csv.DictWriter(f,list(obj[0]));w.writeheader();w.writerows(obj)
def triangles(a,types):
    out=Counter()
    for u,v in zip(*np.where(np.triu(a,1))):
        for w in np.flatnonzero(a[u]&a[v]):
            if w>v:out[tuple(sorted(map(int,types[[u,v,w]])))]+=1
    return out
def connected(a):
    seen=set();sizes=[]
    for u in range(len(a)):
        if u in seen:continue
        todo=[u];seen.add(u);n=0
        while todo:
            v=todo.pop();n+=1
            for w in np.flatnonzero(a[v]):
                if int(w) not in seen:seen.add(int(w));todo.append(int(w))
        sizes.append(n)
    return dict(LCC=max(sizes),components=len(sizes),isolates=int((a.sum(1)==0).sum()))
def main():
    OUT.mkdir(exist_ok=False,parents=True);signal.alarm(300)
    save('protocol',dict(scope='all172 existing PubMed GD exchanges, fixed7 newly admitted triangle-cost cases; no new candidate or resampling',
        cases='old explicit delta>=-1e-12, geometric delta<-1e-12, triangle AE delta>1e-12',
        comparison='componentwise no-worse six actual score components, lower triangle cost; also report all archived metrics plus connectivity dominance',
        ranking='ascending existing geometric score change; competition ranks and ties; alternatives kept in full, illustrative slot chosen lowest ID',
        mass='sum max(-score_change,0), descriptive score benefit only, NOT probability-weighted model gradient',
        budget=dict(seconds=300,threads=1,RSS_MiB=1024,GPU=0)))
    bind(Path(__file__));prev=read(SRC/'binding.json');rows=read(SRC/'exchanges.json')
    def checked(p):
        p=bind(p);assert prev[str(p)]==BIND[str(p)],str(p);return p
    jobs=read(ROOT/'results/rooted_connection_20261109T015142/evaluation_jobs.json')
    job=next(j for j in jobs if j['dataset']=='PubMed' and j['train_seed']==2026110421 and j['arm']=='A' and j['sample_seed']==2026110232)
    path=checked(Path(job['folder'])/'graph.npz');meta=read(path.parent/'binding.json');assert meta['graph_sha256']==BIND[str(path)]
    with np.load(checked(ROOT/'results/rooted_connection_20261109T015142/PubMed_cpu_cost_panel.npz')) as z:q=z['query'].copy();f=z['families'].copy();types=z['types'].copy();refbits=z['reference'].copy()
    with np.load(path) as z:
        assert np.array_equal(z['types'],types);a=np.zeros((len(types),len(types)),dtype=bool);a[z['edges'][0],z['edges'][1]]=True;a|=a.T
    ref=np.zeros_like(a);ref[q[0],q[1]]=refbits.astype(bool);ref|=ref.T
    oldtri=triangles(a,types);reftri=triangles(ref,types);T=sum(oldtri.values());RT=sum(reftri.values())
    ai=3489;u,v=map(int,q[:,ai]);assert not a[u,v]
    addcommon=np.flatnonzero(a[u]&a[v]);withadd=a.copy();withadd[u,v]=withadd[v,u]=True
    addedtypes=Counter(tuple(sorted(map(int,types[[u,v,w]]))) for w in addcommon)
    oldconn=connected(a);refconn=connected(ref);degrees=a.sum(1)
    detail=[];checks=[];endpoints=[];classrows=[]
    for r in rows:
        assert time.monotonic()-START<300 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024<1024
        e=int(r['removed_slot']);x,y=map(int,q[:,e]);assert a[x,y] and f[e]==f[ai]==1
        removedcommon=np.flatnonzero(withadd[x]&withadd[y]);removedtypes=Counter(tuple(sorted(map(int,types[[x,y,w]]))) for w in removedcommon)
        b=withadd.copy();b[x,y]=b[y,x]=False;nt=T+len(addcommon)-len(removedcommon)
        assert abs((abs(nt-RT)-abs(T-RT))-r['triangle_error_change'])<EPS
        assert int(b.sum())==int(a.sum())
        nowconn=connected(b);conndiff={k+'_error_change':abs(nowconn[k]-refconn[k])-abs(oldconn[k]-refconn[k]) for k in oldconn}
        contribution=np.array([r[k] for k in PARTS])*WEIGHTS
        assert abs(contribution.sum()-r['geometric_total_change'])<1e-10
        case=bool(r['explicit_change']>=-EPS and r['geometric_total_change']<-EPS and r['triangle_error_change']>EPS)
        rank=1+sum(s['geometric_total_change']<r['geometric_total_change']-EPS for s in rows)
        row=dict(r,**conndiff,case=case,add_common=int(len(addcommon)),remove_common_after_add=int(len(removedcommon)),
            triangle_before=T,triangle_after=nt,triangle_reference=RT,score_rank=rank,
            degree_benefit=-float(contribution[:4].sum()),closure_cost=float(contribution[4:].sum()),
            added_endpoints=[u,v],deleted_endpoints=[x,y])
        for key,val in zip(PARTS,contribution):row[key+'_weighted']=float(val)
        detail.append(row)
        for node in sorted({u,v,x,y}):
            opposite_type=1 if types[node]==0 else 0
            gd_before=int(a[node,types==opposite_type].sum());gd_after=int(b[node,types==opposite_type].sum())
            assert gd_before<=int(degrees[node]) and gd_after<=int(b[node].sum())
            endpoints.append(dict(removed_slot=e,node=node,type=int(types[node]),total_degree_before=int(degrees[node]),total_degree_after=int(b[node].sum()),
                GD_before=gd_before,GD_after=gd_after,
                note='node IDs identify current state only; no reference identity target'))
        # Typed delta exact additive accounting; reference composition remains separate.
        allclasses=sorted(set(oldtri)|set(reftri)|set(addedtypes)|set(removedtypes))
        typed_L1_before=sum(abs(oldtri[c]-reftri[c]) for c in allclasses)
        typed_L1_after=0
        for c in allclasses:
            count=oldtri[c]+addedtypes[c]-removedtypes[c];typed_L1_after+=abs(count-reftri[c])
            classrows.append(dict(removed_slot=e,triangle_type='-'.join(map(str,c)),before=oldtri[c],after=count,reference=reftri[c],added=addedtypes[c],removed=removedtypes[c]))
        assert abs((typed_L1_after-typed_L1_before)/RT-r['explicit_typed_triangle_L1_change'])<1e-10
        if case:
            full=triangles(b,types);assert sum(full.values())==nt
            assert all(full[c]==oldtri[c]+addedtypes[c]-removedtypes[c] for c in allclasses)
            checks.append(dict(removed_slot=e,full_triangle_enumeration=True))
    save('all_exchanges',detail);save('endpoint_details',endpoints);save('typed_triangle_changes',classrows)
    cases=[r for r in detail if r['case']];assert len(cases)==7
    pairs=[];case_summary=[]
    for c in cases:
        alternatives=[]
        for s in detail:
            if s['removed_slot']==c['removed_slot']:continue
            no_score_cost=all(s[k]<=c[k]+EPS for k in PARTS)
            close_safe=s['triangle_error_change']<=EPS and s['triangle_error_change']<c['triangle_error_change']-EPS
            dominates=no_score_cost and close_safe
            extra=dominates and all(s[k+'_change']<=c[k+'_change']+EPS for k in METRICS) and all(s[k+'_error_change']<=c[k+'_error_change']+EPS for k in ['LCC','components','isolates'])
            if dominates:
                assert s['geometric_total_change']<c['geometric_total_change']+EPS
                alternatives.append(s)
            pairs.append(dict(case_removed=c['removed_slot'],alternative_removed=s['removed_slot'],all_six_components_no_worse=no_score_cost,
                triangle_nonworse_than_start=close_safe,dominates_six_and_triangle=dominates,dominates_including_reported_metrics=extra,
                alternative_score=s['geometric_total_change'],alternative_triangle_change=s['triangle_error_change']))
        strict=[p for p in pairs if p['case_removed']==c['removed_slot'] and p['dominates_including_reported_metrics']]
        exemplar=min(alternatives,key=lambda s:s['removed_slot']) if alternatives else None
        case_summary.append(dict(removed_slot=c['removed_slot'],deleted_endpoints=c['deleted_endpoints'],added_triangles=c['add_common'],removed_triangles=c['remove_common_after_add'],
            net_triangle_change=c['triangle_after']-c['triangle_before'],score=c['geometric_total_change'],rank=c['score_rank'],
            degree_benefit=c['degree_benefit'],closure_cost=c['closure_cost'],dominant_alternatives=len(alternatives),
            dominant_all_reported=len(strict),example_removed=exemplar['removed_slot'] if exemplar else None,
            example_triangle_change=exemplar['triangle_error_change'] if exemplar else None,example_score=exemplar['geometric_total_change'] if exemplar else None))
    save('cases',case_summary);save('all_case_alternatives',pairs);save('tests',dict(case_enumerations=checks,
        all172_triangle_formula=True,all172_typed_delta_matches_archive=True,all172_score_decomposition=True,all172_edge_relation_counts_fixed=True))
    good=[r for r in detail if r['geometric_total_change']<-EPS];mass=sum(-r['geometric_total_change'] for r in good)
    bad=[r for r in good if r['triangle_error_change']>EPS]
    top=sorted(detail,key=lambda r:(r['geometric_total_change'],r['removed_slot']))
    save('summary',dict(graph=job['folder'],reference_triangles=RT,actual_triangles=T,absolute_error_before=abs(T-RT),
        added_slot=ai,added_endpoints=[u,v],added_common_neighbors=addcommon.tolist(),added_triangle_types={'-'.join(map(str,k)):v for k,v in addedtypes.items()},
        case_count=len(cases),rewarded=len(good),harmful_rewarded=len(bad),
        seven_score_benefit_fraction=sum(-r['geometric_total_change'] for r in cases)/mass,
        all_triangle_cost_score_benefit_fraction=sum(-r['geometric_total_change'] for r in bad)/mass,
        all_seven_have_dominating_alternative=all(r['dominant_alternatives']>0 for r in case_summary),
        all_seven_have_dominating_all_reported=all(r['dominant_all_reported']>0 for r in case_summary),
        top5=[dict(removed_slot=r['removed_slot'],score=r['geometric_total_change'],triangle_change=r['triangle_error_change']) for r in top[:5]],
        limitation='benefit fractions are not sampling probabilities, gradients, or actual update frequencies; case selected after observing outcome'))
    # File names only: do not load weights or infer probabilities from unrelated states.
    save('available_sample_files',sorted(p.name for p in path.parent.iterdir() if p.is_file()))
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in BIND.items())
    save('binding',BIND);save('receipt',dict(status='complete',seconds=time.monotonic()-START,RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        exchanges=172,cases=7,models=0,GPU_hours=0,updates=0,new_graph_samples=0,assets_unchanged=True))
    print((OUT/'cases.json').read_text());print((OUT/'summary.json').read_text());print((OUT/'receipt.json').read_text())
if __name__=='__main__':
    try:main()
    except BaseException as e:
        if OUT.exists():save('failure',dict(error=repr(e),seconds=time.monotonic()-START))
        raise
