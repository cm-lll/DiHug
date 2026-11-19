"""Decompose an unchanged archived CPU score; no model or candidate search."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,json,csv,time,hashlib,resource,datetime
from pathlib import Path
from collections import Counter
import numpy as np
import torch
torch.set_num_threads(1)
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'research_branches/rooted_connection_20261109'))
sys.path.insert(0,str(ROOT/'research_branches/rooted_quantile_cpu_20261109'))
from score import RootedScore,RootedIncremental
from quantile import QuantileScore
RUN=ROOT/'results/rooted_connection_20261109T015142'
PREV=ROOT/'results/rooted_quantile_cpu_20261109'
OUT=ROOT/'results/type_price_audit_20261109'
START=time.time();BIND={};EPS=1e-12
def read(p):
    p=Path(p);BIND[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    return json.loads(p.read_text())
def bind(p):
    p=Path(p);BIND[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest();return p
def save(n,obj):
    (OUT/(n+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False))
    if isinstance(obj,list) and obj:
        with (OUT/(n+'.csv')).open('w') as f:
            w=csv.DictWriter(f,list(obj[0]));w.writeheader();w.writerows(obj)
def by_type(core,score,raw,ref):
    out={};raw=raw[None] if raw.ndim==2 else raw
    for t,c in score.cache.items():
        if c is None:continue
        ids=c['nodes'];cols=np.flatnonzero(c['mask']);x=raw[:,ids];y=ref[ids]
        z=np.log1p(x[:,:,c['mask']])/c['scale']
        a=np.square(np.sort(z,axis=1)-c['target_axis']).mean(1)
        m=np.square(np.sort(z@c['u'],axis=1)-c['target_mixed']).mean((1,2))
        vals={'score':.5*(a.mean(1)+m)/len(score.cache),
              'axis_score':.5*a.mean(1)/len(score.cache),
              'mixed_score':.5*m/len(score.cache)}
        for block,mask in [('degree',cols<core.r),('triangle',(cols>=core.r)&(cols<core.base_width)),('path',cols>=core.base_width)]:
            active=cols[mask]
            vals[block+'_axis_score']=.5*a[:,mask].sum(1)/len(cols)/len(score.cache)
            vals[block+'_W1']=abs(np.sort(x[:,:,active],axis=1)-np.sort(y[:,active],axis=0)).mean((1,2)) if len(active) else np.zeros(len(raw))
        deg=x[:,:,:core.r].sum(-1);rdeg=y[:,:core.r].sum(-1)
        dev=abs(np.sort(deg,axis=1)-np.sort(rdeg))
        vals['total_degree_W1']=dev.mean(1)
        vals['hub_error']=dev[:,-2:].mean(1)
        vals['lower_half_degree_error']=dev[:,:len(ids)//2].mean(1)
        tv=[];target=Counter(map(tuple,y[:,:core.r]))
        for d in x[:,:,:core.r]:
            cnt=Counter(map(tuple,d));tv.append(.5*sum(abs(cnt.get(k,0)-target.get(k,0)) for k in cnt.keys()|target.keys())/len(ids))
        vals['joint_degree_TV']=np.array(tv)
        out[t]=vals
    return out
def main():
    OUT.mkdir(exist_ok=False)
    save('protocol',dict(start_BJT=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        scope='unchanged quantile score; exact same64 archived final graphs and4096 selected flips',
        exclusion='no new objective, fitting, random graphs, model calls, GPU or updates',
        limits=dict(worker_wall_seconds=600,threads=1,RSS_MiB=2048),sign_tolerance=EPS,
        statistics='descriptive selected-slot counts, not independent samples/population rates; per-type score includes equal type weight',
        degree_groups='sorted degree ranks: lower floor(n/2), largest2; anonymous distribution comparison, not node-identity matching',
        price_example='previously declared PubMed slot3489; examine all changed sorted coordinate ranks, not a new selected case'))
    for p in [Path(__file__),ROOT/'research_branches/rooted_quantile_cpu_20261109/quantile.py',ROOT/'research_branches/rooted_connection_20261109/score.py']:bind(p)
    previous_bind=read(PREV/'binding.json');edges=read(PREV/'final_edge_comparison.json');jobs=read(RUN/'evaluation_jobs.json')
    cores={}
    for ds in ['IMDB','PubMed']:
        p=bind(RUN/f'{ds}_cpu_cost_panel.npz');assert BIND[str(p)]==previous_bind[str(p)]
        with np.load(p) as z:cores[ds]=RootedScore(z['types'],z['query'],z['families'],z['reference'],10)
    type_rows=[];edge_rows=[];example=[];maxerr=0.
    for job in jobs:
        assert time.time()-START<600 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024<2048
        ident={k:job[k] for k in ('dataset','train_seed','arm','sample_seed')}
        core=cores[job['dataset']];score=QuantileScore(core);ref=core.features(core.reference).numpy()
        p=bind(Path(job['folder'])/'graph.npz');b=read(p.parent/'binding.json')
        assert BIND[str(p)]==b['graph_sha256']==previous_bind[str(p)]
        with np.load(p) as z:
            assert np.array_equal(z['types'],core.types)
            adj=np.zeros((core.n,core.n));adj[z['edges'][0],z['edges'][1]]=1;adj+=adj.T
        q=core.query.numpy();inc=RootedIncremental(core,adj[q[0],q[1]])
        old=by_type(core,score,inc.raw,ref)
        selected=[r for r in edges if all(r[k]==v for k,v in ident.items())]
        ix=np.array([r['slot'] for r in selected]);newraw,_=inc.raw_flips(ix);new=by_type(core,score,newraw,ref)
        for j,r in enumerate(selected):
            deltas={t:{k:float(v[j]-old[t][k][0]) for k,v in vals.items()} for t,vals in new.items()}
            recomposed=sum(v['score'] for v in deltas.values());maxerr=max(maxerr,abs(recomposed-r['new_score_change']))
            assert abs(recomposed-r['new_score_change'])<1e-10
            for t,d in deltas.items():
                type_rows.append(dict(**ident,slot=int(ix[j]),relation=r['relation'],present=r['present'],type=int(t),node_count=len(core.nodes[t]),rewarded=r['new_score_change']<-EPS,
                    **{k+'_change':v for k,v in d.items()}))
            checked=[v for k,v in r.items() if k.endswith('_change') and k not in ('score_change','new_score_change')]
            edge_rows.append(dict(**ident,slot=int(ix[j]),rewarded=recomposed<-EPS,
                score_improved_types=sum(v['score']<-EPS for v in deltas.values()),score_worse_types=sum(v['score']>EPS for v in deltas.values()),
                degree_improved_types=sum(v['total_degree_W1']<-EPS for v in deltas.values()),degree_worse_types=sum(v['total_degree_W1']>EPS for v in deltas.values()),
                hub_worse_types=sum(v['hub_error']>EPS for v in deltas.values()),lower_half_worse_types=sum(v['lower_half_degree_error']>EPS for v in deltas.values()),
                aggregate_no_gain=all(v>=-EPS for v in checked) and any(v>EPS for v in checked)))
            if job['dataset']=='PubMed' and job['train_seed']==2026110421 and job['arm']=='A' and job['sample_seed']==2026110232 and ix[j]==3489:
                for t,c in score.cache.items():
                    ids=c['nodes'];cols=np.flatnonzero(c['mask']);n=len(ids);d=len(cols)
                    x=np.sort(inc.raw[ids][:,cols],axis=0);xx=np.sort(newraw[j,ids][:,cols],axis=0);y=np.sort(ref[ids][:,cols],axis=0)
                    for i,k in zip(*np.where(x!=xx)):
                        coef=.5/len(score.cache)/n/d/(c['scale'][k]**2)
                        exact=coef*((np.log1p(xx[i,k])-np.log1p(y[i,k]))**2-(np.log1p(x[i,k])-np.log1p(y[i,k]))**2)
                        example.append(dict(type=int(t),feature=int(cols[k]),rank=int(i),node_count=n,active_dimensions=d,scale=float(c['scale'][k]),
                            before=float(x[i,k]),after=float(xx[i,k]),reference=float(y[i,k]),raw_abs_change=float(abs(xx[i,k]-y[i,k])-abs(x[i,k]-y[i,k])),
                            contribution=float(exact),local_derivative=float(2*coef*(np.log1p(x[i,k])-np.log1p(y[i,k]))/(1+x[i,k]))))
    save('per_type_edges',type_rows);save('per_edge',edge_rows);save('example_rank_prices',example)
    summary=[]
    for ds in cores:
        rr=[r for r in edge_rows if r['dataset']==ds];good=[r for r in rr if r['rewarded']]
        summary.append(dict(dataset=ds,selected_flips=len(rr),rewarded=len(good),
            rewarded_cross_type_score_tradeoff=sum(r['score_improved_types']>0 and r['score_worse_types']>0 for r in good),
            rewarded_cross_type_degree_tradeoff=sum(r['degree_improved_types']>0 and r['degree_worse_types']>0 for r in good),
            rewarded_any_type_degree_worse=sum(r['degree_worse_types']>0 for r in good),
            rewarded_any_type_hub_worse=sum(r['hub_worse_types']>0 for r in good),
            rewarded_any_type_lower_half_worse=sum(r['lower_half_worse_types']>0 for r in good),
            rewarded_aggregate_no_gain=sum(r['aggregate_no_gain'] for r in good)))
    save('summary',summary)
    groups=[]
    for ds in cores:
        for t in cores[ds].nodes:
            rr=[r for r in type_rows if r['dataset']==ds and r['type']==t and r['rewarded']]
            groups.append(dict(dataset=ds,type=int(t),rewarded_flips=len(rr),**{k+'_'+direction:sum(r[k+'_change']*sgn>EPS for r in rr) for k in ('score','total_degree_W1','joint_degree_TV','hub_error','lower_half_degree_error') for direction,sgn in [('worse',1),('better',-1)]}))
    save('type_summary',groups)
    assert not torch.cuda.is_initialized()
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in BIND.items())
    save('binding',BIND);save('receipt',dict(status='complete',wall_seconds=time.time()-START,peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
         graphs=len(jobs),flips=len(edge_rows),type_rows=len(type_rows),recomposition_max_abs_error=maxerr,GPU_hours=0,model_calls=0,updates=0,new_graphs=0,assets_unchanged=True))
    print(json.dumps(summary,indent=2));print((OUT/'receipt.json').read_text())
if __name__=='__main__':main()
