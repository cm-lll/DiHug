"""CPU bookkeeping on all new eight-summary adverse edits, no new candidate.

Freeze old memberships/residual coefficients as value counterfactuals; symmetric
two-switch attribution sums exactly, not a causal or training claim.
"""
from run import *
START=time.time()
def evaluate(sc,raw,old_raw,move_mu,move_q):
    c0=sc.core;dsum=0.;hsum=0.
    for c in sc.cache.values():
        ids=c['ids'];d,h=sc.local(raw,ids)
        od,oh=sc.local(old_raw,ids)
        if not move_mu:
            p0=old_raw[ids,c0.base_width:].reshape(len(ids),c0.r,c0.r)
            den=od.sum(0);mu=np.divide(p0.sum(0),den[:,None],out=np.zeros((c0.r,c0.r)),where=den[:,None]>0)
            p=raw[ids,c0.base_width:].reshape(len(ids),c0.r,c0.r)
            h=np.concatenate([raw[ids,c0.r:c0.base_width],(p-d[:,:,None]*mu).reshape(len(ids),-1)],axis=1)
        d=d[:,c['dc']]/c['ds'];od=od[:,c['dc']]/c['ds'];h=h[:,c['hc']]/c['hs']
        q=membership(d if move_q else od,c['anchors'],c['h2']);w=q/q.sum(0)
        uni=np.ones((len(d),1))/len(d)
        dsum+=cdf_distance(d,c['d'],uni,uni,np.ones(1),c['ud'])
        hsum+=cdf_distance(h,c['h'],w,c['weights'],c['prior'],c['uh'])
    return .5*(dsum+hsum)/len(sc.cache)
rows=read(OUT/'final_edges.json')
bad=[r for r in rows if r['conditional_change']<-EPS and all(r[m+'_change']>=-EPS for m in METRICS) and any(r[m+'_change']>EPS for m in METRICS)]
ans=[];jobs=read(RUN/'evaluation_jobs.json');cores={}
for ds in ['IMDB','PubMed']:
    with np.load(bind(RUN/f'{ds}_cpu_cost_panel.npz')) as z:
        core=RootedScore(z['types'],z['query'],z['families'],z['reference'],10)
    cores[ds]=(core,ConditionalScore(core))
for job in jobs:
    ident={k:job[k] for k in ['dataset','train_seed','arm','sample_seed']}
    rr=[r for r in bad if all(r[k]==v for k,v in ident.items())]
    if not rr:continue
    core,sc=cores[job['dataset']]
    with np.load(bind(Path(job['folder'])/'graph.npz')) as z:
        a=np.zeros((core.n,core.n));a[z['edges'][0],z['edges'][1]]=1;a+=a.T
    q=core.query.numpy();inc=RootedIncremental(core,a[q[0],q[1]]);base=sc.terms(inc.raw)['total'][0]
    for r in rr:
        limits();raw,_=inc.raw_flips([r['slot']]);raw=raw[0]
        f00=evaluate(sc,raw,inc.raw,False,False);f10=evaluate(sc,raw,inc.raw,True,False)
        f01=evaluate(sc,raw,inc.raw,False,True);f11=evaluate(sc,raw,inc.raw,True,True)
        b=f00-base;mu=.5*(f10-f00+f11-f01);member=.5*(f01-f00+f11-f10)
        assert abs(f11-base-r['conditional_change'])<1e-10
        assert abs(b+mu+member-r['conditional_change'])<1e-10
        ans.append(dict(**r,held_geometry_change=float(b),residual_recenter_contribution=float(mu),
            membership_contribution=float(member),becomes_rewarded_after_geometry_moves=bool(b>=-EPS)))
save('adverse_attribution',ans)
summary=[]
for ds in cores:
    rr=[r for r in ans if r['dataset']==ds]
    summary.append(dict(dataset=ds,eight_summary_adverse_cases=len(rr),
        remains_rewarded_with_geometry_held=sum(r['held_geometry_change']<-EPS for r in rr),
        becomes_rewarded_after_geometry_moves=sum(r['becomes_rewarded_after_geometry_moves'] for r in rr),
        recenter_favors_edit=sum(r['residual_recenter_contribution']<-EPS for r in rr),
        membership_favors_edit=sum(r['membership_contribution']<-EPS for r in rr),
        sum_actual_change=sum(r['conditional_change'] for r in rr),
        sum_held_geometry_change=sum(r['held_geometry_change'] for r in rr),
        sum_recenter_contribution=sum(r['residual_recenter_contribution'] for r in rr),
        sum_membership_contribution=sum(r['membership_contribution'] for r in rr)))
save('attribution_summary',summary)
assert not torch.cuda.is_initialized()
save('attribution_receipt',dict(wall_seconds=time.time()-START,peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
    cases=len(ans),scope='all8-summary adverse edits; algebraic value attribution only',GPU_hours=0,
    failures=[],new_candidate_tested=False))
save('attribution_binding',BIND)
print(json.dumps(summary,indent=2));print((OUT/'attribution_receipt.json').read_text())
