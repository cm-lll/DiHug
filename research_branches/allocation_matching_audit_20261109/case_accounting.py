"""Account for exact matching gains/costs in every previously fixed adverse case."""
from run import *
started=time.time();cases=json.loads((OUT/'all24_previous_adverse_cases.json').read_text());jobs=json.loads((RUN/'evaluation_jobs.json').read_text())
def ledger(audit,raw):
    out={};core=audit.base.core;nt=len(core.nodes)
    for t,c in audit.cache.items():
        z=raw[c['ids']][:,c['mask']]/c['scale'];mu=z.mean(0);center=z-mu
        yz=core.features(core.reference).numpy()[c['ids']][:,c['mask']]/c['scale'];ym=yz.mean(0);yc=yz-ym
        xx=center@c['w'];yy=yc@c['w'];cost=((xx[:,None]-yy[None])**2).sum(-1);ri,ci=linear_sum_assignment(cost)
        delta=center[ri]-yc[ci];d=len(c['cols'])
        means=(mu-c['mu'])**2/d/nt;axis=.5*(delta**2).mean(0)/d/nt
        mixed=.5*((delta@c['u'])**2).mean()/nt
        cols=c['cols']
        for block,mask in [('degree',cols<core.r),('triangle',(cols>=core.r)&(cols<core.base_width)),('path',cols>=core.base_width)]:
            out[(int(t),block,'mean')]=float(means[mask].sum())
            out[(int(t),block,'joint_matched_coordinate')]=float(axis[mask].sum())
        out[(int(t),'all','joint_matched_mixed')]=float(mixed)
    return out
rows=[];ops=[]
for job in jobs:
    ident={k:job[k] for k in ['dataset','train_seed','arm','sample_seed']};rr=[r for r in cases if all(r[k]==v for k,v in ident.items())]
    if not rr:continue
    with np.load(RUN/f'{job["dataset"]}_cpu_cost_panel.npz') as z:core=RootedScore(z['types'],z['query'],z['families'],z['reference'],10)
    audit=JointAudit(core)
    with np.load(Path(job['folder'])/'graph.npz') as z:
        adj=np.zeros((core.n,core.n));adj[z['edges'][0],z['edges'][1]]=1;adj+=adj.T
    q=core.query.numpy();inc=RootedIncremental(core,adj[q[0],q[1]]);old=ledger(audit,inc.raw);newraw,_=inc.raw_flips([r['slot'] for r in rr])
    for j,r in enumerate(rr):
        new=ledger(audit,newraw[j]);dd={k:new[k]-old[k] for k in old}
        assert abs(sum(dd.values())-r['joint_change'])<1e-9
        for k,v in dd.items():rows.append(dict(**ident,slot=r['slot'],type=k[0],block=k[1],component=k[2],delta=v))
        winner=min(dd,key=dd.get)
        coord=sum(v for k,v in dd.items() if k[2]=='joint_matched_coordinate');mix=sum(v for k,v in dd.items() if k[2]=='joint_matched_mixed');mass=sum(v for k,v in dd.items() if k[2]=='mean')
        ops.append(dict(**r,exact_matched_axis_change=coord,exact_matched_mixed_change=mix,exact_mass_change=mass,largest_gain_type=winner[0],largest_gain_block=winner[1],largest_gain_component=winner[2],largest_gain_delta=dd[winner]))
save('case_components',rows);save('case_ledger',ops)
summary=[]
for ds in ['IMDB','PubMed']:
    rr=[r for r in ops if r['dataset']==ds and r['joint_change']<-EPS]
    summary.append(dict(dataset=ds,remaining_rewarded=len(rr),joint_matched_axis_also_improves=sum(r['exact_matched_axis_change']<-EPS for r in rr),
        joint_matched_mixed_improves=sum(r['exact_matched_mixed_change']<-EPS for r in rr),mass_also_improves=sum(r['exact_mass_change']<-EPS for r in rr)))
save('case_ledger_summary',dict(rows=summary,CPU_seconds=time.time()-started,peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024))
print(json.dumps(summary,indent=2))
