"""Explain ALL newly rewarded adverse operations; do not change the candidate."""
from run import *
started=time.time();edges=json.loads((OUT/'final_edges.json').read_text());jobs=json.loads((RUN/'evaluation_jobs.json').read_text())
bad=[r for r in edges if r['candidate_change']<-EPS and r['new_score_change']>=-EPS and all(r[k+'_change']>EPS for k in ('total_degree_W1','joint_degree_TV','triangle_error'))]
def parts(core,s,raw):
    raw=raw[None] if raw.ndim==2 else raw;out={}
    for t,c in s.cache.items():
        z=raw[:,c['ids']][:,:,c['mask']]/c['scale'];m=z.mean(1);cen=z-m[:,None]
        ma=(m-c['mu'])**2;ax=.5*np.square(np.sort(cen,axis=1)-c['target_axis']).mean(1)
        cols=c['cols'];den=len(core.nodes)*len(cols)
        for block,mask in [('degree',cols<core.r),('triangle',(cols>=core.r)&(cols<core.base_width)),('path',cols>=core.base_width)]:
            out[(t,block,'mean')]=ma[:,mask].sum(1)/den
            out[(t,block,'centered_axis')]=ax[:,mask].sum(1)/den
        out[(t,'all','centered_mixed')]=.5*np.square(np.sort(cen@c['u'],axis=1)-c['target_mixed']).mean((1,2))/len(core.nodes)
    return out
rows=[];operations=[]
for j in jobs:
    ident={k:j[k] for k in ('dataset','train_seed','arm','sample_seed')};rr=[r for r in bad if all(r[k]==v for k,v in ident.items())]
    if not rr:continue
    with np.load(RUN/f'{j["dataset"]}_cpu_cost_panel.npz') as z:core=RootedScore(z['types'],z['query'],z['families'],z['reference'],10)
    s=CountShapeScore(core)
    with np.load(Path(j['folder'])/'graph.npz') as z:
        a=np.zeros((core.n,core.n));a[z['edges'][0],z['edges'][1]]=1;a+=a.T
    q=core.query.numpy();inc=RootedIncremental(core,a[q[0],q[1]]);old=parts(core,s,inc.raw);newraw,_=inc.raw_flips([r['slot'] for r in rr]);new=parts(core,s,newraw)
    for i,r in enumerate(rr):
        contributions=[]
        for key in old:
            d=float(new[key][i]-old[key][0]);rows.append(dict(**ident,slot=r['slot'],type=int(key[0]),block=key[1],component=key[2],delta=d));contributions.append((key,d))
        assert abs(sum(x[1] for x in contributions)-r['candidate_change'])<1e-9
        most=min(contributions,key=lambda x:x[1]);operations.append(dict(**r,largest_gain_type=int(most[0][0]),largest_gain_block=most[0][1],largest_gain_component=most[0][2],largest_gain_delta=most[1]))
save('new_adverse_decomposition',rows);save('new_adverse_operations',operations)
counts={}
for r in operations:
    k=(r['dataset'],r['largest_gain_block'],r['largest_gain_component']);counts[k]=counts.get(k,0)+1
summary=dict(new_adverse_total=len(operations),largest_gain_groups=[dict(dataset=k[0],block=k[1],component=k[2],operations=v) for k,v in counts.items()],CPU_seconds=time.time()-started)
save('new_adverse_summary',summary);print(json.dumps(summary,indent=2))
