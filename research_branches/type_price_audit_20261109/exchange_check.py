"""Exhaustive finite counterfactual on ONE declared case; no production edits."""
from run import *
from feedback_audit_cpu import metrics
started=time.time()
save('exchange_protocol',dict(case='PubMed0421 A sample2026110232, fixed absent slot3489',
    enumeration='add3489 and remove EACH existing relation1 edge; no deletion selected using outcomes',
    purpose='determine whether observed distribution/count tradeoff is unavoidable for this case',
    limitation='one known selected case, not a new objective or learned update; no changes to diffusion transitions',
    GPU=0,model_calls=0,random_graph_samples=0,wall_seconds=120))
jobs=json.loads((RUN/'evaluation_jobs.json').read_text())
j=next(j for j in jobs if j['dataset']=='PubMed' and j['train_seed']==2026110421 and j['arm']=='A' and j['sample_seed']==2026110232)
with np.load(RUN/'PubMed_cpu_cost_panel.npz') as z:core=RootedScore(z['types'],z['query'],z['families'],z['reference'],10)
with np.load(Path(j['folder'])/'graph.npz') as z:
    adj=np.zeros((core.n,core.n));adj[z['edges'][0],z['edges'][1]]=1;adj+=adj.T
q=core.query.numpy();bits=adj[q[0],q[1]];inc=RootedIncremental(core,bits);ref=core.features(core.reference).numpy();score=QuantileScore(core)
base=metrics(core,inc.raw,ref);base_types=by_type(core,score,inc.raw,ref);b=bits.copy();assert b[3489]==0;b[3489]=1
added=RootedIncremental(core,b);remove=np.flatnonzero((core.families.numpy()==1)&(bits==1))
raw,_=added.raw_flips(remove);m=metrics(core,raw,ref);ty=by_type(core,score,raw,ref);s=score.raw_score(raw)[0]-score.raw_score(inc.raw)[0]
rows=[]
for k,e in enumerate(remove):
    rows.append(dict(removed_slot=int(e),removed_u=int(q[0,e]),removed_v=int(q[1,e]),same_Gene=bool(21 in q[:,e]),same_Disease=bool(74 in q[:,e]),
        score_change=float(s[k]),any_type_degree_worse=any(ty[t]['total_degree_W1'][k]-base_types[t]['total_degree_W1'][0]>EPS for t in ty),
        any_type_hub_worse=any(ty[t]['hub_error'][k]-base_types[t]['hub_error'][0]>EPS for t in ty),
        **{key+'_change':float(v[k]-base[key][0]) for key,v in m.items()}))
save('exchange_all',rows)
summary=[]
for group,rr in [('all_relation_deletions',rows),('same_Gene_deletions',[r for r in rows if r['same_Gene']])]:
    good=[r for r in rr if all(r[k+'_change']<=EPS for k in base) and any(r[k+'_change']<-EPS for k in base)]
    summary.append(dict(group=group,enumerated=len(rr),all8_metrics_nonworse_some_better=len(good),
        additionally_no_type_degree_or_hub_worse=sum(not r['any_type_degree_worse'] and not r['any_type_hub_worse'] for r in good),
        pareto_score_rewarded=sum(r['score_change']<-EPS for r in good)))
save('exchange_summary',dict(rows=summary,CPU_seconds=time.time()-started,peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
    interpretation='finite existence of redistribution alternatives; not proof model will learn them, not connectivity guarantee or complete motif dominance'))
assert time.time()-started<120 and not torch.cuda.is_initialized()
print((OUT/'exchange_summary.json').read_text())
