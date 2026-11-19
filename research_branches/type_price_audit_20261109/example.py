"""Read-only count/quantile ledger for the predeclared example, not new scoring."""
from run import *
started=time.time()
jobs=json.loads((RUN/'evaluation_jobs.json').read_text())
j=next(j for j in jobs if j['dataset']=='PubMed' and j['train_seed']==2026110421 and j['arm']=='A' and j['sample_seed']==2026110232)
with np.load(RUN/'PubMed_cpu_cost_panel.npz') as z:core=RootedScore(z['types'],z['query'],z['families'],z['reference'],10)
with np.load(Path(j['folder'])/'graph.npz') as z:
    adj=np.zeros((core.n,core.n));adj[z['edges'][0],z['edges'][1]]=1;adj+=adj.T
q=core.query.numpy();inc=RootedIncremental(core,adj[q[0],q[1]]);ref=core.features(core.reference).numpy()
new=inc.raw_flips([3489])[0][0];score=QuantileScore(core);rows=[]
for t,c in score.cache.items():
    ids=c['nodes'];a=inc.raw[ids];b=new[ids];y=ref[ids];cols=np.flatnonzero(c['mask'])
    for block,sl in [('degree',slice(0,core.r)),('triangle',slice(core.r,core.base_width)),('path',slice(core.base_width,None))]:
        rows.append(dict(type=int(t),block=block,reference_sum=float(y[:,sl].sum()),before_sum=float(a[:,sl].sum()),after_sum=float(b[:,sl].sum())))
rel=int(core.families[3489]);edges_before=int(inc.bits[core.families.numpy()==rel].sum());edges_ref=int(core.reference[core.families==rel].sum())
prices=json.loads((OUT/'example_rank_prices.json').read_text());g=next(r for r in prices if r['type']==0 and r['feature']==1);d=next(r for r in prices if r['type']==1 and r['feature']==1)
ratio=abs(d['contribution']/g['contribution'])
# Factor exactly into population, reference scale, and finite log-residual change.
population=g['node_count']/d['node_count'];scale=(g['scale']/d['scale'])**2
raw_term=lambda r:abs((np.log1p(r['after'])-np.log1p(r['reference']))**2-(np.log1p(r['before'])-np.log1p(r['reference']))**2)
residual=raw_term(d)/raw_term(g)
assert abs(ratio-population*scale*residual)<1e-10
result=dict(slot=3489,relation=rel,endpoints=q[:,3489].tolist(),endpoint_types=core.types[q[:,3489]].tolist(),
    relation_edges_reference=edges_ref,relation_edges_before=edges_before,relation_edges_after=edges_before+1,
    degree_price_ratio_Disease_gain_over_Gene_cost=ratio,factors=dict(population=population,reference_scale=scale,finite_log_residual=residual),
    caution='ranks are distribution quantiles, not reference node identities; triangle sums count node participations, not unique triangles',
    block_totals=rows,CPU_seconds=time.time()-started)
save('example_count_ledger',result);print(json.dumps(result,indent=2))
