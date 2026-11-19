"""Independent saved-table checks and descriptive aggregation; no model calls."""
import json,hashlib,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/pair_allocation_signal_20261109'
def read(n):return json.loads((OUT/(n+'.json')).read_text())
def save(n,x):(OUT/(n+'.json')).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
rows=read('pairs');summaries=read('summary');binding=read('binding');start=time.monotonic();eps=1e-12
max_error=0.;agg=[]
for summary in summaries:
    tag=summary['state'];rr=[r for r in rows if r['state']==tag]
    with np.load(OUT/(tag+'_vectors.npz')) as z:p=z['probability'];stored=z['pair'];rel=z['relation']
    with np.load(ROOT/'results/joint_degree_full_signal_20261109'/(tag+'.npz')) as z:delta=z['new_delta'];bits=z['bits']
    g=np.zeros(len(p));un=np.zeros(len(p));opposed=0;aligned=0
    for r in rr:
        e,f=r['remove'],r['add'];assert bits[e]==1 and bits[f]==0 and rel[e]==rel[f]==r['relation']
        # Constant F10 may be set to zero: derivatives do not depend on it.
        F00,F10,F01,F11=float(delta[e]),0.,r['score_change'],float(delta[f])
        def risk(ze,zf):
            pe,pf=1/(1+np.exp(-ze)),1/(1+np.exp(-zf))
            return (1-pe)*(1-pf)*F00+pe*(1-pf)*F10+(1-pe)*pf*F01+pe*pf*F11
        ze,zf=np.log(p[e]/(1-p[e])),np.log(p[f]/(1-p[f]));h=1e-5
        ge=(risk(ze+h,zf)-risk(ze-h,zf))/(2*h);gf=(risk(ze,zf+h)-risk(ze,zf-h))/(2*h)
        max_error=max(max_error,abs(ge-r['independent_g_remove']),abs(gf-r['independent_g_add']))
        g[e]+=r['pair_g_remove']/160;g[f]+=r['pair_g_add']/160
        un[e]+=r['independent_g_remove']/160;un[f]+=r['independent_g_add']/160
        direction=r['independent_g_add']-r['independent_g_remove'];pair=r['pair_g_add']
        if abs(pair)>eps and abs(direction)>eps:
            aligned+=pair*direction>0;opposed+=pair*direction<0
        assert abs(sum(r[k]*w for k,w in [('joint_distance_change',.125),('relation_W1_change',.125),
            ('low_change',.25),('hub_change',.25),('typed_triangle_L1_change',.125),('node_triangle_W1_change',.125)])-r['score_change'])<1e-10
    assert np.allclose(g,stored,rtol=0,atol=1e-18)
    good=[r for r in rr if r['score_change']<-eps];bad=[r for r in good if r['triangle_error_change']>eps]
    agg.append(dict(state=tag,aligned_with_same_pair_independent_position=aligned,opposed=opposed,
        q_min=min(r['q'] for r in rr),q_max=max(r['q'] for r in rr),
        both_reference_nonedge_pairs=sum(r['label_remove']==0 and r['label_add']==0 for r in rr),
        triangle_cost_total=sum(r['triangle_error_change'] for r in bad),
        triangle_cost_max=max([r['triangle_error_change'] for r in bad],default=0)))
assert max_error<1e-10
assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in binding.items())
save('independent_checks',dict(original_two_slot_risk_finite_differences=1280,max_error=max_error,
    score_decompositions=640,vector_reconstructions=4,assets_unchanged=True,seconds=time.monotonic()-start))
save('additional_summary',agg)
save('aggregate',dict(pairs=len(rows),rewarded=sum(x['rewarded'] for x in summaries),
    triangle_cost=sum(x['triangle_cost'] for x in summaries),hub_cost=sum(x['hub_cost'] for x in summaries),
    low_cost=sum(x['low_cost'] for x in summaries),joint_cost=sum(x['joint_cost'] for x in summaries),
    isolates_cost=sum(x['isolates_cost'] for x in summaries),
    conditional_probability_graphs=False,training_seeds=1,parents=2,archived_hard_draws=4,
    adopted=False,formal_training_authorized=False))
print((OUT/'independent_checks.json').read_text());print((OUT/'aggregate.json').read_text())
