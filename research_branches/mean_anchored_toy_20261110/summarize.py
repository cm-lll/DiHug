"""Summaries only, no extra scientific computation."""
import csv,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/mean_anchored_toy_20261110_attempt2'
finals=json.loads((OUT/'finals.json').read_text());updates=json.loads((OUT/'updates.json').read_text());flat=[dict(seed=x['seed'],arm=x['arm'],updates=x['updates'],readout=x['readout'],**x['terminal']) for x in finals];keys=list(finals[0]['terminal']);agg=[];comparisons=[]
def mean_sd(v):
    m=sum(v)/len(v);return m,math.sqrt(sum((x-m)**2 for x in v)/(len(v)-1))
for arm in ['average','cagrad','same_norm_average']:
    for n in [20,50,100]:
        for rd in ['raw','EMA']:
            rr=[x for x in flat if (x['arm'],x['updates'],x['readout'])==(arm,n,rd)];row=dict(arm=arm,updates=n,readout=rd,n=len(rr))
            for k in keys:row[k],row[k+'_SD']=mean_sd([x[k] for x in rr])
            agg.append(row)
for x in flat:
    if x['arm']!='cagrad':continue
    for control in ['average','same_norm_average']:
        y=next(y for y in flat if y['seed']==x['seed'] and y['updates']==x['updates'] and y['readout']==x['readout'] and y['arm']==control)
        comparisons.append(dict(seed=x['seed'],updates=x['updates'],readout=x['readout'],control=control,**{k:x[k]-y[k] for k in keys}))
geometry=[]
for arm in ['average','cagrad','same_norm_average']:
    rr=[x for x in updates if x['arm']==arm]
    geometry.append(dict(arm=arm,updates=len(rr),finite_mean_harm=sum(sum(x['after_costs'])>sum(x['before_costs'])+5e-12 for x in rr),finite_component_harm=[sum(x['after_costs'][k]>x['before_costs'][k]+1e-12 for x in rr) for k in range(5)],mean_norm=sum(x['actual_norm'] for x in rr)/len(rr)))
for name,rows in [('aggregate.csv',agg),('terminal_metrics.csv',flat),('paired_differences.csv',comparisons)]:
    with (OUT/name).open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(OUT/'descriptive_summary.json').write_text(json.dumps(dict(aggregate=agg,per_seed=flat,comparisons=comparisons,geometry=geometry),indent=2))
print(json.dumps(dict(complete=True,raw100=[x for x in agg if x['updates']==100 and x['readout']=='raw'],geometry=geometry),indent=2))
