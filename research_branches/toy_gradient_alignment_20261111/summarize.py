import json
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parents[2]/'results/toy_gradient_alignment_20261111'
rows=json.loads((P/'conditions.json').read_text());probes=json.loads((P/'probes.json').read_text());summary=[]
for arm in ('original','typed_endpoint'):
    for n in (0,20,50,100):
        rr=[x for x in rows if x['arm']==arm and x['update']==n]
        row=dict(arm=arm,update=n,seeds=[x['seed'] for x in rr],cosines=[x['cos_train_terminal'] for x in rr],norm_ratios=[x['terminal_vs_local_scale'] for x in rr])
        for d in ('train','terminal'):
            pp=[x for x in probes if x['arm']==arm and x['update']==n and x['direction']==d and x['norm']==.001]
            row[d]=dict(mean_terminal_change=float(np.mean([x['terminal_mean_change'] for x in pp])),per_risk_mean_change=np.mean([x['terminal_change'] for x in pp],axis=0).tolist(),per_seed_mean_change=[x['terminal_mean_change'] for x in pp])
        summary.append(row)
totals={}
for d in ('train','terminal'):
    for length in (.001,.0001):
        pp=[x for x in probes if x['direction']==d and x['norm']==length]
        changes=np.array([x['terminal_change'] for x in pp])
        totals[f'{d}_{length}']=dict(n=len(pp),mean_improved=int((changes.mean(1)<0).sum()),per_risk_increased=(changes>0).sum(0).tolist(),all_five_improved=int((changes<0).all(1).sum()))
(P/'aggregate.json').write_text(json.dumps(dict(groups=summary,totals=totals),indent=2))
print(json.dumps(totals,indent=2))
for x in summary:
    if x['update']==100:print(json.dumps(x))
