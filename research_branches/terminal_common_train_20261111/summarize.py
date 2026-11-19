import json
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parents[2]/'results/terminal_common_train_20261111'
R=json.loads((P/'finals.json').read_text());U=json.loads((P/'updates.json').read_text());groups=[]
keys=['mean_five_cost','relation_count','joint_degree','triangle001','triangle011','component_cost','connected','reference_isomorph_probability','triangle_MAE','isomorph_entropy']
for arm in ('common','mean','norm_matched_mean'):
    for n in (0,5,10,20):
        for rd in ('raw','EMA'):
            rr=[r for r in R if r['arm']==arm and r['update']==n and r['readout']==rd]
            groups.append(dict(arm=arm,update=n,readout=rd,n=len(rr),means={k:float(np.mean([r[k] for r in rr])) for k in keys},sd={k:float(np.std([r[k] for r in rr],ddof=1)) for k in keys}))
update_stats=[]
for arm in ('common','mean','norm_matched_mean'):
    rr=[r for r in U if r['arm']==arm];ch=np.array([r['actual_change'] for r in rr]);sl=np.array([r['first_order'] for r in rr])
    update_stats.append(dict(arm=arm,n=len(rr),strict_five_decrease=int((ch<0).all(1).sum()),per_cost_increase=(ch>0).sum(0).tolist(),max_actual_increase=float(ch.max()),per_cost_positive_first_order=(sl>0).sum(0).tolist(),sum_update_norm=float(sum(r['norm'] for r in rr))))
(P/'aggregate.json').write_text(json.dumps(dict(groups=groups,update_stats=update_stats),indent=2))
print(json.dumps(update_stats,indent=2))
for g in groups:
    if g['update'] in (0,20) and (g['arm']=='common' or g['update']==20):print(g['arm'],g['update'],g['readout'],g['means'],g['sd'])
