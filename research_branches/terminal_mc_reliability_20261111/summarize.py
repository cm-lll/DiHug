import json,collections
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parents[2]/'results/terminal_mc_reliability_20261111';R=json.loads((P/'estimates.json').read_text());T=json.loads((P/'truth.json').read_text())
groups=[]
for seed,n in [(None,None)]+[(s,t) for s in [2026110901,2026110902,2026110903] for t in [0,20]]:
    for K in [32,128,512]:
        rows=[r for r in R if r['K']==K and (seed is None or (r['seed']==seed and r['update']==n))];valid=[r for r in rows if r['status']=='strict_descent']
        group=dict(seed=seed,update=n,K=K,n=len(rows),statuses=dict(collections.Counter(r['status'] for r in rows)),relative_gradient_RMSE=float(np.sqrt(np.mean([r['relative_gradient_error']**2 for r in rows]))),mean_gradient_cos_median=float(np.median([r['mean_gradient_cos'] for r in rows if r['mean_gradient_cos'] is not None])),common_cos_median=float(np.median([r['cos_to_exact_common'] for r in valid])) if valid else None,true_any_harm=sum(r['true_any_harm'] for r in valid),true_all_improve=sum(r['true_all_strict_improve'] for r in valid),true_mean_worse=sum(r['true_mean_slope']>0 for r in valid),true_per_cost_harm=(np.array([r['true_normalized_slopes'] for r in valid])>1e-8).sum(0).tolist() if valid else [],max_true_normalized_harm=float(max(max(r['true_normalized_slopes']) for r in valid)) if valid else None)
        groups.append(group)
(P/'aggregate.json').write_text(json.dumps(groups,indent=2))
for r in groups:print(json.dumps(r))
