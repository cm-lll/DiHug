import csv,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/mean_anchored_pilot_20261110_attempt2'
rows=list(csv.DictReader((OUT/'scores.csv').open()));dec=list(csv.DictReader((OUT/'decisions.csv').open()));comparisons=list(csv.DictReader((OUT/'contrasts.csv').open()))
metrics=['relation_count','joint_degree','present_triangles','absent_triangles','components','mean_five']
summary={'models':[],'comparisons':{},'conditions':len(dec),'reporting_floor':1e-7,'not_significance_test':True}
for ds,seed in sorted(set((r['dataset'],r['seed']) for r in rows)):
    model={'dataset':ds,'seed':seed,'six_time_means':{},'current_input':{}}
    for arm in ('unchanged','original','average','cagrad','same_norm_average'):
        other=[r for r in rows if r['dataset']==ds and r['seed']==seed and r['arm']==arm and r['panel']!='same']
        current=[r for r in rows if r['dataset']==ds and r['seed']==seed and r['arm']==arm and r['panel']=='same']
        model['six_time_means'][arm]={m:sum(float(r[m]) for r in other)/len(other) for m in metrics} if other else None
        model['current_input'][arm]={m:float(current[0][m]) for m in metrics} if current else None
    summary['models'].append(model)
for control in ('average','same_norm_average','original','unchanged'):
    rr=[r for r in comparisons if r['control']==control];summary['comparisons'][control]={}
    for m in metrics:
        vals=[float(r[m]) for r in rr]
        summary['comparisons'][control][m]={'lower':sum(v< -1e-7 for v in vals),'higher':sum(v>1e-7 for v in vals),'within_floor':sum(abs(v)<=1e-7 for v in vals),'mean_delta':sum(vals)/len(vals)}
summary['direction']={k:{'min':min(float(r[k]) for r in dec),'max':max(float(r[k]) for r in dec),'mean':sum(float(r[k]) for r in dec)/len(dec)} for k in ('norm_ratio','mean_descent_ratio','cosine_to_average','solver_gap')}
summary['finite_mean_worse_than_unchanged']=sum(float(r['finite_mean_change'])>1e-7 for r in dec)
summary['nearly_zero']=sum(float(r['norm_ratio'])<1e-8 for r in dec)
summary['all_five_not_worse_than_same_norm']=sum(all(float(r[m])<=1e-7 for m in metrics[:-1]) for r in comparisons if r['control']=='same_norm_average')
(OUT/'summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
