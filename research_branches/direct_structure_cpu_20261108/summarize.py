import csv,json
from pathlib import Path
from collections import defaultdict
import numpy as np
R=Path(__file__).resolve().parents[2];O=R/'results/direct_structure_cpu_20261108'
rows=list(csv.DictReader((O/'graph_summary.csv').open()))
def csvsave(n,x):
    with (O/n).open('w') as f:w=csv.DictWriter(f,list(x[0]));w.writeheader();w.writerows(x)
def aggregate(keys):
    groups=defaultdict(list)
    for x in rows:groups[tuple(x[k] for k in keys)].append(x)
    out=[]
    for key,g in sorted(groups.items()):
        rec=dict(zip(keys,key));rec['graphs']=len(g)
        for k in ['slots','improving','common_improvements','common_missed','nonbetter_damage_rewarded','tradeoffs_rewarded']:
            rec[k]=sum(int(x[k]) for x in g)
        rec['mass']=sum(float(x['mass']) for x in g)
        for k in ['triangle_abs_error','total_degree_W1','joint_degree_TV']:
            bad=sum(int(x[k+'_bad']) for x in g);mass=sum(float(x[k+'_bad_mass']) for x in g)
            rec[k+'_bad']=bad;rec[k+'_bad_fraction']=bad/rec['improving'] if rec['improving'] else None
            rec[k+'_bad_mass_fraction']=mass/rec['mass'] if rec['mass'] else None
        out.append(rec)
    return out
for name,keys in [('dataset_summary.csv',['dataset','method']),('time_summary.csv',['dataset','t','method']),('parent_summary.csv',['condition','arm','method']),('triangle_state_summary.csv',['dataset','triangle_state','method'])]:csvsave(name,aggregate(keys))
costs=list(csv.DictReader((O/'costs.csv').open()));summary=[]
for ds in ['IMDB','PubMed']:
    group=[x for x in costs if x['dataset']==ds];a=np.array([float(x['seconds']) for x in group])
    summary.append(dict(dataset=ds,graphs=len(a),min_seconds=float(a.min()),median_seconds=float(np.median(a)),mean_seconds=float(a.mean()),max_seconds=float(a.max()),degree_scale=float(group[0]['degree_scale']),triangle_scale=float(group[0]['triangle_scale'])))
(O/'cost_summary.json').write_text(json.dumps(summary,indent=2))
gradient_rows=[]
for row in costs:
    name=f"{row['condition']}_{row['arm']}_t{row['t']}_draw{row['draw']}"
    with np.load(O/f'{name}_direct.npz') as z:g=z['logit_conditional_estimate'].copy()
    with np.load(R/f'results/hard_structure_edge_direction_cpu_20261108/{name}_flips.npz') as z:
        old=z['gradient_mass']*np.sign(z['score_flip'])*z['flip_sign']
    n=float(np.linalg.norm(g));n0=float(np.linalg.norm(old))
    gradient_rows.append(dict(condition=row['condition'],arm=row['arm'],t=row['t'],draw=row['draw'],direct_L2=n,MMD_L2=n0,ratio=n/n0 if n0 else None,cosine=float(np.dot(g,old)/(n*n0)) if n*n0 else None))
csvsave('free_logit_gradient_comparison.csv',gradient_rows)
print(json.dumps(dict(datasets=aggregate(['dataset','method']),times=aggregate(['dataset','t','method']),costs=summary),indent=2))
