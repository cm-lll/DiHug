"""Reproduce descriptive summaries from the CPU decomposition CSVs."""
import csv,json,statistics as st
from pathlib import Path
P=Path(__file__).resolve().parents[2]/'results/joint_transport_decomposition_20261113'
def rows(name):return list(csv.DictReader((P/(name+'.csv')).open()))
def summary(name,keys,fields):
    rr=rows(name);out=[]
    for key in sorted(set(tuple(r[k] for k in keys) for r in rr)):
        xs=[r for r in rr if tuple(r[k] for k in keys)==key];x=dict(zip(keys,key));x['n']=len(xs)
        for f in fields:
            v=[float(r[f]) for r in xs];x[f+'_mean']=st.mean(v);x[f+'_sd']=st.stdev(v) if len(v)>1 else None
        out.append(x)
    (P/(name+'_summary.json')).write_text(json.dumps(out,indent=2))
    return out
summary('graphs',['dataset','train_seed','arm'],['score','degree_cost','heat_cost','triangles','triangle_error','MP3_error','degree_W1','joint_TV','isolates','LCC'])
summary('types',['dataset','train_seed','arm','type'],['degree_cost','heat_cost','return_cost','arrival_cost','low_count','isolates','max_degree','triangle_participations'])
b=rows('training_batches');u=rows('training_updates');report={}
for ds in ('IMDB','PubMed'):
    x=[r for r in b if r['dataset']==ds];v=[float(r['heat_degree_ratio']) for r in x]
    report[ds]=dict(batches=len(x),heat_advantage_larger=sum(y>1 for y in v),ratio_min=min(v),ratio_median=st.median(v),ratio_max=max(v),lowest_cost_graph_more_triangle_error_than_highest=sum(float(r['best_cost_triangle_error'])>float(r['worst_cost_triangle_error']) for r in x))
report['updates']=dict(n=len(u),clipped=sum(float(r['clip_scale'])<1 for r in u),norm_min=min(float(r['gradient_norm_before_clip']) for r in u),norm_max=max(float(r['gradient_norm_before_clip']) for r in u))
report['limits']=['scalar advantage decomposition is not parameter-gradient decomposition','same graph samples and batch pairs are not independent training replicates','posthoc observations, no tuned reward or new models','same joint assignment per graph; isolated-node group membership changes across graphs']
(P/'summary.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
