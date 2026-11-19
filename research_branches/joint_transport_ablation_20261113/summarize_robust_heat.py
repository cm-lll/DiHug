"""CPU summaries, not additional fitting or model computation."""
import csv,json,statistics as st
from pathlib import Path
P=Path(__file__).resolve().parents[2]/'results/robust_heat_transport_20261113'
r=list(csv.DictReader((P/'fresh_and_severity.csv').open()));summary=[]
for ds in ('IMDB','PubMed'):
    for route in ('isolation','hub','closure','ordinary','two_halves'):
        a=[x for x in r if x['dataset']==ds and x['route']==route];x=dict(dataset=ds,route=route,n=len(a))
        for k in ('original_total','robust_total','robust_degree','robust_heat','isolates','LCC','triangles'):
            vals=[float(z[k]) for z in a];x[k+'_mean']=st.mean(vals);x[k+'_sd']=st.stdev(vals) if len(vals)>1 else None
        summary.append(x)
with (P/'summary.csv').open('w') as f:
    w=csv.DictWriter(f,list(summary[0]));w.writeheader();w.writerows(summary)
paths=[]
for ds in ('IMDB','PubMed'):
    for seed in sorted(set(x['index'] for x in r if x['dataset']==ds and x['route']=='isolation_path')):
        a=sorted([x for x in r if x['dataset']==ds and x['route']=='isolation_path' and x['index']==seed],key=lambda x:int(x['level']));vals=[float(x['robust_total']) for x in a]
        paths.append(dict(dataset=ds,seed=int(seed),levels=[int(x['level']) for x in a],costs=vals,strictly_increasing=all(x<y for x,y in zip(vals,vals[1:]))))
(P/'isolation_path_checks.json').write_text(json.dumps(paths,indent=2))
old=list(csv.DictReader((P/'old_graphs_rescored.csv').open()));models=[]
for key in sorted(set((x['dataset'],x['train_seed'],x['arm']) for x in old)):
    a=[x for x in old if (x['dataset'],x['train_seed'],x['arm'])==key];v=dict(zip(('dataset','train_seed','arm'),key))
    for k in ('original','robust','robust_degree','robust_heat','robust_isolate_contribution','triangles'):
        vals=[float(x[k]) for x in a];v[k+'_mean']=st.mean(vals);v[k+'_sampling_sd']=st.stdev(vals)
    models.append(v)
(P/'model_rescore_summary.json').write_text(json.dumps(models,indent=2));print('summaries complete; monotonic paths',sum(x['strictly_increasing'] for x in paths),'/',len(paths))
