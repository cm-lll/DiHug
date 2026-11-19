"""Read-only summary of predeclared CPU sensitivity panel, no model calls."""
import csv,json,statistics as st,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];P=ROOT/'results/transport_sensitivity_20261113'
rows=[]
for version in ('original','candidate'):
    rr=list(csv.DictReader((P/(version+'.csv')).open()))
    for ds in ('IMDB','PubMed'):
        for split in ('development','heldout'):
            for route in ('isolation','hub','closure','ordinary'):
                v=[r for r in rr if r['dataset']==ds and r['split']==split and r['route']==route];assert len(v)==8
                out=dict(version=version,dataset=ds,split=split,route=route,n=8)
                for k in ('cost','degree','heat','triangle_delta','isolates','max_degree'):
                    a=[float(r[k]) for r in v];out.update({k+'_mean':st.mean(a),k+'_median':st.median(a),k+'_sd':st.stdev(a),k+'_min':min(a),k+'_max':max(a)})
                rows.append(out)
with (P/'summary.csv').open('w') as f:
    w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
(P/'summary.json').write_text(json.dumps(rows,indent=2))
ratios=[]
for ds in ('IMDB','PubMed'):
    for version in ('original','candidate'):
        by={x['route']:x for x in rows if x['dataset']==ds and x['version']==version and x['split']=='heldout'}
        ratios.append(dict(dataset=ds,version=version,isolation_over_hub=by['isolation']['cost_mean']/by['hub']['cost_mean'],isolation_over_closure=by['isolation']['cost_mean']/by['closure']['cost_mean'],isolation_over_ordinary=by['isolation']['cost_mean']/by['ordinary']['cost_mean']))
(P/'response_ratios.json').write_text(json.dumps(ratios,indent=2));print(json.dumps(ratios,indent=2))
