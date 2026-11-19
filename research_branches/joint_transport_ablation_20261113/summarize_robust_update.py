"""CPU-only paired summaries; all signs retained, no selection."""
import json,csv
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/robust_heat_update_check_20261113T095140'
def mean(xs):
 xs=[x for x in xs if x is not None]
 return sum(xs)/len(xs) if xs else None
def metrics(rows):
 def pool(key):return mean([r[key] for t in rows for r in t['relations']])
 return dict(CE=mean([x['CE'] for x in rows]),node_degree_MAE=mean([x['node_total_degree_MAE'] for x in rows]),
  positive_CE=pool('positive_CE'),negative_CE=pool('negative_CE'),
  relation_edge_AE=mean([abs(r['expected_edges']-r['reference_edges']) for t in rows for r in t['relations']]),
  missed_mass=pool('missed_mass'),false_mass=pool('false_mass'),
  low_degree_MAE=mean([r['low_MAE'] for t in rows for r in t['type_groups']]),
  hub_degree_MAE=mean([r['hub_MAE'] for t in rows for r in t['type_groups']]))
summaries=[];times=[];relations=[];types=[]
for path in sorted(OUT.glob('*_A_transport.json')):
 x=json.loads(path.read_text())
 if len(x['arms'])!=2 or 'comparisons' not in x:continue
 result=dict(dataset=x['dataset'],seed=x['seed'],baseline=metrics(x['baseline']),comparisons=x['comparisons'])
 for arm,a in x['arms'].items():
  result[arm]=metrics(a['validation']);result[arm]['gradient_norm_before_clip']=a['gradient_norm_before_clip'];result[arm]['clip_scale']=a['clip_scale']
 result['robust_minus_original']={k:result['robust'][k]-result['original'][k] for k in result['baseline']}
 for arm,rows in [('baseline',x['baseline'])]+[(k,v['validation']) for k,v in x['arms'].items()]:
  for t in rows:
   common=dict(dataset=x['dataset'],seed=x['seed'],arm=arm,t=t['t'],input_hash=t['input_hash'])
   times.append(dict(**common,CE=t['CE'],node_total_degree_MAE=t['node_total_degree_MAE']))
   for r in t['relations']:relations.append(dict(**common,**r))
   for r in t['type_groups']:types.append(dict(**common,**r))
 summaries.append(result)
(OUT/'summary.json').write_text(json.dumps(summaries,indent=2))
for name,rows in [('per_time.csv',times),('per_relation.csv',relations),('per_type_low_hub.csv',types)]:
 if rows:
  with (OUT/name).open('w') as f:
   w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print(json.dumps(summaries,indent=2))
