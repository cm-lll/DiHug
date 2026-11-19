"""CPU summaries of frozen-mask diagnostic; no model calls."""
import json,csv
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/partial_attention_check_20261113T101200'
def mean(xs):
 xs=[x for x in xs if x is not None]
 return sum(xs)/len(xs) if xs else None
def metrics(rows):
 def pool(k):return mean([r[k] for t in rows for r in t['relations']])
 return dict(CE=mean([r['CE'] for r in rows]),node_degree_MAE=mean([r['node_total_degree_MAE'] for r in rows]),positive_CE=pool('positive_CE'),negative_CE=pool('negative_CE'),relation_edge_AE=mean([abs(r['expected_edges']-r['reference_edges']) for t in rows for r in t['relations']]),missed_mass=pool('missed_mass'),false_mass=pool('false_mass'),low_degree_MAE=mean([r['low_MAE'] for t in rows for r in t['type_groups']]),hub_degree_MAE=mean([r['hub_MAE'] for t in rows for r in t['type_groups']]))
summary=[];times=[];relations=[];types=[]
for path in sorted(OUT.glob('*_A_transport.json')):
 x=json.loads(path.read_text());d=dict(dataset=x['dataset'],seed=x['seed'])
 for arm,key in [('baseline','baseline'),('full','full_validation'),('partial','validation')]:
  d[arm]=metrics(x[key])
  for t in x[key]:
   common=dict(dataset=x['dataset'],seed=x['seed'],arm=arm,t=t['t'],input_hash=t['input_hash'])
   times.append(dict(**common,CE=t['CE'],node_total_degree_MAE=t['node_total_degree_MAE']))
   for r in t['relations']:relations.append(dict(**common,**r))
   for r in t['type_groups']:types.append(dict(**common,**r))
 d['partial_minus_full']={k:d['partial'][k]-d['full'][k] for k in d['baseline']}
 d['partial_minus_baseline']={k:d['partial'][k]-d['baseline'][k] for k in d['baseline']}
 for k in ['trainable_count','total_parameter_count','gradient_norm_before_clip','clip_scale','update_seconds','partial_update_norm','full_update_norm','update_cosine','frozen_raw_exact','buffers_exact','restore_exact','RNG_unchanged','EMA_updates']:d[k]=x[k]
 summary.append(d)
(OUT/'summary.json').write_text(json.dumps(summary,indent=2))
for name,rows in [('per_time.csv',times),('per_relation.csv',relations),('per_type_low_hub.csv',types)]:
 if rows:
  with (OUT/name).open('w') as f:
   w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print(json.dumps(summary,indent=2))
