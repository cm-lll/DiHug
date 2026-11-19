"""CPU-only paired results against previously saved same-state updates."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import json,csv,sys
from pathlib import Path
OUT=Path(sys.argv[1]);ROOT=Path(__file__).resolve().parents[2];REF=ROOT/'results/update_response_20261108T142622'
read=lambda p:json.loads(Path(p).read_text())
def dump(n,x):
 with (OUT/n).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in x for k in r)));w.writeheader();w.writerows(x)
r=read(OUT/'receipt.json');assert r['status']=='complete' and (r['updates'],r['forwards'],r['extra_gradients'])==(4,16,8)
old=read(REF/'results.json');new=read(OUT/'results.json');totals=[];relations=[];geometries=[];summaries=[]
for x in new:
 prev=next(q for q in old if q['condition']==x['condition']);condition=x['condition']
 for t in ['1','500','1000']:
  variants={'NONE':prev['base'][t],'CE':prev['outcomes']['CE']['responses'][t],'PS':prev['outcomes']['PS']['responses'][t],'RANK':x['responses'][t]}
  assert len({v['input_sha256'] for v in variants.values()})==1
  for arm,v in variants.items():
   totals.append(dict(condition=condition,dataset=x['dataset'],seed=x['seed'],parent_branch=x['parent_branch'],t=int(t),arm=arm,**{k:v[k] for k in ['CE','positive_CE','negative_CE','expected_edges']}))
   for z in v['relations']:relations.append(dict(condition=condition,t=int(t),arm=arm,**z))
  rank=variants['RANK'];ce=variants['CE'];base=variants['NONE'];ps=variants['PS']
  s=dict(condition=condition,t=int(t))
  for m in ['CE','positive_CE','negative_CE','expected_edges']:
   s[m+'_RANK']=rank[m];s[m+'_CE']=ce[m];s[m+'_PS']=ps[m];s[m+'_before']=base[m]
   s[m+'_RANK_minus_CE']=rank[m]-ce[m];s[m+'_RANK_minus_PS']=rank[m]-ps[m];s[m+'_RANK_change']=rank[m]-base[m]
  s['relation_degree_improved']=sum(a['expected_relation_degree_MAE']<b['expected_relation_degree_MAE'] for a,b in zip(rank['relations'],ce['relations']))
  s['relation_degree_worsened']=sum(a['expected_relation_degree_MAE']>b['expected_relation_degree_MAE'] for a,b in zip(rank['relations'],ce['relations']))
  for m in ['missed_positive_mass','false_positive_mass']:
   s[m+'_RANK_minus_CE']=sum(a[m]-b[m] for a,b in zip(rank['relations'],ce['relations']))
  summaries.append(s)
 for group,v in x['geometry'].items():geometries.append(dict(condition=condition,group=group,**v))
dump('validation_total.csv',totals);dump('validation_relations.csv',relations);dump('paired_summary.csv',summaries);dump('parameter_geometry.csv',geometries)
paired_rel=[]
for q in relations:
 if q['arm']!='RANK':continue
 ce=next(z for z in relations if z['condition']==q['condition'] and z['t']==q['t'] and z['family']==q['family'] and z['arm']=='CE')
 row={k:q[k] for k in ['condition','t','family','positive_count','negative_count']}
 for m in ['positive_CE','negative_CE','expected_edges','missed_positive_mass','false_positive_mass','expected_relation_degree_MAE']:
  row[m+'_CE']=ce[m];row[m+'_RANK']=q[m];row[m+'_difference']=None if q[m] is None else q[m]-ce[m]
 paired_rel.append(row)
dump('paired_relations.csv',paired_rel)
(OUT/'summary.json').write_text(json.dumps(dict(rows=summaries,all_parameter_groups=geometries,limitations=['two original training seeds, four local parents','raw fixed conditional predictions only','no current-training after-step forward; linear prediction only','no complete graph quality evaluation']),indent=2))
print(json.dumps({'paired':summaries,'geometry':[x for x in geometries if x['group']=='all']}))
