"""CPU summary of actual single updates; no model calls."""
import json,csv,hashlib,time
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[2];O=R/'results/direct_one_step_20261108T212756';S=R/'results/direct_structure_response_20261108T205816'
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
probes=read(O/'probes.json');res=read(O/'results.json');pairs=[];relations=[];geometry=[]
def csvout(name,rows):
 with (O/name).open('w') as f:
  w=csv.DictWriter(f,list(dict.fromkeys(k for x in rows for k in x)));w.writeheader();w.writerows(rows)
for x in res:
 c=x['condition']
 for which in ['training','t500']:
  p={z['arm']:z for z in probes if z['condition']==c and z['input']==which};before=p['CE_before'] if which=='training' else p['start'];a,b=p['CE'],p['CE_STRUCT'];assert a['input_sha256']==b['input_sha256']==before['input_sha256']
  if which=='training':
   with np.load(O/f'{c}_CE_before_training.npz') as u,np.load(O/f'{c}_CE_STRUCT_before_training.npz') as v:assert np.array_equal(u['probability'],v['probability']) and np.array_equal(u['CE'],v['CE'])
  row=dict(condition=c,input=which,t=a['t'])
  for key in ['CE','positive_CE','negative_CE','expected_edges','expected_triangles','expected_degree_MAE']:
   row[key+'_before']=before[key];row[key+'_CE']=a[key];row[key+'_DIRECT']=b[key];row[key+'_CE_delta']=a[key]-before[key];row[key+'_DIRECT_delta']=b[key]-before[key];row[key+'_DIRECT_minus_CE']=b[key]-a[key]
  pairs.append(row)
  for aa,bb in zip(a['relations'],b['relations']):
   assert aa['family']==bb['family'];rr=dict(condition=c,input=which,t=a['t'],family=aa['family'],positive_count=aa['positive_count'],negative_count=aa['negative_count'])
   for k in ['positive_CE','negative_CE','missed_positive_mass','false_positive_mass']:rr[k+'_CE']=aa[k];rr[k+'_DIRECT']=bb[k];rr[k+'_delta']=None if aa[k] is None else bb[k]-aa[k]
   relations.append(rr)
 for z in x['groups']:
  row=dict(condition=c,group=z['group'])
  for k in ['CE_move','DIRECT_move','incremental_move','weighted_structure_gradient','CE_gradient']:row[k+'_L2']=z[k]['L2']
  for k in ['structure_dot_CE_move','structure_dot_DIRECT_move','structure_dot_incremental_move','CE_dot_incremental_move']:row[k]=z[k]['dot'];row[k+'_cosine']=z[k]['cosine']
  row['move_cosine']=z['move_cosine'];geometry.append(row)
csvout('paired_probes.csv',pairs);csvout('paired_relations.csv',relations);csvout('gradient_update_geometry.csv',geometry)
checks={}
for f in [O/'source_binding.json',S/'source_binding.json',S/'reused_assets.json']:
 checks[str(f)]={path:sha(path)==h for path,h in read(f).items()}
for c,v in read(S/'control_binding.json').items():checks[c]={v['parent']:sha(v['parent'])==v['parent_sha256']}
assert all(all(x.values()) for x in checks.values())
audit=dict(bindings_unchanged=True,checks=checks,matching_historical_first_steps=all(x['actual_first_step_matches_history'] for x in res),gradient_probe_residuals={x['condition']:x['gradient_sum_residual'] for x in res},limits='raw conditional marginal predictions; one step and one auxiliary draw per parent, not expected objective or full generation. Two local histories per dataset share one training seed. Total displacement includes Adam state/weight decay; incremental displacement is paired difference, not an additive standalone optimizer component.')
(O/'audit.json').write_text(json.dumps(audit,indent=2));(O/'summary.json').write_text(json.dumps(dict(pairs=pairs,geometry=geometry,audit=audit),indent=2))
print('PAIRED PROBES')
for x in pairs:print(x['condition'],x['input'],*[f'{k}:{x[k+"_before"]:.8g}->{x[k+"_CE"]:.8g}->{x[k+"_DIRECT"]:.8g}' for k in ['CE','expected_triangles','expected_degree_MAE']])
print('GEOMETRY')
for x in geometry:
 if x['group']=='all':print(json.dumps(x))
