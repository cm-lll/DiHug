"""CPU timeline and saved-gradient description; no model calls or new draws."""
import json,csv,hashlib,time,sys
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[2];O=R/'results/direct_step_trace_20261108T213749';S=R/'results/direct_structure_response_20261108T205816'
sys.path.insert(0,str(R/'research_branches/direct_structure_cpu_20261108'));from score import DirectScore
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
probes=read(O/'probes.json');updates=read(O/'updates.json');old=read(S/'updates.json')+read(S/'control_updates.json');rows=[];rel=[];gradrows=[];summary=[]
def out(name,rows):
 with (O/name).open('w') as f:
  w=csv.DictWriter(f,list(dict.fromkeys(k for x in rows for k in x)));w.writeheader();w.writerows(rows)
metrics=['CE','positive_CE','negative_CE','expected_edges','expected_triangles','expected_degree_MAE']
for c in sorted(set(x['condition'] for x in probes)):
 trajectory=[]
 for offset in range(9):
  p={x['arm']:x for x in probes if x['condition']==c and x['offset']==offset}
  if len(p)!=2:continue
  a,b=p['CE'],p['CE_STRUCT'];assert a['input_sha256']==b['input_sha256'];rec=next((x for x in old if x['condition']==c and x['arm']=='CE' and x['offset']==offset),None)
  r=dict(condition=c,offset=offset,training_t=rec['t'] if rec else None,reference_triangles=a['reference_triangles'])
  for k in metrics:r[k+'_CE']=a[k];r[k+'_DIRECT']=b[k];r[k+'_delta']=b[k]-a[k]
  r['triangle_error_CE']=abs(a['expected_triangles']-a['reference_triangles']);r['triangle_error_DIRECT']=abs(b['expected_triangles']-b['reference_triangles']);r['triangle_error_delta']=r['triangle_error_DIRECT']-r['triangle_error_CE'];rows.append(r);trajectory.append(r)
  for x,y in zip(a['relations'],b['relations']):
   assert x['family']==y['family'];rr=dict(condition=c,offset=offset,training_t=r['training_t'],family=x['family'],positive_count=x['positive_count'],negative_count=x['negative_count'])
   for k in ['positive_CE','negative_CE','missed_positive_mass','false_positive_mass']:
    rr[k+'_CE']=x[k];rr[k+'_DIRECT']=y[k];rr[k+'_delta']=None if x[k] is None else y[k]-x[k]
   rel.append(rr)
 changes=[]
 for a,b in zip(trajectory,trajectory[1:]):
  if abs(a['expected_triangles_delta'])>1e-9 and abs(b['expected_triangles_delta'])>1e-9 and np.sign(a['expected_triangles_delta'])!=np.sign(b['expected_triangles_delta']):changes.append(dict(offset=b['offset'],training_t=b['training_t'],before=a['expected_triangles_delta'],after=b['expected_triangles_delta']))
 summary.append(dict(condition=c,recorded_points=len(trajectory),triangle_count_gap_sign_changes=changes,final=trajectory[-1]))
 for offset in range(2,9):
  path=O/f'{c}_gradient_offset{offset}.npz'
  if not path.exists():continue
  with np.load(path) as z:grad=z['gradient'];bits=z['bits']
  with np.load(O/f'{c}_CE_STRUCT_train_offset{offset}.npz') as z:q=z['query'];f=z['families'];y=z['labels'];types=z['types'];p=z['probability'];t=int(z['t'])
  n=len(types);A=np.zeros((n,n));A[q[0],q[1]]=A[q[1],q[0]]=p
  rr=next(x for x in probes if x['condition']==c);r=len(rr['relations']);obj=DirectScore(types,q,f,y,bits,r);m=obj.metrics;record=next(x for x in updates if x['condition']==c and x['arm']=='CE_STRUCT' and x['offset']==offset)
  assert abs(obj.base-record['hard_score_sample'])<1e-12
  changes=m.changes();factor=p*(1-p)*changes['flip_sign']/3
  components={k:factor*changes[k]/scale for k,scale in [('total_degree_W1',obj.degree_scale),('joint_degree_TV',1.),('triangle_abs_error',obj.triangle_scale)]}
  residual=float(abs(sum(components.values())-grad).max());assert residual<1e-14,'saved gradient component reconstruction mismatch'
  row=dict(condition=c,offset=offset,t=t,hard_edges=int(bits.sum()),reference_edges=int(y.sum()),hard_triangles=m.triangles,reference_triangles=m.ref_triangles,probability_expected_triangles=float((A*(A@A)).sum()/6),hard_W1=m.base_w1,hard_joint_TV=m.base_tv,hard_score=obj.base,gradient_sum=float(grad.sum()),positive_gradient_sum=float(grad[y==1].sum()),negative_gradient_sum=float(grad[y==0].sum()),gradient_L2=float(np.linalg.norm(grad)),component_reconstruction_max_abs=residual,note='Unweighted sampled structure derivative under uniform active-logit perturbation; not actual parameter update or unbiased estimate of the eight-step effect')
  for k,v in components.items():row[k+'_gradient_sum']=float(v.sum());row[k+'_gradient_L2']=float(np.linalg.norm(v))
  gradrows.append(row)
out('timeline.csv',rows);out('relations.csv',rel);out('sampled_gradient_context.csv',gradrows)
checks={}
for name in ['source_binding.json','asset_binding.json']:
 checks[name]={path:sha(path)==h for path,h in read(O/name).items()}
assert all(all(x.values()) for x in checks.values())
assert all(x['recorded_points']==9 for x in summary) and len(updates)==56 and all(x['historical_exact'] for x in updates)
receipt=read(O/'receipt.json');assert receipt['status']=='complete'
(O/'audit.json').write_text(json.dumps(dict(assets_and_sources_unchanged=True,checks=checks,all_56_updates_match_history=True,all_8_endpoints_exact=all(all(x['complete_endpoint_exact'].values()) for x in read(O/'checks.json')),sign_tolerance=1e-9,scope='descriptive transition localization, one training seed per dataset with two local parent histories; raw conditional probabilities only'),indent=2))
(O/'summary.json').write_text(json.dumps(summary,indent=2))
for s in summary:
 print(s['condition'],'crossings',s['triangle_count_gap_sign_changes'])
 for r in rows:
  if r['condition']==s['condition']:print(r['offset'],r['training_t'],'tri',round(r['expected_triangles_CE'],6),round(r['expected_triangles_DIRECT'],6),'delta',round(r['expected_triangles_delta'],6),'error_delta',round(r['triangle_error_delta'],6),'CEdelta',round(r['CE_delta'],9))
print('SAVED GRADIENT CONTEXT')
for x in gradrows:
 if x['condition'].startswith('PubMed'):print(x['condition'],x['offset'],x['t'],'hardtri',x['hard_triangles'],'gradient sum',x['gradient_sum'])
