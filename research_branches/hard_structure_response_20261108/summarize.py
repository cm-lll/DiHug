"""CPU-only audit and paired summaries; no model imports."""
import json,csv,hashlib,math,time,subprocess
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[2];O=R/'results/hard_structure_response_20261108T182154'
read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
probes=read(O/'probes.json');updates=read(O/'updates.json');pairs=[];relations=[];groups=[]
metrics=['CE','positive_CE','negative_CE','expected_edges','edge_abs_error','expected_degree_MAE','expected_triangles','triangle_abs_error','expected_triangle_participation_MAE','hard_score_mean']
for condition in sorted(set(x['condition'] for x in probes)):
 for t in [1,500,1000]:
  rows={x['arm']:x for x in probes if x['condition']==condition and x['t']==t}
  if not all(k in rows for k in ['start','CE','CE_STRUCT']):continue
  a,b=rows['CE'],rows['CE_STRUCT'];assert a['input_sha256']==b['input_sha256']==rows['start']['input_sha256']
  d=np.asarray(b['hard_score_samples'])-np.asarray(a['hard_score_samples']);row=dict(condition=condition,t=t,seed=a['seed'],input_sha256=a['input_sha256'],hard_score_MC_pairs=len(d),hard_score_paired_delta=float(d.mean()),hard_score_paired_MC_SE=float(d.std(ddof=1)/np.sqrt(len(d))))
  for key in metrics:
   row[key+'_start']=rows['start'][key];row[key+'_CE']=a[key];row[key+'_STRUCT']=b[key];row[key+'_delta']=b[key]-a[key]
  pairs.append(row)
  for ra,rb in zip(a['relations'],b['relations']):
   assert ra['family']==rb['family'];rr=dict(condition=condition,t=t,family=ra['family'],positive_count=ra['positive_count'],negative_count=ra['negative_count'])
   for key in ['positive_CE','negative_CE','expected_edges','missed_positive_mass','false_positive_mass','expected_relation_degree_MAE']:
    rr[key+'_CE']=ra[key];rr[key+'_STRUCT']=rb[key];rr[key+'_delta']=None if ra[key] is None or rb[key] is None else rb[key]-ra[key]
   relations.append(rr)
  for ra,rb in zip(a['nodegroups'],b['nodegroups']):
   assert (ra['type'],ra['group'],ra['support'])==(rb['type'],rb['group'],rb['support']);rr=dict(condition=condition,t=t,type=ra['type'],group=ra['group'],support=ra['support'])
   for key in ['degree_MAE','triangle_participation_MAE']:
    rr[key+'_CE']=ra[key];rr[key+'_STRUCT']=rb[key];rr[key+'_delta']=None if ra[key] is None else rb[key]-ra[key]
   groups.append(rr)
def writecsv(name,rows):
 if not rows:return
 fields=list(dict.fromkeys(k for row in rows for k in row))
 with (O/name).open('w') as f:
  w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)
writecsv('paired_metrics.csv',pairs);writecsv('paired_relations.csv',relations);writecsv('paired_nodegroups.csv',groups)
writecsv('training_summary.csv',[{k:x[k] for k in ['condition','arm','offset','update','t','unweighted_CE','gradient_norm','update_seconds','auxiliary_seconds','auxiliary_loss','input_sha256']} for x in updates])
summary=[]
for condition in sorted(set(x['condition'] for x in pairs)):
 rows=[x for x in pairs if x['condition']==condition];r=dict(condition=condition,times=len(rows))
 for key in metrics:r[key+'_mean_delta']=float(np.mean([x[key+'_delta'] for x in rows]))
 r['score_improved_times']=[x['t'] for x in rows if x['hard_score_mean_delta']<0];r['degree_improved_times']=[x['t'] for x in rows if x['expected_degree_MAE_delta']<0];r['triangle_improved_times']=[x['t'] for x in rows if x['triangle_abs_error_delta']<0]
 summary.append(r)
checks={}
for src in [O/'source_binding.json',R/'results/hard_structure_incremental_cpu_20261108/source_binding.json',R/'results/joint_output_formal_projected_20261108T043000/source_binding.json']:
 checks[str(src)]={path:sha(path)==value for path,value in read(src).items()}
for a in read(O/'assets.json'):checks[a['condition']]={a['parent']:sha(a['parent'])==a['sha256']}
input_pairs=[]
for condition in sorted(set(x['condition'] for x in updates)):
 a=[x for x in updates if x['condition']==condition and x['arm']=='CE'];b=[x for x in updates if x['condition']==condition and x['arm']=='CE_STRUCT']
 input_pairs.append(dict(condition=condition,CE_steps=len(a),STRUCT_steps=len(b),input_pairing_all=all(x['input_sha256']==y['input_sha256'] for x,y in zip(a,b)),times=[x['t'] for x in a]))
old=read(R/'results/hard_structure_gradient_gpu_20261108T180629/environment_active.json');env=read(O/'environment_active.json');env_match={k:env[k]==old[k] for k in ['python_version','torch_runtime_version','packages','cuda','device_name','execution_policy']}
audit=dict(at_epoch=time.time(),all_bindings_unchanged=all(all(x.values()) for x in checks.values()),checks=checks,paired_inputs=input_pairs,environment_matches_previous=env_match,git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip(),local_checkpoint_sha256={str(p):sha(p) for p in O.glob('*temporary.pt')},auxiliary_warning='Independent auxiliary RNG saved in envelope; production main RNG preserved. New endpoint is diagnostic-only; no uninterrupted/resumed next candidate update equivalence tested this ticket.')
(O/'audit.json').write_text(json.dumps(audit,indent=2));(O/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2));print(json.dumps(dict(bindings_unchanged=audit['all_bindings_unchanged'],input_pairs=input_pairs)))
assert audit['all_bindings_unchanged'] and all(env_match.values())
