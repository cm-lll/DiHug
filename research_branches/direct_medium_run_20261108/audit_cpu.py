"""No-model inventory/schema/label and resource receipt audit."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import sys,json,hashlib
from pathlib import Path
import numpy as np
import torch
torch.set_num_threads(1)
R=Path(__file__).resolve().parents[2];O=Path(sys.argv[1]);P=json.loads((O/'protocol.json').read_text())
tasks=json.loads((R/'results/output_boundary_replay_20261108T081654/tasks.json').read_text());rows=[];bindings={}
package=R/'research_branches/typed_structure_early_cn_expansion_review/experiments/typed_structure_early_cn_expansion_v1/package'
for ds in ['IMDB','PubMed']:
 cfg=next(x for x in tasks if x['dataset']==ds)['row']['config'];data=package/cfg['datasets'][ds]['file']
 actual=hashlib.sha256(data.read_bytes()).hexdigest();assert actual==cfg['datasets'][ds]['sha256'];bindings[str(data)]=actual
 with np.load(data) as z:types=z['types'];edges=z['edges'];schema=z['schema']
 target={tuple(sorted(map(int,e))) for e in edges};assert len(target)==len(edges)
 support={tuple(sorted(map(int,pair))) for pair in schema}
 full_expected={ (i,j) for i in range(len(types)) for j in range(i+1,len(types)) if tuple(sorted((int(types[i]),int(types[j])))) in support }
 # Production common.make_case: active = reference family counts > 0.
 active_support={tuple(sorted((int(types[i]),int(types[j])))) for i,j in target}
 expected={pair for pair in full_expected if tuple(sorted((int(types[pair[0]]),int(types[pair[1]])))) in active_support}
 assert not (target & (full_expected-expected)), 'inactive fixed-zero support contains reference edge'
 bundle=torch.load(O/f'{ds}_inputs.pt',map_location='cpu');assert np.array_equal(types,bundle['types'])
 for item in bundle['fixed']:
  inp=item['input'];q=np.asarray(inp['query']);actual_pairs={tuple(sorted(map(int,pair))) for pair in q.T}
  assert len(actual_pairs)==q.shape[1] and actual_pairs==expected
  yy=np.array([tuple(sorted(map(int,pair))) in target for pair in q.T],dtype=int);assert np.array_equal(yy,inp['labels'])
  assert 100*(item['layer']-1)+1<=item['t']<=100*item['layer']
  assert item['noise_seed']==P['evaluation'][ds]['inputs'][item['index']]['noise_seed']
  rows.append(dict(dataset=ds,index=item['index'],t=item['t'],slots=q.shape[1],full_schema_slots=len(full_expected),fixed_zero_omitted=len(full_expected-expected),positive_slots=int(yy.sum()),query_coverage=True,labels_match_X0=True,layer_valid=True))
assert len(rows)==40
checks=json.loads((O/'checks.json').read_text());assert all(all(x['state_exact'].values()) for x in checks)
receipt=json.loads((O/'receipt.json').read_text());lease=json.loads((O/'lease.json').read_text())
assert lease['gpu_hours']<=.75+1e-6
assert receipt['end_epoch']<=P['science_stop_epoch']
for name in ['source_binding.json','asset_binding.json']:
 for p,h in json.loads((O/name).read_text()).items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
updates=json.loads((O/'updates.json').read_text());pairing=[]
for c in sorted({x['condition'] for x in updates}):
 a=[x for x in updates if x['condition']==c and x['arm']=='CE'];b=[x for x in updates if x['condition']==c and x['arm']=='CE_STRUCT']
 if len(a)==len(b)==56:
  assert [x['input_sha256'] for x in a]==[x['input_sha256'] for x in b]
  times=[x['t'] for x in a];counts=np.histogram(times,bins=np.arange(1,1002,100))[0].tolist()
  pairing.append(dict(condition=c,matched=True,updates=56,time_layer_counts=counts))
manifest=[]
for task in P['tasks']:
 for offset in [32,64]:
  cp=O/f"{task['condition']}_{task['arm']}_offset{offset}.pt"
  if not cp.exists():continue
  env=torch.load(cp,map_location='cpu');st=env['state'];expected=(1032 if task['dataset']=='IMDB' else 2032)+offset
  assert env['diagnostic_only'] and env['source_resume']==task['resume_path']
  assert st['updates']==st['ema']['updates']==expected
  assert (env['auxiliary_rng'] is not None)==(task['arm']=='CE_STRUCT')
  steps=sorted({int(v['step'].item()) for v in st['optimizer']['state'].values() if 'step' in v})
  assert steps==[expected]
  manifest.append(dict(path=str(cp.resolve()),sha256=hashlib.sha256(cp.read_bytes()).hexdigest(),condition=task['condition'],arm=task['arm'],
   local_offset=offset,updates=expected,optimizer_steps=steps,EMA_updates=st['ema']['updates'],parent_resume=task['resume_path'],
   diagnostic_only=True,coefficient=.001 if task['arm']=='CE_STRUCT' else 0,
   binding_note='original parent scientific identity retained in state.binding; diagnostic objective/lineage explicitly recorded here and source_resume/protocol'))
  del env,st
(O/'checkpoint_manifest.json').write_text(json.dumps(manifest,indent=2))
out=dict(inputs=rows,state_restores=len(checks),state_checks_pass=True,assets_unchanged=True,data_binding=bindings,paired_updates=pairing,complete_new_checkpoints=len(manifest),
 within_GPU_budget=True,within_science_deadline=True,CUDA_initialized=torch.cuda.is_initialized(),model_calls=0)
(O/'audit.json').write_text(json.dumps(out,indent=2));print(json.dumps(dict(restores=len(checks),inputs=len(rows),paired_updates=pairing),indent=2))
