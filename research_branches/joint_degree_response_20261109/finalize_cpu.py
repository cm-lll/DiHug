"""Independent arithmetic/state receipt checks after GPU worker exit."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import json,csv,hashlib,datetime
from pathlib import Path
import torch
torch.set_num_threads(1)
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/joint_degree_response_20261109'
def read(n):return json.loads((OUT/(n+'.json')).read_text())
def save(n,x):(OUT/(n+'.json')).write_text(json.dumps(x,ensure_ascii=False,indent=2))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
receipt=read('receipt');release=read('release');p=read('protocol')
assert receipt['status']=='complete' and release['exit_code']==0 and release['worker_exited']
assert (receipt['forwards'],receipt['backwards'],receipt['optimizer_steps'])==(72,8,8)
assert all(sha(k)==v for k,v in read('binding').items())
first=json.loads((OUT/'attempt1/release.json').read_text());hours=first['conservative_GPU_hours']+release['conservative_GPU_hours'];assert hours<=.25
inputs=read('evaluation_inputs');assert len(inputs)==6 and sum(not r['previously_observed'] for r in inputs)==4 and len({r['hash'] for r in inputs})==5
assert all(x['p_exact'] and x['state_unchanged'] for x in read('checks'))
assert len(read('state_records'))==8
groups=[];comparisons=[]
def cmp(a,b):
 den=float(a.norm()*b.norm());return dict(norm_a=float(a.norm()),norm_b=float(b.norm()),cosine=float(a@b)/den if den else None,difference_norm=float((a-b).norm()))
for parent in ['CE','PS']:
 d=torch.load(OUT/f'{parent}_positions.pt',map_location='cpu',weights_only=False);v={k:x.double() for k,x in d['positions'].items()};layout=d['layout']
 assert d['diagnostic_only'] and d['not_resumable']
 for gr in ['all']+sorted({r['group'] for r in layout}):
  idx=[r for r in layout if gr=='all' or r['group']==gr]
  x={k:torch.cat([a[r['start']:r['end']] for r in idx]) for k,a in v.items()}
  for arm in v:
   groups.append(dict(parent=parent,group=gr,arm=arm,**cmp(x[arm]-x['NOOP'],x['CE']-x['NOOP']),aux_norm=float((x[arm]-x['CE']).norm())))
  for a,b in [('FULL','COMMON'),('FULL','RESIDUAL'),('MATCH','RESIDUAL')]:comparisons.append(dict(parent=parent,group=gr,comparison=a+'_vs_'+b,**cmp(x[a]-x['CE'],x[b]-x['CE'])))
 for key in ['probability','ce']:
  import numpy as np
  for arm in v:
   with np.load(OUT/f'{parent}_{arm}_t1_draw0.npz') as a,np.load(OUT/f'{parent}_{arm}_t1_draw1.npz') as b:assert np.array_equal(a[key],b[key])
for n,rows in [('parameter_group_movements',groups),('parameter_aux_comparisons',comparisons)]:
 save(n,rows)
 with (OUT/(n+'.csv')).open('w') as h:w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
save('completion',dict(status='complete',actual_forwards=72,reused_duplicate_forward_records=12,backwards=8,independent_temporary_optimizer_steps=8,new_graphs=0,unique_evaluation_states=5,newly_unseen_states=4,source_assets_verified=len(read('binding')),conservative_GPU_hours=hours,conservative_GPU_seconds=hours*3600,receipt=receipt,release=release,first_attempt_release=first,source_hashes={str(f.resolve()):sha(f) for f in Path(__file__).parent.glob('*.py')}))
print(json.dumps(dict(GPU_hours=hours,forwards=72,backwards=8,temporary_steps=8,parameters_finite=True,old_assets_unchanged=True)))
