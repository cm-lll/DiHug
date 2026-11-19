"""Read archived tensors/metadata only. No model import, CUDA, or optimizer step."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import json,hashlib,time,resource
from pathlib import Path
import torch
torch.set_num_threads(1)
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/joint_degree_response_plan_20261109'
OUT.mkdir(exist_ok=True)
PARAM=ROOT/'results/joint_degree_parameter_signal_20261109'
REL=ROOT/'results/joint_degree_relation_signal_20261109'
OLD=ROOT/'results/hard_structure_response_20261108T182154'
START=time.time();bindings={}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def bind(p):
 p=Path(p);bindings[str(p)]=sha(p);return p
def read(p):return json.loads(bind(p).read_text())
def save(n,x):(OUT/(n+'.json')).write_text(json.dumps(x,ensure_ascii=False,indent=2))
for p,h in read(REL/'binding.json').items():
 assert sha(p)==h,('changed_asset',p);bindings[p]=h
production=read(PARAM/'production_source_binding.json')
for p,h in production.items():assert sha(p)==h,('production_changed',p)
rows=[]
for parent in ['CE','PS']:
 cp=bind(OLD/f'PubMed_{parent}_CE_temporary.pt')
 assert sha(cp)==read(cp.with_suffix('.sha256.json'))['sha256']
 envelope=torch.load(cp,map_location='cpu',weights_only=False);s=envelope['state'];o=s['optimizer']
 layout=read(PARAM/f'{parent}_parameter_layout.json')
 gfile=bind(PARAM/f'{parent}_parameter_gradients.pt')
 assert sha(gfile)==read(PARAM/'completion.json')['output_hashes'][str(gfile)]
 gp=torch.load(gfile,map_location='cpu',weights_only=False)
 assert gp['checkpoint_sha256']==sha(cp)
 group=o['param_groups'][0];ids=group['params']
 assert len(o['param_groups'])==1 and len(ids)==len(layout)==474
 assert s['updates']==s['ema']['updates']==2040
 missing=[]
 for pid,row in zip(ids,layout):
  assert list(s['model'][row['name']].shape)==row['shape']
  if pid in o['state']:
   st=o['state'][pid]
   assert list(st['exp_avg'].shape)==row['shape']==list(st['exp_avg_sq'].shape)
   assert float(st['step'])==2040
  else:missing.append(row['name'])
 nce=float(gp['gradients']['CE'].double().norm());ng=float(gp['gradients']['new_mean'].double().norm())
 assert nce>0 and ng>0
 seeds={f't{t}_draw{j}':int.from_bytes(hashlib.sha256(f'joint_degree_response_v1|PubMed|{t}|{j}'.encode()).digest()[:8],'little') for t in [1,500,1000] for j in [0,1]}
 rows.append(dict(parent=parent,checkpoint=str(cp),sha256=sha(cp),dataset=s['binding']['dataset'],train_seed=s['binding']['train_seed'],readout='raw',updates=2040,optimizer={k:v for k,v in group.items() if k!='params'},parameter_count=len(ids),optimizer_state_count=len(o['state']),missing_optimizer_state_names=missing,previous_None_gradient_names=gp['none_parameters'],CE_gradient_norm=nce,full_structural_gradient_norm=ng,proposed_local_unit_balance_lambda=nce/ng,evaluation_noise_seeds=seeds,execution_policy=envelope['execution_policy'],parent_sha256=envelope['parent_sha256']))
 del envelope,s,o,gp
save('assets',rows);save('binding',bindings);save('production_source_binding',production)
save('receipt',dict(status='complete',CPU_seconds=time.time()-START,peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,models_called=0,optimizer_steps=0,new_graphs=0,GPU_hours=0,production_files_verified=len(production),assets_verified=len(bindings)))
print(json.dumps(rows,ensure_ascii=False,indent=2))
