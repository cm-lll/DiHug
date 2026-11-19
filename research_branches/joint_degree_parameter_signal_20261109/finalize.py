"""Read-only CPU verification and summaries of saved parameter vectors."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import json,hashlib,math,csv,datetime
from pathlib import Path
import torch
torch.set_num_threads(1)
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/joint_degree_parameter_signal_20261109'
def read(n):return json.loads((OUT/(n+'.json')).read_text())
def save(n,x):
 (OUT/(n+'.json')).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
 if isinstance(x,list) and x:
  with (OUT/(n+'.csv')).open('w',newline='') as f:
   w=csv.DictWriter(f,list(x[0]));w.writeheader();w.writerows(x)
receipt=read('receipt');release=read('release');assert receipt['status']=='complete' and receipt['forwards']==2 and receipt['gradient_queries']==10
assert receipt['optimizer_updates']==receipt['new_graphs']==0 and release['worker_exited'] and release['conservative_GPU_hours']<.25
for r in read('probability_matching'):assert r['p_exact'] and r['p_max_abs']==0
for r in read('state_checks'):assert all(r['loaded_exact'].values()) and all(r['after_unchanged'].values())
rows=read('comparisons');summary=[];checks=[]
for parent in ['CE','PS']:
 z=torch.load(OUT/f'{parent}_parameter_gradients.pt',map_location='cpu',weights_only=False);gs=z['gradients'];layout=read(parent+'_parameter_layout')
 assert len({r['name'] for r in layout})==len(layout) and layout[0]['start']==0
 assert all(a['end']==b['start'] for a,b in zip(layout,layout[1:]))
 assert all(v.numel()==layout[-1]['end'] and bool(torch.isfinite(v).all()) for v in gs.values())
 for method in ['old','new']:assert torch.equal(gs[method+'_mean'],(gs[method+'_draw0']+gs[method+'_draw1'])/2)
 for r in [r for r in rows if r['parent']==parent]:
  if r['comparison'].endswith('_draw_pair'):a,b=r['comparison'].split('_')[0]+'_draw0',r['comparison'].split('_')[0]+'_draw1'
  else:a,b=r['comparison'].split('_vs_')
  ids=[v for v in layout if r['group']=='all' or v['group']==r['group']]
  x=torch.cat([gs[a][v['start']:v['end']] for v in ids]).double();y=torch.cat([gs[b][v['start']:v['end']] for v in ids]).double()
  den=float(x.norm()*y.norm());c=float(x@y/den) if den else None
  assert c==r['cosine'] or (c is not None and abs(c-r['cosine'])<1e-12)
 old=gs['old_mean'].double();new=gs['new_mean'].double();ce=gs['CE'].double()
 k=float(new@old/(old@old));res=float((new-k*old).norm()/new.norm())
 shares={gr:sum(float(new[v['start']:v['end']].square().sum()) for v in layout if v['group']==gr)/float(new.square().sum()) for gr in sorted({v['group'] for v in layout})}
 summary.append(dict(parent=parent,train_seed=2026110422,updates=2040,t=1000,readout='raw',parameter_elements=new.numel(),
  new_to_CE_gradient_norm=float(new.norm()/ce.norm()),old_to_CE_gradient_norm=float(old.norm()/ce.norm()),
  best_scalar_old_to_new=k,new_residual_fraction_after_best_old_scaling=res,joint_head_squared_norm_share=shares['joint_head'],
  all_group_squared_norm_share=shares,limitation='unweighted gradient norms, not AdamW displacement or recommended lambda'))
 checks.append(dict(parent=parent,layout_partition=True,finite=True,mean_identity=True,summary_recomputed=True))
save('scale_direction_summary',summary);save('independent_checks',checks)
save('comparisons',rows);save('per_parameter',read('per_parameter'))
binding=read('binding');assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in binding.items())
formal=ROOT/'results/joint_output_formal_projected_20261108T043000/source_binding.json'
manifest=json.loads(formal.read_text());assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in manifest.items())
save('production_source_binding',manifest)
save('completion',dict(status='complete',source_assets_verified=len(binding),production_sources_verified=len(manifest),
 restored_and_unchanged=True,probabilities_bitwise=True,model_calls=2,gradient_queries=10,optimizer_steps=0,new_graphs=0,
 resource_receipt=receipt,release=release,output_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.glob('*_parameter_gradients.pt')}))
print(json.dumps(summary,ensure_ascii=False))
