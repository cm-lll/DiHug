"""CPU-only audit of decomposition and delivery tables; no model imports."""
import json,csv,hashlib,time,datetime
from pathlib import Path
import numpy as np
import torch
torch.set_num_threads(1)
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/joint_degree_relation_signal_20261109'
PREV=ROOT/'results/joint_degree_parameter_signal_20261109'
def read(n):return json.loads((OUT/(n+'.json')).read_text())
def save(n,x):(OUT/(n+'.json')).write_text(json.dumps(x,ensure_ascii=False,indent=2))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert read('receipt')['status']=='complete'
assert read('receipt')['forwards']==read('receipt')['gradient_queries']==2
assert read('release')['worker_exited'] and read('release')['exit_code']==0
assert all(sha(p)==h for p,h in read('binding').items())
assert all(r['p_exact'] for r in read('probability_matching'))
assert all(all(r['loaded_exact'].values()) and all(r['after_unchanged'].values()) for r in read('state_checks'))
rows=[];checks=[]
for parent in ['CE','PS']:
 with np.load(OUT/f'{parent}_logit_decomposition.npz') as z:
  t,c,r,f=[z[k].copy() for k in ['total','common','residual','relation']]
  assert np.allclose(t,c+r,rtol=0,atol=1e-15)
  assert all(abs(r[f==k].mean())<1e-15 for k in np.unique(f))
  logit_common_energy=float(c@c/(t@t));logit_residual_energy=float(r@r/(t@t))
  checks.append(dict(parent=parent,logit_orthogonality=float(c@r),max_relation_residual_mean=max(abs(r[f==k].mean()) for k in np.unique(f))))
 payload=torch.load(OUT/f'{parent}_decomposition.pt',map_location='cpu',weights_only=False)
 g=payload['gradients'];t,c,r=[g[k].double() for k in ['total','common','residual']]
 old=torch.load(PREV/f'{parent}_parameter_gradients.pt',map_location='cpu',weights_only=False)
 assert torch.equal(t,old['gradients']['new_mean'].double())
 assert torch.equal(r,t-c)
 assert bool(torch.isfinite(c).all())
 row=next(v.copy() for v in read('comparisons') if v['parent']==parent and v['group']=='all')
 row.update(logit_common_squared_norm_fraction=logit_common_energy,logit_residual_squared_norm_fraction=logit_residual_energy,
  parameter_common_squared_norm_ratio=float(c@c/(t@t)),parameter_residual_squared_norm_ratio=float(r@r/(t@t)),parameter_cross_term_ratio=float(2*(c@r)/(t@t)))
 rows.append(row)
save('summary',rows);save('independent_checks',dict(checks=checks,previous_total_exact=True,bindings_verified=len(read('binding')),state_unchanged=True,limitation='residual VJP derived by subtraction; separate residual backward not run; archived total averages two FP32 VJPs, common uses FP32 cast of averaged logit vector'))
for name,rs in [('summary',rows),('comparisons',read('comparisons')),('per_parameter',read('per_parameter'))]:
 with (OUT/(name+'.csv')).open('w') as h:
  writer=csv.DictWriter(h,fieldnames=list(rs[0]));writer.writeheader();writer.writerows(rs)
save('completion',dict(status='complete',forwards=2,gradient_queries=2,optimizer_steps=0,new_graphs=0,receipt=read('receipt'),release=read('release'),output_hashes={str(p):sha(p) for p in OUT.glob('*decomposition.*')}))
print(json.dumps(rows,ensure_ascii=False,indent=2))
