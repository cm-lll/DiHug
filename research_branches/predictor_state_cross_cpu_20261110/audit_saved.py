"""Arithmetic audit of saved arrays and the full state partition; no model evaluation."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import hashlib,json,time
from pathlib import Path
import numpy as np
START=time.monotonic();ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/predictor_state_cross_cpu_20261110'
def read(n):return json.loads((OUT/(n+'.json')).read_text())
a=np.load(OUT/'distributions.npz');panels=read('panels');decomp=read('decomposition');rows=read('state_details')
names=['relation_count','anonymous_joint_degree','triangle001','triangle011','components','total_cost','connected_probability','triangle_count_MAE']
err=[]
for inp,key in [('0','mu_old'),('20','mu_new'),('qref','qref')]:
 for step,pkey in [(0,'clean_old'),(20,'clean_new')]:
  dist=np.einsum('i,ij->j',a[key],a[pkey]);r=next(x for x in panels if x['input_source']==inp and x['predictor_step']==step)
  err.append(abs(dist.sum()-1))
  for j,name in enumerate(names):err.append(abs(float(dist@a['cost'][:,j])-r[name]))
for j,name in enumerate(names):
 h0=a['clean_old']@a['cost'][:,j];h1=a['clean_new']@a['cost'][:,j]
 d=next(x for x in decomp if x['metric']==name)
 values=[float(a['mu_old']@(h1-h0)),float((a['mu_new']-a['mu_old'])@h0),float((a['mu_new']-a['mu_old'])@(h1-h0))]
 for k,v in zip(['predictor_at_old_states','state_at_old_predictor','interaction'],values):err.append(abs(v-d[k]))
for path,sha in read('bindings').items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha,path
s=[x for x in rows if x['metric']=='components'];sparse=[x for x in s if x['edges']<=2];dense=[x for x in s if x['edges']>=3]
assert len(s)==64 and len(sparse)==22 and len(dense)==42
assert all(x['change']>0 for x in sparse) and all(x['change']<0 for x in dense)
assert max(err)<1e-10
result=dict(passed=True,checks=len(err),max_error=max(err),seconds=time.monotonic()-START,
 sparse_states=len(sparse),other_states=len(dense),
 sparse_qref_mass=sum(x['qref'] for x in sparse),sparse_old_generated_mass=sum(x['mu_old'] for x in sparse),
 sparse_new_generated_mass=sum(x['mu_new'] for x in sparse),sources_unchanged=True)
(OUT/'independent_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
