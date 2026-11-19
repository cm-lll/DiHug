"""Verify persisted results without repeating the scientific panel."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import json,itertools,time,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/joint_degree_full_signal_20261109'
def read(n):return json.loads((OUT/(n+'.json')).read_text())
def save(n,x):(OUT/(n+'.json')).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
pairs=read('paired_draws');panels=read('panel_summary');tests=read('tests')
assert len(pairs)==28 and len(panels)==56 and len(tests)==72
assert max(r['full_feature_delta_error'] for r in tests)<1e-10
old=json.loads((ROOT/'results/joint_degree_signal_20261109/verified/slot_feedback.json').read_text())
bykey={(f"{r['dataset']}_{r['parent']}_CE_t{r['t']}",r['draw'],r['slot']):r for r in old}
found=0;coords=0;files=sorted(OUT.glob('*_draw[01].npz'));assert len(files)==24
for path in files:
 tag,draw=path.stem.rsplit('_draw',1);draw=int(draw)
 with np.load(path) as z:
  ix=z['slots'];coords+=len(ix);p=z['probability'];b=z['bits'];factor=p[ix]*(1-p[ix])*(1-2*b[ix])
  for method in ['old','new']:
   assert np.isfinite(z[method+'_g'][ix]).all()
   assert np.max(abs(factor*z[method+'_delta'][ix]-z[method+'_g'][ix]))<1e-13
  for e in ix:
   r=bykey.get((tag,draw,int(e)))
   if r is not None:
    found+=1;assert abs(z['new_g'][e]-r['new_logit_derivative'])<1e-11
assert found==2765 and coords<=78874
# Exhaustive fixed-vector HT variance identity; no random draws or graphs.
g=np.array([.2,-.7,1.4,0.,.3]);M=len(g);m=2;errors=[]
for ss in itertools.combinations(range(M),m):
 h=np.zeros(M);h[list(ss)]=g[list(ss)]*M/m;errors.append(float(np.square(h-g).sum()))
analytic=(M/m-1)*float(g@g);err=abs(np.mean(errors)-analytic);assert err<1e-12
save('independent_checks',dict(pairs=28,panel_summaries=56,feature_checks=72,files=24,coordinates=coords,prior_slots_reproduced=found,HT_exhaustive_subsets=10,HT_identity_error=err))
aggregate=[]
for ds in ['IMDB','PubMed']:
 for t in [1,500,1000]:
  for scope in ['union','full']:
   for method in ['old','new']:
    rr=[r for r in panels if r['dataset']==ds and r['t']==t and r['scope']==scope and r['method']==method]
    if not rr:continue
    mass=sum(r['rewarded_abs_mass'] for r in rr);cost=sum(r['triangle_cost_abs_mass'] for r in rr)
    aggregate.append(dict(dataset=ds,t=t,scope=scope,method=method,states=len(rr),slots=sum(r['slots'] for r in rr),rewarded=sum(r['rewarded'] for r in rr),triangle_cost=sum(r['triangle_cost'] for r in rr),
     hub_cost=sum(r['hub_cost'] for r in rr),isolates_cost=sum(r['isolates_cost'] for r in rr),triangle_cost_mass_share=cost/mass if mass else None,CE_opposed=sum(r['CE_opposed'] for r in rr)))
save('aggregate',aggregate)
bindings=json.loads((ROOT/'results/joint_degree_signal_20261109/verified/binding.json').read_text())
assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in bindings.items())
save('verified_prior_bindings',bindings)
failure=read('failure') if (OUT/'failure.json').exists() else None
if failure:
 assert failure['error']=='AssertionError()' and 'len(pairrows)==28*2' in failure['traceback'],failure
 save('engineering_notes',dict(issue='Final count assertion mistakenly expected56 paired-draw records; correct count is12 union panels*2methods+2 full panels*2methods=28. 56 is per-draw summary count.',
  resolution='All24 outputs and72 full-feature checks were already saved. Independent finalizer verified correct28 pairs/56 summaries and reproduced2765 prior slots. No science recomputation, no formula change. Original failure and executed source retained.',
  failure_file='failure.json'))
save('completion',dict(status='complete_after_independent_output_verification' if failure else 'complete',coordinates=coords,assets_verified=len(bindings),
 worker_seconds=failure['worker_seconds'] if failure else read('receipt')['worker_seconds'],
 memory='Worker checked RSS<2GiB each32 coordinates; exact peak unavailable if final receipt assertion failed',
 GPU_hours=0,models=0,updates=0,new_graphs=0,
 output_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files+[OUT/'paired_draws.json',OUT/'panel_summary.json',OUT/'relation_label_summary.json']}))
print(json.dumps(dict(complete=True,coordinates=coords,regressions=found,full_pairs=[r for r in pairs if r['scope']=='full']),ensure_ascii=False))
