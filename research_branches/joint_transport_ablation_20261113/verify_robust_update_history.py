"""CPU-only comparison of original-score diagnostic with saved historical update."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/robust_heat_update_check_20261113T095140'
OLD=ROOT/'results/joint_transport_ablation_20261113T011928'
def equal(a,b):
 if isinstance(a,torch.Tensor):return bool(torch.equal(a,b))
 if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
 if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
 if hasattr(a,'shape'):
  import numpy as np
  return bool(np.array_equal(a,b))
 return a==b
rows=[]
for f in sorted(OUT.glob('*_original_temporary.pt')):
 job=f.name.removesuffix('_original_temporary.pt');ds=job.split('_')[0]
 a=torch.load(f,map_location='cpu',weights_only=False)
 b=torch.load(OLD/job/f'{ds}_T_eager/update_0001_complete.pt',map_location='cpu',weights_only=False)
 eq={k:bool(equal(a[k],b[k])) for k in a if k!='binding' and k in b}
 md=max(float((a['model'][k]-b['model'][k]).abs().max()) for k in a['model'] if a['model'][k].numel() and a['model'][k].is_floating_point())
 rows.append(dict(job=job,exact_by_component=eq,model_max_abs_difference=md))
(OUT/'historical_original_update_comparison.json').write_text(json.dumps(rows,indent=2))
print(json.dumps(rows,indent=2))
