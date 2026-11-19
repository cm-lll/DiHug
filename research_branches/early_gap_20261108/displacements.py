"""Analyze saved actual updates on CPU; no model construction."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']=os.environ['OPENBLAS_NUM_THREADS']='1'
import torch,json,csv,sys,math
from pathlib import Path
torch.set_num_threads(1);out=Path(sys.argv[1]);rows=[];cumulative={}
def cat(n):
 if n.startswith('common_neighbor.'):return 'CN_exclusive'
 if n.startswith('joint_head.'):return 'joint_head'
 if n.startswith('branch.'):return 'degree_branch'
 return 'shared_backbone_other'
for step in range(1,9):
 a=torch.load(out/f'PubMed_CE_{step}_parameter_displacement.pt',map_location='cpu')
 b=torch.load(out/f'PubMed_PS_{step}_parameter_displacement.pt',map_location='cpu')
 assert a.keys()==b.keys()
 for n in a:
  d=b[n].double()-a[n].double();cumulative[n]=cumulative.get(n,0)+d
 for group in ['all','CN_exclusive','joint_head','degree_branch','shared_backbone_other']:
  names=[n for n in a if group=='all' or cat(n)==group]
  aa=bb=dot=diff=gap=0.
  for n in names:
   x=a[n].double();y=b[n].double();aa+=float(x.square().sum());bb+=float(y.square().sum());dot+=float((x*y).sum());diff+=float((x-y).square().sum());gap+=float(cumulative[n].square().sum())
  rows.append(dict(offset=step,group=group,CE_update_L2=math.sqrt(aa),PS_update_L2=math.sqrt(bb),update_cosine=dot/math.sqrt(aa*bb) if aa*bb else None,update_difference_L2=math.sqrt(diff),cumulative_parameter_gap_L2=math.sqrt(gap)))
with (out/'displacement_geometry.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
assert not torch.cuda.is_initialized()
print(json.dumps([r for r in rows if r['group']=='all']))
