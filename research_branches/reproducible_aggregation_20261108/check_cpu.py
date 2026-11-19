"""Bounded math/gradient checks; no denoiser import or CUDA initialization."""
import os,sys,json,ast,inspect
from pathlib import Path
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'research_branches/joint_output_projected_review_20261108'))
import torch
torch.set_num_threads(1)
from pb_degree import _GroupedCDF
from adapter import ReproducibleGroupedCDF,prefix_sum,execution_policy
OUT=ROOT/'results/reproducible_aggregation_20261108T113027';OUT.mkdir(exist_ok=True)
rows=[]
for width in [1,7,21]:
 for style in ['interior','binary']:
  generator=torch.Generator().manual_seed(2026110800+width)
  p=torch.rand((5,width),generator=generator,dtype=torch.float64)
  if style=='binary':p=(p>.5).double()
  target=torch.linspace(.1,.9,width,dtype=torch.float64).repeat(2,1);bounds=((0,2),(2,5));caps=(width,width)
  results=[]
  for cls in [_GroupedCDF,ReproducibleGroupedCDF]:
   q=p.clone().requires_grad_();v=cls.apply(q,target,bounds,caps);grad=torch.autograd.grad(v.sum(),q)[0];results.append((v.detach(),grad))
  # Independent differentiable polynomial reference, same inclusive CDF target.
  q=p.clone().requires_grad_();h=q.new_ones((5,1))
  for j in range(width):
   v=q[:,j:j+1];h=torch.nn.functional.pad(h*(1-v),(0,1))+torch.nn.functional.pad(h*v,(1,0))
  values=torch.stack([.5*(h[lo:hi].mean(0).cumsum(0)[:width]-target[g]).square().mean() for g,(lo,hi) in enumerate(bounds)])
  gradient=torch.autograd.grad(values.sum(),q)[0]
  r=dict(width=width,style=style,legacy_value_exact=torch.equal(results[0][0],results[1][0]),legacy_gradient_exact=torch.equal(results[0][1],results[1][1]),independent_value_max=float((values-results[1][0]).abs().max()),independent_gradient_max=float((gradient-results[1][1]).abs().max()))
  assert r['legacy_value_exact'] and r['legacy_gradient_exact']
  assert r['independent_value_max']<1e-12 and r['independent_gradient_max']<1e-12
  rows.append(r)
# Verify only the two registered scans differ from the original class math.
def normalized(cls):
 src=inspect.getsource(cls).replace(cls.__name__,'CDF').replace('h[lo:hi].mean(0).cumsum(0)','prefix_sum(h[lo:hi].mean(0))').replace('dcdf.flip(0).cumsum(0)','prefix_sum(dcdf.flip(0))')
 tree=ast.parse(src)
 for node in ast.walk(tree):
  if isinstance(node,(ast.ClassDef,ast.FunctionDef)) and node.body and isinstance(node.body[0],ast.Expr) and isinstance(node.body[0].value,ast.Constant) and isinstance(node.body[0].value.value,str):node.body.pop(0)
 return ast.dump(tree,include_attributes=False)
assert normalized(_GroupedCDF)==normalized(ReproducibleGroupedCDF)
import pb_degree,projected_degree
old=(pb_degree._GroupedCDF,projected_degree._GroupedCDF,torch.are_deterministic_algorithms_enabled())
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
try:
 with execution_policy(True):
  assert torch.are_deterministic_algorithms_enabled() and pb_degree._GroupedCDF is ReproducibleGroupedCDF
  raise RuntimeError('injected scope exit')
except RuntimeError as e:assert str(e)=='injected scope exit'
assert old==(pb_degree._GroupedCDF,projected_degree._GroupedCDF,torch.are_deterministic_algorithms_enabled())
assert not torch.cuda.is_initialized()
result=dict(status='passed',tests=rows,AST_equivalent_except_two_prefix_calls=True,policy_exception_restore=True,CUDA_initialized=False)
(OUT/'cpu_checks.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
