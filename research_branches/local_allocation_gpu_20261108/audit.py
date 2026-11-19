"""Audit actual saved Adam state on CPU, without creating a denoiser."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']=os.environ['OPENBLAS_NUM_THREADS']='1'
import torch,json,sys,hashlib,time
from pathlib import Path
torch.set_num_threads(1);OUT=Path(sys.argv[1]);ROOT=Path(__file__).resolve().parents[2];start=time.time()
read=lambda p:json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
r=read(OUT/'receipt.json');lease=read(OUT/'lease.json');P=read(OUT/'protocol.json')
assert r['status']=='complete' and lease['returncode']==0 and lease['gpu_hours']<=P['gpu_hours_limit'] and lease['end_epoch']<=P['science_stop_epoch']
source=read(ROOT/'results/joint_output_formal_projected_20261108T043000/source_binding.json');assert all(sha(p)==h for p,h in source.items())
checks=[]
for asset in read(OUT/'assets.json'):
 assert sha(asset['parent'])==asset['sha256'] and sha(asset['reused_CE_state'])==asset['reused_CE_sha256']
 cond=asset['condition'];parent=torch.load(asset['parent'],map_location='cpu')['state'];livefile=OUT/f'{cond}_RANK_temporary.pt';live=torch.load(livefile,map_location='cpu')['state'];t=torch.load(OUT/f'{cond}_tensors.pt',map_location='cpu')
 assert live['updates']==live['ema']['updates']==asset['update']
 ids=live['optimizer']['param_groups'][0]['params'];beta1,beta2=live['optimizer']['param_groups'][0]['betas'];maxerr={'exp_avg':0.,'exp_avg_sq':0.}
 for j,(k,span) in enumerate(zip(ids,t['spans'])):
  n,group,lo,hi=span
  assert torch.isfinite(live['model'][n]).all()
  if t['gradient_none'][j]:continue
  grad=t['actual_gradient'][lo:hi].double();old=parent['optimizer']['state'][k];new=live['optimizer']['state'][k]
  assert int(new['step'])==int(old['step'])+1
  for field,beta in [('exp_avg',beta1),('exp_avg_sq',beta2)]:
   expected=beta*old[field].flatten().double()+(1-beta)*(grad if field=='exp_avg' else grad.square());actual=new[field].flatten().double()
   assert torch.allclose(actual,expected,atol=2e-6,rtol=1e-5)
   maxerr[field]=max(maxerr[field],float((actual-expected).abs().max()))
 checks.append(dict(condition=cond,moment_max_abs=maxerr,checkpoint_sha256=sha(livefile),tensors_sha256=sha(OUT/f'{cond}_tensors.pt')))
 del parent,live,t
assert not torch.cuda.is_initialized()
result=dict(status='passed',source_files_unchanged=len(source),parents_and_CE_references_unchanged=4,states=checks,seconds=time.time()-start,model_calls=0,gpu_hours=lease['gpu_hours'])
(OUT/'audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
