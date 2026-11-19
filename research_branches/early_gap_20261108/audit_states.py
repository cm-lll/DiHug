"""CPU full-state and recorded displacement audit, no model calls."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']=os.environ['OPENBLAS_NUM_THREADS']='1'
import torch,numpy as np,json,sys
from pathlib import Path
torch.set_num_threads(1);out=Path(sys.argv[1])
def equal(a,b):
 if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
 if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and np.array_equal(a,b)
 if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
 if isinstance(a,(list,tuple)):return type(a)==type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
 return a==b
a=torch.load(out/'PubMed_CE_diagnostic_state.pt',map_location='cpu')['state'];b=torch.load(out/'PubMed_PS_diagnostic_state.pt',map_location='cpu')['state']
assert a['updates']==b['updates']==2008 and a['ema']['updates']==b['ema']['updates']==2008
assert equal(a['rng'],b['rng'])
sumgap={}
for i in range(1,9):
 x=torch.load(out/f'PubMed_CE_{i}_parameter_displacement.pt',map_location='cpu');y=torch.load(out/f'PubMed_PS_{i}_parameter_displacement.pt',map_location='cpu')
 for k in x:sumgap[k]=sumgap.get(k,0)+(y[k].double()-x[k].double())
maxerr=max(float((v-(b['model'][k].double()-a['model'][k].double())).abs().max()) for k,v in sumgap.items())
assert maxerr<1e-7
assert not torch.cuda.is_initialized()
r={'status':'passed','endpoint_updates':2008,'EMA_updates':2008,'paired_endpoint_full_RNG_exact':True,'sum_recorded_delta_vs_final_parameter_gap_max_abs':maxerr,'model_calls':0,'GPU_calls':0}
(out/'state_audit.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
