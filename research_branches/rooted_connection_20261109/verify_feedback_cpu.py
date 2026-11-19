"""Independent finite algebra and score decomposition; no model or optimizer."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import itertools,json,time,hashlib
from pathlib import Path
import numpy as np
import torch
from score import RootedScore,RootedIncremental,kernel
torch.set_num_threads(1)
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/rooted_feedback_audit_20261109';RUN=ROOT/'results/rooted_connection_20261109T015142'
def main():
    start=time.time()
    assert not (OUT/'independent_checks.json').exists()
    g=np.array([.1,-.2,0.,.4,-.7,.9]);m=2;M=len(g);samples=[]
    for ix in itertools.combinations(range(M),m):
        h=np.zeros(M);h[list(ix)]=M/m*g[list(ix)];samples.append(h)
    samples=np.array(samples);mean_error=float(abs(samples.mean(0)-g).max())
    measured=float(((samples-g)**2).sum(1).mean());predicted=(M/m-1)*float(g@g)
    assert mean_error<1e-14 and abs(measured-predicted)<1e-14
    # The production surrogate differentiates logit1-logit0; sorting must map
    # canonical slots back to original query order. Distinct probabilities and
    # differences prevent a permutation from accidentally passing.
    permutation=np.array([3,0,5,2,1,4]);order=np.argsort(permutation)
    logits=torch.tensor([[.3,.1],[.2,-.6],[1.1,.4],[-.5,.8],[.2,.9],[.7,-.2]],dtype=torch.float64,requires_grad=True)
    p=torch.softmax(logits,dim=1)[:,1].detach().numpy()[order]
    delta=np.array([.1,.2,-.4,.9,-.3,.5]);ix=np.array([1,4])
    derivative=p[ix]*(1-p[ix])*delta[ix]*(M/len(ix))
    surrogate=((logits[order[ix],1]-logits[order[ix],0])*torch.tensor(derivative)).sum()
    actual=torch.autograd.grad(surrogate,logits)[0].numpy();expected=np.zeros_like(actual)
    expected[order[ix],1]=derivative;expected[order[ix],0]=-derivative
    assert np.array_equal(actual,expected)
    check=dict(subsets=len(samples),mean_error=mean_error,empirical_MSE=measured,predicted_MSE=predicted,
        surrogate_query_permutation_error=float(abs(actual-expected).max()),model_calls=0,GPU=0,
        seconds=time.time()-start,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (OUT/'independent_checks.json').write_text(json.dumps(check,indent=2))
    print(json.dumps(check))
if __name__=='__main__':main()
