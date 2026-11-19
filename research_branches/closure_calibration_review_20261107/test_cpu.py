"""Small exact checks; CPU tensors only, no denoiser/training or graph generation."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import time,json,itertools,hashlib
from pathlib import Path
import numpy as np
import torch
from closure_calibration import WedgeCalibration,combine
torch.set_num_threads(1)
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/pb_trajectory_repair_20261107';START=time.time();checks=[]
def build(types,edges,schema):
    edges=np.asarray(edges,dtype=int).reshape(-1,2).T;n=len(types);u,v=np.triu_indices(n,1)
    fam=np.array([schema.index(tuple(sorted((types[a],types[b])))) for a,b in zip(u,v)])
    E=set(map(tuple,edges.T));y=np.array([(a,b) in E for a,b in zip(u,v)])
    prior=[float(y[fam==i].mean()) if np.any(fam==i) else 0 for i in range(len(schema))]
    return WedgeCalibration(types,schema,prior,edges,data_sha256='a'*64)
o=build([0,0,0,1,1],[(0,1),(1,2),(0,3),(1,3),(2,4)],[(0,0),(0,1),(1,1)])
xt=np.array([(0,1),(0,2),(1,3),(2,3),(3,4)]).T
# Last pair is fixed zero: reject instead of silently treating query edges as Xt.
try:o.context(xt);raise AssertionError('fixed-support violation accepted')
except ValueError:checks.append('fixed_zero_rejected')
xt=xt[:,:-1];ctx=o.context(xt);q=np.stack([o.keys//o.n,o.keys%o.n]);f=o.families
for i,g in enumerate(ctx.groups):
    edges=set(map(tuple,xt.T));cn=[]
    for a,b in q.T:
        cn.append(sum(tuple(sorted((a,k))) in edges and tuple(sorted((b,k))) in edges for k in range(o.n) if k not in (a,b)))
    mask=(f==g['relation'])&np.array([int(tuple(x) in edges)==g['Xt_present'] for x in q.T]);v=np.array(cn)*mask
    assert np.allclose(ctx.weights[i],v/v.sum())
checks.append('hand_counted_frozen_Xt_CN_and_groups')
p=torch.linspace(.1,.9,len(o.keys),dtype=torch.float64,requires_grad=True)
loss,_=ctx(p,q,f);grad=torch.autograd.grad(loss,p)[0]
expected=ctx.weights.T@(ctx.weights@p.detach().numpy()-ctx.targets)/len(ctx.groups)
assert np.allclose(grad.numpy(),expected,atol=1e-12)
assert torch.autograd.gradcheck(lambda z:ctx(z,q,f)[0],(p,),eps=1e-6,atol=1e-7,rtol=1e-5)
checks.append('analytic_gradient_and_finite_difference')
truth=torch.tensor(o.labels,dtype=torch.float64,requires_grad=True);v,_=ctx(truth,q,f);assert float(v)==0 and torch.count_nonzero(torch.autograd.grad(v,truth)[0])==0
checks.append('zero_target_at_X0_probabilities')
perm=np.arange(len(o.keys))[::-1].copy();assert torch.allclose(ctx(p[perm],q[:,perm],f[perm])[0],loss)
try:ctx(p[:-1],q[:,:-1],f[:-1]);raise AssertionError('subset accepted')
except ValueError:pass
checks.append('query_permutation_and_full_coverage')
empty=o.context(np.zeros((2,0),dtype=int));v,_=empty(p,q,f);assert float(v)==0 and torch.count_nonzero(torch.autograd.grad(v,p)[0])==0
checks.append('zero_wedge_empty_groups_finite_zero')
# Verify mixture expectation exactly without assuming independence of closing edges:
# expectation of a linear weighted edge sum is always sum(weights*marginal).
weights=ctx.weights[0];joint=[(.35,np.zeros(len(o.keys))),(.65,np.ones(len(o.keys)))]
assert np.allclose(sum(w*(weights@x) for w,x in joint),weights@sum(w*x for w,x in joint))
checks.append('correlated_edge_joint_linear_expectation')
# Node permutation maps topology, labels, priors and probabilities together.
pi=np.array([2,4,1,0,3]);typ=np.empty(o.n,dtype=int);typ[pi]=o.types
clean=np.stack([o.keys[o.labels==1]//o.n,o.keys[o.labels==1]%o.n]);clean=np.sort(pi[clean],axis=0)
oo=WedgeCalibration(typ,o.schema,o.prior,clean,data_sha256='b'*64)
xx=np.sort(pi[xt],axis=0);qq=np.sort(pi[q],axis=0);other=oo.context(xx)
assert torch.allclose(other(p,qq,f)[0],loss,atol=1e-12)
checks.append('node_relabeling_invariance')
assert torch.equal(combine(p.sum(),18336,p.sum()*0,loss),p.sum()/18336+loss)
checks.append('CE_weight_unmodified_adapter')
assert not torch.cuda.is_initialized()
receipt=dict(passed=True,checks=checks,seconds=time.time()-START,CUDA_initialized=False,model_calls=0,training_updates=0,new_generated_graphs=0)
(OUT/'cpu_tests.json').write_text(json.dumps(receipt,indent=2)+'\n');print(receipt)
