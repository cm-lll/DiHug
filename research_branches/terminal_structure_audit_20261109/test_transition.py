"""Bounded CPU checks of the isolated likelihood adapter, not O/A recovery tests."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import hashlib, importlib.util, itertools, json, resource, signal, time
from pathlib import Path
import numpy as np
import torch
from transition_objective import reverse_probability,transition_log_probability,score_function_loss

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/terminal_structure_audit_20261109/adapter'
K=ROOT/'research_branches/cn_multiseed_20261104/research_branches/typed_structure_early_cn_expansion_review/experiments/typed_structure_early_cn_expansion_v1/package/hsgd/kernel.py'

def main():
    start=time.monotonic();OUT.mkdir(exist_ok=False)
    signal.alarm(60);torch.set_num_threads(1);torch.set_num_interop_threads(1)
    spec=importlib.util.spec_from_file_location('production_kernel',K);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    prior=np.array([0.,1.,.03,.4,.95]);alpha=np.array([1.,.65,.25,0.]);kernel=mod.MarginalKernel(prior,alpha)
    tests=[]
    def check(name,error,tol=1e-10):
        tests.append(dict(name=name,error=float(error),tolerance=tol,passed=bool(error<tol)))
        assert error<tol,(name,error)
    family=torch.arange(5,dtype=torch.long)
    for dtype in [torch.float64,torch.float32]:
        for t in [1,2,3]:
            for bits in itertools.product([0,1],repeat=3):
                x=torch.tensor([0,1,*bits]);p=torch.tensor([0.,1.,.02,.6,.997],dtype=dtype,requires_grad=True)
                q=reverse_probability(p,x,family,prior,alpha,t-1,t)
                expected=kernel.reverse(p.detach().numpy(),x.numpy(),family.numpy(),t-1,t)
                check(f'kernel_{dtype}_{t}_{bits}',np.max(abs(q.detach().numpy()-expected)))
                y=torch.tensor([0,1,1,0,1]);lp=transition_log_probability(p,x,y,family,prior,alpha,t-1,t)
                expectedlp=np.log(np.where(y.numpy()>0,expected,1-expected)).sum()
                check(f'logprob_{dtype}_{t}_{bits}',abs(lp.item()-expectedlp))
                g=torch.autograd.grad(lp,p)[0].detach().numpy()
                check(f'fixed_gradient_{dtype}_{t}_{bits}',abs(g[:2]).max())
                if dtype==torch.float64:
                    for e in range(2,5):
                        h=1e-6;pa=p.detach().numpy().copy();pb=pa.copy();pa[e]+=h;pb[e]-=h
                        qa=kernel.reverse(pa,x.numpy(),family.numpy(),t-1,t);qb=kernel.reverse(pb,x.numpy(),family.numpy(),t-1,t)
                        fd=(np.log(np.where(y.numpy()>0,qa,1-qa)).sum()-np.log(np.where(y.numpy()>0,qb,1-qb)).sum())/(2*h)
                        check(f'gradient_{t}_{bits}_{e}',abs(fd-g[e]),1e-4)
    # Saturated clean probability is allowed if the realized event has support.
    p=torch.tensor([0.,1.,0.,1.,.5],dtype=torch.float64,requires_grad=True);x=torch.tensor([0,1,0,1,0]);y=x.clone()
    lp=transition_log_probability(p,x,y,family,prior,alpha,0,1);g=torch.autograd.grad(lp,p)[0]
    check('saturation_realized_gradient_finite',0 if torch.isfinite(g).all() else 1)
    for name,change in [('fixed_support',lambda:transition_log_probability(p,x,torch.tensor([1,1,0,1,0]),family,prior,alpha,0,1)),
                        ('zero_probability',lambda:transition_log_probability(p,x,torch.tensor([0,1,1,1,0]),family,prior,alpha,0,1))]:
        rejected=False
        try:change()
        except ValueError:rejected=True
        check(name+'_reject',0 if rejected else 1)
    # Independent two-rollout enumeration: leave-one-out retains expectation.
    theta=torch.tensor(.3,dtype=torch.float64,requires_grad=True);prob=theta.sigmoid();expected_grad=prob*(1-prob)*2
    estimated=0.
    for a,b in itertools.product([0.,1.],repeat=2):
        bits=torch.tensor([a,b],dtype=torch.float64)
        logps=torch.log(torch.where(bits.bool(),prob,1-prob))[:,None]
        cost=2*bits+3
        loss=score_function_loss(logps,cost,1,1)
        grad=torch.autograd.grad(loss,theta,retain_graph=True)[0]
        mass=torch.prod(torch.where(bits.bool(),prob,1-prob)).detach()
        estimated+=float(mass*grad)
    check('independent_leave_one_out_expectation',abs(estimated-float(expected_grad)))
    lp=torch.tensor([[-.3,-.2],[-.1,-.4]],dtype=torch.float64,requires_grad=True)
    c=torch.tensor([2.,5.],requires_grad=True)
    loss=score_function_loss(lp,c,10,2)
    gl,gc=torch.autograd.grad(loss,(lp,c),allow_unused=True)
    check('uniform_time_factor',abs(gl.detach().numpy()-np.array([[-7.5,-7.5],[7.5,7.5]])).max())
    check('cost_and_baseline_detached',0 if gc is None else 1)
    receipt=dict(status='complete_stopped',tests=len(tests),passed=all(t['passed'] for t in tests),
                 seconds=time.monotonic()-start,RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                 torch=torch.__version__,CUDA_initialized=torch.cuda.is_initialized(),GPU_hours=0,
                 production_model_calls=0,production_updates=0,real_generated_graphs=0,
                 limitation='CPU adapter only; no model random-context replay, saturation frequency or GPU trajectory test',
                 hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [K,Path(__file__),Path(__file__).with_name('transition_objective.py')]})
    (OUT/'tests.json').write_text(json.dumps(tests,indent=2)+'\n');(OUT/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))
if __name__=='__main__':
    try:main()
    except BaseException as e:
        if OUT.exists():(OUT/'failure.json').write_text(json.dumps(dict(error=repr(e))))
        raise
