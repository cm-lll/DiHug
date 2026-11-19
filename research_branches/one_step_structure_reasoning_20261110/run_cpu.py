"""Exhaustive finite arithmetic only; no O/A network or optimizer."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import csv, hashlib, importlib.util, json, resource, signal, time
from pathlib import Path
import numpy as np
from scipy.special import logit
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/one_step_structure_reasoning_20261110'
START=time.monotonic();BIND={}
def bind(p):
    p=Path(p);BIND[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest();return p
def load(name,p):
    spec=importlib.util.spec_from_file_location(name,bind(p));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def save(name,x):
    (OUT/(name+'.json')).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    if isinstance(x,list) and x:
        with (OUT/(name+'.csv')).open('w',newline='') as f:
            w=csv.DictWriter(f,list(x[0]));w.writeheader();w.writerows(x)
def cosine(x,y):
    den=np.linalg.norm(x)*np.linalg.norm(y);return float(x@y/den) if den else None
def main():
    OUT.mkdir(parents=True,exist_ok=False);signal.alarm(180);resource.setrlimit(resource.RLIMIT_AS,(1024**3,1024**3))
    cfg=json.loads(bind(Path(__file__).with_name('protocol.json')).read_text());save('protocol',cfg);bind(Path(__file__))
    m=load('finite_one_step',ROOT/'research_branches/terminal_structure_audit_20261109/audit_cpu.py');m.kernel_module=load('bound_kernel_one_step',m.KERNEL)
    old=json.loads(bind(m.PROTOCOL).read_text());f=m.FiniteDiffusion(old)
    ids=np.arange(f.S);flags=2**np.arange(f.M-1,-1,-1)
    ids0=ids[:,None] & ~flags[None,:];ids1=ids0 | flags[None,:]
    diff=f.cost[ids1]-f.cost[ids0]
    checks=[];contexts=[];summary=[]
    def check(name,error,tol=1e-10):
        checks.append(dict(name=name,error=float(error),tolerance=tol,passed=bool(error<tol)));assert error<tol,(name,error)
    def risk(theta):
        _,_,clean,_=f.matrices(theta)
        return sum(float(f.qref[t]@np.exp(clean[t])@f.cost) for t in range(1,f.T+1))/f.T
    for seed in old['initializations']:
        theta=np.random.default_rng(seed).normal(0,.15,f.D);theta[:3]+=logit(f.pi)
        ev=f.evaluate(theta);_,_,clean,pred=f.matrices(theta)
        total_gradient=np.zeros(f.D);by_time=[]
        for t in range(1,f.T+1):
            P=np.exp(clean[t]);p=pred[t];F=f.features[t]
            G=np.einsum('ije,ied->ijd',f.B[None,:,:]-p[:,None,:],F)
            means=np.einsum('ij,j,ijd->id',P,f.cost,G)
            total_gradient+=f.qref[t]@means/f.T
            # Conditional edge difference integrates over all other Bernoulli edges.
            dlogit=p*(1-p)*(P@diff)
            check(f'edge_difference_{seed}_{t}',np.max(abs(np.einsum('ie,ied->id',dlogit,F)-means)))
            check(f'normalization_{seed}_{t}',np.max(abs(P.sum(1)-1)))
            for x in range(f.S):
                dist=P[x];mean=means[x];center=float(dist@f.cost)
                vanilla=f.cost[:,None]*G[x]
                centered=(f.cost-center)[:,None]*G[x]
                contributions=(p[x]*(1-p[x]))[None,:,None]*diff[:,:,None]*F[x][None,:,:]
                conditional=contributions.sum(1)
                check(f'conditional_mean_{seed}_{t}_{x}',np.max(abs(dist@conditional-mean)))
                # Choose a uniformly random coordinate, multiply by M; no coordinate drawn here.
                coord_second=float(np.einsum('s,sed,sed->',dist,contributions,contributions)*f.M)
                def variance(a):return float(dist@np.sum((a-mean)**2,axis=1))
                contexts.append(dict(seed=seed,t=t,input_state=x,input_forward_mass=float(f.qref[t][x]),hard_risk=center,
                    gradient_norm=float(np.linalg.norm(mean)),RF_variance=variance(vanilla),
                    oracle_centered_RF_variance=variance(centered),all_coordinate_difference_variance=variance(conditional),
                    one_coordinate_difference_variance=coord_second-float(mean@mean)))
            blocks=f.qref[t]@P@f.blocks
            by_time.append(dict(t=t,hard_clean_risk=float(f.qref[t]@P@f.cost),**dict(zip(f.block_names,blocks.tolist()))))
        eps=cfg['fixed_hypothetical_derivative_step'];fd=[]
        for j in range(f.D):
            d=np.eye(f.D)[j]*eps;fd.append((risk(theta+d)-risk(theta-d))/(2*eps))
        check(f'finite_difference_{seed}',np.max(abs(np.array(fd)-total_gradient)),1e-7)
        # Directional derivative only; neither branch is optimized or retained as a trained state.
        direction=-total_gradient/max(np.linalg.norm(total_gradient),1e-15)
        terminal_fd=(f.evaluate(theta+eps*direction)['metrics']['terminal_cost']-f.evaluate(theta-eps*direction)['metrics']['terminal_cost'])/(2*eps)
        check(f'terminal_directional_derivative_{seed}',abs(terminal_fd-float(ev['g']@direction)),1e-7)
        local_rows=[r for r in contexts if r['seed']==seed]
        variances={k:sum(r['input_forward_mass']*r[k]/f.T for r in local_rows) for k in ['RF_variance','oracle_centered_RF_variance','all_coordinate_difference_variance','one_coordinate_difference_variance']}
        summary.append(dict(seed=seed,one_step_forward_risk=risk(theta),terminal_chain_risk=ev['metrics']['terminal_cost'],
            local_gradient_norm=float(np.linalg.norm(total_gradient)),local_vs_terminal_gradient_cosine=cosine(total_gradient,ev['g']),
            local_vs_original_CE_gradient_cosine=cosine(total_gradient,ev['ce']),
            local_descent_terminal_directional_derivative=terminal_fd,
            conditional_on_Xt_expected_variance=variances,by_time=by_time))
    check('reference_cost_zero',abs(f.cost[f.refid]))
    minima=np.flatnonzero(abs(f.cost-f.cost.min())<1e-12)
    save('input_dependence_witness',dict(minimum_cost=float(f.cost.min()),minimizer_indices=minima.tolist(),
        minimizer_isomorph_classes=int(len(np.unique(f.orbits[minima]))),
        input_independent_reference_hard_risk=float(f.cost[f.refid]),
        note='Objective-level unrestricted product-family witness; not claiming the actual equivariant A architecture can output a fixed labeled template without input. Pure structural risk does not identify a conditional denoising distribution.'))
    # Two exchangeable candidate edges. Good graphs are 10 and 01 (one edge).
    bits=np.array([[0.,0.],[0.,1.],[1.,0.],[1.,1.]])
    target=np.array([0.,.5,.5,0.]);c=(bits.sum(1)-1)**2;tworows=[]
    for alpha in [.65,.25,0.]:
        qprob=alpha*bits+(1-alpha)*.5
        Q=np.prod(np.where(bits[None,:,:]>0,qprob[:,None,:],1-qprob[:,None,:]),axis=-1)
        px=target@Q;posterior=target[:,None]*Q/px[None,:]
        marginals=posterior.T@bits
        indep=np.prod(np.where(bits[None,:,:]>0,marginals[:,None,:],1-marginals[:,None,:]),axis=-1)
        check(f'coupled_posterior_normalization_{alpha}',np.max(abs(posterior.sum(0)-1)))
        check(f'marginals_preserved_{alpha}',np.max(abs(indep@bits-marginals)))
        for x in range(4):
            tworows.append(dict(retention=alpha,input_bits=bits[x].astype(int).tolist(),input_mass=float(px[x]),
                correct_marginals=marginals[x].tolist(),true_joint_hard_risk=float(posterior[:,x]@c),
                independent_hard_risk=float(indep[x]@c),soft_edge_count_squared_error=float((marginals[x].sum()-1)**2),
                input_independent_deterministic_good_graph_cost=0.))
    save('two_slot_uncertainty',tworows);save('summary',summary);save('per_state',contexts);save('checks',checks)
    for p,h in BIND.items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
    save('bindings',BIND)
    save('receipt',dict(status='complete_stopped',seconds=time.monotonic()-START,RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        tests=len(checks),tests_passed=all(c['passed'] for c in checks),finite_contexts=len(contexts),GPU_hours=0,
        model_calls=0,optimizer_updates=0,new_generated_graphs=0,CPU_threads=1,original_assets_unchanged=True))
    print(json.dumps(summary))
if __name__=='__main__':
    try:main()
    except BaseException as e:
        if OUT.exists():save('failure',dict(error=repr(e),seconds=time.monotonic()-START))
        raise
