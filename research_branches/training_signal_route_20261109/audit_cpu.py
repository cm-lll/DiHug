"""Exact archived-universe audit of clean-target training versus posterior labels.

No random draws, models, autograd, optimizer, teacher chain or new graph files.
"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import csv,hashlib,json,resource,signal,time
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from scipy.special import expit
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/training_signal_route_20261109'
START=time.monotonic();HASHES={}
def bind(p):
    p=ROOT/p;HASHES[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest();return p
def save(n,o):
    (OUT/(n+'.json')).write_text(json.dumps(o,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    if isinstance(o,list) and o:
        with (OUT/(n+'.csv')).open('w',newline='') as f:
            w=csv.DictWriter(f,list(o[0]));w.writeheader();w.writerows(o)
def main():
    OUT.mkdir(parents=True,exist_ok=False);signal.alarm(120);resource.setrlimit(resource.RLIMIT_AS,(1024**3,1024**3))
    save('protocol',dict(start_utc=datetime.now(timezone.utc).isoformat(),
        work='Exact finite sums using saved graph universe and saved targets; no model or synthetic graph sampling',
        targets=['bounded_expected_error','tail_constrained'],retentions=[0.,.5,.95,1.],
        noise='relation refresh q(x_e=1|g_e)=alpha*g_e+(1-alpha)*pi_r; pi from same reference, fixed 0/1 support',
        logits='two predetermined three-coefficient arithmetic test functions of observations; no trained network',
        checks=['risk equality','analytic expected gradient equality','finite differences of direct joint risk','conditional label variance','reference-only input distribution mismatch'],
        tolerance=1e-10,CPU_seconds=120,CPU_threads=1,memory_GiB=1,GPU=0,
        not_done=['real tolerance choice','target fitting','new graphs','teacher chain','model call','training','generation']))
    with np.load(bind('results/structure_target_tail_20261109/targets.npz')) as z:
        codes=z['codes'].copy();targets={k:z[k].copy() for k in ['bounded_expected_error','tail_constrained']}
    with np.load(bind('results/structure_without_edge_ce_theory_20261109/finite_universe.npz')) as z:
        B=z['bits'][codes].astype(float);fam=z['families'].copy();ref=z['reference'].astype(float)
    old=json.loads(bind('results/structure_target_constraints_20261109/completed/binding.json').read_text())
    for p,h in old['assets'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
    R=np.unique(fam);pi=np.array([ref[fam==r].mean() for r in R]);pe=pi[fam];X=B.copy();E=B.shape[1]
    refidx=int(np.flatnonzero(np.all(B==ref,axis=1))[0]);assert len(B)==512
    # Fixed analytic predictors: only evaluate scalar arithmetic for expectation tests.
    feat=np.stack([np.ones_like(X),X, np.broadcast_to((X.mean(1)-.5)[:,None],X.shape)],axis=2)
    thetas=[np.array([-1.1,.6,.2]),np.array([.35,-.4,.7])]
    rows=[];details=[];checks=[]
    for alpha in [0.,.5,.95,1.]:
        prob=alpha*B+(1-alpha)*pe
        Q=np.prod(np.where(X[None,:,:]>0,prob[:,None,:],1-prob[:,None,:]),axis=2)
        assert abs(Q.sum(1)-1).max()<1e-12
        # Independent refresh calculation as probability of equality/non-equality.
        alt=np.ones_like(Q)
        for e in range(E):
            alt*=alpha*(B[:,e,None]==X[None,:,e])+(1-alpha)*np.where(X[None,:,e]>0,pe[e],1-pe[e])
        assert abs(Q-alt).max()<1e-12
        for name,rho in targets.items():
            assert abs(rho.sum()-1)<1e-12 and (rho>0).all()
            J=rho[:,None]*Q;px=J.sum(0);assert (px>0).all()
            m=(J.T@B)/px[:,None]
            conditional_var=np.maximum(0,m*(1-m))
            assert np.max(abs(px@m-rho@B))<1e-12
            qref=Q[refidx];TV=float(abs(px-qref).sum()/2)
            gap=float(np.sum(qref[:,None]*abs(m-ref))/E)
            relation_var=[]
            for r in R:
                ids=fam==r;vv=float(px@conditional_var[:,ids].mean(1));relation_var.append(vv)
                details.append(dict(target=name,retention=alpha,relation=int(r),slots=int(ids.sum()),
                    expected_label_variance=vv,mean_clean_edge_probability=float((rho@B)[ids].mean()),
                    reference_only_label_vs_target_posterior_MAE=float(qref@abs(m[:,ids]-ref[ids]).mean(1))))
            for si,theta in enumerate(thetas):
                z=feat@theta;p=expit(z)
                # Direct clean-graph expectation: loop reduction order G -> X, no soft label used.
                def hard_risk(th):
                    zz=feat@th;ln=np.logaddexp(0,zz)
                    return float(rho@(Q@ln.sum(1)-np.sum(B*(Q@zz),axis=1))/E)
                hard=hard_risk(theta);soft=float(px@np.sum(np.logaddexp(0,z)-m*z,axis=1)/E)
                gh=np.array([rho@(Q@np.sum(p*feat[:,:,k],axis=1)-np.sum(B*(Q@feat[:,:,k]),axis=1))/E for k in range(3)])
                gs=np.sum(px[:,None,None]*(p-m)[:,:,None]*feat,axis=(0,1))/E
                eps=1e-5;fd=np.array([(hard_risk(theta+np.eye(3)[k]*eps)-hard_risk(theta-np.eye(3)[k]*eps))/(2*eps) for k in range(3)])
                err=float(abs(gh-gs).max());errfd=float(abs(fd-gh).max())
                assert abs(hard-soft)<1e-10 and err<1e-10 and errfd<1e-8
                # Per-input expected logit gradient covariance trace is sum of Bernoulli variances;
                # this is NOT parameter-gradient variance or long-run MCMC variance.
                checks.append(dict(target=name,retention=alpha,test_function=si,joint_risk=hard,posterior_label_risk=soft,
                    risk_error=abs(hard-soft),gradient_error=err,finite_difference_error=errfd,
                    hard_gradient=gh.tolist(),soft_gradient=gs.tolist()))
            rows.append(dict(target=name,retention=alpha,clean_states=len(B),observations=len(X),
                forward_input_TV_reference_only=TV,reference_hard_label_vs_target_posterior_MAE=gap,
                mean_conditional_logit_gradient_variance=float(px@conditional_var.mean(1)),
                mean_target_edges=float(rho@B.sum(1)),
                boundary='variance of hard clean labels conditional on Xt; exact posterior soft labels remove this part only'))
    save('checks',checks);save('summary',rows);save('relations',details)
    cost=json.loads(bind('results/component_target_cpu_20261109/verified/completion.json').read_text())
    sec=(cost['timing_seconds']['component']+cost['timing_seconds']['old_terms'])/cost['flips']
    save('cost_accounting',dict(measured_single_independent_score_seconds=sec,
        posterior_label_illustration=dict(training_inputs=3000,coordinates=128,backgrounds=100,score_calls=38400000,CPU_hours=38400000*sec/3600),
        clean_library_route=dict(online_structure_score_calls=0,clean_library_generation_seconds=None,chain_mixing=None,required_library_size=None,
            warning='Not zero total cost. Clean target sampling, sequential scoring, diversity and finite library bias remain unmeasured.'),
        break_even_scoring_calls=38400000,
        not_promised='No training speedup factor until target-library cost and statistical quality are measured'))
    save('next_protocol',dict(status='design_only_not_training_ready',
        route='Generate a reference-side structural target library offline; during training choose a clean state, corrupt with original q, use that state as clean labels.',
        exact_condition='The sampled clean-state law must be rho. IID samples give unbiased gradients; stationary correlated samples preserve marginal expectation but affect effective sample size and optimizer trajectory.',
        library_condition='Uniform sampling from a finite saved library trains its empirical law, not exactly rho; repeated corruptions and node permutations do not add independent structural examples.',
        target_tolerances=None,adopted_teacher=None,
        preserve=['diffusion model family','legal sparse support','original q/reverse interface'],
        not_equivalent=['using edited labels on original X0-noise only','using a single favorable edit as a posterior marginal','deploy-time reference-driven edit policy'],
        next_required=['freeze reference-side acceptable structure contract','bound cost and mixing of offline clean-state sampler','verify diverse nonisomorphic support and all structure costs before any model training'],
        resource_proposal=dict(preparation='CPU-only source/protocol work',offline_pilot='Separate bounded proposal after target freeze; no GPU needed',formal_training='not authorized by this audit')))
    for p,h in HASHES.items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
    save('binding',dict(source=str(Path(__file__)),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),assets=HASHES))
    save('receipt',dict(status='complete_stopped',seconds=time.monotonic()-START,RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        checks=len(checks),posterior_conditions=len(rows),model_calls=0,new_random_draws=0,new_graphs=0,training_updates=0,GPU_hours=0,
        max_risk_error=max(r['risk_error'] for r in checks),max_gradient_error=max(r['gradient_error'] for r in checks),
        max_finite_difference_error=max(r['finite_difference_error'] for r in checks),assets_unchanged=True,formal_training_authorized=False))
    print(json.dumps(dict(receipt=json.loads((OUT/'receipt.json').read_text()),summary=rows),ensure_ascii=False))
if __name__=='__main__':
    try:main()
    except BaseException as e:
        if OUT.exists():save('failure',dict(error=repr(e),seconds=time.monotonic()-START))
        raise
