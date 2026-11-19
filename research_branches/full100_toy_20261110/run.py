"""100-step finite typed diffusion: matched toy training and exact/sampled chains.

No production denoiser, GPU, graph editing, count constraint, or new loss.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import sys,time,json,csv,hashlib,signal,resource,traceback,importlib.util,itertools
from pathlib import Path
import numpy as np
from scipy.special import expit
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/full100_toy_20261110'
sys.path.insert(0,str(ROOT/'research_branches/hard_clean_zero_safe_20261110'))
from combine import project_mean
SEEDS=[2026110901,2026110902,2026110903];T=100;K=8;STEPS=20;LR=.2;SAMPLES=1024
BIND={}
def bind(p):
    p=Path(p);BIND[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest();return p
def load(name,p):
    spec=importlib.util.spec_from_file_location(name,bind(p));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def dump(n,v):(OUT/n).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False))
def table(n,rr):
    if not rr:return
    with (OUT/n).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)

def setup():
    m=load('finite100',ROOT/'research_branches/terminal_structure_audit_20261109/audit_cpu.py')
    m.kernel_module=load('production100_kernel',m.KERNEL)
    cfg=json.loads(bind(m.PROTOCOL).read_text());old=np.array(cfg['noise']['cumulative_retention'])
    # Same piecewise-linear cumulative-retention curve in normalized time;
    # retrain toy arms on this 100-time grid, do not reuse three-time trained arms.
    cfg['noise']['cumulative_retention']=np.interp(np.linspace(0,1,T+1),np.linspace(0,1,len(old)),old).tolist()
    f=m.FiniteDiffusion(cfg)
    tri=np.zeros((f.S,2))
    for s,A in enumerate(f.adj):
        for nodes in itertools.combinations(range(4),3):
            if all(A[i,j] for i,j in itertools.combinations(nodes,2)):tri[s,int(f.types[list(nodes)].sum())-1]+=1
    C=np.column_stack([f.blocks[:,:2],(tri-tri[f.refid])**2,f.blocks[:,3]])
    return f,C,tri,cfg,m.kernel_module

def atomic(f,C,theta,times,states):
    features=np.stack([f.features[int(t)][int(x)] for t,x in zip(times,states)])
    p=expit(features@theta)
    cp=np.exp(np.einsum('se,ie->is',f.B,np.log(p))+np.einsum('se,ie->is',1-f.B,np.log1p(-p)))
    costs=cp@C
    dz=np.einsum('is,sk,ise->kie',cp,C,f.B[None,:,:]-p[:,None,:])
    G=np.einsum('kie,ied->kd',dz,features)/len(times)
    ce=float(-np.mean(f.ref*np.log(p)+(1-f.ref)*np.log1p(-p)))
    gc=np.einsum('ie,ied->d',p-f.ref,features)/(len(times)*f.M)
    return costs.mean(0),G,ce,gc

def transitions(f,theta,t):
    p=expit(f.features[t]@theta)
    q=f.k[t][0]+p*(f.k[t][1]-f.k[t][0])
    if not np.isfinite(q).all() or (q<=0).any() or (q>=1).any():raise FloatingPointError('invalid learned reverse probability')
    logP=np.einsum('je,ie->ij',f.B,np.log(q))+np.einsum('je,ie->ij',1-f.B,np.log1p(-q))
    P=np.exp(logP)
    clean=np.exp(np.einsum('je,ie->ij',f.B,np.log(p))+np.einsum('je,ie->ij',1-f.B,np.log1p(-p)))
    np.testing.assert_allclose(P.sum(1),1.,atol=1e-12,rtol=1e-12)
    return P,clean,p

def metrics(f,C,tri,dist):
    mass=dist.sum();assert abs(mass-1)<1e-10 and dist.min()>-1e-14
    orbit=np.bincount(f.orbits,weights=dist);positive=orbit>0
    row={name:float(dist@C[:,j]) for j,name in enumerate(['relation_count','joint_degree','triangle001','triangle011','component_cost'])}
    row.update(mean_five_cost=float(dist@C.mean(1)),edges=float(dist@f.B.sum(1)),triangles=float(dist@f.tri),
        triangle_MAE=float(dist@abs(f.tri-f.tri[f.refid])),connected=float(dist@(f.components==1)),
        components=float(dist@f.components),triangle001_mean=float(dist@tri[:,0]),triangle011_mean=float(dist@tri[:,1]),
        isomorph_entropy=float(-np.sum(orbit[positive]*np.log(orbit[positive]))),isomorph_collision=float(orbit@orbit),
        reference_isomorph_probability=float(orbit[f.orbits[f.refid]]))
    return row

def gates(f,C,theta,kernel):
    errors=[]
    for t in range(1,T+1):
        # Independent scalar Bayes calculation for both clean hypotheses.
        a=f.alpha[t-1];b=f.alpha[t];r=b/a
        for clean in [0.,1.]:
            for observed in [0,1]:
                for fam,pi in enumerate(f.pi):
                    prior1=a*clean+(1-a)*pi
                    lik1=r*observed+(1-r)*(pi if observed else 1-pi)
                    lik0=r*(1-observed)+(1-r)*(pi if observed else 1-pi)
                    den=prior1*lik1+(1-prior1)*lik0
                    answer=prior1*lik1/den
                    got=kernel.MarginalKernel(f.pi,f.alpha).reverse(np.array([clean]),np.array([observed]),np.array([fam]),t-1,t)[0]
                    errors.append(abs(answer-got))
    assert max(errors)<1e-12
    # Exact forward/true-posterior identity for every time; true X0 oracle
    # is a correctness test, NOT an available learned baseline at generation.
    dist=f.prior.copy();stationary=f.prior.copy();max_forward=0.
    for t in range(T,0,-1):
        q=f.k[t][0]+f.ref*(f.k[t][1]-f.k[t][0])
        P=np.prod(np.where(f.B[None,:,:]>0,q[:,None,:],1-q[:,None,:]),axis=2)
        dist=dist@P
        expected=np.eye(f.S)[f.refid] if t==1 else f.qref[t-1]
        max_forward=max(max_forward,float(abs(dist-expected).max()))
        # Stationary-prior Bayes predictor depends on Xt; a constant p0=pi
        # is NOT generally that posterior for intermediate noise times.
        prior_p0=f.alpha[t]*f.B+(1-f.alpha[t])*f.pi[f.f]
        q0=f.k[t][0]+prior_p0*(f.k[t][1]-f.k[t][0])
        P0=np.prod(np.where(f.B[None,:,:]>0,q0[:,None,:],1-q0[:,None,:]),axis=2)
        stationary=stationary@P0
    assert max_forward<1e-12 and abs(stationary-f.prior).max()<1e-12
    # t=1 reverse probability equals predicted clean probability.
    p=expit(f.features[1]@theta);q=f.k[1][0]+p*(f.k[1][1]-f.k[1][0]);assert abs(q-p).max()<1e-12
    fixed=kernel.MarginalKernel([0,1],f.alpha)
    np.testing.assert_array_equal(fixed.reverse(np.array([0.,1.]),np.array([0,1]),np.array([0,1]),0,1),[0,1])
    try:fixed.reverse(np.array([.1,1.]),np.array([0,1]),np.array([0,1]),0,1)
    except ValueError:pass
    else:raise AssertionError('fixed support violation accepted')
    ts=np.array([1,25,75,100]);xs=np.array([0,17,36,63]);_,G,_,gc=atomic(f,C,theta,ts,xs);max_fd=0.
    for j in range(len(theta)):
        step=np.eye(len(theta))[j]*1e-5
        a,_,ca,_=atomic(f,C,theta+step,ts,xs);b,_,cb,_=atomic(f,C,theta-step,ts,xs)
        max_fd=max(max_fd,float(abs((a-b)/2e-5-G[:,j]).max()),abs((ca-cb)/2e-5-gc[j]))
    assert max_fd<1e-8
    return dict(passed=True,scalar_Bayes_checks=len(errors),max_Bayes_error=max(errors),max_forward_oracle_error=max_forward,
        prior_stationarity_error=float(abs(stationary-f.prior).max()),gradient_finite_difference_error=max_fd,
        t1_and_fixed_support=True,oracle_uses_reference_only_for_test=True)

def evaluate(f,C,tri,theta,meta,uniforms,rows,paths,distributions):
    dist=f.prior.copy();states=np.searchsorted(np.cumsum(dist),uniforms[0],side='right');history=[states.copy()]
    grid={100,75,50,25,10,5,1};one_shot=None;step_metrics=[]
    for t in range(T,0,-1):
        P,clean,p=transitions(f,theta,t)
        if t==T:one_shot=metrics(f,C,tri,dist@clean)
        if t in grid:
            clean_on=metrics(f,C,tri,dist@clean);clean_forward=metrics(f,C,tri,f.qref[t]@clean)
            row=dict(**meta,t=t,occupancy_TV_to_forward=float(abs(dist-f.qref[t]).sum()/2),
                **{'state_'+k:v for k,v in metrics(f,C,tri,dist).items()},
                **{'clean_onpolicy_'+k:v for k,v in clean_on.items()},
                **{'clean_forward_'+k:v for k,v in clean_forward.items()})
            rows.append(row)
        probs=np.cumsum(P[states],axis=1);probs[:,-1]=1.
        states=(uniforms[T-t+1,:,None]>=probs).sum(1);history.append(states.copy())
        dist=dist@P
    terminal=metrics(f,C,tri,dist);emp=np.bincount(states,minlength=f.S)/len(states);sample=metrics(f,C,tri,emp)
    rows.append(dict(**meta,t=0,occupancy_TV_to_forward=float(1-dist[f.refid]),
        **{'state_'+k:v for k,v in terminal.items()},**{'clean_onpolicy_'+k:v for k,v in terminal.items()},**{'clean_forward_'+k:v for k,v in terminal.items()}))
    # At t=0 no new clean prediction is evaluated. Fields above mirror terminal
    # only for rectangular CSV; this explicit flag prevents qref-validation claims.
    rows[-1]['terminal_placeholder']=True
    for r in rows:
        if 'terminal_placeholder' not in r:r['terminal_placeholder']=False
    paths.append(np.array(history,dtype=np.uint8));distributions.append(dist)
    return dict(**meta,one_shot_t100=one_shot,terminal=terminal,sampled_terminal=sample,
        sampling_TV=float(abs(emp-dist).sum()/2),terminal_parameter_count=len(theta))

def main():
    start=time.time();OUT.mkdir(exist_ok=False)
    def stop(*_):raise TimeoutError('bounded CPU stop')
    signal.signal(signal.SIGALRM,stop);signal.signal(signal.SIGTERM,stop);signal.alarm(180)
    receipt=dict(start_epoch=start,GPU_hours=0,production_model_calls=0,production_updates=0,toy_updates=0)
    curves=[];updates=[];finals=[];paths=[];distributions=[];theta_bank=[];index=[]
    try:
        bind(__file__);bind(ROOT/'research_branches/hard_clean_zero_safe_20261110/combine.py')
        f,C,tri,cfg,kernel=setup()
        previous=ROOT/'results/one_step_to_terminal_cpu_20261110'
        saved=np.load(bind(previous/'states_and_distributions.npz'));oldindex=json.loads(bind(previous/'distribution_index.json').read_text())
        initials={seed:saved['parameters'][next(i for i,x in enumerate(oldindex) if x['seed']==seed and x['step']==0 and x['arm']=='exact_forward_clean_structure')].copy() for seed in SEEDS}
        banks={};randoms={}
        for seed in SEEDS:
            rng=np.random.default_rng(np.random.SeedSequence([2026111020,seed]));ts=rng.integers(1,T+1,size=(STEPS,K));u=rng.random((STEPS,K,f.M));xs=np.empty((STEPS,K),dtype=int)
            for j in range(STEPS):
                for k,t in enumerate(ts[j]):
                    p=f.alpha[t]*f.ref+(1-f.alpha[t])*f.pi[f.f];bits=(u[j,k]<p).astype(float)
                    xs[j,k]=np.flatnonzero((f.B==bits).all(1))[0]
            banks[str(seed)+'_t']=ts;banks[str(seed)+'_x']=xs
            randoms[str(seed)]=np.random.default_rng(np.random.SeedSequence([2026111021,seed])).random((T+1,SAMPLES))
        np.savez_compressed(OUT/'input_banks.npz',**banks);np.savez_compressed(OUT/'sampling_uniforms.npz',**randoms)
        dump('protocol.json',dict(graph=cfg['graph'],noise=cfg['noise'],T=T,training_updates=STEPS,inputs_per_update=K,learning_rate=LR,
            optimizer='plain SGD matching prior toy LR; no EMA or AdamW; cannot validate production optimizer',
            arms=['reference_CE','hard_five_project_mean'],initializations=SEEDS,sampling_paths=SAMPLES,
            objective='same five hard graph costs, exact 64-output expectation; input IID time/Xt minibatch K8; no CE in structure arm',
            grid_rationale='piecewise-linear interpolation of old toy retention vs normalized time; all new arms trained on same 100-time grid',
            source_of_initials='existing untrained 9-parameter toy states only; do not reuse 3-time trained parameters',
            evaluation='full 100 reverse transitions, each uses current generated state; exact 64-state distribution and 1024 paired sampled paths',
            no_count_constraint=True,no_reference_at_inference=True,validation_grid=[100,75,50,25,10,5,1,0],
            limits=dict(CPU_threads=1,wall_seconds=240,worker_seconds=180,RSS_GiB=2,GPU=0),bindings=BIND))
        dump('gates.json',gates(f,C,initials[SEEDS[0]],kernel))
        for seed in SEEDS:
            ts=banks[str(seed)+'_t'];xs=banks[str(seed)+'_x'];u=randoms[str(seed)]
            theta=initials[seed];meta=dict(seed=seed,arm='initial',updates=0)
            finals.append(evaluate(f,C,tri,theta,meta,u,curves,paths,distributions));theta_bank.append(theta.copy());index.append(meta)
            for arm in ['reference_CE','hard_five_project_mean']:
                theta=initials[seed].copy()
                for step in range(STEPS):
                    cost,G,ce,gce=atomic(f,C,theta,ts[step],xs[step])
                    if arm=='reference_CE':g=gce;coeff=[];res=0.
                    else:
                        coeff,solver=project_mean(G@G.T);g=G.T@coeff;coeff=coeff.tolist();res=solver['residual']
                    if not np.isfinite(g).all():raise FloatingPointError('nonfinite gradient')
                    nxt=theta-LR*g;after,_,ce_after,_=atomic(f,C,nxt,ts[step],xs[step])
                    updates.append(dict(seed=seed,arm=arm,update=step+1,gradient_norm=float(np.linalg.norm(g)),
                        displacement=float(np.linalg.norm(nxt-theta)),coefficients=coeff,KKT=res,zero_direction=bool(np.linalg.norm(g)<=1e-12),
                        before_costs=cost.tolist(),after_costs=after.tolist(),before_CE=ce,after_CE=ce_after,
                        times=ts[step].tolist(),states=xs[step].tolist()))
                    theta=nxt;receipt['toy_updates']+=1
                meta=dict(seed=seed,arm=arm,updates=STEPS)
                finals.append(evaluate(f,C,tri,theta,meta,u,curves,paths,distributions));theta_bank.append(theta.copy());index.append(meta)
                dump('progress.json',dict(completed_models=len(finals),seconds=time.time()-start));print(json.dumps(dict(seed=seed,arm=arm,seconds=time.time()-start)),flush=True)
        for p,h in BIND.items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
        receipt.update(status='complete',bindings_unchanged=len(BIND),full_toy_chains=len(paths)*SAMPLES,exact_terminal_distributions=len(distributions))
    except BaseException as exc:receipt.update(status='failed_or_stopped',error=repr(exc),traceback=traceback.format_exc())
    finally:
        dump('finals.json',finals);dump('updates.json',updates);table('trajectory_metrics.csv',curves);dump('state_index.json',index)
        if paths:np.savez_compressed(OUT/'states_and_paths.npz',parameters=np.array(theta_bank),distributions=np.array(distributions),paths=np.array(paths))
        receipt.update(end_epoch=time.time(),seconds=time.time()-start,RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
        dump('receipt.json',receipt);print(json.dumps(receipt),flush=True)
    return 0 if receipt['status']=='complete' else 1
if __name__=='__main__':raise SystemExit(main())
