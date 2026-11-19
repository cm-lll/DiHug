"""One-step clean structural loss: input vs hard-output sampling allocation."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for n in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[n]='1'
import importlib.util,json,hashlib,time,signal,resource,traceback
from pathlib import Path
import numpy as np
from scipy.special import expit
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/single_step_budget_20261111'
sp=importlib.util.spec_from_file_location('typed',ROOT/'research_branches/toy_typed_endpoint_20261110/run.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);r=m.r
ALLOC=[(1,64),(4,16),(16,4),(1,256),(4,64),(16,16),(64,4)]
def full(f,C,w):
    F=np.stack([f.features[t] for t in range(1,101)]);prob=expit(F@w)
    law=np.prod(np.where(f.B[None,None,:,:]>0,prob[:,:,None,:],1-prob[:,:,None,:]),axis=-1)
    dz=np.einsum('txy,yc,txye->txce',law,C,f.B[None,None,:,:]-prob[:,:,None,:]);G=np.einsum('txce,txed->txcd',dz,F)
    q=np.stack([f.qref[t] for t in range(1,101)])/100
    mean=np.einsum('tx,txcd->cd',q,G);cost=np.einsum('tx,txy,yc->c',q,law,C)
    var=float(np.einsum('tx,txcd->',q,(G-mean)**2))
    return mean,cost,G,var
def main():
    start=time.time();OUT.mkdir(exist_ok=False);r.OUT=OUT;signal.alarm(180);resource.setrlimit(resource.RLIMIT_AS,(2*1024**3,2*1024**3))
    rows=[];truth=[];arrays={};tests=[];receipt=dict(status='running',GPU_hours=0,new_full_chains=0,updates=0,production_calls=0,unique_scored_hard_graphs=0)
    try:
        f,C,*_=r.setup();f=m.augment(f);src=ROOT/'results/toy_typed_endpoint_20261110';hist=np.load(src/'histories.npz')
        for p in [Path(__file__),Path(__file__).with_name('audit.py'),src/'histories.npz',src/'protocol.json',src/'gates.json',ROOT/'research_branches/toy_typed_endpoint_20261110/run.py',ROOT/'research_branches/full100_toy_20261110/run.py']:r.bind(p)
        assert json.loads((src/'gates.json').read_text())['passed']
        protocol=dict(track='CORE single-step hard-clean gradient cost diagnostic',seeds=r.SEEDS,updates=[0,100],arm='typed_endpoint raw, previously trained SINGLE-STEP objective; not terminal-trained weights',conditions=6,repeats=128,allocations=ALLOC,objective='E_t uniform1..100 E_Xt~q(.|reference) E_Y~product Bernoulli p_theta(clean|Xt,t) original five hard costs; no CE, reverse step or path value',estimator='per-input independent output LOO score; average B inputs; score=(Y-p) times current feature Jacobian',random='SeedSequence [2026111121,seed,update,repeat] input; [2026111122,seed,update,repeat,input_index] hard; shared prefixes',decomposition='exact per-input gradient average vs population gradient isolates input noise; sampled gradient minus exact per-input average isolates output noise; finite-sample cross term retained',cost='B network predictions, B*K hard scores; equal output counts NOT equal GPU compute; toy cache speed not production throughput',decision='No winner or formal training gate from gradient accuracy alone. Report tradeoffs for every saved state and both budgets. No additional allocations based on outcomes.',limits=dict(worker_seconds=180,wall_minutes=20,threads=1,memory_GiB=2,GPU=0),bindings=r.BIND)
        r.dump('protocol.json',protocol);ph=hashlib.sha256((OUT/'protocol.json').read_bytes()).hexdigest();r.dump('freeze.json',dict(epoch=time.time(),sha256=ph))
        powers=2**np.arange(5,-1,-1);cdfs={t:np.cumsum(f.qref[t]) for t in range(1,101)}
        for seed in r.SEEDS:
            for step in (0,100):
                key=f'{seed}_{step}';w=hist[f'{seed}_typed_endpoint_parameters'][step];G,cost,gcond,var=full(f,C,w);norm=np.linalg.norm(G);gn=np.linalg.norm(G,axis=1)
                truth.append(dict(seed=seed,update=step,gradient=G.tolist(),cost=cost.tolist(),input_gradient_variance=var,gradient_norm=float(norm)))
                h=1e-5;d=np.arange(1,11,dtype=float);d/=np.linalg.norm(d);fd=(full(f,C,w+h*d)[1]-full(f,C,w-h*d)[1])/(2*h);err=float(abs(fd-G@d).max());assert err<1e-8;tests.append(dict(seed=seed,update=step,finite_difference_error=err))
                alltx=[];ally=[]
                for rep in range(128):
                    irng=np.random.default_rng(np.random.SeedSequence([2026111121,seed,step,rep]));ts=irng.integers(1,101,64);u=irng.random(64);xs=np.array([np.searchsorted(cdfs[int(t)],v) for t,v in zip(ts,u)])
                    F=np.stack([f.features[int(t)][int(x)] for t,x in zip(ts,xs)]);p=expit(F@w);alltx.append(np.stack([ts,xs],axis=1));ys=np.full((64,256),255,dtype=np.uint8);conditional=gcond[ts-1,xs];grad={}
                    for i in range(64):
                        nk=max(K for B,K in ALLOC if i<B);hrng=np.random.default_rng(np.random.SeedSequence([2026111122,seed,step,rep,i]));bits=hrng.random((nk,6))<p[i];y=(bits@powers).astype(int);ys[i,:nk]=y;receipt['unique_scored_hard_graphs']+=nk
                        score=(bits-p[i])@F[i];c=C[y]
                        for K in sorted(set(K for B,K in ALLOC if i<B)):
                            centered=c[:K]-c[0];adv=(K*centered-centered.sum(0))/(K-1);grad[i,K]=adv.T@score[:K]/K
                    ally.append(ys)
                    for B,K in ALLOC:
                        estimate=np.mean([grad[i,K] for i in range(B)],axis=0);oracle=conditional[:B].mean(0);ie=oracle-G;oe=estimate-oracle;total=estimate-G;mean=estimate.mean(0);direction=-mean/np.linalg.norm(mean) if np.linalg.norm(mean)>0 else np.zeros(10);slopes=G@direction
                        row=dict(seed=seed,update=step,repeat=rep,B=B,K=K,hard_budget=B*K,gradient=estimate.tolist(),conditional_gradient=oracle.tolist(),relative_MSE=float(np.sum(total**2)/norm**2),input_relative_MSE=float(np.sum(ie**2)/norm**2),output_relative_MSE=float(np.sum(oe**2)/norm**2),relative_cross=float(2*np.sum(ie*oe)/norm**2),mean_direction_true_slopes=slopes.tolist(),mean_objective_descent=bool(slopes.mean()<-1e-10),any_cost_harm=bool(max(slopes)>1e-10),relative_row_errors=(np.linalg.norm(total,axis=1)/gn).tolist(),mean_gradient_cos=float(mean@G.mean(0)/(np.linalg.norm(mean)*np.linalg.norm(G.mean(0)))) if np.linalg.norm(mean)>0 else None)
                        assert abs(row['relative_MSE']-row['input_relative_MSE']-row['output_relative_MSE']-row['relative_cross'])<1e-10;rows.append(row)
                arrays[key+'_inputs']=np.array(alltx);arrays[key+'_outputs']=np.array(ally);r.dump('progress.json',dict(conditions=len(truth),seconds=time.time()-start))
        np.savez_compressed(OUT/'evidence.npz',**arrays)
        for name,h in r.BIND.items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==h
        assert hashlib.sha256((OUT/'protocol.json').read_bytes()).hexdigest()==ph
        receipt.update(status='complete',conditions=6,estimates=len(rows),bindings_unchanged=True)
    except BaseException as e:receipt.update(status='failed',error=repr(e),traceback=traceback.format_exc())
    finally:
        r.dump('estimates.json',rows);r.dump('truth.json',truth);r.dump('tests.json',tests);receipt.update(seconds=time.time()-start,RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024);r.dump('receipt.json',receipt);print(json.dumps(receipt))
    return 0 if receipt['status']=='complete' else 1
if __name__=='__main__':raise SystemExit(main())
