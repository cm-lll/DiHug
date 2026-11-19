"""Frozen finite-state LOO gradient accuracy; no parameter updates or chains."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import sys,time,json,hashlib,resource,signal,traceback,importlib.util
from pathlib import Path
import numpy as np
from scipy.special import expit
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/finite_feedback_precision_20261110';OLD=ROOT/'results/full100_direction_long_20261110'
spec=importlib.util.spec_from_file_location('precision_toy',ROOT/'research_branches/full100_toy_20261110/run.py')
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r);r.OUT=OUT
from combine import loo_gradient
KS=[4,16,64];REPS=64;TIMES=[5,50,95];BATCH=8
def main():
    start=time.time();OUT.mkdir(exist_ok=False)
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('300s bound')));signal.alarm(300)
    receipt=dict(status='running',start_epoch=start,GPU_hours=0,parameter_updates=0,production_calls=0,reverse_chains=0)
    exact=[];estimates=[];directions=[];idsall=[];features=[];probs=[];metadata=[];timings=[];rows=[]
    try:
        for p in [Path(__file__),Path(__file__).with_name('audit.py'),Path(__file__).with_name('watch.py'),
            ROOT/'research_branches/full100_toy_20261110/run.py',ROOT/'research_branches/hard_clean_zero_safe_20261110/combine.py',
            *[OLD/n for n in ('training_trajectories.npz','protocol.json','independent_audit.json')]]:r.bind(p)
        assert json.loads((OLD/'independent_audit.json').read_text())['passed']
        f,C,tri,cfg,kernel=r.setup();hist=np.load(OLD/'training_trajectories.npz');inputs={}
        for t in TIMES:
            rng=np.random.default_rng(np.random.SeedSequence([2026111053,t]));p=f.alpha[t]*f.ref+(1-f.alpha[t])*f.pi[f.f]
            bits=(rng.random((BATCH,f.M))<p).astype(float);xs=np.array([np.flatnonzero((f.B==b).all(1))[0] for b in bits]);inputs[str(t)]=xs.tolist()
        r.dump('inputs.json',dict(stream=2026111053,times=TIMES,states=inputs,reference_id=int(f.refid),reused_across_models=True));r.bind(OUT/'inputs.json')
        protocol=dict(initializations=r.SEEDS,arms=['common','mean'],updates=[20,50,100],readout='raw',times=TIMES,inputs_per_time=BATCH,
            hard_graph_samples_per_input=KS,independent_replicates=REPS,hard_stream=2026111054,condition_count=54,
            estimator='per-input unbiased leave-one-out score estimator, average 8 inputs; same hard samples used by all five costs',
            reuse='nested first4/16/64 of same64 samples per input and repeat; repeats independent, K arms paired',
            directions='mean gradient and project_mean estimated Gram; evaluate true exact first-order change at each frozen state; no actual update or AdamW',
            scope='4 nodes/6 slots/64 outputs/9 parameters; finite arithmetic toy; fixed noisy input batches, not uncertainty over new training seeds',
            no_online_selection=True,zero_threshold=1e-14,first_order_harm_tolerance=1e-12,
            costs='original five hard risks; table lookup used for timing, not representative of full real graph scoring',
            budget=dict(CPU_threads=1,RSS_GiB=2,worker_seconds=300,watch_seconds=360,GPU=0),bindings=r.BIND)
        r.dump('protocol.json',protocol);ph=hashlib.sha256((OUT/'protocol.json').read_bytes()).hexdigest();r.dump('freeze.json',dict(epoch=time.time(),protocol_SHA256=ph))
        # Explicit LOO tests: constant cost has exact zero feedback; per-input conventional baseline is identical.
        test_rng=np.random.default_rng(2026111055);err=0.
        for k in KS:
            p=test_rng.random(6);bits=(test_rng.random((k,6))<p).astype(float);c=test_rng.normal(size=(k,5))
            traditional=(c-(c.sum(0)-c)/(k-1)).T@(bits-p)/k
            err=max(err,float(abs(traditional-loo_gradient(p,bits,c)).max()))
            assert np.array_equal(loo_gradient(p,bits,np.ones((k,5))),np.zeros((5,6)))
        assert err<1e-12;r.dump('tests.json',dict(passed=True,three_Ks=True,constant_zero=True,max_algebra_error=err))
        weights=2**np.arange(f.M-1,-1,-1)
        for seed in r.SEEDS:
            for arm in ('common','mean'):
                for step in (20,50,100):
                    theta=hist[f'{seed}_{arm}_parameters'][step-20]
                    for t in TIMES:
                        ci=len(metadata);xs=np.array(inputs[str(t)]);ts=np.full(BATCH,t);F=np.stack([f.features[t][x] for x in xs]);p=expit(F@theta)
                        tick=time.perf_counter();risk,G,ce,_=r.atomic(f,C,theta,ts,xs);exact_s=time.perf_counter()-tick
                        co,_=r.project_mean(G@G.T);true_dirs={'mean':G.mean(0),'common':G.T@co}
                        rng=np.random.default_rng(np.random.SeedSequence([2026111054,seed,0 if arm=='common' else 1,step,t]))
                        tick=time.perf_counter();bits=(rng.random((REPS,BATCH,64,f.M))<p[None,:,None,:]).astype(np.uint8)
                        ids=np.tensordot(bits,weights,axes=([-1],[0])).astype(np.uint8);sampling_s=time.perf_counter()-tick
                        current=[];current_dirs=[]
                        for k in KS:
                            tick=time.perf_counter();cost=C[ids[:,:,:k]];center=cost-cost[:,:,:1];adv=(k*center-center.sum(2,keepdims=True))/(k-1)
                            score=np.einsum('rbke,bed->rbkd',bits[:,:,:k]-p[None,:,None,:],F)
                            ghat=np.einsum('rbkj,rbkd->rjd',adv,score)/(BATCH*k);estimate_s=time.perf_counter()-tick
                            current.append(ghat);dd=[];projection_s=0.
                            for rep in range(REPS):
                                tick=time.perf_counter();coef,info=r.project_mean(ghat[rep]@ghat[rep].T);gc=ghat[rep].T@coef;projection_s+=time.perf_counter()-tick
                                gd={'mean':ghat[rep].mean(0),'common':gc};dd.append(np.stack([gd['mean'],gd['common']]))
                                for name,g in gd.items():
                                    norm=float(np.linalg.norm(g));true=true_dirs[name];tn=float(np.linalg.norm(true));zero=norm<=1e-14
                                    direction=np.zeros_like(g) if zero else -g/norm
                                    slope=G@direction;estimated=ghat[rep]@direction
                                    rows.append(dict(condition=ci,seed=seed,state_arm=arm,update=step,t=t,K=k,replicate=rep,direction=name,zero=zero,
                                        norm=norm,true_norm=tn,cosine=float(g@true/(norm*tn)) if norm>1e-14 and tn>1e-14 else None,
                                        relative_error=float(np.linalg.norm(g-true)/tn) if tn>1e-14 else None,
                                        estimated_harm=bool(max(estimated)>1e-12),true_any_harm=bool(max(slope)>1e-12),
                                        true_mean_improves=bool(slope.mean()<-1e-12),**{f'true_slope_{j}':float(v) for j,v in enumerate(slope)},
                                        **{f'estimated_zero_group_{j}':bool(np.linalg.norm(ghat[rep,j])<=1e-14) for j in range(5)}))
                            current_dirs.append(np.array(dd));timings.append(dict(condition=ci,K=k,replicates=REPS,estimate_seconds=estimate_s,projection_seconds=projection_s,sampling64_seconds=sampling_s,exact_gradient_seconds=exact_s))
                        metadata.append(dict(condition=ci,seed=seed,arm=arm,update=step,t=t,risk=risk.tolist(),CE=ce,exact_directions={k:v.tolist() for k,v in true_dirs.items()}))
                        exact.append(G);estimates.append(np.array(current));directions.append(np.array(current_dirs));idsall.append(ids);features.append(F);probs.append(p)
        for path,h in r.BIND.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==h
        assert hashlib.sha256((OUT/'protocol.json').read_bytes()).hexdigest()==ph
        receipt.update(status='complete',conditions=len(metadata),unique_hard_draws=len(metadata)*REPS*BATCH*64,estimated_gradient_batches=len(metadata)*len(KS)*REPS)
    except BaseException as e:receipt.update(status='failed',error=repr(e),traceback=traceback.format_exc())
    finally:
        r.dump('conditions.json',metadata);r.table('replicate_metrics.csv',rows);r.table('timings.csv',timings)
        if exact:np.savez_compressed(OUT/'arrays.npz',exact=np.array(exact),estimates=np.array(estimates),directions=np.array(directions),ids=np.array(idsall),features=np.array(features),probabilities=np.array(probs),costs=C,bits=f.B)
        receipt.update(seconds=time.time()-start,end_epoch=time.time(),RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024);r.dump('receipt.json',receipt);print(json.dumps(receipt),flush=True)
    return 0 if receipt['status']=='complete' else 1
if __name__=='__main__':raise SystemExit(main())
