"""CPU-only, fixed continuation 20 -> 100 updates of two archived toy arms."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,time,json,hashlib,resource,signal,traceback,importlib.util
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/full100_direction_long_20261110'
COMMON=ROOT/'results/full100_adam_scalar_20261110';MEAN=ROOT/'results/full100_mean_direction_20261110'
BASE=ROOT/'results/full100_toy_20261110';ADAM=ROOT/'results/full100_adam_scale_20261110'
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
helper=load('mean_long_helpers',ROOT/'research_branches/full100_mean_direction_20261110/run.py')
q=helper.q;r=q.r;r.OUT=OUT
scalar=load('scalar_long_helpers',ROOT/'research_branches/full100_adam_scalar_20261110/scalar.py')
ARMS=['common','mean']
def parent(seed,arm):return COMMON/f'{seed}_scalar_direction_u20.pt' if arm=='common' else MEAN/f'{seed}_mean_u20.pt'
def gradient(G,arm):
    if arm=='mean':return G.mean(0)
    co,_=r.project_mean(G@G.T);return G.T@co
def step(p,o,ema,g,arm,opt,norm):
    theta=p.detach().numpy().copy();p.grad=torch.from_numpy(g.copy());o.step();native=p.detach().numpy()-theta
    d=scalar.direction_step(g,native,theta,opt)[0] if arm=='common' else helper.displacement(g,theta,norm,opt)
    with torch.no_grad():p.copy_(torch.from_numpy(theta+d))
    actual=p.detach().numpy()-theta
    return .99*ema+.01*p.detach().numpy(),actual,native
def main():
    start=time.time();OUT.mkdir(exist_ok=False);torch.set_num_threads(1);torch.set_num_interop_threads(1)
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('300s CPU limit')));signal.alarm(300)
    receipt=dict(status='running',start_epoch=start,GPU_hours=0,production_calls=0,toy_updates=0,recovery_steps=0)
    finals=[];idx=[];params=[];dists=[];paths=[];curves=[];updates=[];fixed=[];hist={};recover=[]
    try:
        for p in [Path(__file__),Path(__file__).with_name('audit.py'),Path(__file__).with_name('watch.py'),
            ROOT/'research_branches/full100_mean_direction_20261110/run.py',ROOT/'research_branches/full100_adam_scalar_20261110/scalar.py',
            ROOT/'research_branches/full100_adam_scale_20261110/run.py',ROOT/'research_branches/full100_toy_20261110/run.py',
            ROOT/'research_branches/hard_clean_zero_safe_20261110/combine.py',
            *[d/n for d in (COMMON,MEAN) for n in ('protocol.json','finals.json','independent_audit.json','training_trajectories.npz')],
            *[BASE/n for n in ('input_banks.npz','sampling_uniforms.npz','finals.json')],ADAM/'calibration_and_eval.npz',
            *[parent(s,a) for s in r.SEEDS for a in ARMS]]:r.bind(p)
        for d in (COMMON,MEAN):assert json.loads((d/'independent_audit.json').read_text())['passed']
        f,C,tri,cfg,kernel=r.setup();oldbank=np.load(BASE/'input_banks.npz');banks={}
        # Keep old first 20 inputs exactly. New continuation is frozen before updates.
        for seed in r.SEEDS:
            rng=np.random.default_rng(np.random.SeedSequence([2026111052,seed]));ts=rng.integers(1,101,size=(80,8));us=rng.random((80,8,f.M));xs=np.empty((80,8),int)
            for j in range(80):
                for k,t in enumerate(ts[j]):
                    prob=f.alpha[t]*f.ref+(1-f.alpha[t])*f.pi[f.f];bits=(us[j,k]<prob).astype(float)
                    xs[j,k]=np.flatnonzero((f.B==bits).all(1))[0]
            banks[f'{seed}_t']=np.concatenate([oldbank[f'{seed}_t'],ts]);banks[f'{seed}_x']=np.concatenate([oldbank[f'{seed}_x'],xs])
        np.savez_compressed(OUT/'input_banks.npz',**banks);r.bind(OUT/'input_banks.npz')
        bh=hashlib.sha256((OUT/'input_banks.npz').read_bytes()).hexdigest();oldbh=hashlib.sha256((BASE/'input_banks.npz').read_bytes()).hexdigest()
        opt=json.loads((COMMON/'protocol.json').read_text())['optimizer']
        protocol=dict(seeds=r.SEEDS,arms=ARMS,start_update=20,end_update=100,K=8,T=100,readouts=[20,50,100],EMA_readouts=[20,50,100],
            optimizer=opt,EMA=.99,continuation_input_stream=2026111052,input_hash=bh,parent_input_hash=oldbh,
            rule='common: unchanged projected gradient and native adaptive norm scalar; mean: arithmetic mean gradient, same per-index actual displacement norm as common',
            scope='4-node 64-state 9-parameter float64 CPU toy; not production O/A, not generalization',
            reuse='restore all six complete update20 checkpoints including m/v/step/EMA/RNG; reuse update20 terminal results, no refit',
            inference='unchanged original posterior T100, own intermediate states, exact terminal probabilities plus 1024 paired paths; no fixed edge counts',
            evaluation='old 96-input development panel each update; no acceptance or parameter selection; report all five risks and costs',
            mean_state='own mean-gradient Adam moments advance but do not select direction or norm; experimental equal-norm control, not deployable optimizer',
            budget=dict(CPU_threads=1,RSS_GiB=2,worker_seconds=300,watch_seconds=360,GPU=0),bindings=r.BIND)
        r.dump('protocol.json',protocol);ph=hashlib.sha256((OUT/'protocol.json').read_bytes()).hexdigest();r.dump('freeze.json',dict(protocol_SHA256=ph,epoch=time.time()))
        r.dump('tests.json',dict(mean=helper.tests(),common=scalar.tests()))
        uniforms=np.load(BASE/'sampling_uniforms.npz');ev=np.load(ADAM/'calibration_and_eval.npz')
        for seed in r.SEEDS:
            ts,xs=banks[f'{seed}_t'],banks[f'{seed}_x'];et,ex=ev[f'{seed}_eval_t'],ev[f'{seed}_eval_x'];norms={}
            for arm in ARMS:
                z=torch.load(parent(seed,arm),weights_only=False);p,o,ema,cursor=q.restore(z,opt,oldbh)
                assert cursor==20 and o.state[p]['step'].item()==20
                oldhist=np.load((COMMON if arm=='common' else MEAN)/'training_trajectories.npz')
                prefix=f'{seed}_scalar_direction' if arm=='common' else str(seed)
                np.testing.assert_array_equal(p.detach().numpy(),oldhist[prefix+'_parameters'][20])
                np.testing.assert_array_equal(np.stack([o.state[p]['exp_avg'].numpy(),o.state[p]['exp_avg_sq'].numpy(),ema]),oldhist[prefix+'_moments_ema'][20])
                hh=[];ss=[]
                for n in range(20,101):
                    theta=p.detach().numpy().copy();st=o.state[p];hh.append(theta);ss.append(np.stack([st['exp_avg'].numpy().copy(),st['exp_avg_sq'].numpy().copy(),ema.copy()]))
                    val,_,ce,_=r.atomic(f,C,theta,et,ex);fixed.append(dict(seed=seed,arm=arm,update=n,CE=ce,**dict(zip(['relation_count','joint_degree','triangle001','triangle011','component_cost'],val))))
                    if n in (20,50,100):
                        for rd,w in [('raw',theta),('EMA',ema)]:
                            meta=dict(seed=seed,arm=arm,updates=n,readout=rd)
                            if n==20:
                                prior=json.loads(((COMMON if arm=='common' else MEAN)/'finals.json').read_text());oldarm='scalar_direction' if arm=='common' else 'mean_direction_matched_norm'
                                row=next(x for x in prior if x['seed']==seed and x['arm']==oldarm and x['updates']==20 and x['readout']==rd)
                                finals.append(dict(**meta,terminal=row['terminal'],reused=True))
                            else:
                                finals.append(dict(**r.evaluate(f,C,tri,w,meta,uniforms[str(seed)],curves,paths,dists),reused=False));idx.append(meta);params.append(w.copy())
                        torch.save(q.snapshot(p,o,ema,n,bh),OUT/f'{seed}_{arm}_u{n}.pt')
                    if n==100:break
                    cost,G,ce,_=r.atomic(f,C,theta,ts[n],xs[n]);g=gradient(G,arm)
                    if n==50:
                        snap=q.snapshot(p,o,ema,n,bh);loaded=torch.load(OUT/f'{seed}_{arm}_u50.pt',weights_only=False);outs=[]
                        for source in (snap,loaded):
                            pp,oo,ee,cc=q.restore(source,opt,bh);assert cc==50
                            _,GG,_,_=r.atomic(f,C,pp.detach().numpy(),ts[cc],xs[cc]);gg=gradient(GG,arm)
                            ee,_,_=step(pp,oo,ee,gg,arm,opt,norms.get(n));outs.append(q.snapshot(pp,oo,ee,cc+1,bh))
                        torch.testing.assert_close(outs[0]['parameter'],outs[1]['parameter'],atol=0,rtol=0);np.testing.assert_array_equal(outs[0]['ema'],outs[1]['ema'])
                        for k in ('step','exp_avg','exp_avg_sq'):torch.testing.assert_close(outs[0]['optimizer']['state'][0][k],outs[1]['optimizer']['state'][0][k],atol=0,rtol=0)
                        assert outs[0]['python_rng']==outs[1]['python_rng'] and torch.equal(outs[0]['torch_rng'],outs[1]['torch_rng'])
                        for a,b in zip(outs[0]['numpy_rng'],outs[1]['numpy_rng']):np.testing.assert_array_equal(a,b)
                        recover.append(dict(seed=seed,arm=arm,passed=True));receipt['recovery_steps']+=2
                    ema,d,native=step(p,o,ema,g,arm,opt,norms.get(n));norm=float(np.linalg.norm(d))
                    if arm=='common':norms[n]=norm
                    else:assert abs(norm-norms[n])<1e-13
                    after,_,ace,_=r.atomic(f,C,p.detach().numpy(),ts[n],xs[n])
                    if not np.isfinite(p.detach().numpy()).all() or not np.isfinite(after).all():raise FloatingPointError('nonfinite')
                    updates.append(dict(seed=seed,arm=arm,update=n+1,actual_norm=norm,native_norm=float(np.linalg.norm(native)),
                        gradient_norm=float(np.linalg.norm(g)),inner=(G@d).tolist(),before_costs=cost.tolist(),after_costs=after.tolist(),before_CE=ce,after_CE=ace,times=ts[n].tolist(),states=xs[n].tolist()))
                    receipt['toy_updates']+=1
                hist[f'{seed}_{arm}_parameters']=np.array(hh);hist[f'{seed}_{arm}_moments_ema']=np.array(ss)
        for path,h in r.BIND.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==h
        assert hashlib.sha256((OUT/'protocol.json').read_bytes()).hexdigest()==ph
        receipt.update(status='complete',new_terminal_distributions=len(idx),new_toy_paths=len(idx)*1024)
    except BaseException as exc:receipt.update(status='failed',error=repr(exc),traceback=traceback.format_exc())
    finally:
        r.dump('updates.json',updates);r.dump('finals.json',finals);r.dump('state_index.json',idx);r.dump('recovery.json',recover);r.table('fixed_validation.csv',fixed);r.table('trajectory_metrics.csv',curves)
        if hist:np.savez_compressed(OUT/'training_trajectories.npz',**hist)
        if params:np.savez_compressed(OUT/'states_and_paths.npz',parameters=np.array(params),distributions=np.array(dists),paths=np.array(paths))
        receipt.update(seconds=time.time()-start,end_epoch=time.time(),RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024);r.dump('receipt.json',receipt);print(json.dumps(receipt),flush=True)
    return 0 if receipt['status']=='complete' else 1
if __name__=='__main__':raise SystemExit(main())
