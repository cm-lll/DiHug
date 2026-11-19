"""CPU toy training with K64 LOO feedback, fixed archived displacement budget."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,time,json,hashlib,resource,signal,traceback,importlib.util,copy
from pathlib import Path
import numpy as np
import torch
from scipy.special import expit
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/finite_feedback_training_20261110'
LONG=ROOT/'results/full100_direction_long_20261110';COMMON=ROOT/'results/full100_adam_scalar_20261110';MEAN=ROOT/'results/full100_mean_direction_20261110'
BASE=ROOT/'results/full100_toy_20261110';ADAM=ROOT/'results/full100_adam_scale_20261110';PREC=ROOT/'results/finite_feedback_precision_20261110'
spec=importlib.util.spec_from_file_location('finite_train_helpers',ROOT/'research_branches/full100_mean_direction_20261110/run.py')
helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper);q=helper.q;r=q.r;r.OUT=OUT
ARMS=['common','mean'];K=64
def sampled(f,C,theta,ts,xs,rng):
    F=np.stack([f.features[int(t)][int(x)] for t,x in zip(ts,xs)]);p=expit(F@theta)
    bits=(rng.random((len(ts),K,f.M))<p[:,None,:]).astype(np.uint8);ids=(bits@(2**np.arange(f.M-1,-1,-1))).astype(np.uint8)
    costs=C[ids];center=costs-costs[:,:1];adv=(K*center-center.sum(1,keepdims=True))/(K-1)
    score=np.einsum('bke,bed->bkd',bits-p[:,None,:],F);G=np.einsum('bkj,bkd->jd',adv,score)/(len(ts)*K)
    return G,ids,costs.mean((0,1))
def gradient(G,arm):
    if arm=='mean':return G.mean(0)
    co,info=r.project_mean(G@G.T);return G.T@co
def move(g,theta,norm,opt):
    if np.linalg.norm(g)<=1e-14:return -opt['lr']*opt['weight_decay']*theta,True
    return helper.displacement(g,theta,norm,opt),False
def snapshot(p,o,ema,n,bh,rng):
    z=q.snapshot(p,o,ema,n,bh);z['hard_rng']=copy.deepcopy(rng.bit_generator.state);return z
def restore(z,opt,bh):
    p,o,e,n=q.restore(z,opt,bh);rng=np.random.default_rng();rng.bit_generator.state=copy.deepcopy(z['hard_rng']);return p,o,e,n,rng
def main():
    start=time.time();OUT.mkdir(exist_ok=False);torch.set_num_threads(1);torch.set_num_interop_threads(1)
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('300s bound')));signal.alarm(300)
    receipt=dict(status='running',start_epoch=start,GPU_hours=0,production_calls=0,toy_updates=0,recovery_updates=0)
    hist={};samplehist={};updates=[];fixed=[];finals=[];idx=[];params=[];paths=[];dists=[];curves=[];recover=[]
    try:
        for p in [Path(__file__),Path(__file__).with_name('audit.py'),Path(__file__).with_name('watch.py'),
            ROOT/'research_branches/full100_mean_direction_20261110/run.py',ROOT/'research_branches/full100_adam_scale_20261110/run.py',
            ROOT/'research_branches/full100_toy_20261110/run.py',ROOT/'research_branches/hard_clean_zero_safe_20261110/combine.py',
            *[d/n for d in (LONG,COMMON,MEAN) for n in ('training_trajectories.npz','protocol.json','finals.json','independent_audit.json')],
            LONG/'input_banks.npz',BASE/'sampling_uniforms.npz',BASE/'finals.json',ADAM/'calibration_and_eval.npz',PREC/'independent_audit.json']:r.bind(p)
        for d in (LONG,COMMON,MEAN,PREC):assert json.loads((d/'independent_audit.json').read_text())['passed']
        f,C,tri,cfg,kernel=r.setup();h0=np.load(COMMON/'training_trajectories.npz');h1=np.load(LONG/'training_trajectories.npz')
        banks=np.load(LONG/'input_banks.npz');bh=hashlib.sha256((LONG/'input_banks.npz').read_bytes()).hexdigest();norms={};initial={}
        for seed in r.SEEDS:
            prefix=h0[f'{seed}_scalar_direction_parameters'];tail=h1[f'{seed}_common_parameters'];np.testing.assert_array_equal(prefix[-1],tail[0])
            full=np.concatenate([prefix,tail[1:]]);norms[seed]=np.linalg.norm(np.diff(full,axis=0),axis=1);initial[seed]=prefix[0].copy()
        opt=json.loads((COMMON/'protocol.json').read_text())['optimizer'];np.savez_compressed(OUT/'fixed_budgets.npz',**{str(k):v for k,v in norms.items()});r.bind(OUT/'fixed_budgets.npz')
        protocol=dict(seeds=r.SEEDS,arms=ARMS,updates=100,inputs_per_update=8,hard_graphs_per_input=K,hard_stream=2026111056,T=100,
            optimizer=opt,EMA=.99,input_hash=bh,readouts=[20,50,100],raw_and_EMA=True,
            direction='existing arithmetic mean or project_mean of K64 per-input LOO gradients; no CE, no online acceptance',
            magnitude='both finite arms use identical archived exact common-direction displacement norm at each seed/update; isolate direction estimation, not deployment or adaptive magnitude response',
            state='own estimated gradients advance Adam m/v/step once; proposal not used to set direction/budget; decay once, EMA once; complete hard RNG saved',
            zero_rule='estimated direction norm<=1e-14 commits only decay and advances optimizer/EMA, records missed norm budget; no exact-gradient fallback',
            diagnostic='exact gradients/costs evaluated read-only after direction chosen; never choose direction/size/checkpoint or reject step',
            sampling='shared uniforms between finite arms, separate hard stream from input and generation; differing weights need not yield same hard graph',
            scope='4-node 64-state 9-parameter CPU float64 toy; fixed developmental seeds, one hard stream per seed; no real-model evidence',
            budget=dict(CPU_threads=1,RSS_GiB=2,worker_seconds=300,watch_seconds=360,GPU=0),bindings=r.BIND)
        r.dump('protocol.json',protocol);ph=hashlib.sha256((OUT/'protocol.json').read_bytes()).hexdigest();r.dump('freeze.json',dict(protocol_SHA256=ph,epoch=time.time()));r.dump('tests.json',helper.tests())
        ev=np.load(ADAM/'calibration_and_eval.npz');uniform=np.load(BASE/'sampling_uniforms.npz')
        for seed in r.SEEDS:
            ts,xs=banks[f'{seed}_t'],banks[f'{seed}_x'];et,ex=ev[f'{seed}_eval_t'],ev[f'{seed}_eval_x']
            for arm in ARMS:
                p,o=q.new_optimizer(initial[seed],opt);ema=initial[seed].copy();rng=np.random.default_rng(np.random.SeedSequence([2026111056,seed]));hh=[];ss=[];samples=[];gh=[]
                torch.save(snapshot(p,o,ema,0,bh,rng),OUT/f'{seed}_{arm}_u0.pt')
                for n in range(101):
                    theta=p.detach().numpy().copy();st=o.state.get(p,{});m=np.zeros_like(theta) if not st else st['exp_avg'].numpy().copy();v=np.zeros_like(theta) if not st else st['exp_avg_sq'].numpy().copy()
                    hh.append(theta);ss.append(np.stack([m,v,ema.copy()]));val,_,ce,_=r.atomic(f,C,theta,et,ex)
                    fixed.append(dict(seed=seed,arm=arm,update=n,CE=ce,**dict(zip(['relation_count','joint_degree','triangle001','triangle011','component_cost'],val))))
                    if n in (20,50,100):
                        for rd,w in [('raw',theta),('EMA',ema)]:
                            meta=dict(seed=seed,arm=arm,updates=n,readout=rd);finals.append(r.evaluate(f,C,tri,w,meta,uniform[str(seed)],curves,paths,dists));idx.append(meta);params.append(w.copy())
                        torch.save(snapshot(p,o,ema,n,bh,rng),OUT/f'{seed}_{arm}_u{n}.pt')
                    if n==100:break
                    if n==50:
                        outs=[]
                        for z in (snapshot(p,o,ema,n,bh,rng),torch.load(OUT/f'{seed}_{arm}_u50.pt',weights_only=False)):
                            pp,oo,ee,cc,rr=restore(z,opt,bh);assert cc==50;th=pp.detach().numpy().copy();GG,ii,_=sampled(f,C,th,ts[n],xs[n],rr);gg=gradient(GG,arm)
                            pp.grad=torch.from_numpy(gg.copy());oo.step();dd,_=move(gg,th,norms[seed][n],opt)
                            with torch.no_grad():pp.copy_(torch.from_numpy(th+dd))
                            ee=.99*ee+.01*pp.detach().numpy();outs.append((snapshot(pp,oo,ee,n+1,bh,rr),ii))
                        a,b=outs[0][0],outs[1][0];np.testing.assert_array_equal(outs[0][1],outs[1][1]);torch.testing.assert_close(a['parameter'],b['parameter'],atol=0,rtol=0);np.testing.assert_array_equal(a['ema'],b['ema'])
                        for key in ('step','exp_avg','exp_avg_sq'):torch.testing.assert_close(a['optimizer']['state'][0][key],b['optimizer']['state'][0][key],atol=0,rtol=0)
                        assert a['hard_rng']==b['hard_rng'] and a['python_rng']==b['python_rng'] and torch.equal(a['torch_rng'],b['torch_rng'])
                        for x,y in zip(a['numpy_rng'],b['numpy_rng']):np.testing.assert_array_equal(x,y)
                        recover.append(dict(seed=seed,arm=arm,passed=True));receipt['recovery_updates']+=2
                    Ghat,ids,chat=sampled(f,C,theta,ts[n],xs[n],rng);g=gradient(Ghat,arm);d,zero=move(g,theta,norms[seed][n],opt)
                    # Exact quantities are diagnostic only; cannot enter g or d above.
                    cost,G,ce,_=r.atomic(f,C,theta,ts[n],xs[n]);exactg=gradient(G,arm)
                    p.grad=torch.from_numpy(g.copy());o.step();pred,_,_,_=q.formula(theta,g,m,v,n,opt);np.testing.assert_allclose(p.detach().numpy(),pred,atol=1e-14,rtol=0)
                    with torch.no_grad():p.copy_(torch.from_numpy(theta+d))
                    actual=p.detach().numpy()-theta;ema=.99*ema+.01*p.detach().numpy();after,_,ace,_=r.atomic(f,C,p.detach().numpy(),ts[n],xs[n])
                    if not np.isfinite(after).all() or not np.isfinite(g).all():raise FloatingPointError('nonfinite')
                    norm=float(np.linalg.norm(actual));assert zero or abs(norm-norms[seed][n])<1e-13
                    cos=float(g@exactg/(np.linalg.norm(g)*np.linalg.norm(exactg))) if np.linalg.norm(g)>1e-14 and np.linalg.norm(exactg)>1e-14 else None
                    updates.append(dict(seed=seed,arm=arm,update=n+1,budget=float(norms[seed][n]),actual_norm=norm,zero=zero,cosine=cos,
                        estimated_inner=(Ghat@actual).tolist(),true_inner=(G@actual).tolist(),before_costs=cost.tolist(),after_costs=after.tolist(),sample_costs=chat.tolist(),before_CE=ce,after_CE=ace,times=ts[n].tolist(),states=xs[n].tolist()))
                    samples.append(ids);gh.append(Ghat);receipt['toy_updates']+=1
                hist[f'{seed}_{arm}_parameters']=np.array(hh);hist[f'{seed}_{arm}_moments_ema']=np.array(ss);samplehist[f'{seed}_{arm}_ids']=np.array(samples);samplehist[f'{seed}_{arm}_estimated_gradients']=np.array(gh)
        for path,h in r.BIND.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==h
        assert hashlib.sha256((OUT/'protocol.json').read_bytes()).hexdigest()==ph
        receipt.update(status='complete',new_terminal_distributions=len(idx),new_reverse_toy_paths=len(idx)*1024,training_hard_draws=receipt['toy_updates']*8*K)
    except BaseException as e:receipt.update(status='failed',error=repr(e),traceback=traceback.format_exc())
    finally:
        r.dump('updates.json',updates);r.dump('finals.json',finals);r.dump('state_index.json',idx);r.dump('recovery.json',recover);r.table('fixed_validation.csv',fixed);r.table('trajectory_metrics.csv',curves)
        if hist:np.savez_compressed(OUT/'training_trajectories.npz',**hist);np.savez_compressed(OUT/'training_samples.npz',**samplehist)
        if params:np.savez_compressed(OUT/'states_and_paths.npz',parameters=np.array(params),distributions=np.array(dists),paths=np.array(paths))
        receipt.update(seconds=time.time()-start,end_epoch=time.time(),RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024);r.dump('receipt.json',receipt);print(json.dumps(receipt),flush=True)
    return 0 if receipt['status']=='complete' else 1
if __name__=='__main__':raise SystemExit(main())
