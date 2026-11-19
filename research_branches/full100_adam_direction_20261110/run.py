"""Only CPU finite toy; actual-Adam direction projection and equal-norm control."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,time,json,hashlib,resource,signal,traceback,importlib.util,copy
from pathlib import Path
import numpy as np
import torch
from projection import project_displacement,tests
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/full100_adam_direction_20261110'
PREV=ROOT/'results/full100_adam_scale_20261110';BASE=ROOT/'results/full100_toy_20261110'
spec=importlib.util.spec_from_file_location('adam_previous',ROOT/'research_branches/full100_adam_scale_20261110/run.py')
q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q);r=q.r;r.OUT=OUT
ARMS=['original_AdamW','project_actual','matched_norm_AdamW']
def main():
    start=time.time();OUT.mkdir(exist_ok=False);torch.set_num_threads(1);torch.set_num_interop_threads(1)
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('300s CPU limit')));signal.alarm(300)
    receipt=dict(start_epoch=start,status='running',GPU_hours=0,production_calls=0,toy_updates=0,recovery_probe_steps=0)
    updates=[];local=[];fixed=[];finals=[];index=[];params=[];paths=[];dists=[];curves=[];hist={};recover=[]
    try:
        for p in [Path(__file__),Path(__file__).with_name('projection.py'),ROOT/'research_branches/full100_adam_scale_20261110/run.py',
          ROOT/'research_branches/full100_toy_20261110/run.py',ROOT/'research_branches/hard_clean_zero_safe_20261110/combine.py',
          *[BASE/n for n in ('input_banks.npz','sampling_uniforms.npz')],
          *[PREV/n for n in ('protocol.json','training_trajectories.npz','calibration_and_eval.npz','finals.json')]]:r.bind(p)
        f,C,tri,cfg,kernel=r.setup();old=np.load(PREV/'training_trajectories.npz');banks=np.load(BASE/'input_banks.npz')
        uniforms=np.load(BASE/'sampling_uniforms.npz');ev=np.load(PREV/'calibration_and_eval.npz');oldfin=json.loads((PREV/'finals.json').read_text())
        opt=json.loads((PREV/'protocol.json').read_text())['optimizer'];bankhash=hashlib.sha256((BASE/'input_banks.npz').read_bytes()).hexdigest()
        protocol=dict(scope='4 nodes/9 parameters exact hard risks; not a production denoiser',seeds=r.SEEDS,updates=20,K=8,T=100,
            optimizer=opt,backend='native torch CPU float64 foreach/fused=False',EMA=.99,arms=ARMS,
            candidate='project full native AdamW displacement onto Gd<=0; G is existing five current-input risk gradients; no added probes or acceptance',
            state='original combined hard gradient updates m/v/step once; project parameter proposal incl decay, then EMA once; zero projection still advances optimizer',
            comparison='matched_norm arm has independent trajectory and moments; each index uses project_actual arm committed norm budget; multiplier may exceed1, explicitly recorded, no cap',
            local_control='at each project_actual pre-state, native/projected/equal-norm-radial proposals on exactly same input and optimizer proposal, no state advance for shadow evaluations',
            budget=dict(worker_seconds=300,supervisor_seconds=360,RSS_GiB=2,CPU_threads=1,GPU=0),bindings=r.BIND)
        r.dump('protocol.json',protocol);phash=hashlib.sha256((OUT/'protocol.json').read_bytes()).hexdigest();r.dump('freeze.json',dict(protocol_SHA256=phash,epoch=time.time()))
        r.dump('projection_tests.json',tests())
        for seed in r.SEEDS:
            ts=banks[f'{seed}_t'];xs=banks[f'{seed}_x'];et=ev[f'{seed}_eval_t'];ex=ev[f'{seed}_eval_x'];initial=old[f'{seed}_hard_AdamW_parameters'][0].copy();targetnorms=[]
            for arm in ARMS:
                p,o=q.new_optimizer(initial,opt);ema=initial.copy();history=[];states=[]
                for step in range(21):
                    theta=p.detach().numpy().copy();history.append(theta.copy());state=o.state.get(p,{})
                    m=np.zeros_like(theta) if not state else state['exp_avg'].numpy().copy();v=np.zeros_like(theta) if not state else state['exp_avg_sq'].numpy().copy();states.append(np.stack([m,v,ema]))
                    val,_,ce,_=r.atomic(f,C,theta,et,ex)
                    fixed.append(dict(seed=seed,arm=arm,update=step,CE=ce,mean_five_cost=float(val.mean()),**{k:float(val[j]) for j,k in enumerate(['relation_count','joint_degree','triangle001','triangle011','component_cost'])}))
                    if step in (10,20):
                        for readout,w in [('raw',theta)]+([('EMA',ema)] if step==20 else []):
                            meta=dict(seed=seed,arm=arm,updates=step,readout=readout)
                            if arm=='original_AdamW':
                                prior=next(x for x in oldfin if x['seed']==seed and x['arm']=='hard_AdamW' and x['updates']==step and x['readout']==readout)
                                finals.append(dict(**meta,terminal=prior['terminal'],reused=True))
                            else:
                                finals.append(dict(**r.evaluate(f,C,tri,w,meta,uniforms[str(seed)],curves,paths,dists),reused=False));index.append(meta);params.append(w.copy())
                    if step==20:break
                    cost,G,bce,_=r.atomic(f,C,theta,ts[step],xs[step]);coef,sol=r.project_mean(G@G.T);g=G.T@coef
                    if step==10:
                        z=q.snapshot(p,o,ema,step,bankhash);fn=OUT/f'{seed}_{arm}_u10.pt';torch.save(z,fn)
                        outputs=[]
                        for source in (z,torch.load(fn,weights_only=False)):
                            pp,oo,ee,cursor=q.restore(source,opt,bankhash);assert cursor==step
                            pp.grad=torch.from_numpy(g.copy());oo.step();dd=pp.detach().numpy()-theta
                            if arm=='project_actual':dd,_=project_displacement(G,dd)
                            if arm=='matched_norm_AdamW':dd=dd*(targetnorms[step]/np.linalg.norm(dd))
                            actual=theta+dd;ee=.99*ee+.01*actual
                            outputs.append((actual,ee,copy.deepcopy(oo.state_dict())))
                        np.testing.assert_array_equal(outputs[0][0],outputs[1][0]);np.testing.assert_array_equal(outputs[0][1],outputs[1][1])
                        for k in ('exp_avg','exp_avg_sq','step'):torch.testing.assert_close(outputs[0][2]['state'][0][k],outputs[1][2]['state'][0][k],atol=0,rtol=0)
                        recover.append(dict(seed=seed,arm=arm,next_parameter_optimizer_EMA_exact=True));receipt['recovery_probe_steps']+=2
                    p.grad=torch.from_numpy(g.copy());o.step();proposal=p.detach().numpy().copy();native=proposal-theta
                    pred,_,_,_=q.formula(theta,g,m,v,step,opt);assert abs(pred-proposal).max()<1e-14
                    projection_info=None;factor=1.
                    if arm=='project_actual':
                        d,projection_info=project_displacement(G,native);nxt=theta+d
                        # Equal-norm shadow uses actual written projected displacement.
                        projected_norm=float(np.linalg.norm(nxt-theta));factor=projected_norm/np.linalg.norm(native) if np.linalg.norm(native)>0 else 0.
                        radial=theta+factor*native
                        for name,w in [('before',theta),('native',proposal),('project_actual',nxt),('equal_norm_radial',radial)]:
                            lc,_,lce,_=r.atomic(f,C,w,ts[step],xs[step]);vc,_,vce,_=r.atomic(f,C,w,et,ex)
                            local.append(dict(seed=seed,update=step+1,arm=name,norm=float(np.linalg.norm(w-theta)),train_costs=lc.tolist(),fixed_costs=vc.tolist(),train_CE=lce,fixed_CE=vce))
                    elif arm=='matched_norm_AdamW':
                        norm=float(np.linalg.norm(native))
                        if norm==0 and targetnorms[step]>1e-15:raise RuntimeError('zero native direction cannot match positive budget')
                        factor=targetnorms[step]/norm if norm else 0.;nxt=theta+factor*native
                    else:nxt=proposal
                    actual=nxt-theta
                    if arm=='project_actual':
                        tol=64*np.finfo(float).eps*np.linalg.norm(G,axis=1)*max(np.linalg.norm(theta),1.)
                        assert (G@actual<=tol).all();targetnorms.append(float(np.linalg.norm(actual)))
                    if arm=='matched_norm_AdamW':assert abs(np.linalg.norm(actual)-targetnorms[step])<1e-13
                    with torch.no_grad():p.copy_(torch.from_numpy(nxt))
                    ema=.99*ema+.01*nxt;after,_,ace,_=r.atomic(f,C,nxt,ts[step],xs[step])
                    updates.append(dict(seed=seed,arm=arm,update=step+1,gradient_norm=float(np.linalg.norm(g)),native_norm=float(np.linalg.norm(native)),actual_norm=float(np.linalg.norm(actual)),
                      radial_factor=factor,native_inner=(G@native).tolist(),actual_inner=(G@actual).tolist(),projection=projection_info,
                      before_costs=cost.tolist(),after_costs=after.tolist(),before_CE=bce,after_CE=ace,times=ts[step].tolist(),states=xs[step].tolist()))
                    receipt['toy_updates']+=1
                hist[f'{seed}_{arm}_parameters']=np.array(history);hist[f'{seed}_{arm}_moments_ema']=np.array(states)
                if arm=='original_AdamW':
                    np.testing.assert_array_equal(hist[f'{seed}_{arm}_parameters'],old[f'{seed}_hard_AdamW_parameters'])
                    np.testing.assert_array_equal(hist[f'{seed}_{arm}_moments_ema'],old[f'{seed}_hard_AdamW_moments_ema'])
                torch.save(q.snapshot(p,o,ema,20,bankhash),OUT/f'{seed}_{arm}_u20.pt')
        assert hashlib.sha256((OUT/'protocol.json').read_bytes()).hexdigest()==phash
        for path,h in r.BIND.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==h
        r.dump('recovery_checks.json',recover)
        receipt.update(status='complete',bindings_unchanged=len(r.BIND),new_terminal_distributions=len(index),new_toy_paths=len(index)*1024)
    except BaseException as exc:receipt.update(status='failed',error=repr(exc),traceback=traceback.format_exc())
    finally:
        r.dump('updates.json',updates);r.dump('local_shadows.json',local);r.dump('finals.json',finals);r.dump('state_index.json',index);r.table('fixed_validation.csv',fixed);r.table('trajectory_metrics.csv',curves)
        if hist:np.savez_compressed(OUT/'training_trajectories.npz',**hist)
        if params:np.savez_compressed(OUT/'states_and_paths.npz',parameters=np.array(params),distributions=np.array(dists),paths=np.array(paths))
        receipt.update(seconds=time.time()-start,end_epoch=time.time(),RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024);r.dump('receipt.json',receipt);print(json.dumps(receipt),flush=True)
    return 0 if receipt['status']=='complete' else 1
if __name__=='__main__':raise SystemExit(main())
