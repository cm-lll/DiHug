"""Bounded CPU ablation: arithmetic mean direction vs common-descent direction.

Use archived per-step common-descent displacement norms; this is a mechanism
control, not a deployable optimizer or an independent final validation.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,time,json,hashlib,resource,signal,traceback,importlib.util,copy
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/full100_mean_direction_20261110'
PREV=ROOT/'results/full100_adam_scalar_20261110'
BASE=ROOT/'results/full100_toy_20261110'
ADAM=ROOT/'results/full100_adam_scale_20261110'
spec=importlib.util.spec_from_file_location('adam_helpers_mean',ROOT/'research_branches/full100_adam_scale_20261110/run.py')
q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q);r=q.r;r.OUT=OUT

def displacement(g,theta,norm,opt):
    """Keep decay once and solve ||-a*g/||g||+decay||=norm, a>=0."""
    decay=-opt['lr']*opt['weight_decay']*theta
    gn=np.linalg.norm(g)
    if gn==0:raise RuntimeError('zero mean gradient: cannot silently match a nonzero budget')
    u=g/gn;b=float(u@decay)
    rad=norm*norm-float(decay@decay)+b*b
    if rad<0:raise RuntimeError('norm budget incompatible with decay')
    a=b+np.sqrt(rad)
    if a<0:raise RuntimeError('no nonnegative direction scale')
    d=-a*u+decay
    assert np.isfinite(d).all() and abs(np.linalg.norm(d)-norm)<1e-13
    return d

def tests():
    rng=np.random.default_rng(2026111051);errs=[]
    for wd in (0.,1e-12,.1):
        for _ in range(24):
            g=rng.normal(size=9);theta=rng.normal(size=9);opt=dict(lr=.0002,weight_decay=wd)
            d=displacement(g,theta,.001,opt);dc=-opt['lr']*wd*theta
            errs.append(abs(np.linalg.norm(d)-.001))
            assert (d-dc)@g<0
            np.testing.assert_allclose(displacement(7*g,theta,.001,opt),d,atol=1e-14,rtol=0)
    return dict(passed=True,cases=len(errs),max_norm_error=float(max(errs)))

def main():
    start=time.time();OUT.mkdir(exist_ok=False);torch.set_num_threads(1);torch.set_num_interop_threads(1)
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('300s bound')));signal.alarm(300)
    receipt=dict(status='running',start_epoch=start,GPU_hours=0,production_calls=0,toy_updates=0,recovery_steps=0)
    history={};updates=[];fixed=[];shadows=[];finals=[];index=[];params=[];paths=[];dists=[];curves=[];recover=[]
    try:
        for p in [Path(__file__),Path(__file__).with_name('audit.py'),Path(__file__).with_name('watch.py'),
            ROOT/'research_branches/full100_adam_scale_20261110/run.py',ROOT/'research_branches/full100_toy_20261110/run.py',
            ROOT/'research_branches/hard_clean_zero_safe_20261110/combine.py',
            *[BASE/n for n in ('input_banks.npz','sampling_uniforms.npz','finals.json')],
            *[PREV/n for n in ('training_trajectories.npz','protocol.json','finals.json','independent_audit.json')],
            *[ADAM/n for n in ('calibration_and_eval.npz','finals.json')]]:r.bind(p)
        assert json.loads((PREV/'independent_audit.json').read_text())['passed']
        f,C,tri,cfg,kernel=r.setup();old=np.load(PREV/'training_trajectories.npz')
        banks=np.load(BASE/'input_banks.npz');ev=np.load(ADAM/'calibration_and_eval.npz');uniforms=np.load(BASE/'sampling_uniforms.npz')
        opt=json.loads((PREV/'protocol.json').read_text())['optimizer'];bankhash=hashlib.sha256((BASE/'input_banks.npz').read_bytes()).hexdigest()
        protocol=dict(seeds=r.SEEDS,updates=20,K=8,T=100,CPU_threads=1,worker_seconds=300,watch_seconds=360,RSS_GiB=2,GPU=0,
            optimizer=opt,EMA=.99,backend='CPU float64; finite 4-node/6-slot/9-parameter model; not actual O/A',
            new_arm='mean_direction_matched_norm',reference_arm='scalar_direction (reused)',
            objective='identical five exact hard graph risks, no CE auxiliary, arithmetic mean gradient instead of project_mean',
            magnitude='per seed/update archived scalar_direction total actual displacement norm; same decay once, solve positive scalar to match total norm',
            state='mean gradient advances own AdamW moments/step once, its proposal discarded for controlled direction/norm; own EMA once; saved RNG and cursor',
            comparison='same-state shadows on all 60 archived common-direction pre-states plus independent 60-update mean trajectory; not a deployable standalone rule',
            readouts=['raw10','raw20','EMA20'],fixed_evaluation='old 96 inputs/seed; development only',
            stop='nonfinite, zero direction or norm infeasibility, binding/replay failure; no search or extension',bindings=r.BIND)
        r.dump('protocol.json',protocol);phash=hashlib.sha256((OUT/'protocol.json').read_bytes()).hexdigest()
        r.dump('freeze.json',dict(epoch=time.time(),protocol_SHA256=phash));r.dump('tests.json',tests())
        for seed in r.SEEDS:
            oldtheta=old[f'{seed}_scalar_direction_parameters'];oldmom=old[f'{seed}_scalar_direction_moments_ema']
            norms=np.linalg.norm(np.diff(oldtheta,axis=0),axis=1);ts=banks[f'{seed}_t'];xs=banks[f'{seed}_x']
            et,ex=ev[f'{seed}_eval_t'],ev[f'{seed}_eval_x'];p,o=q.new_optimizer(oldtheta[0],opt);ema=oldtheta[0].copy();hh=[];ss=[]
            for step in range(21):
                theta=p.detach().numpy().copy();st=o.state.get(p,{})
                m=np.zeros_like(theta) if not st else st['exp_avg'].numpy().copy()
                v=np.zeros_like(theta) if not st else st['exp_avg_sq'].numpy().copy()
                hh.append(theta);ss.append(np.stack([m,v,ema]));val,_,ce,_=r.atomic(f,C,theta,et,ex)
                fixed.append(dict(seed=seed,update=step,CE=ce,**dict(zip(['relation_count','joint_degree','triangle001','triangle011','component_cost'],val))))
                if step in (10,20):
                    for rd,w in [('raw',theta)]+([('EMA',ema)] if step==20 else []):
                        meta=dict(seed=seed,arm='mean_direction_matched_norm',updates=step,readout=rd)
                        finals.append(r.evaluate(f,C,tri,w,meta,uniforms[str(seed)],curves,paths,dists));index.append(meta);params.append(w)
                if step==20:break
                costs,G,bce,_=r.atomic(f,C,theta,ts[step],xs[step]);g=G.mean(0)
                d=displacement(g,theta,norms[step],opt)
                if step==10:
                    z=q.snapshot(p,o,ema,step,bankhash);fn=OUT/f'{seed}_mean_u10.pt';torch.save(z,fn);outs=[]
                    for source in (z,torch.load(fn,weights_only=False)):
                        pp,oo,ee,cursor=q.restore(source,opt,bankhash);assert cursor==step
                        tt=pp.detach().numpy().copy();_,gg,_,_=r.atomic(f,C,tt,ts[cursor],xs[cursor]);gg=gg.mean(0)
                        pp.grad=torch.from_numpy(gg);oo.step();dd=displacement(gg,tt,norms[cursor],opt)
                        with torch.no_grad():pp.copy_(torch.from_numpy(tt+dd))
                        outs.append(q.snapshot(pp,oo,.99*ee+.01*(tt+dd),cursor+1,bankhash))
                    torch.testing.assert_close(outs[0]['parameter'],outs[1]['parameter'],atol=0,rtol=0)
                    np.testing.assert_array_equal(outs[0]['ema'],outs[1]['ema'])
                    for k in ('step','exp_avg','exp_avg_sq'):torch.testing.assert_close(outs[0]['optimizer']['state'][0][k],outs[1]['optimizer']['state'][0][k],atol=0,rtol=0)
                    assert outs[0]['python_rng']==outs[1]['python_rng'];assert torch.equal(outs[0]['torch_rng'],outs[1]['torch_rng'])
                    for a,b in zip(outs[0]['numpy_rng'],outs[1]['numpy_rng']):np.testing.assert_array_equal(a,b)
                    recover.append(dict(seed=seed,passed=True));receipt['recovery_steps']+=2
                p.grad=torch.from_numpy(g.copy());o.step();pred,_,_,_=q.formula(theta,g,m,v,step,opt)
                np.testing.assert_allclose(p.detach().numpy(),pred,atol=1e-14,rtol=0)
                with torch.no_grad():p.copy_(torch.from_numpy(theta+d))
                actual=p.detach().numpy()-theta;ema=.99*ema+.01*p.detach().numpy();after,_,ace,_=r.atomic(f,C,p.detach().numpy(),ts[step],xs[step])
                updates.append(dict(seed=seed,update=step+1,budget=float(norms[step]),actual_norm=float(np.linalg.norm(actual)),
                    before_costs=costs.tolist(),after_costs=after.tolist(),before_CE=bce,after_CE=ace,inner=(G@actual).tolist(),times=ts[step].tolist(),states=xs[step].tolist()))
                # Same-state direction isolation: use archived common pre-boundary, no optimizer advance.
                base=oldtheta[step];bc,BG,_,_=r.atomic(f,C,base,ts[step],xs[step]);dc=oldtheta[step+1]-base
                coef,_=r.project_mean(BG@BG.T);cg=BG.T@coef;mg=BG.mean(0);dm=displacement(mg,base,norms[step],opt)
                for name,dd in [('unchanged',np.zeros_like(dc)),('common',dc),('mean',dm)]:
                    cc,_,ce1,_=r.atomic(f,C,base+dd,ts[step],xs[step]);vv,_,ce2,_=r.atomic(f,C,base+dd,et,ex)
                    shadows.append(dict(seed=seed,update=step+1,arm=name,norm=float(np.linalg.norm(dd)),train_costs=cc.tolist(),fixed_costs=vv.tolist(),train_CE=ce1,fixed_CE=ce2,
                        linear=(BG@dd).tolist(),mean_to_common_cosine=float(mg@cg/(np.linalg.norm(mg)*np.linalg.norm(cg)))))
                receipt['toy_updates']+=1
            history[f'{seed}_parameters']=np.array(hh);history[f'{seed}_moments_ema']=np.array(ss)
            torch.save(q.snapshot(p,o,ema,20,bankhash),OUT/f'{seed}_mean_u20.pt')
        for path,h in r.BIND.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==h
        assert hashlib.sha256((OUT/'protocol.json').read_bytes()).hexdigest()==phash
        receipt.update(status='complete',new_terminal_distributions=len(index),new_toy_paths=len(index)*1024)
    except BaseException as exc:receipt.update(status='failed',error=repr(exc),traceback=traceback.format_exc())
    finally:
        r.dump('updates.json',updates);r.dump('shadows.json',shadows);r.dump('finals.json',finals);r.dump('state_index.json',index);r.dump('recovery.json',recover)
        r.table('fixed_validation.csv',fixed);r.table('trajectory_metrics.csv',curves)
        if history:np.savez_compressed(OUT/'training_trajectories.npz',**history)
        if params:np.savez_compressed(OUT/'states_and_paths.npz',parameters=np.array(params),distributions=np.array(dists),paths=np.array(paths))
        receipt.update(seconds=time.time()-start,end_epoch=time.time(),RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
        r.dump('receipt.json',receipt);print(json.dumps(receipt),flush=True)
    return 0 if receipt['status']=='complete' else 1
if __name__=='__main__':raise SystemExit(main())
