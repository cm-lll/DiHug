"""Fixed shared-parameter CPU toy CAGrad versus average descent, T100."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import time,json,hashlib,signal,resource,traceback,importlib.util,random,copy
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=Path(os.environ.get('TOY_OUT',str(ROOT/'results/mean_anchored_toy_20261110')))
def load(n,p):
    s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
r=load('finite_helper',ROOT/'research_branches/full100_toy_20261110/run.py');r.OUT=OUT
solver=load('cagrad_helper',ROOT/'research_branches/mean_anchored_pilot_20261110/run.py')
certified=load('certified_solver',Path(__file__).with_name('solver.py'))
BASE=ROOT/'results/full100_toy_20261110';LONG=ROOT/'results/full100_direction_long_20261110';EVAL=ROOT/'results/full100_adam_scale_20261110/calibration_and_eval.npz'
ARMS=['average','cagrad','same_norm_average'];LR=.2
def update(theta,G,arm,target=None):
    mean=G.mean(0);certificate={}
    if arm=='cagrad':d,certificate=certified.solve(G,solver.solve)
    else:d=mean
    delta=-LR*d
    if arm=='same_norm_average':
        if np.linalg.norm(delta)==0:
            if target!=0:raise ArithmeticError('cannot norm-match zero average gradient')
        else:delta=delta*(target/np.linalg.norm(delta))
    return theta+delta,certificate
def snapshot(theta,ema,cursor,bh):
    return dict(theta=theta.copy(),ema=ema.copy(),cursor=cursor,bank_hash=bh,python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=solver.old.torch.get_rng_state())
def main():
    start=time.time();OUT.mkdir(exist_ok=True);solver.old.torch.set_num_threads(1);solver.old.torch.set_num_interop_threads(1)
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('600s CPU science limit')));signal.alarm(600)
    updates=[];finals=[];index=[];params=[];paths=[];dists=[];curves=[];fixed=[];hist={};recovery=[];fds=[]
    receipt=dict(status='running',GPU_hours=0,production_calls=0,toy_updates=0,recovery_probe_updates=0)
    try:
        files=[Path(__file__),Path(__file__).with_name('solver.py'),Path(__file__).with_name('audit.py'),Path(__file__).with_name('watch.py'),ROOT/'research_branches/mean_anchored_pilot_20261110/run.py',ROOT/'research_branches/coordinate_joint_pilot_20261110/run.py',ROOT/'research_branches/full100_toy_20261110/run.py',BASE/'states_and_paths.npz',BASE/'state_index.json',BASE/'sampling_uniforms.npz',LONG/'input_banks.npz',EVAL,ROOT/'results/mean_anchored_pilot_20261110_attempt2/cpu_gate.json',ROOT/'results/mean_anchored_pilot_20261110_attempt2/audit.json']
        for p in files:r.bind(p)
        assert json.loads(files[-2].read_text())['passed'] and json.loads(files[-1].read_text())['passed']
        f,C,tri,cfg,kernel=r.setup();banks=np.load(LONG/'input_banks.npz');bh=r.BIND[str(LONG/'input_banks.npz')];uniforms=np.load(BASE/'sampling_uniforms.npz');ev=np.load(EVAL)
        old=np.load(BASE/'states_and_paths.npz');ix=json.loads((BASE/'state_index.json').read_text());initial={s:old['parameters'][next(i for i,x in enumerate(ix) if x['seed']==s and x['arm']=='initial')].copy() for s in r.SEEDS}
        protocol=dict(seeds=r.SEEDS,arms=ARMS,T=100,updates=100,K=8,c=.5,learning_rate=LR,optimizer='direct SGD without moments or weight decay; existing toy LR .2, not production AdamW',EMA=.99,readouts=[0,20,50,100],graph=cfg['graph'],noise=cfg['noise'],risk_names=['relation_count','joint_degree','triangle001','triangle011','component_cost'],objective='unchanged exact expected five hard costs over 64 output graphs, no CE term',input_bank_hash=bh,training='shared 9-parameter predictor across edges, states and times; archived 100x8 IID forward inputs',matching='same_norm_average follows its own mean gradient with per-seed per-update norm from cagrad; external experimental control, not deployable optimizer',evaluation='raw/EMA full100 exact distribution plus 1024 paired paths; 96 old separate inputs, developer evidence not independent confirmation',geometry='CAGrad acts on parameter gradients and is applied directly; no Adam transformation',budgets=dict(CPU_threads=1,RSS_MiB=2048,worker_seconds=600,watch_seconds=660,GPU=0),bindings=r.BIND)
        r.dump('protocol.json',protocol);ph=hashlib.sha256((OUT/'protocol.json').read_bytes()).hexdigest();r.dump('freeze.json',dict(protocol_SHA256=ph,epoch=time.time()))
        # Exact analytic case and old numerical cases share the same primal problem.
        rng=np.random.default_rng(2026111082);tests=[]
        for j in range(30):
            GG=rng.normal(size=(5,9))
            if j%3==0:GG[1]=GG[0]
            dd,cc=certified.solve(GG,solver.solve);mm=GG.mean(0);ww=np.array(cc['multipliers']);gg=ww@GG
            assert np.linalg.norm(dd-mm)<=.5*np.linalg.norm(mm)+1e-8
            assert abs(gg@mm+.5*np.linalg.norm(mm)*np.linalg.norm(gg)-np.min(GG@dd))<cc['tolerance']*(mm@mm)+1e-10
            tests.append(cc)
        r.dump('solver_gate.json',dict(passed=True,cases=len(tests),certificates=tests))
        for seed in r.SEEDS:
            ts,xs=banks[f'{seed}_t'],banks[f'{seed}_x'];et,ex=ev[f'{seed}_eval_t'],ev[f'{seed}_eval_x'];norms={}
            theta0=initial[seed];meta=dict(seed=seed,arm='initial',updates=0,readout='raw')
            finals.append(r.evaluate(f,C,tri,theta0,meta,uniforms[str(seed)],curves,paths,dists));params.append(theta0.copy());index.append(meta)
            for arm in ['average','cagrad','same_norm_average']:
                theta=theta0.copy();ema=theta.copy();history=[];eh=[]
                for n in range(101):
                    history.append(theta.copy());eh.append(ema.copy());val,_,ce,_=r.atomic(f,C,theta,et,ex)
                    fixed.append(dict(seed=seed,arm=arm,update=n,CE=ce,mean_five_cost=float(val.mean()),**dict(zip(protocol['risk_names'],map(float,val)))))
                    if n in (20,50,100):
                        for rd,w in [('raw',theta),('EMA',ema)]:
                            meta=dict(seed=seed,arm=arm,updates=n,readout=rd);finals.append(r.evaluate(f,C,tri,w,meta,uniforms[str(seed)],curves,paths,dists));params.append(w.copy());index.append(meta)
                        solver.old.torch.save(snapshot(theta,ema,n,bh),OUT/f'{seed}_{arm}_u{n}.pt')
                    if n==100:break
                    cost,G,ce,_=r.atomic(f,C,theta,ts[n],xs[n]);nxt,cert=update(theta,G,arm,norms.get(n));delta=nxt-theta
                    if arm=='cagrad':norms[n]=float(np.linalg.norm(delta))
                    if arm=='same_norm_average':assert abs(np.linalg.norm(delta)-norms[n])<1e-12
                    if n in (0,99) and arm=='average':
                        for j in range(9):
                            h=np.eye(9)[j]*1e-5
                            fd=(r.atomic(f,C,theta+h,ts[n],xs[n])[0].mean()-r.atomic(f,C,theta-h,ts[n],xs[n])[0].mean())/2e-5
                            fds.append(abs(fd-G.mean(0)[j]));assert fds[-1]<1e-8
                    if n==50:
                        snap=snapshot(theta,ema,n,bh);loaded=solver.old.torch.load(OUT/f'{seed}_{arm}_u50.pt',weights_only=False)
                        assert loaded['cursor']==50 and loaded['bank_hash']==bh;np.testing.assert_array_equal(loaded['theta'],theta);np.testing.assert_array_equal(loaded['ema'],ema)
                        assert loaded['python_rng']==snap['python_rng'] and solver.old.torch.equal(loaded['torch_rng'],snap['torch_rng'])
                        for a,b in zip(loaded['numpy_rng'],snap['numpy_rng']):np.testing.assert_array_equal(a,b)
                        _,GG,_,_=r.atomic(f,C,loaded['theta'],ts[n],xs[n]);nn,_=update(loaded['theta'],GG,arm,norms.get(n))
                        np.testing.assert_array_equal(nn,nxt);np.testing.assert_array_equal(.99*loaded['ema']+.01*nn,.99*ema+.01*nxt)
                        recovery.append(dict(seed=seed,arm=arm,passed=True));receipt['recovery_probe_updates']+=1
                    after,_,ace,_=r.atomic(f,C,nxt,ts[n],xs[n]);assert np.isfinite(nxt).all() and np.isfinite(after).all()
                    updates.append(dict(seed=seed,arm=arm,update=n+1,times=ts[n].tolist(),states=xs[n].tolist(),gradient=G.tolist(),actual_norm=float(np.linalg.norm(delta)),inner=(G@delta).tolist(),before_costs=cost.tolist(),after_costs=after.tolist(),before_CE=ce,after_CE=ace,certificate=cert))
                    theta=nxt;ema=.99*ema+.01*theta;receipt['toy_updates']+=1
                hist[f'{seed}_{arm}_parameters']=np.array(history);hist[f'{seed}_{arm}_ema']=np.array(eh)
                r.dump('progress.json',dict(seed=seed,arm=arm,updates=receipt['toy_updates'],seconds=time.time()-start))
        for p,h in r.BIND.items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
        assert hashlib.sha256((OUT/'protocol.json').read_bytes()).hexdigest()==ph
        receipt.update(status='complete',exact_terminal_distributions=len(finals),paired_toy_paths=1024*len(finals),finite_difference_checks=len(fds),max_finite_difference=max(fds))
    except BaseException as exc:receipt.update(status='failed',error=repr(exc),traceback=traceback.format_exc())
    finally:
        r.dump('updates.json',updates);r.dump('finals.json',finals);r.dump('state_index.json',index);r.dump('recovery.json',recovery);r.table('fixed_validation.csv',fixed);r.table('trajectory_metrics.csv',curves)
        if hist:np.savez_compressed(OUT/'training_trajectories.npz',**hist)
        if params:np.savez_compressed(OUT/'states_and_paths.npz',parameters=np.array(params),distributions=np.array(dists),paths=np.array(paths))
        receipt.update(seconds=time.time()-start,end_epoch=time.time(),RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024);r.dump('receipt.json',receipt);print(json.dumps(receipt))
    return 0 if receipt['status']=='complete' else 1
if __name__=='__main__':raise SystemExit(main())
