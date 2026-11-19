"""CPU toy: calibrate actual AdamW displacement, not preconditioned gradient."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import sys,time,json,hashlib,resource,signal,traceback,importlib.util,copy,random
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];OLD=ROOT/'results/full100_toy_20261110'
OUT=ROOT/'results/full100_adam_scale_20261110'
spec=importlib.util.spec_from_file_location('scalehelper',ROOT/'research_branches/full100_scale_20261110/run.py')
helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
r=helper.r;r.OUT=OUT
CONFIG=ROOT/'research_branches/cn_multiseed_20261104/configs/AE_late_20261102.json'
AUDIT=ROOT/'results/count_adam_audit_cpu_20261110/summary.json'
ARMS=['CE_AdamW','hard_AdamW','hard_scaled_displacement']

def new_optimizer(theta,opt):
    p=torch.nn.Parameter(torch.tensor(theta,dtype=torch.float64))
    o=torch.optim.AdamW([p],lr=opt['lr'],betas=opt['betas'],eps=opt['eps'],weight_decay=opt['weight_decay'],
        amsgrad=False,foreach=False,fused=False)
    return p,o

def formula(theta,g,m,v,step,opt):
    b1,b2=opt['betas'];mm=b1*m+(1-b1)*g;vv=b2*v+(1-b2)*g*g;k=step+1
    nxt=theta*(1-opt['lr']*opt['weight_decay'])-opt['lr']*(mm/(1-b1**k))/(np.sqrt(vv/(1-b2**k))+opt['eps'])
    return nxt,mm,vv,k

def snapshot(p,o,ema,cursor,bank_hash):
    return dict(parameter=p.detach().clone(),optimizer=copy.deepcopy(o.state_dict()),ema=ema.copy(),cursor=cursor,bank_hash=bank_hash,
      python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state())

def restore(z,opt,bank_hash):
    assert z['bank_hash']==bank_hash
    p,o=new_optimizer(z['parameter'].numpy(),opt);o.load_state_dict(copy.deepcopy(z['optimizer']))
    random.setstate(z['python_rng']);np.random.set_state(z['numpy_rng']);torch.set_rng_state(z['torch_rng'])
    return p,o,z['ema'].copy(),z['cursor']

def advance(p,o,ema,g,scale):
    before=p.detach().numpy().copy();p.grad=torch.from_numpy(g.copy());o.step();proposal=p.detach().numpy().copy()
    with torch.no_grad():p.copy_(torch.from_numpy(before+scale*(proposal-before)))
    actual=p.detach().numpy().copy();ema=.99*ema+.01*actual
    return actual,ema,proposal

def main():
    start=time.time();OUT.mkdir(exist_ok=False);torch.set_num_threads(1);torch.set_num_interop_threads(1)
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('300s CPU bound')));signal.alarm(300)
    receipt=dict(start_epoch=start,GPU_hours=0,production_calls=0,toy_updates=0,recovery_probe_updates=0,status='running')
    updates=[];fixed=[];index=[];finals=[];curves=[];paths=[];dists=[];params=[];trajectories=[];recovery=[]
    try:
        for path in [Path(__file__),CONFIG,AUDIT,ROOT/'research_branches/hard_clean_pilot_20261110/runtime.py',
          ROOT/'research_branches/full100_scale_20261110/run.py',ROOT/'research_branches/full100_toy_20261110/run.py',
          ROOT/'research_branches/hard_clean_zero_safe_20261110/combine.py',
          *[OLD/n for n in ('states_and_paths.npz','state_index.json','input_banks.npz','sampling_uniforms.npz')]]:r.bind(path)
        f,C,tri,cfg,kernel=r.setup()
        # Bind production numerical settings recorded from four real AdamW states.
        records=json.loads(AUDIT.read_text());records=records if isinstance(records,list) else records['tasks']
        opts=[z['optimizer'] for z in records];opt=opts[0]
        assert all(o==opt for o in opts)
        assert opt==dict(lr=.0002,betas=[.9,.999],eps=1e-8,weight_decay=1e-12,foreach=None,fused=None)
        for task in json.loads(CONFIG.read_text())['tasks']:
            native=task['config']['optimizer']
            for key in ('lr','betas','eps','weight_decay'):assert native[key]==opt[key]
            assert native['type']=='AdamW' and native['amsgrad'] is False
        oldidx=json.loads((OLD/'state_index.json').read_text());old=np.load(OLD/'states_and_paths.npz');banks=np.load(OLD/'input_banks.npz');uniforms=np.load(OLD/'sampling_uniforms.npz')
        initial={s:old['parameters'][next(i for i,x in enumerate(oldidx) if x['seed']==s and x['arm']=='initial')].copy() for s in r.SEEDS}
        protocol=dict(initializations=r.SEEDS,T=100,toy_updates_per_arm=20,K=8,arms=ARMS,optimizer=opt,
          actual_backend=dict(device='CPU',dtype='float64',foreach=False,fused=False,amsgrad=False,torch=str(torch.__version__)),EMA=.99,
          calibration=dict(stream=2026111040,batches_per_initialization=24,inputs_per_batch=8,
            rule='median of fresh-state actual AdamW CE displacement norm / hard displacement norm; one global scalar clipped [1,10]',
            zero_norm_stop=1e-12,note='same calibration bank as previous SGD diagnostic, now use actual AdamW; not independent confirmation'),
          submission='single native AdamW proposal; moments/step advance once with unscaled hard gradient; commit old+s*(proposal-old), including decay; then EMA once',
          evaluation='raw at updates10/20 and EMA at20, exact full100 and 1024 paired paths; 96 old separate fixed inputs each update',
          restrictions='no new loss, no raw-gradient multiplier, no projection after AdamW, no online acceptance, no LR search; oldSGD not matched baseline',
          budgets=dict(CPU_threads=1,RSS_GiB=2,worker_seconds=300,supervisor_seconds=360,GPU=0),bindings=r.BIND)
        r.dump('protocol.json',protocol)
        calibration=[];bankout={};formula_errors=[]
        for s in r.SEEDS:
            ts,xs=helper.bank(f,s,2026111040,(24,8));et,ex=helper.bank(f,s,2026111041,(96,))
            bankout[f'{s}_cal_t']=ts;bankout[f'{s}_cal_x']=xs;bankout[f'{s}_eval_t']=et;bankout[f'{s}_eval_x']=ex
            for j in range(24):
                _,G,_,gc=r.atomic(f,C,initial[s],ts[j],xs[j]);coef,sol=r.project_mean(G@G.T);g=G.T@coef;norms=[]
                for grad in (gc,g):
                    p,o=new_optimizer(initial[s],opt);p.grad=torch.from_numpy(grad.copy());o.step();got=p.detach().numpy().copy()
                    predicted,_,_,_=formula(initial[s],grad,np.zeros_like(grad),np.zeros_like(grad),0,opt)
                    err=float(abs(got-predicted).max());assert err<1e-14;formula_errors.append(err)
                    norms.append(float(np.linalg.norm(got-initial[s])))
                if norms[1]<=1e-12:raise RuntimeError('zero calibration displacement, no substitution')
                calibration.append(dict(seed=s,batch=j,CE_norm=norms[0],hard_norm=norms[1],ratio=norms[0]/norms[1],gradient_ratio=float(np.linalg.norm(gc)/np.linalg.norm(g))))
        np.savez_compressed(OUT/'calibration_and_eval.npz',**bankout)
        raw=float(np.median([x['ratio'] for x in calibration]));scale=float(np.clip(raw,1,10))
        r.dump('calibration.json',dict(scale=scale,unclipped_median=raw,cap_active=scale!=raw,records=calibration,epoch=time.time()))
        frozen={n:hashlib.sha256((OUT/n).read_bytes()).hexdigest() for n in ('protocol.json','calibration.json')};r.dump('freeze.json',frozen)
        r.dump('gates.json',r.gates(f,C,initial[r.SEEDS[0]],kernel))
        bank_hash=hashlib.sha256((OLD/'input_banks.npz').read_bytes()).hexdigest()
        for s in r.SEEDS:
            ts=banks[f'{s}_t'];xs=banks[f'{s}_x'];et=bankout[f'{s}_eval_t'];ex=bankout[f'{s}_eval_x']
            for arm in ARMS:
                factor=scale if arm=='hard_scaled_displacement' else 1.
                p,o=new_optimizer(initial[s],opt);ema=initial[s].copy();history=[];moment_history=[]
                for step in range(21):
                    theta=p.detach().numpy().copy();history.append(theta.copy())
                    state=o.state.get(p,{});m=np.zeros_like(theta) if not state else state['exp_avg'].numpy().copy();v=np.zeros_like(theta) if not state else state['exp_avg_sq'].numpy().copy()
                    moment_history.append(np.stack([m,v,ema]))
                    val,_,vce,_=r.atomic(f,C,theta,et,ex)
                    fixed.append(dict(seed=s,arm=arm,update=step,CE=vce,mean_five_cost=float(val.mean()),**{k:float(val[j]) for j,k in enumerate(['relation_count','joint_degree','triangle001','triangle011','component_cost'])}))
                    if step in (10,20):
                        for readout,weights in [('raw',theta)]+([('EMA',ema)] if step==20 else []):
                            meta=dict(seed=s,arm=arm,updates=step,readout=readout)
                            finals.append(r.evaluate(f,C,tri,weights,meta,uniforms[str(s)],curves,paths,dists));index.append(meta);params.append(weights.copy())
                    if step==20:break
                    cost,G,ce,gc=r.atomic(f,C,theta,ts[step],xs[step]);coef,sol=r.project_mean(G@G.T);g=gc if arm=='CE_AdamW' else G.T@coef
                    if step==10:
                        z=snapshot(p,o,ema,step,bank_hash);ck=OUT/f'{s}_{arm}_u10.pt';torch.save(z,ck)
                        pc,oc=new_optimizer(theta,opt);oc.load_state_dict(copy.deepcopy(z['optimizer']))
                        a,ea,pa=advance(pc,oc,ema.copy(),g,factor)
                        loaded=torch.load(ck,weights_only=False);pr,orr,er,cursor=restore(loaded,opt,bank_hash);b,eb,pb=advance(pr,orr,er,g,factor)
                        assert cursor==step;np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(ea,eb)
                        for k in ('exp_avg','exp_avg_sq','step'):torch.testing.assert_close(oc.state[pc][k],orr.state[pr][k],atol=0,rtol=0)
                        assert random.getstate()==z['python_rng'];np.testing.assert_array_equal(np.random.get_state()[1],z['numpy_rng'][1]);assert torch.equal(torch.get_rng_state(),z['torch_rng'])
                        recovery.append(dict(seed=s,arm=arm,parameters_optimizer_EMA_RNG_cursor_exact=True));receipt['recovery_probe_updates']+=2
                    pred,mm,vv,k=formula(theta,g,m,v,step,opt)
                    nxt,ema,proposal=advance(p,o,ema,g,factor);err=float(abs(pred-proposal).max());assert err<1e-14;formula_errors.append(err)
                    np.testing.assert_allclose(o.state[p]['exp_avg'].numpy(),mm,atol=1e-14,rtol=0);np.testing.assert_allclose(o.state[p]['exp_avg_sq'].numpy(),vv,atol=1e-14,rtol=0)
                    after,_,cea,_=r.atomic(f,C,nxt,ts[step],xs[step]);delta=nxt-theta
                    updates.append(dict(seed=s,arm=arm,update=step+1,scale=factor,gradient_norm=float(np.linalg.norm(g)),
                        proposal_norm=float(np.linalg.norm(proposal-theta)),actual_norm=float(np.linalg.norm(delta)),
                        negative_gradient_inner_products=(G@(-g)).tolist(),actual_inner_products=(G@delta).tolist(),
                        before_costs=cost.tolist(),after_costs=after.tolist(),before_CE=ce,after_CE=cea,formula_error=err,
                        times=ts[step].tolist(),states=xs[step].tolist()))
                    receipt['toy_updates']+=1
                trajectories.append(dict(seed=s,arm=arm,parameters=np.array(history),moments_ema=np.array(moment_history)))
                torch.save(snapshot(p,o,ema,20,bank_hash),OUT/f'{s}_{arm}_u20.pt')
        for n,h in frozen.items():assert hashlib.sha256((OUT/n).read_bytes()).hexdigest()==h
        for p,h in r.BIND.items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
        r.dump('recovery_checks.json',recovery)
        receipt.update(status='complete',scale=scale,formula_max_error=max(formula_errors),bindings_unchanged=len(r.BIND),new_terminal_distributions=len(index),new_toy_paths=len(index)*1024)
    except BaseException as exc:receipt.update(status='failed',error=repr(exc),traceback=traceback.format_exc())
    finally:
        r.dump('updates.json',updates);r.dump('finals.json',finals);r.dump('state_index.json',index);r.table('fixed_validation.csv',fixed);r.table('trajectory_metrics.csv',curves)
        if params:np.savez_compressed(OUT/'states_and_paths.npz',parameters=np.array(params),distributions=np.array(dists),paths=np.array(paths))
        if trajectories:
            np.savez_compressed(OUT/'training_trajectories.npz',**{f'{x["seed"]}_{x["arm"]}_{k}':x[k] for x in trajectories for k in ('parameters','moments_ema')})
        receipt.update(seconds=time.time()-start,end_epoch=time.time(),RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024);r.dump('receipt.json',receipt);print(json.dumps(receipt),flush=True)
    return 0 if receipt['status']=='complete' else 1
if __name__=='__main__':raise SystemExit(main())
