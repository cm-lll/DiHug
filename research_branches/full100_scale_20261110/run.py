"""One frozen global scale, matched finite toy; no production model/GPU."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import sys,time,json,hashlib,resource,signal,traceback,importlib.util
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
OLD=ROOT/'results/full100_toy_20261110'
OUT=ROOT/'results/full100_scale_20261110'
spec=importlib.util.spec_from_file_location('base100',ROOT/'research_branches/full100_toy_20261110/run.py')
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
r.OUT=OUT
CAL_STREAM=2026111040;EVAL_STREAM=2026111041

def bank(f,seed,stream,shape):
    rng=np.random.default_rng(np.random.SeedSequence([stream,seed]))
    ts=rng.integers(1,101,size=shape);u=rng.random((*shape,f.M));xs=np.empty(shape,dtype=int)
    for ix in np.ndindex(shape):
        t=ts[ix];p=f.alpha[t]*f.ref+(1-f.alpha[t])*f.pi[f.f]
        xs[ix]=np.flatnonzero((f.B==(u[ix]<p)).all(1))[0]
    return ts,xs

def main():
    start=time.time();OUT.mkdir(exist_ok=False)
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('180s CPU budget')));signal.alarm(180)
    receipt=dict(start_epoch=start,GPU_hours=0,production_calls=0,toy_updates=0,status='running')
    updates=[];fixed=[];index=[];finals=[];curves=[];paths=[];dists=[];params=[]
    try:
        for p in [Path(__file__),ROOT/'research_branches/full100_toy_20261110/run.py',ROOT/'research_branches/hard_clean_zero_safe_20261110/combine.py',*[OLD/n for n in ('protocol.json','states_and_paths.npz','state_index.json','input_banks.npz','sampling_uniforms.npz','finals.json','updates.json')]]:r.bind(p)
        f,C,tri,cfg,kernel=r.setup()
        oldidx=json.loads((OLD/'state_index.json').read_text());old=np.load(OLD/'states_and_paths.npz')
        oldfin=json.loads((OLD/'finals.json').read_text());oldbanks=np.load(OLD/'input_banks.npz');uniform=np.load(OLD/'sampling_uniforms.npz')
        initials={s:old['parameters'][next(i for i,x in enumerate(oldidx) if x['seed']==s and x['arm']=='initial')].copy() for s in r.SEEDS}
        protocol=dict(initializations=r.SEEDS,T=100,updates=20,inputs_per_update=8,base_SGD_LR=.2,
          calibration=dict(stream=CAL_STREAM,batches_per_initialization=24,inputs_per_batch=8,
            rule='one global median of ||g_CE||/||project_mean(g_hard)|| at frozen untrained states, clipped to [1,10]',
            zero_threshold=1e-12,zero_rule='stop calibration, do not omit zero cases',
            cap_reason='predeclared diagnostic bound, not optimized for outcomes'),
          evaluation=dict(stream=EVAL_STREAM,inputs_per_initialization=96,full_chain_checkpoints=[10,20],sample_paths=1024,
            note='new random streams, same reference; finite states can repeat; historical seeds are development evidence'),
          arms=['hard_original','hard_scaled'],reuse='old initial/CE20/hard20 full-chain results; hard20 replay required exact',
          restrictions='no CE in hard objective, no direction change at same state, no rescaling per step, no target or sampler change',
          limits=dict(CPU_threads=1,RSS_GiB=2,worker_seconds=180,supervisor_seconds=240,GPU=0),bindings=r.BIND)
        r.dump('protocol.json',protocol)
        banks={};cal=[];ratios=[]
        for s in r.SEEDS:
            ts,xs=bank(f,s,CAL_STREAM,(24,8));et,ex=bank(f,s,EVAL_STREAM,(96,))
            banks[f'{s}_cal_t']=ts;banks[f'{s}_cal_x']=xs;banks[f'{s}_eval_t']=et;banks[f'{s}_eval_x']=ex
            for j in range(24):
                _,G,_,gc=r.atomic(f,C,initials[s],ts[j],xs[j]);coef,sol=r.project_mean(G@G.T);g=G.T@coef
                n=float(np.linalg.norm(g));mean=float(np.linalg.norm(G.mean(0)))
                if n<=1e-12:raise RuntimeError('calibration zero direction: no automatic substitution')
                ratio=float(np.linalg.norm(gc)/n);ratios.append(ratio)
                cal.append(dict(seed=s,batch=j,CE_norm=float(np.linalg.norm(gc)),hard_norm=n,mean_norm=mean,
                    projected_to_mean=n/max(mean,1e-30),ratio=ratio,coefficients=coef.tolist(),KKT=sol['residual']))
        np.savez_compressed(OUT/'calibration_and_eval_inputs.npz',**banks)
        raw=float(np.median(ratios));scale=float(np.clip(raw,1.,10.))
        frozen=dict(scale=scale,unclipped_median=raw,cap_active=scale!=raw,ratios_quantiles=np.quantile(ratios,[0,.25,.5,.75,1]).tolist(),
                    samples=len(cal),frozen_epoch=time.time(),rule=protocol['calibration']['rule'],records=cal)
        r.dump('calibration.json',frozen)
        # Freeze checksum before any new training or terminal evaluation.
        cal_hash=hashlib.sha256((OUT/'calibration.json').read_bytes()).hexdigest()
        r.dump('freeze.json',dict(calibration_SHA256=cal_hash,protocol_SHA256=hashlib.sha256((OUT/'protocol.json').read_bytes()).hexdigest(),epoch=time.time()))
        r.dump('gates.json',r.gates(f,C,initials[r.SEEDS[0]],kernel))
        replay=[]
        for s in r.SEEDS:
            ts=oldbanks[f'{s}_t'];xs=oldbanks[f'{s}_x'];et=banks[f'{s}_eval_t'];ex=banks[f'{s}_eval_x']
            for arm,factor in [('hard_original',1.),('hard_scaled',scale)]:
                theta=initials[s].copy()
                for step in range(21):
                    cost,_,ce,_=r.atomic(f,C,theta,et,ex)
                    fixed.append(dict(seed=s,arm=arm,update=step,CE=ce,**{name:float(cost[k]) for k,name in enumerate(['relation_count','joint_degree','triangle001','triangle011','component_cost'])},mean_five_cost=float(cost.mean())))
                    if step in (10,20):
                        meta=dict(seed=s,arm=arm,updates=step)
                        if arm=='hard_original' and step==20:
                            ix=next(i for i,x in enumerate(oldidx) if x['seed']==s and x['arm']=='hard_five_project_mean')
                            np.testing.assert_array_equal(theta,old['parameters'][ix])
                            replay.append(dict(seed=s,parameter_bitwise=True))
                            finals.append(dict(**meta,terminal=oldfin[ix]['terminal'],reused=True))
                        else:
                            result=r.evaluate(f,C,tri,theta,meta,uniform[str(s)],curves,paths,dists)
                            result['reused']=False;finals.append(result);index.append(meta);params.append(theta.copy())
                    if step==20:break
                    before,G,ce,gc=r.atomic(f,C,theta,ts[step],xs[step]);coef,sol=r.project_mean(G@G.T);g=G.T@coef
                    displacement=-r.LR*factor*g;nxt=theta+displacement
                    if not np.isfinite(nxt).all():raise FloatingPointError('nonfinite candidate')
                    after,_,cea,_=r.atomic(f,C,nxt,ts[step],xs[step])
                    inner=G@displacement
                    assert inner.max()<1e-9
                    updates.append(dict(seed=s,arm=arm,update=step+1,factor=factor,displacement=float(np.linalg.norm(displacement)),
                        hard_norm=float(np.linalg.norm(g)),mean_norm=float(np.linalg.norm(G.mean(0))),CE_norm=float(np.linalg.norm(gc)),
                        coefficients=coef.tolist(),KKT=sol['residual'],first_order=inner.tolist(),before_costs=before.tolist(),after_costs=after.tolist(),
                        before_CE=ce,after_CE=cea,times=ts[step].tolist(),states=xs[step].tolist()))
                    theta=nxt;receipt['toy_updates']+=1
        assert hashlib.sha256((OUT/'calibration.json').read_bytes()).hexdigest()==cal_hash
        for p,h in r.BIND.items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
        r.dump('replay_checks.json',replay)
        receipt.update(status='complete',scale=scale,original_bindings_unchanged=len(r.BIND),new_terminal_distributions=len(index),new_toy_paths=len(index)*1024)
    except BaseException as exc:
        receipt.update(status='failed',error=repr(exc),traceback=traceback.format_exc())
    finally:
        r.dump('updates.json',updates);r.dump('finals.json',finals);r.dump('state_index.json',index)
        r.table('fixed_validation.csv',fixed);r.table('trajectory_metrics.csv',curves)
        if params:np.savez_compressed(OUT/'states_and_paths.npz',parameters=np.array(params),distributions=np.array(dists),paths=np.array(paths))
        receipt.update(seconds=time.time()-start,end_epoch=time.time(),RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
        r.dump('receipt.json',receipt);print(json.dumps(receipt),flush=True)
    return 0 if receipt['status']=='complete' else 1
if __name__=='__main__':raise SystemExit(main())
