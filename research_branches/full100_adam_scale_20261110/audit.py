"""Independent NumPy Adam equations and direct production reverse-kernel check."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import json,time,csv,hashlib
import numpy as np
from scipy.special import expit
import run as q
r=q.r;OUT=q.OUT
def read(n):return json.loads((OUT/n).read_text())
def main():
    start=time.time();assert read('receipt.json')['status']=='complete'
    f,C,tri,cfg,kernel=r.setup();protocol=read('protocol.json');opt=protocol['optimizer'];scale=read('calibration.json')['scale']
    state=np.load(OUT/'states_and_paths.npz');ix=read('state_index.json');finals=read('finals.json');uniforms=np.load(q.OLD/'sampling_uniforms.npz');checks=[]
    for j,meta in enumerate(ix):
        theta=state['parameters'][j];dist=f.prior.copy();u=uniforms[str(meta['seed'])]
        bits=np.searchsorted(np.cumsum(dist),u[0],side='right');np.testing.assert_array_equal(bits,state['paths'][j,0])
        for t in range(100,0,-1):
            p=expit(f.features[t]@theta)
            z=f.kernel.reverse(p.ravel(),f.B.ravel(),np.tile(f.f,f.S),t-1,t).reshape(f.S,f.M)
            P=np.prod(np.where(f.B[None,:,:]>0,z[:,None,:],1-z[:,None,:]),axis=2)
            assert abs(P.sum(1)-1).max()<1e-12
            cumulative=np.cumsum(P[bits],axis=1);cumulative[:,-1]=1.
            bits=(u[101-t,:,None]>=cumulative).sum(1);np.testing.assert_array_equal(bits,state['paths'][j,101-t]);dist=dist@P
        err=float(abs(dist-state['distributions'][j]).max());assert err<1e-12
        for k,v in r.metrics(f,C,tri,dist).items():assert abs(v-finals[j]['terminal'][k])<1e-12
        checks.append(dict(**meta,max_probability_error=err,all1024paths_identical=True))
    history=np.load(OUT/'training_trajectories.npz');banks=np.load(q.OLD/'input_banks.npz');updates=read('updates.json');replay=[]
    for seed in r.SEEDS:
        for arm in q.ARMS:
            theta=history[f'{seed}_{arm}_parameters'][0].copy();m=np.zeros_like(theta);v=m.copy();ema=theta.copy();fac=scale if arm=='hard_scaled_displacement' else 1.
            max_error=0.;cosines=[]
            for step in range(20):
                ts=banks[f'{seed}_t'][step];xs=banks[f'{seed}_x'][step];before,G,ce,gc=r.atomic(f,C,theta,ts,xs)
                coef,sol=r.project_mean(G@G.T);g=gc if arm=='CE_AdamW' else G.T@coef
                # Independently spell out standard AdamW with bias corrections.
                m=.9*m+.1*g;v=.999*v+.001*g*g
                delta=-opt['lr']*opt['weight_decay']*theta-opt['lr']*(m/(1-.9**(step+1)))/(np.sqrt(v/(1-.999**(step+1)))+opt['eps'])
                theta=theta+fac*delta;ema=.99*ema+.01*theta
                target=history[f'{seed}_{arm}_parameters'][step+1];targetstate=history[f'{seed}_{arm}_moments_ema'][step+1]
                error=max(float(abs(theta-target).max()),float(abs(np.stack([m,v,ema])-targetstate).max()));max_error=max(max_error,error)
                assert error<1e-12
                row=next(x for x in updates if x['seed']==seed and x['arm']==arm and x['update']==step+1)
                assert row['times']==ts.tolist() and row['states']==xs.tolist()
                cosines.append(float((-g)@delta/(np.linalg.norm(g)*np.linalg.norm(delta))))
            rows=[x for x in updates if x['seed']==seed and x['arm']==arm]
            replay.append(dict(seed=seed,arm=arm,max_parameter_moment_EMA_error=max_error,
              mean_update_norm=float(np.mean([x['actual_norm'] for x in rows])),mean_gradient_norm=float(np.mean([x['gradient_norm'] for x in rows])),
              mean_negative_gradient_update_cosine=float(np.mean(cosines)),
              first_order_any_damage=sum(max(x['actual_inner_products'])>1e-12 for x in rows),
              finite_any_damage=sum(any(b>a+1e-12 for a,b in zip(x['before_costs'],x['after_costs'])) for x in rows),
              negative_gradient_any_damage=sum(max(x['negative_gradient_inner_products'])>1e-12 for x in rows)))
    # Verify the initial calibration from fresh NumPy moments, without torch optimizer.
    cal=read('calibration.json');cb=np.load(OUT/'calibration_and_eval.npz');ratios=[];cal_error=0.
    for row in cal['records']:
        seed=row['seed'];batch=row['batch'];theta=history[f'{seed}_CE_AdamW_parameters'][0]
        _,G,_,gc=r.atomic(f,C,theta,cb[f'{seed}_cal_t'][batch],cb[f'{seed}_cal_x'][batch]);coef,_=r.project_mean(G@G.T);g=G.T@coef;norms=[]
        for grad in (gc,g):
            delta=-opt['lr']*opt['weight_decay']*theta-opt['lr']*grad/(abs(grad)+opt['eps']);norms.append(np.linalg.norm(delta))
        rat=norms[0]/norms[1];ratios.append(rat);cal_error=max(cal_error,abs(rat-row['ratio']))
    assert cal_error<1e-10 and abs(np.clip(np.median(ratios),1,10)-scale)<1e-10
    for n,h in read('freeze.json').items():assert hashlib.sha256((OUT/n).read_bytes()).hexdigest()==h
    for path,h in protocol['bindings'].items():assert hashlib.sha256(q.Path(path).read_bytes()).hexdigest()==h
    flat=[dict(**m,**z['terminal']) for m,z in zip(ix,finals)];aggregate=[];keys=list(finals[0]['terminal'])
    for arm,step,rd in sorted({(x['arm'],x['updates'],x['readout']) for x in flat}):
        rows=[x for x in flat if x['arm']==arm and x['updates']==step and x['readout']==rd];a=dict(arm=arm,updates=step,readout=rd,n_initializations=len(rows))
        for k in keys:
            vals=np.array([x[k] for x in rows]);a[k]=float(vals.mean());a[k+'_SD']=float(vals.std(ddof=1))
        aggregate.append(a)
    oldfin=json.loads((q.OLD/'finals.json').read_text());difference=[]
    for row in flat:
        if row['updates']!=20:continue
        initial=next(x for x in oldfin if x['seed']==row['seed'] and x['arm']=='initial')['terminal']
        difference.append(dict(seed=row['seed'],arm=row['arm'],readout=row['readout'],**{k:row[k]-initial[k] for k in keys}))
    r.table('terminal_metrics.csv',flat);r.table('aggregate.csv',aggregate);r.table('update_geometry.csv',replay);r.table('minus_initial.csv',difference)
    r.dump('summary.json',dict(scale=scale,aggregate=aggregate,per_initialization=flat,minus_initial=difference,update_geometry=replay,
       scope='fresh-state float64 CPU AdamW 20-update arithmetic toy; no inference about trained GPU model quality'))
    r.dump('independent_audit.json',dict(passed=True,seconds=time.time()-start,checks=checks,replay=replay,calibration_max_ratio_error=cal_error,
        all_bindings_unchanged=True,frozen_before_training=True,all_paths_reproduced=True,GPU_hours=0))
    print(json.dumps(dict(passed=True,seconds=time.time()-start,scale=scale)))
if __name__=='__main__':main()
