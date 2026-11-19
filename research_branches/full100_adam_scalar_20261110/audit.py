"""Independent scalar formula, NumPy optimizer replay and full-kernel audit."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import json,time,hashlib,csv
import numpy as np
from scipy.special import expit
import torch
import run as q
r=q.r;OUT=q.OUT
def read(n):return json.loads((OUT/n).read_text())
def independent_direction(g,d,theta,opt):
    decay=-opt['lr']*opt['weight_decay']*theta
    if np.linalg.norm(g)==0:return decay
    return -(g/np.linalg.norm(g))*np.linalg.norm(d-decay)+decay
def main():
    start=time.time();assert read('receipt.json')['status']=='complete'
    f,C,tri,cfg,kernel=r.setup();protocol=read('protocol.json');opt=protocol['optimizer'];finals=read('finals.json');ix=read('state_index.json')
    saved=np.load(OUT/'states_and_paths.npz');uniform=np.load(q.BASE/'sampling_uniforms.npz');chainchecks=[]
    for j,meta in enumerate(ix):
        theta=saved['parameters'][j];dist=f.prior.copy();u=uniform[str(meta['seed'])];st=np.searchsorted(np.cumsum(dist),u[0],side='right');np.testing.assert_array_equal(st,saved['paths'][j,0])
        for t in range(100,0,-1):
            p=expit(f.features[t]@theta);z=f.kernel.reverse(p.ravel(),f.B.ravel(),np.tile(f.f,f.S),t-1,t).reshape(f.S,f.M)
            P=np.prod(np.where(f.B[None,:,:]>0,z[:,None,:],1-z[:,None,:]),axis=2);assert abs(P.sum(1)-1).max()<1e-12
            cum=np.cumsum(P[st],axis=1);cum[:,-1]=1.;st=(u[101-t,:,None]>=cum).sum(1);np.testing.assert_array_equal(st,saved['paths'][j,101-t]);dist=dist@P
        err=float(abs(dist-saved['distributions'][j]).max());assert err<1e-12
        row=next(z for z in finals if all(z[k]==v for k,v in meta.items()))
        for k,v in r.metrics(f,C,tri,dist).items():assert abs(row['terminal'][k]-v)<1e-12
        chainchecks.append(dict(**meta,probability_error=err,all1024paths_exact=True))
    history=np.load(OUT/'training_trajectories.npz');banks=np.load(q.BASE/'input_banks.npz');upd=read('updates.json');replay=[]
    for seed in r.SEEDS:
        projected=history[f'{seed}_scalar_direction_parameters'];targets=np.linalg.norm(np.diff(projected,axis=0),axis=1)
        for arm in q.ARMS:
            theta=history[f'{seed}_{arm}_parameters'][0].copy();m=np.zeros_like(theta);v=m.copy();ema=theta.copy();maxerr=0.
            for step in range(20):
                ts=banks[f'{seed}_t'][step];xs=banks[f'{seed}_x'][step];_,G,_,_=r.atomic(f,C,theta,ts,xs);co,_=r.project_mean(G@G.T);g=G.T@co
                m=.9*m+.1*g;v=.999*v+.001*g*g
                d=-opt['lr']*opt['weight_decay']*theta-opt['lr']*(m/(1-.9**(step+1)))/(np.sqrt(v/(1-.999**(step+1)))+opt['eps'])
                if arm=='scalar_direction':d=independent_direction(g,d,theta,opt)
                if arm=='matched_norm_AdamW':d=d*targets[step]/np.linalg.norm(d)
                theta+=d;ema=.99*ema+.01*theta
                err=max(float(abs(theta-history[f'{seed}_{arm}_parameters'][step+1]).max()),float(abs(np.stack([m,v,ema])-history[f'{seed}_{arm}_moments_ema'][step+1]).max()));maxerr=max(maxerr,err);assert err<1e-11
                row=next(x for x in upd if x['seed']==seed and x['arm']==arm and x['update']==step+1)
                assert row['times']==ts.tolist() and row['states']==xs.tolist()
            z10=torch.load(OUT/f'{seed}_{arm}_u10.pt',weights_only=False);z20=torch.load(OUT/f'{seed}_{arm}_u20.pt',weights_only=False)
            assert z10['cursor']==10 and z20['cursor']==20 and z10['bank_hash']==z20['bank_hash']==hashlib.sha256((q.BASE/'input_banks.npz').read_bytes()).hexdigest()
            assert z10['python_rng']==z20['python_rng'];assert torch.equal(z10['torch_rng'],z20['torch_rng'])
            for a,b in zip(z10['numpy_rng'],z20['numpy_rng']):np.testing.assert_array_equal(a,b)
            replay.append(dict(seed=seed,arm=arm,max_parameter_moment_EMA_error=maxerr,CPU_RNG_and_cursor_bound=True))
    flat=[dict(seed=x['seed'],arm=x['arm'],updates=x['updates'],readout=x['readout'],**x['terminal']) for x in finals]
    for x in json.loads((q.PREV/'finals.json').read_text()):
        if x['arm']=='CE_AdamW':flat.append(dict(seed=x['seed'],arm=x['arm'],updates=x['updates'],readout=x['readout'],**x['terminal']))
    keys=list(finals[0]['terminal']);aggregate=[]
    for arm,step,rd in sorted({(x['arm'],x['updates'],x['readout']) for x in flat}):
        rows=[x for x in flat if x['arm']==arm and x['updates']==step and x['readout']==rd];a=dict(arm=arm,updates=step,readout=rd,n_initializations=len(rows))
        for k in keys:
            vals=np.array([x[k] for x in rows]);a[k]=float(vals.mean());a[k+'_SD']=float(vals.std(ddof=1))
        aggregate.append(a)
    geometry=[]
    for seed in r.SEEDS:
        for arm in q.ARMS:
            rows=[x for x in upd if x['seed']==seed and x['arm']==arm]
            d=np.array([np.array(x['after_costs'])-x['before_costs'] for x in rows]);info=[x['direction'] for x in rows if x['direction']]
            geometry.append(dict(seed=seed,arm=arm,mean_norm=float(np.mean([x['actual_norm'] for x in rows])),
                mean_native_norm=float(np.mean([x['native_norm'] for x in rows])),mean_norm_ratio=float(np.mean([x['actual_norm']/x['native_norm'] for x in rows])),
                max_radial_factor=max(x['radial_factor'] for x in rows),min_radial_factor=min(x['radial_factor'] for x in rows),
                first_order_damage=sum(max(x['actual_inner'])>1e-12 for x in rows),finite_damage=int(sum(x.max()>1e-12 for x in d)),
                finite_group_increase_counts=(d>1e-12).sum(0).tolist(),max_finite_increase=float(d.max()),zero_steps=sum(x['actual_norm']<=1e-12 for x in rows),
                mean_cosine=float(np.mean([x['cosine'] for x in info])) if info else None,
                mean_effective_scalar=float(np.mean([x['effective_scalar'] for x in info])) if info else None))
    locals=read('local_shadows.json');localdiff=[]
    for seed in r.SEEDS:
        for step in range(1,21):
            row={x['arm']:x for x in locals if x['seed']==seed and x['update']==step}
            assert abs(row['scalar_direction']['norm']-row['equal_norm_radial']['norm'])<1e-13
            localdiff.append(dict(seed=seed,update=step,train_project_minus_radial=(np.array(row['scalar_direction']['train_costs'])-row['equal_norm_radial']['train_costs']).tolist(),
                fixed_project_minus_radial=(np.array(row['scalar_direction']['fixed_costs'])-row['equal_norm_radial']['fixed_costs']).tolist()))
    for path,h in protocol['bindings'].items():assert hashlib.sha256(q.Path(path).read_bytes()).hexdigest()==h
    assert hashlib.sha256((OUT/'protocol.json').read_bytes()).hexdigest()==read('freeze.json')['protocol_SHA256']
    r.table('terminal_metrics.csv',flat);r.table('aggregate.csv',aggregate);r.table('geometry.csv',geometry)
    r.dump('summary.json',dict(aggregate=aggregate,per_initialization=flat,geometry=geometry,local_same_state_differences=localdiff))
    r.dump('independent_audit.json',dict(passed=True,seconds=time.time()-start,chains=chainchecks,replay=replay,
        independent_solver='independent scalar normalization and NumPy Adam replay',bindings_unchanged=True,protocol_unchanged=True,GPU_hours=0))
    print(json.dumps(dict(passed=True,seconds=time.time()-start,chains=len(chainchecks))))
if __name__=='__main__':main()
