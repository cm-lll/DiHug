"""Independent finite differences, saved-trajectory and production-kernel audit."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import time,json,hashlib,resource
import numpy as np
from scipy.special import expit
import torch
import run as q
r=q.r;OUT=q.OUT
def read(n):return json.loads((OUT/n).read_text())
def main():
    start=time.time();assert read('receipt.json')['status']=='complete'
    f,C,tri,cfg,kernel=r.setup();protocol=read('protocol.json');opt=protocol['optimizer']
    saved=np.load(OUT/'states_and_paths.npz');ix=read('state_index.json');finals=read('finals.json');uniform=np.load(q.BASE/'sampling_uniforms.npz')
    chainchecks=[]
    for j,meta in enumerate(ix):
        theta=saved['parameters'][j];dist=f.prior.copy();u=uniform[str(meta['seed'])];st=np.searchsorted(np.cumsum(dist),u[0],side='right')
        np.testing.assert_array_equal(st,saved['paths'][j,0])
        for t in range(100,0,-1):
            p=expit(f.features[t]@theta);z=f.kernel.reverse(p.ravel(),f.B.ravel(),np.tile(f.f,f.S),t-1,t).reshape(f.S,f.M)
            P=np.prod(np.where(f.B[None,:,:]>0,z[:,None,:],1-z[:,None,:]),axis=2);assert abs(P.sum(1)-1).max()<1e-12
            cum=np.cumsum(P[st],axis=1);cum[:,-1]=1.;st=(u[101-t,:,None]>=cum).sum(1)
            np.testing.assert_array_equal(st,saved['paths'][j,101-t]);dist=dist@P
        err=float(abs(dist-saved['distributions'][j]).max());assert err<1e-12
        for k,v in r.metrics(f,C,tri,dist).items():assert abs(finals[j]['terminal'][k]-v)<1e-12
        chainchecks.append(dict(**meta,max_probability_error=err,paths_verified=1024))
    hist=np.load(OUT/'training_trajectories.npz');old=np.load(q.PREV/'training_trajectories.npz');banks=np.load(q.BASE/'input_banks.npz')
    replay=[];fd=[];normerrors=[]
    for seed in r.SEEDS:
        target=hist[f'{seed}_parameters'];states=hist[f'{seed}_moments_ema'];theta=target[0].copy();ema=theta.copy();m=np.zeros_like(theta);v=m.copy()
        norms=np.linalg.norm(np.diff(old[f'{seed}_scalar_direction_parameters'],axis=0),axis=1);maxerr=0.
        for step in range(20):
            ts=banks[f'{seed}_t'][step];xs=banks[f'{seed}_x'][step];_,G,_,_=r.atomic(f,C,theta,ts,xs);g=G.mean(0)
            if step in (0,10,19):
                for k in range(len(theta)):
                    eps=np.eye(len(theta))[k]*1e-5
                    pa=r.atomic(f,C,theta+eps,ts,xs)[0].mean();pb=r.atomic(f,C,theta-eps,ts,xs)[0].mean()
                    fd.append(abs((pa-pb)/2e-5-g[k]))
            # Independent quadratic form for common positive step and decay.
            decay=-opt['lr']*opt['weight_decay']*theta
            aa=float(g@g);bb=-2*float(g@decay);cc=float(decay@decay)-norms[step]**2
            scale=(-bb+np.sqrt(bb*bb-4*aa*cc))/(2*aa)
            delta=-scale*g+decay
            m=.9*m+.1*g;v=.999*v+.001*g*g;theta=theta+delta;ema=.99*ema+.01*theta
            err=max(float(abs(theta-target[step+1]).max()),float(abs(np.stack([m,v,ema])-states[step+1]).max()));maxerr=max(maxerr,err);assert err<1e-11
            normerrors.append(abs(np.linalg.norm(target[step+1]-target[step])-norms[step]))
        a=torch.load(OUT/f'{seed}_mean_u10.pt',weights_only=False);b=torch.load(OUT/f'{seed}_mean_u20.pt',weights_only=False)
        assert a['cursor']==10 and b['cursor']==20
        assert a['bank_hash']==b['bank_hash']==hashlib.sha256((q.BASE/'input_banks.npz').read_bytes()).hexdigest()
        torch.testing.assert_close(b['parameter'],torch.from_numpy(target[20]),atol=0,rtol=0)
        for k,col in [('exp_avg',0),('exp_avg_sq',1)]:np.testing.assert_array_equal(b['optimizer']['state'][0][k].numpy(),states[20,col])
        assert b['optimizer']['state'][0]['step'].item()==20
        assert a['python_rng']==b['python_rng'] and torch.equal(a['torch_rng'],b['torch_rng'])
        for x,y in zip(a['numpy_rng'],b['numpy_rng']):np.testing.assert_array_equal(x,y)
        replay.append(dict(seed=seed,max_state_error=maxerr,checkpoint_RNG_verified=True))
    assert max(fd)<1e-8 and max(normerrors)<1e-13
    flat=[]
    for source,allowed in [(finals,None),(json.loads((q.PREV/'finals.json').read_text()),{'scalar_direction','matched_norm_AdamW'}),
            (json.loads((q.ADAM/'finals.json').read_text()),{'CE_AdamW'}),(json.loads((q.BASE/'finals.json').read_text()),{'initial'})]:
        for x in source:
            if allowed is None or x['arm'] in allowed:flat.append(dict(seed=x['seed'],arm=x['arm'],updates=x.get('updates',0),readout=x.get('readout','raw'),**x['terminal']))
    keys=list(finals[0]['terminal']);aggregate=[]
    for arm,step,rd in sorted({(x['arm'],x['updates'],x['readout']) for x in flat}):
        rows=[x for x in flat if (x['arm'],x['updates'],x['readout'])==(arm,step,rd)];z=dict(arm=arm,updates=step,readout=rd,n=len(rows))
        for k in keys:
            vals=np.array([x[k] for x in rows]);z[k]=float(vals.mean());z[k+'_SD']=float(vals.std(ddof=1))
        aggregate.append(z)
    upd=read('updates.json');shadows=read('shadows.json');geo=[];same=[];risk=['relation_count','joint_degree','triangle001','triangle011','component_cost']
    for seed in r.SEEDS:
        rr=[x for x in upd if x['seed']==seed];diff=np.array([np.array(x['after_costs'])-x['before_costs'] for x in rr])
        geo.append(dict(seed=seed,first_order_harm_counts=(np.array([x['inner'] for x in rr])>1e-12).sum(0).tolist(),finite_harm_counts=(diff>1e-12).sum(0).tolist(),mean_norm=float(np.mean([x['actual_norm'] for x in rr]))))
        for step in range(1,21):
            rows={x['arm']:x for x in shadows if x['seed']==seed and x['update']==step}
            assert abs(rows['mean']['norm']-rows['common']['norm'])<1e-13
            same.append(dict(seed=seed,update=step,cosine=rows['mean']['mean_to_common_cosine'],
                train_mean_minus_common=(np.array(rows['mean']['train_costs'])-rows['common']['train_costs']).tolist(),
                fixed_mean_minus_common=(np.array(rows['mean']['fixed_costs'])-rows['common']['fixed_costs']).tolist()))
    for path,h in protocol['bindings'].items():assert hashlib.sha256(q.Path(path).read_bytes()).hexdigest()==h
    assert hashlib.sha256((OUT/'protocol.json').read_bytes()).hexdigest()==read('freeze.json')['protocol_SHA256']
    r.table('terminal_metrics.csv',flat);r.table('aggregate.csv',aggregate);r.table('geometry.csv',geo)
    r.dump('summary.json',dict(aggregate=aggregate,per_seed=flat,geometry=geo,same_state=same,risk_order=risk))
    result=dict(passed=True,seconds=time.time()-start,RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        finite_difference_checks=len(fd),finite_difference_max_error=float(max(fd)),norm_max_error=float(max(normerrors)),chains=chainchecks,replay=replay,
        source_bindings_unchanged=True,protocol_unchanged=True,GPU_hours=0)
    r.dump('independent_audit.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ('chains','replay')}))
if __name__=='__main__':main()
