"""Independent sample regeneration, LOO, NNLS, update/state and full-chain audit."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import time,json,hashlib,resource
import numpy as np
from scipy.special import expit
from scipy.optimize import nnls
import torch
import run as q
r=q.r;OUT=q.OUT
def read(n):return json.loads((OUT/n).read_text())
def main():
    start=time.time();assert read('receipt.json')['status']=='complete';f,C,tri,cfg,kernel=r.setup();prot=read('protocol.json');opt=prot['optimizer']
    saved=np.load(OUT/'states_and_paths.npz');ix=read('state_index.json');finals=read('finals.json');uniform=np.load(q.BASE/'sampling_uniforms.npz');checks=[]
    for j,meta in enumerate(ix):
        theta=saved['parameters'][j];dist=f.prior.copy();u=uniform[str(meta['seed'])];st=np.searchsorted(np.cumsum(dist),u[0],side='right');np.testing.assert_array_equal(st,saved['paths'][j,0])
        for t in range(100,0,-1):
            p=expit(f.features[t]@theta);z=f.kernel.reverse(p.ravel(),f.B.ravel(),np.tile(f.f,f.S),t-1,t).reshape(f.S,f.M)
            P=np.prod(np.where(f.B[None,:,:]>0,z[:,None,:],1-z[:,None,:]),axis=2);assert abs(P.sum(1)-1).max()<1e-12
            cum=np.cumsum(P[st],axis=1);cum[:,-1]=1.;st=(u[101-t,:,None]>=cum).sum(1);np.testing.assert_array_equal(st,saved['paths'][j,101-t]);dist=dist@P
        err=float(abs(dist-saved['distributions'][j]).max());assert err<1e-12
        for k,v in r.metrics(f,C,tri,dist).items():assert abs(finals[j]['terminal'][k]-v)<1e-12
        checks.append(dict(**meta,max_probability_error=err,paths_verified=1024))
    hist=np.load(OUT/'training_trajectories.npz');samples=np.load(OUT/'training_samples.npz');banks=np.load(q.LONG/'input_banks.npz');bud=np.load(OUT/'fixed_budgets.npz')
    replay=[];gradient_error=0.;projection_error=0.;norm_error=0.
    for seed in r.SEEDS:
        for arm in q.ARMS:
            key=f'{seed}_{arm}';target=hist[key+'_parameters'];stored=hist[key+'_moments_ema'];theta=target[0].copy();m=np.zeros(9);v=m.copy();ema=theta.copy()
            rng=np.random.default_rng(np.random.SeedSequence([2026111056,seed]));maxerr=0.
            for n in range(100):
                ts,xs=banks[f'{seed}_t'][n],banks[f'{seed}_x'][n]
                # Use saved pre-boundary for sample comparisons; independent recurrence is checked below.
                F=np.stack([f.features[int(t)][int(x)] for t,x in zip(ts,xs)]);p=expit(F@target[n]);bits=(rng.random((8,64,6))<p[:,None,:]).astype(np.uint8)
                np.testing.assert_array_equal(bits,f.B[samples[key+'_ids'][n]])
                costs=C[samples[key+'_ids'][n]];adv=costs-(costs.sum(1,keepdims=True)-costs)/63
                score=np.einsum('bke,bed->bkd',bits-p[:,None,:],F);G=np.einsum('bkj,bkd->jd',adv,score)/(8*64)
                gradient_error=max(gradient_error,float(abs(G-samples[key+'_estimated_gradients'][n]).max()));assert gradient_error<1e-12
                if arm=='common':
                    lam,_=nnls(G.T,-G.mean(0),maxiter=1000);g=G.mean(0)+G.T@lam
                    co,_=r.project_mean(G@G.T);projection_error=max(projection_error,float(abs(g-G.T@co).max()));assert projection_error<1e-8
                else:g=G.mean(0)
                decay=-opt['lr']*opt['weight_decay']*theta
                if np.linalg.norm(g)<=1e-14:delta=decay
                else:
                    a=float(g@g);b=-2*float(g@decay);c=float(decay@decay)-bud[str(seed)][n]**2
                    scale=(-b+np.sqrt(b*b-4*a*c))/(2*a);delta=-scale*g+decay
                m=.9*m+.1*g;v=.999*v+.001*g*g;theta+=delta;ema=.99*ema+.01*theta
                err=max(float(abs(theta-target[n+1]).max()),float(abs(np.stack([m,v,ema])-stored[n+1]).max()));maxerr=max(maxerr,err);assert err<1e-10
                norm_error=max(norm_error,abs(np.linalg.norm(target[n+1]-target[n])-bud[str(seed)][n]))
                if n+1 in (20,50,100):
                    z=torch.load(OUT/f'{seed}_{arm}_u{n+1}.pt',weights_only=False);assert z['hard_rng']==rng.bit_generator.state
                    assert z['cursor']==n+1 and z['optimizer']['state'][0]['step'].item()==n+1 and z['bank_hash']==prot['input_hash']
                    np.testing.assert_array_equal(z['parameter'].numpy(),target[n+1]);np.testing.assert_array_equal(z['ema'],stored[n+1,2])
                    for k,col in [('exp_avg',0),('exp_avg_sq',1)]:np.testing.assert_array_equal(z['optimizer']['state'][0][k].numpy(),stored[n+1,col])
            init=torch.load(OUT/f'{seed}_{arm}_u0.pt',weights_only=False);end=torch.load(OUT/f'{seed}_{arm}_u100.pt',weights_only=False)
            assert init['python_rng']==end['python_rng'] and torch.equal(init['torch_rng'],end['torch_rng'])
            for a,b in zip(init['numpy_rng'],end['numpy_rng']):np.testing.assert_array_equal(a,b)
            replay.append(dict(seed=seed,arm=arm,max_state_error=maxerr))
    flat=[dict(seed=x['seed'],direction=x['arm'],feedback='finite64',updates=x['updates'],readout=x['readout'],**x['terminal']) for x in finals]
    for x in json.loads((q.LONG/'finals.json').read_text()):flat.append(dict(seed=x['seed'],direction=x['arm'],feedback='exact',updates=x['updates'],readout=x['readout'],**x['terminal']))
    initial=[x for x in json.loads((q.BASE/'finals.json').read_text()) if x['arm']=='initial'];agg=[];diffs=[];keys=list(finals[0]['terminal'])
    for feedback in ('exact','finite64'):
        for arm in q.ARMS:
            for n in (20,50,100):
                for rd in ('raw','EMA'):
                    rr=[x for x in flat if (x['feedback'],x['direction'],x['updates'],x['readout'])==(feedback,arm,n,rd)];z=dict(feedback=feedback,direction=arm,updates=n,readout=rd,n_initializations=len(rr))
                    for k in keys:
                        vals=np.array([x[k] for x in rr]);z[k]=float(vals.mean());z[k+'_SD']=float(vals.std(ddof=1))
                    agg.append(z)
    for x in flat:
        if x['feedback']!='finite64':continue
        ref=next(y for y in flat if y['feedback']=='exact' and all(y[k]==x[k] for k in ('seed','direction','updates','readout')))
        diffs.append(dict(seed=x['seed'],direction=x['direction'],updates=x['updates'],readout=x['readout'],**{k:x[k]-ref[k] for k in keys}))
    geometry=[];updates=read('updates.json')
    for seed in r.SEEDS:
        for arm in q.ARMS:
            rr=[x for x in updates if x['seed']==seed and x['arm']==arm];dc=np.array([np.array(x['after_costs'])-x['before_costs'] for x in rr])
            geometry.append(dict(seed=seed,direction=arm,mean_cosine=float(np.mean([x['cosine'] for x in rr if x['cosine'] is not None])),
                zero_steps=sum(x['zero'] for x in rr),estimated_any_harm=sum(max(x['estimated_inner'])>1e-12 for x in rr),true_any_harm=sum(max(x['true_inner'])>1e-12 for x in rr),
                finite_harm_counts=(dc>1e-12).sum(0).tolist(),max_finite_increase=float(dc.max())))
    for path,h in prot['bindings'].items():assert hashlib.sha256(q.Path(path).read_bytes()).hexdigest()==h
    assert hashlib.sha256((OUT/'protocol.json').read_bytes()).hexdigest()==read('freeze.json')['protocol_SHA256']
    r.table('terminal_metrics.csv',flat);r.table('aggregate.csv',agg);r.table('finite_minus_exact.csv',diffs);r.table('geometry.csv',geometry)
    r.dump('summary.json',dict(aggregate=agg,per_seed=flat,finite_minus_exact=diffs,initial=initial,geometry=geometry))
    result=dict(passed=True,seconds=time.time()-start,RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,chains=checks,replay=replay,
        independent_LOO_error=gradient_error,independent_NNLS_error=projection_error,max_norm_error=norm_error,bindings_unchanged=True,hard_rng_reproduced=True,GPU_hours=0)
    r.dump('independent_audit.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ('chains','replay')}))
if __name__=='__main__':main()
