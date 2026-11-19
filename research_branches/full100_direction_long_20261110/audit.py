"""Independent NumPy update replay and direct production-kernel verification."""
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
    start=time.time();assert read('receipt.json')['status']=='complete';f,C,tri,cfg,kernel=r.setup()
    prot=read('protocol.json');opt=prot['optimizer'];saved=np.load(OUT/'states_and_paths.npz');ix=read('state_index.json');finals=read('finals.json')
    uniforms=np.load(q.BASE/'sampling_uniforms.npz');chains=[]
    for j,meta in enumerate(ix):
        theta=saved['parameters'][j];dist=f.prior.copy();u=uniforms[str(meta['seed'])];st=np.searchsorted(np.cumsum(dist),u[0],side='right');np.testing.assert_array_equal(st,saved['paths'][j,0])
        for t in range(100,0,-1):
            p=expit(f.features[t]@theta);z=f.kernel.reverse(p.ravel(),f.B.ravel(),np.tile(f.f,f.S),t-1,t).reshape(f.S,f.M)
            P=np.prod(np.where(f.B[None,:,:]>0,z[:,None,:],1-z[:,None,:]),axis=2);assert abs(P.sum(1)-1).max()<1e-12
            cum=np.cumsum(P[st],axis=1);cum[:,-1]=1.;st=(u[101-t,:,None]>=cum).sum(1);np.testing.assert_array_equal(st,saved['paths'][j,101-t]);dist=dist@P
        err=float(abs(dist-saved['distributions'][j]).max());assert err<1e-12
        row=next(x for x in finals if all(x[k]==v for k,v in meta.items()))
        for k,v in r.metrics(f,C,tri,dist).items():assert abs(row['terminal'][k]-v)<1e-12
        chains.append(dict(**meta,max_probability_error=err,paths_verified=1024))
    banks=np.load(OUT/'input_banks.npz');oldbank=np.load(q.BASE/'input_banks.npz');hist=np.load(OUT/'training_trajectories.npz');replays=[];fds=[];normerrs=[]
    for seed in r.SEEDS:
        for typ in ('t','x'):np.testing.assert_array_equal(banks[f'{seed}_{typ}'][:20],oldbank[f'{seed}_{typ}'])
        rng=np.random.default_rng(np.random.SeedSequence([2026111052,seed]));tt=rng.integers(1,101,size=(80,8));uu=rng.random((80,8,f.M))
        np.testing.assert_array_equal(tt,banks[f'{seed}_t'][20:])
        for j in range(80):
            for k,t in enumerate(tt[j]):
                p=f.alpha[t]*f.ref+(1-f.alpha[t])*f.pi[f.f]
                np.testing.assert_array_equal((uu[j,k]<p).astype(float),f.B[banks[f'{seed}_x'][j+20,k]])
        common=hist[f'{seed}_common_parameters'];norms=np.linalg.norm(np.diff(common,axis=0),axis=1)
        for arm in q.ARMS:
            target=hist[f'{seed}_{arm}_parameters'];stored=hist[f'{seed}_{arm}_moments_ema'];z=torch.load(q.parent(seed,arm),weights_only=False)
            theta=z['parameter'].numpy().copy();state=z['optimizer']['state'][0];m=state['exp_avg'].numpy().copy();v=state['exp_avg_sq'].numpy().copy();ema=z['ema'].copy()
            assert state['step'].item()==20;np.testing.assert_array_equal(theta,target[0]);maximum=0.
            for j in range(80):
                n=j+20;ts,xs=banks[f'{seed}_t'][n],banks[f'{seed}_x'][n];_,G,_,_=r.atomic(f,C,theta,ts,xs)
                co,_=r.project_mean(G@G.T);g=G.mean(0) if arm=='mean' else G.T@co
                if arm=='mean' and j in (0,29,79):
                    for d in range(9):
                        h=np.eye(9)[d]*1e-5
                        fds.append(abs((r.atomic(f,C,theta+h,ts,xs)[0].mean()-r.atomic(f,C,theta-h,ts,xs)[0].mean())/2e-5-g[d]))
                m=.9*m+.1*g;v=.999*v+.001*g*g;decay=-opt['lr']*opt['weight_decay']*theta
                adaptive=-opt['lr']*(m/(1-.9**(n+1)))/(np.sqrt(v/(1-.999**(n+1)))+opt['eps'])
                if arm=='common':
                    # Mirror actual floating-point native subtraction, using independent Adam equations.
                    native=theta*(1-opt['lr']*opt['weight_decay'])+adaptive-theta
                    dd=-g/np.linalg.norm(g)*np.linalg.norm(native-decay)+decay if np.linalg.norm(g) else decay
                else:
                    a=float(g@g);b=-2*float(g@decay);c=float(decay@decay)-norms[j]**2
                    scale=(-b+np.sqrt(b*b-4*a*c))/(2*a);dd=-scale*g+decay
                theta=theta+dd;ema=.99*ema+.01*theta
                error=max(float(abs(theta-target[j+1]).max()),float(abs(np.stack([m,v,ema])-stored[j+1]).max()));maximum=max(maximum,error);assert error<1e-10
                if arm=='mean':normerrs.append(abs(np.linalg.norm(target[j+1]-target[j])-norms[j]))
            for n in (20,50,100):
                end=torch.load(OUT/f'{seed}_{arm}_u{n}.pt',weights_only=False);assert end['cursor']==n and end['optimizer']['state'][0]['step'].item()==n
                assert end['bank_hash']==prot['input_hash'];np.testing.assert_array_equal(end['parameter'].numpy(),target[n-20]);np.testing.assert_array_equal(end['ema'],stored[n-20,2])
                for key,col in [('exp_avg',0),('exp_avg_sq',1)]:np.testing.assert_array_equal(end['optimizer']['state'][0][key].numpy(),stored[n-20,col])
                assert end['python_rng']==z['python_rng'] and torch.equal(end['torch_rng'],z['torch_rng'])
                for a,b in zip(end['numpy_rng'],z['numpy_rng']):np.testing.assert_array_equal(a,b)
            replays.append(dict(seed=seed,arm=arm,max_state_error=maximum))
    assert max(fds)<1e-8 and max(normerrs)<1e-13
    for p,h in prot['bindings'].items():assert hashlib.sha256(q.Path(p).read_bytes()).hexdigest()==h
    assert hashlib.sha256((OUT/'protocol.json').read_bytes()).hexdigest()==read('freeze.json')['protocol_SHA256']
    flat=[dict(seed=x['seed'],arm=x['arm'],updates=x['updates'],readout=x['readout'],**x['terminal']) for x in finals]
    initial=[x for x in json.loads((q.BASE/'finals.json').read_text()) if x['arm']=='initial'];keys=list(finals[0]['terminal']);agg=[];deltas=[]
    for arm in q.ARMS:
        for n in (20,50,100):
            for rd in ('raw','EMA'):
                rr=[x for x in flat if (x['arm'],x['updates'],x['readout'])==(arm,n,rd)];z=dict(arm=arm,updates=n,readout=rd,n=3)
                for k in keys:
                    v=np.array([x[k] for x in rr]);z[k]=float(v.mean());z[k+'_SD']=float(v.std(ddof=1))
                agg.append(z)
    for x in flat:
        base=next(y['terminal'] for y in initial if y['seed']==x['seed'])
        deltas.append(dict(seed=x['seed'],arm=x['arm'],updates=x['updates'],readout=x['readout'],**{k:x[k]-base[k] for k in keys}))
    geometry=[]
    for seed in r.SEEDS:
        for arm in q.ARMS:
            rr=[x for x in read('updates.json') if x['seed']==seed and x['arm']==arm];diff=np.array([np.array(x['after_costs'])-x['before_costs'] for x in rr])
            geometry.append(dict(seed=seed,arm=arm,mean_norm=float(np.mean([x['actual_norm'] for x in rr])),first_order_harm=(np.array([x['inner'] for x in rr])>1e-12).sum(0).tolist(),finite_harm=(diff>1e-12).sum(0).tolist(),max_finite_increase=float(diff.max())))
    r.table('terminal_metrics.csv',flat);r.table('aggregate.csv',agg);r.table('minus_initial.csv',deltas);r.table('geometry.csv',geometry)
    r.dump('summary.json',dict(aggregate=agg,per_seed=flat,minus_initial=deltas,geometry=geometry,initial=initial))
    result=dict(passed=True,seconds=time.time()-start,RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,chains=chains,replays=replays,
        finite_difference_checks=len(fds),max_finite_difference=float(max(fds)),max_norm_error=float(max(normerrs)),inputs_reconstructed=True,bindings_unchanged=True,GPU_hours=0)
    r.dump('independent_audit.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ('chains','replays')}))
if __name__=='__main__':main()
