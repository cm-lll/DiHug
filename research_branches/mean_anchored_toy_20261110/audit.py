"""Independent replay using stored gradients and direct reverse-kernel transitions."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import json,time,hashlib,resource
import numpy as np
from scipy.special import expit
import run as q
r=q.r;OUT=q.OUT
def read(n):return json.loads((OUT/n).read_text())
def main():
    start=time.time();assert read('receipt.json')['status']=='complete';prot=read('protocol.json');f,C,tri,cfg,kernel=r.setup()
    hist=np.load(OUT/'training_trajectories.npz');updates=read('updates.json');banks=np.load(q.LONG/'input_banks.npz');maxgap=0.;chains=[]
    for seed in r.SEEDS:
        norm=np.linalg.norm(np.diff(hist[f'{seed}_cagrad_parameters'],axis=0),axis=1)
        for arm in q.ARMS:
            theta=hist[f'{seed}_{arm}_parameters'];ema=hist[f'{seed}_{arm}_ema'];rr=[x for x in updates if x['seed']==seed and x['arm']==arm]
            for j,row in enumerate(rr):
                np.testing.assert_array_equal(row['times'],banks[f'{seed}_t'][j]);np.testing.assert_array_equal(row['states'],banks[f'{seed}_x'][j])
                cost,G,ce,_=r.atomic(f,C,theta[j],row['times'],row['states']);np.testing.assert_allclose(G,row['gradient'],atol=1e-12,rtol=1e-12)
                delta=theta[j+1]-theta[j];m=G.mean(0)
                if arm=='cagrad':
                    d=-delta/.2;assert np.linalg.norm(d-m)<=.5*np.linalg.norm(m)+1e-9
                    w=np.array(row['certificate']['multipliers']);gw=w@G;upper=gw@m+.5*np.linalg.norm(m)*np.linalg.norm(gw);gap=upper-min(G@d);maxgap=max(maxgap,abs(float(gap)))
                    assert abs(gap)<=row['certificate']['tolerance']*(m@m)+1e-10
                else:
                    expected=-.2*m
                    if arm=='same_norm_average':expected*=norm[j]/np.linalg.norm(expected)
                    np.testing.assert_allclose(delta,expected,atol=1e-12,rtol=1e-10)
                np.testing.assert_allclose(ema[j+1],.99*ema[j]+.01*theta[j+1],atol=1e-14,rtol=1e-14)
    saved=np.load(OUT/'states_and_paths.npz');ix=read('state_index.json');finals=read('finals.json');uniforms=np.load(q.BASE/'sampling_uniforms.npz')
    for j,meta in enumerate(ix):
        theta=saved['parameters'][j];dist=f.prior.copy();u=uniforms[str(meta['seed'])];st=np.searchsorted(np.cumsum(dist),u[0],side='right');np.testing.assert_array_equal(st,saved['paths'][j,0])
        for t in range(100,0,-1):
            p=expit(f.features[t]@theta);z=f.kernel.reverse(p.ravel(),f.B.ravel(),np.tile(f.f,f.S),t-1,t).reshape(f.S,f.M)
            P=np.prod(np.where(f.B[None,:,:]>0,z[:,None,:],1-z[:,None,:]),axis=2);assert abs(P.sum(1)-1).max()<1e-12
            cum=np.cumsum(P[st],axis=1);cum[:,-1]=1.;st=(u[101-t,:,None]>=cum).sum(1);np.testing.assert_array_equal(st,saved['paths'][j,101-t]);dist=dist@P
        err=float(abs(dist-saved['distributions'][j]).max());assert err<1e-12
        row=next(x for x in finals if all(x[k]==v for k,v in meta.items()))
        for k,v in r.metrics(f,C,tri,dist).items():assert abs(row['terminal'][k]-v)<1e-12
        chains.append(dict(**meta,max_probability_error=err))
    for p,h in prot['bindings'].items():assert hashlib.sha256(q.Path(p).read_bytes()).hexdigest()==h
    assert hashlib.sha256((OUT/'protocol.json').read_bytes()).hexdigest()==read('freeze.json')['protocol_SHA256']
    flat=[dict(seed=x['seed'],arm=x['arm'],updates=x['updates'],readout=x['readout'],**x['terminal']) for x in finals];agg=[]
    keys=list(finals[0]['terminal'])
    for arm in q.ARMS:
        for n in (20,50,100):
            for rd in ('raw','EMA'):
                rr=[x for x in flat if (x['arm'],x['updates'],x['readout'])==(arm,n,rd)];z=dict(arm=arm,updates=n,readout=rd,n=3)
                for k in keys:
                    v=np.array([x[k] for x in rr]);z[k]=float(v.mean());z[k+'_SD']=float(v.std(ddof=1))
                agg.append(z)
    r.table('terminal_metrics.csv',flat);r.table('aggregate.csv',agg);r.dump('summary.json',dict(aggregate=agg,per_seed=flat))
    result=dict(passed=True,updates_replayed=len(updates),terminal_distributions=len(chains),sampled_paths=len(chains)*1024,max_dual_gap=maxgap,seconds=time.time()-start,RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,GPU_hours=0,bindings_unchanged=True)
    r.dump('audit.json',result);print(json.dumps(result))
if __name__=='__main__':main()
