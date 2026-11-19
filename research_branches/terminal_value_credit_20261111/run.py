"""Frozen-policy conditional-value credit, archived finite paths only."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[k]='1'
import importlib.util,json,time,resource,signal,hashlib,traceback
from pathlib import Path
import numpy as np
from scipy.special import expit
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/terminal_value_credit_20261111'
sp=importlib.util.spec_from_file_location('parent',ROOT/'research_branches/terminal_state_baseline_20261111/run.py');p=importlib.util.module_from_spec(sp);sp.loader.exec_module(p)
r=p.r;old=p.old
def main():
    start=time.time();OUT.mkdir(exist_ok=False);r.OUT=OUT;signal.alarm(180);resource.setrlimit(resource.RLIMIT_AS,(2*1024**3,2*1024**3))
    rows=[];checks=[];receipt=dict(status='running',GPU_hours=0,new_paths=0,updates=0,production_calls=0)
    try:
        src=ROOT/'results/terminal_mc_reliability_20261111';vs=ROOT/'results/terminal_state_baseline_20261111'
        for x in [Path(__file__),Path(__file__).with_name('audit.py'),src/'sample_evidence.npz',src/'truth.json',vs/'oracle_values.npz',vs/'audit.json',ROOT/'results/terminal_common_train_20261111/histories.npz',ROOT/'research_branches/terminal_state_baseline_20261111/run.py']:r.bind(x)
        assert json.loads((vs/'audit.json').read_text())['passed']
        r.dump('protocol.json',dict(track='CORE terminal credit feasibility',formula='mean_paths sum_t (V[t-1](X[t-1])-V[t](X[t])) * grad log P_theta(X[t-1]|X[t]); V is exact frozen-policy terminal cost-to-go, detached',conditions=6,K=[32,128,512],repeats=32,new_paths=0,new_updates=0,comparison='Same archived paths and original five terminal costs, previous LOO and exact-value baseline retained',decision='For practical consideration require at K128 MSE improvement in all six states, pooled >=20% reduction vs LOO and no pooled true common-descent decrease. Oracle is not a production implementation. No full training or larger K.',limits=dict(CPU_seconds=180,threads=1,memory_GiB=2,GPU=0),bindings=r.BIND))
        ph=hashlib.sha256((OUT/'protocol.json').read_bytes()).hexdigest();r.dump('freeze.json',dict(epoch=time.time(),sha256=ph))
        ev=np.load(src/'sample_evidence.npz');truth=json.loads((src/'truth.json').read_text());values=np.load(vs/'oracle_values.npz');hist=np.load(ROOT/'results/terminal_common_train_20261111/histories.npz')
        f,C,*_=r.setup();f=old.m.a.m.augment(f);path_est={}
        for item in truth:
            seed,step=item['seed'],item['update'];key=f'{seed}_{step}';w=hist[f'{seed}_common_parameters'][step];V=values[key];G=np.array(item['gradient']);gn=np.linalg.norm(G,axis=1);U=G/gn[:,None];dt=np.array(item['common_certificate']['direction']);cache={};trans={};expected=np.zeros_like(G);occ=f.prior.copy()
            for t in range(1,101):
                F=f.features[t];prob=expit(F@w);diff=f.k[t][1]-f.k[t][0];q=f.k[t][0]+prob*diff;dq=diff*prob*(1-prob)
                P=np.prod(np.where(f.B[None,:,:]>0,q[:,None,:],1-q[:,None,:]),axis=-1)
                S=np.einsum('ije,ied->ijd',(f.B[None,:,:]-q[:,None,:])/(q*(1-q))[:,None,:],dq[:,:,None]*F)
                np.testing.assert_allclose(P@V[t-1],V[t],atol=1e-12,rtol=1e-12);cache[t]=S;trans[t]=P
            for t in range(100,0,-1):
                advantage=V[t-1][None,:,:]-V[t][:,None,:]
                expected+=np.einsum('i,ij,ijc,ijd->cd',occ,trans[t],advantage,cache[t]);occ=occ@trans[t]
            np.testing.assert_allclose(expected,G,atol=2e-11,rtol=2e-10)
            X=ev[key+'_states'];pg=np.zeros((32,512,5,10))
            for rep in range(32):
                for t in range(100,0,-1):
                    x=X[rep,:,100-t];y=X[rep,:,101-t];pg[rep]+=(V[t-1][y]-V[t][x])[:,:,None]*cache[t][x,y][:,None,:]
                for K in (32,128,512):
                    g=pg[rep,:K].mean(0);en=np.linalg.norm(g,axis=1);d=np.zeros(10);cert={};status='zero_gradient_row'
                    if min(en)>0:cert=old.m.s.simplex_min_norm(g/en[:,None]);status=cert['status'];d=np.array(cert['direction'])
                    slopes=U@d
                    rows.append(dict(seed=seed,update=step,repeat=rep,K=K,method='exact_value_credit',gradient=g.tolist(),relative_error=float(np.linalg.norm(g-G)/np.linalg.norm(G)),relative_row_errors=(np.linalg.norm(g-G,axis=1)/gn).tolist(),status=status,certificate=cert,direction=d.tolist(),cos_to_exact_common=old.cos(d,dt),true_normalized_slopes=slopes.tolist(),true_any_harm=bool(max(slopes)>1e-8),true_all_improve=bool(max(slopes)<-1e-8),true_mean_slope=float(G.mean(0)@d)))
            path_est[key]=pg;checks.append(dict(seed=seed,update=step,expected_gradient_max_error=float(abs(expected-G).max()),bellman=True));r.dump('progress.json',dict(conditions=len(checks),seconds=time.time()-start))
        np.savez_compressed(OUT/'path_gradients.npz',**path_est)
        for name,h in r.BIND.items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==h
        assert hashlib.sha256((OUT/'protocol.json').read_bytes()).hexdigest()==ph
        receipt.update(status='complete',conditions=6,estimates=len(rows),bindings_unchanged=True)
    except BaseException as e:receipt.update(status='failed',error=repr(e),traceback=traceback.format_exc())
    finally:
        r.dump('estimates.json',rows);r.dump('checks.json',checks);receipt.update(seconds=time.time()-start,RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024);r.dump('receipt.json',receipt);print(json.dumps(receipt))
    return 0 if receipt['status']=='complete' else 1
if __name__=='__main__':raise SystemExit(main())
