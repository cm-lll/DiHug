"""Paired oracle baseline study on archived hard paths; no model updates/new paths."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import importlib.util,json,hashlib,time,resource,signal,traceback
from pathlib import Path
import numpy as np
from scipy.special import expit
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/terminal_state_baseline_20261111'
spec=importlib.util.spec_from_file_location('previous',ROOT/'research_branches/terminal_mc_reliability_20261111/run.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
r=old.r
def main():
    start=time.time();OUT.mkdir(exist_ok=False);r.OUT=OUT;signal.alarm(180)
    resource.setrlimit(resource.RLIMIT_AS,(2*1024**3,2*1024**3));rows=[];checks=[];receipt=dict(status='running',GPU_hours=0,network_calls=0,updates=0,new_paths=0)
    try:
        source=ROOT/'results/terminal_mc_reliability_20261111';evidence=np.load(source/'sample_evidence.npz');truth=json.loads((source/'truth.json').read_text());previous=json.loads((source/'estimates.json').read_text())
        hist=np.load(ROOT/'results/terminal_common_train_20261111/histories.npz');f,C,tri,_,_=r.setup();f=old.m.a.m.augment(f)
        paths=[Path(__file__),Path(__file__).with_name('audit.py'),source/'protocol.json',source/'sample_evidence.npz',source/'truth.json',source/'estimates.json',source/'audit.json',ROOT/'research_branches/terminal_mc_reliability_20261111/run.py',ROOT/'research_branches/terminal_common_direction_20261111/run.py',ROOT/'results/terminal_common_train_20261111/histories.npz']
        for p in paths:r.bind(p)
        assert json.loads((source/'audit.json').read_text())['passed']
        protocol=dict(track='CORE variance-reduction feasibility',conditions=6,K=[32,128,512],repeats=32,methods=['LOO_reuse','exact_population_constant','exact_conditional_value'],trajectories='identical archived paths, no new sampling',definition='b_t(x)=E_theta[C(X0)|Xt=x]; use full actual transition score; baseline detached and independent of current action',control='constant oracle distinguishes state dependence from knowledge of exact terminal mean',known_limits='oracle value unavailable on production graphs; not guaranteed variance optimal',continuation_rule='At K128 require paired gradient MSE improvement in all six states and at least 20% pooled reduction versus LOO, no loss of pooled true common-descent rate; otherwise do not deploy this candidate. Engineering resource screen, not significance test.',real_stage='Only after technical evidence supports candidate, check available assets and resources; no fabricated oracle on real graph. No GPU if driver unavailable.',limits=dict(CPU_seconds=180,threads=1,RSS_MiB=2048,overall_wall_minutes=90,max_GPU_hours=1),bindings=r.BIND)
        r.dump('protocol.json',protocol);ph=hashlib.sha256((OUT/'protocol.json').read_bytes()).hexdigest();r.dump('freeze.json',dict(epoch=time.time(),protocol_sha256=ph))
        r.dump('window.json',dict(start='2026-11-11T01:26:29+08:00',deadline='2026-11-11T02:56:29+08:00',GPU_hours_cap=1,GPU_allocated=False,host='GPU-Node01',resource_error='nvidia-smi cannot communicate with NVIDIA driver; no driver mutations'))
        gradients={};baselines={}
        for target in truth:
            seed,step=target['seed'],target['update'];key=f'{seed}_{step}';w=hist[f'{seed}_common_parameters'][step];G=np.array(target['gradient']);gn=np.linalg.norm(G,axis=1);U=G/gn[:,None];dt=np.array(target['common_certificate']['direction'])
            cache={};V={0:C.copy()};maxzero=0.
            for t in range(1,101):
                F=f.features[t];p=expit(F@w);diff=f.k[t][1]-f.k[t][0];q=f.k[t][0]+p*diff;dq=diff*p*(1-p)
                P=np.prod(np.where(f.B[None,:,:]>0,q[:,None,:],1-q[:,None,:]),axis=-1)
                Sc=np.einsum('ije,ied->ijd',(f.B[None,:,:]-q[:,None,:])/(q*(1-q))[:,None,:],dq[:,:,None]*F)
                V[t]=P@V[t-1];cache[t]=Sc;maxzero=max(maxzero,float(abs(np.einsum('ij,ijd->id',P,Sc)).max()))
            mu=f.prior@V[100];np.testing.assert_allclose(mu,target['terminal_costs'],atol=1e-12,rtol=1e-12);assert maxzero<1e-12
            baselines[key]=np.array([V[t] for t in range(101)])
            X=evidence[key+'_states'];SS=evidence[key+'_scores'];Y=evidence[key+'_endpoints'];estimates=[]
            for rep in range(32):
                correction=np.zeros((512,5,10));S=np.zeros((512,10))
                for t in range(100,0,-1):
                    x=X[rep,:,100-t];y=X[rep,:,101-t];sc=cache[t][x,y];S+=sc;correction+=V[t][x,:,None]*sc[:,None,:]
                np.testing.assert_allclose(S,SS[rep],atol=1e-12,rtol=1e-12)
                c=C[Y[rep]]
                for K in (32,128,512):
                    gs={'exact_population_constant':np.mean((c[:K]-mu)[:,:,None]*S[:K,None,:],axis=0),'exact_conditional_value':np.mean(c[:K,:,None]*S[:K,None,:]-correction[:K],axis=0)}
                    prior=next(x for x in previous if x['seed']==seed and x['update']==step and x['repeat']==rep and x['K']==K)
                    gs['LOO_reuse']=np.array(prior['gradient'])
                    for method,g in gs.items():
                        en=np.linalg.norm(g,axis=1);status='zero_gradient_row';d=np.zeros(10);cert={}
                        if min(en)>0:cert=old.m.s.simplex_min_norm(g/en[:,None]);status=cert['status'];d=np.array(cert['direction'])
                        slopes=U@d;err=g-G
                        rows.append(dict(seed=seed,update=step,repeat=rep,K=K,method=method,gradient=g.tolist(),relative_error=float(np.linalg.norm(err)/np.linalg.norm(G)),relative_row_errors=(np.linalg.norm(err,axis=1)/gn).tolist(),status=status,certificate=cert,direction=d.tolist(),cos_to_exact_common=old.cos(d,dt),true_normalized_slopes=slopes.tolist(),true_any_harm=bool(max(slopes)>1e-8),true_all_improve=bool(max(slopes)<-1e-8),true_mean_slope=float(G.mean(0)@d)))
            checks.append(dict(seed=seed,update=step,max_conditional_zero_score=maxzero,terminal_value_identity=True,archived_scores_match=True))
            r.dump('progress.json',dict(conditions=len(checks),seconds=time.time()-start))
        np.savez_compressed(OUT/'oracle_values.npz',**baselines)
        for p,h in r.BIND.items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
        assert hashlib.sha256((OUT/'protocol.json').read_bytes()).hexdigest()==ph
        receipt.update(status='complete',conditions=len(checks),estimates=len(rows),bindings_unchanged=True)
    except BaseException as e:receipt.update(status='failed',error=repr(e),traceback=traceback.format_exc())
    finally:
        r.dump('estimates.json',rows);r.dump('checks.json',checks);receipt.update(seconds=time.time()-start,RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024);r.dump('receipt.json',receipt);print(json.dumps(receipt))
    return 0 if receipt['status']=='complete' else 1
if __name__=='__main__':raise SystemExit(main())
