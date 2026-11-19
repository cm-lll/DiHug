"""Exact clean-training versus terminal-risk gradients. Frozen CPU toy states only."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import hashlib,importlib.util,json,resource,signal,time,traceback
from pathlib import Path
import numpy as np
from scipy.special import expit
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/toy_gradient_alignment_20261111'
spec=importlib.util.spec_from_file_location('typed',ROOT/'research_branches/toy_typed_endpoint_20261110/run.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
r=m.r
NAMES=['relation_count','joint_degree','triangle001','triangle011','component_cost']
def cosine(a,b):
    den=np.linalg.norm(a)*np.linalg.norm(b);return None if den==0 else float(a@b/den)
def full(f,C,w):
    P={};DP={};local={};cost={}
    for t,F in f.features.items():
        p=expit(F@w);q=f.k[t][0]+p*(f.k[t][1]-f.k[t][0])
        P[t]=np.prod(np.where(f.B[None,:,:]>0,q[:,None,:],1-q[:,None,:]),axis=2)
        cp=np.prod(np.where(f.B[None,:,:]>0,p[:,None,:],1-p[:,None,:]),axis=2)
        cost[t]=cp@C
        dl=np.einsum('ij,jk,ije->ike',cp,C,f.B[None,:,:]-p[:,None,:])
        local[t]=np.einsum('ike,ied->ikd',dl,F)
        dq=(f.k[t][1]-f.k[t][0])*p*(1-p)
        score=np.einsum('ije,ied->ijd',(f.B[None,:,:]-q[:,None,:])/(q*(1-q))[:,None,:],dq[:,:,None]*F)
        DP[t]=P[t][:,:,None]*score
    occ={100:f.prior.copy()};der=np.zeros((64,len(w)))
    for t in range(100,0,-1):
        der=P[t].T@der+np.einsum('i,ijd->jd',occ[t],DP[t]);occ[t-1]=occ[t]@P[t]
    Gterm=C.T@der;V=C.copy();stages=[];Gadj=np.zeros_like(Gterm);Gtrain=np.zeros_like(Gterm);Gon=np.zeros_like(Gterm);Ltrain=np.zeros(5)
    for t in range(1,101):
        Gt=np.einsum('i,jk,ijd->kd',occ[t],V,DP[t])
        Gf=np.einsum('i,ikd->kd',f.qref[t],local[t])/100
        Go=np.einsum('i,ikd->kd',occ[t],local[t])/100
        Gadj+=Gt;Gtrain+=Gf;Gon+=Go;Ltrain+=f.qref[t]@cost[t]/100
        stages.append(dict(t=t,occupancy_TV=float(abs(occ[t]-f.qref[t]).sum()/2),terminal_gradient=Gt.tolist(),forward_clean_gradient=Gf.tolist(),frozen_onpolicy_clean_gradient=Go.tolist()))
        V=P[t]@V
    assert np.max(abs(Gadj-Gterm))<1e-11
    return dict(terminal=occ[0]@C,forward_clean=Ltrain,Gterm=Gterm,Gtrain=Gtrain,Gon=Gon,stages=stages,adjoint_error=float(abs(Gadj-Gterm).max()),distribution=occ[0])
def values(f,C,w):
    dist=f.prior.copy();clean=np.zeros(5)
    for t in range(100,0,-1):
        p=expit(f.features[t]@w);q=f.k[t][0]+p*(f.k[t][1]-f.k[t][0])
        # Separate product implementation, no derivative arrays or training helper.
        P=np.prod(np.where(f.B[None,:,:]>0,q[:,None,:],1-q[:,None,:]),axis=-1)
        cp=np.prod(np.where(f.B[None,:,:]>0,p[:,None,:],1-p[:,None,:]),axis=-1)
        dist=dist@P;clean+=f.qref[t]@cp@C/100
    return dist@C,clean
def main():
    start=time.time();OUT.mkdir(exist_ok=False);r.OUT=OUT
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('600s CPU limit')));signal.alarm(600)
    resource.setrlimit(resource.RLIMIT_AS,(2*1024**3,2*1024**3))
    rows=[];stages=[];probes=[];checks=[];receipt=dict(status='running',GPU_hours=0,training_updates=0,new_sampled_paths=0,production_calls=0)
    try:
        f,C,tri,cfg,_=r.setup();models={'original':f,'typed_endpoint':m.augment(f)}
        src=ROOT/'results/toy_typed_endpoint_20261110';paths=[Path(__file__),Path(__file__).with_name('watch.py'),ROOT/'research_branches/toy_typed_endpoint_20261110/run.py',ROOT/'research_branches/full100_toy_20261110/run.py',src/'histories.npz',src/'protocol.json',src/'audit.json',src/'distributions.npz']
        for p in paths:r.bind(p)
        assert json.loads(paths[-2].read_text())['passed']
        hist=np.load(src/'histories.npz');saved=np.load(src/'distributions.npz')
        protocol=dict(track='CORE finite objective-alignment diagnostic',seeds=r.SEEDS,arms=list(models),updates=[0,20,50,100],readout='raw only, no inference about EMA',conditions=24,risks=NAMES,time_distribution='exact uniform 1..100',inputs='all 64 forward states with exact q(Xt|X0) weights; no Monte Carlo',comparison='forward clean objective versus complete T100 terminal objective; frozen-onpolicy clean gradient is stop-gradient diagnostic only',probes=dict(norms=[.001,.0001],directions=['negative exact forward clean mean','negative terminal mean, same norm'],finite_difference_h=1e-5),limits=dict(CPU_threads=1,seconds=600,RSS_MiB=2048,GPU=0),bindings=r.BIND)
        r.dump('protocol.json',protocol);ph=hashlib.sha256((OUT/'protocol.json').read_bytes()).hexdigest();r.dump('freeze.json',dict(epoch=time.time(),protocol_sha256=ph))
        for seed in r.SEEDS:
            for arm,fm in models.items():
                for n in protocol['updates']:
                    meta=dict(seed=seed,arm=arm,update=n);w=hist[f'{seed}_{arm}_parameters'][n];v=full(fm,C,w)
                    np.testing.assert_allclose(v['distribution'],saved[f'{seed}_{arm}_{n}_raw'],rtol=1e-12,atol=1e-12)
                    gt=v['Gtrain'].mean(0);gz=v['Gterm'].mean(0);go=v['Gon'].mean(0)
                    row=dict(**meta,terminal=v['terminal'].tolist(),forward_clean=v['forward_clean'].tolist(),Gtrain=v['Gtrain'].tolist(),Gterm=v['Gterm'].tolist(),Gon=v['Gon'].tolist(),cos_train_terminal=cosine(gt,gz),cos_onpolicy_terminal=cosine(go,gz),train_norm=float(np.linalg.norm(gt)),terminal_norm=float(np.linalg.norm(gz)),terminal_slopes_under_train=(-v['Gterm']@gt).tolist(),per_cost_cos=[cosine(a,b) for a,b in zip(v['Gtrain'],v['Gterm'])],terminal_vs_local_scale=float(np.linalg.norm(gz)/np.linalg.norm(gt)))
                    rows.append(row)
                    for stage in v['stages']:stages.append(dict(**meta,**stage))
                    errs=[]
                    for label,vec in [('train',gt),('terminal',gz)]:
                        d=-vec/np.linalg.norm(vec);h=1e-5
                        plus=values(fm,C,w+h*d);minus=values(fm,C,w-h*d)
                        e1=float(abs((plus[0]-minus[0])/(2*h)-v['Gterm']@d).max());e2=float(abs((plus[1]-minus[1])/(2*h)-v['Gtrain']@d).max());errs.extend([e1,e2]);assert max(e1,e2)<1e-7
                        for length in (.001,.0001):
                            z,l=values(fm,C,w+length*d)
                            probes.append(dict(**meta,direction=label,norm=length,terminal_change=(z-v['terminal']).tolist(),forward_clean_change=(l-v['forward_clean']).tolist(),terminal_mean_change=float((z-v['terminal']).mean()),forward_mean_change=float((l-v['forward_clean']).mean()),first_order_terminal=(length*v['Gterm']@d).tolist()))
                    checks.append(dict(**meta,adjoint_error=v['adjoint_error'],finite_difference_error=max(errs),saved_distribution_match=True))
                    r.dump('progress.json',dict(conditions=len(rows),seconds=time.time()-start))
        for p,h in r.BIND.items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
        assert hashlib.sha256((OUT/'protocol.json').read_bytes()).hexdigest()==ph
        receipt.update(status='complete',conditions=len(rows),finite_probes=len(probes),finite_difference_direction_checks=48,bindings_unchanged=True)
    except BaseException as e:receipt.update(status='failed',error=repr(e),traceback=traceback.format_exc())
    finally:
        r.dump('conditions.json',rows);r.dump('stages.json',stages);r.dump('probes.json',probes);r.dump('checks.json',checks)
        receipt.update(seconds=time.time()-start,RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024);r.dump('receipt.json',receipt);print(json.dumps(receipt))
    return 0 if receipt['status']=='complete' else 1
if __name__=='__main__':raise SystemExit(main())
