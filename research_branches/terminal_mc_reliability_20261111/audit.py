import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import importlib.util,json,hashlib,time,resource,signal
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
start=time.time();signal.alarm(60)
sp=importlib.util.spec_from_file_location('exp',Path(__file__).with_name('run.py'));m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
p=m.OUT;records=json.loads((p/'estimates.json').read_text());data=np.load(p/'sample_evidence.npz');f,C,_,_,_=m.r.setup();f=m.m.a.m.augment(f)
hist=np.load(m.ROOT/'results/terminal_common_train_20261111/histories.npz');maxerr=0.;scoreerr=0.;qp=0
for row in records:
    key=f"{row['seed']}_{row['update']}";rep=row['repeat'];k=row['K'];Y=data[key+'_endpoints'][rep,:k];S=data[key+'_scores'][rep,:k];c=C[Y]
    # Direct leave-one-out baseline, independently of anchored implementation.
    g=np.mean((c-(c.sum(0)-c)/(k-1))[:,:,None]*S[:,None,:],axis=0)
    err=float(abs(g-np.array(row['gradient'])).max());maxerr=max(maxerr,err);assert err<1e-11
    if rep in (0,16,31) and row['status']=='strict_descent':
        U=g/np.linalg.norm(g,axis=1)[:,None]
        opt=minimize(lambda x:.5*np.linalg.norm(x@U)**2,np.ones(5)/5,jac=lambda x:U@(x@U),bounds=[(0,1)]*5,constraints=[{'type':'eq','fun':lambda x:x.sum()-1,'jac':lambda x:np.ones(5)}],method='SLSQP',options=dict(ftol=1e-14,maxiter=1000))
        assert opt.success and abs(np.linalg.norm(opt.x@U)**2-row['certificate']['margin']**2)<1e-8;qp+=1
for seed in m.r.SEEDS:
    for n in (0,20):
        key=f'{seed}_{n}';w=hist[f'{seed}_common_parameters'][n]
        for rep in (0,16,31):
            for path in (0,511):
                states=data[key+'_states'][rep,path];v=sum((m.score(f,w,100-j,int(states[j]),int(states[j+1])) for j in range(100)),np.zeros(10))
                err=float(abs(v-data[key+'_scores'][rep,path]).max());scoreerr=max(scoreerr,err);assert err<1e-11
for path,h in json.loads((p/'protocol.json').read_text())['bindings'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==h
result=dict(passed=True,estimates=len(records),independent_QPs=qp,path_score_checks=36,max_LOO_error=maxerr,max_scalar_path_score_error=scoreerr,bindings_unchanged=True,seconds=time.time()-start,RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,GPU_hours=0)
(p/'audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
