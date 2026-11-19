import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import importlib.util,json,hashlib,time,resource,signal
from pathlib import Path
import numpy as np
from scipy.special import expit
start=time.time();signal.alarm(60)
sp=importlib.util.spec_from_file_location('run',Path(__file__).with_name('run.py'));m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
p=m.OUT;f,C,_,_,_=m.r.setup();f=m.old.m.a.m.augment(f);h=np.load(m.ROOT/'results/terminal_common_train_20261111/histories.npz');data=np.load(m.ROOT/'results/terminal_mc_reliability_20261111/sample_evidence.npz');V=np.load(p/'oracle_values.npz');rows=json.loads((p/'estimates.json').read_text());err=0.;nchecks=0
for seed in m.r.SEEDS:
    for step in (0,20):
        key=f'{seed}_{step}';w=h[f'{seed}_common_parameters'][step];vv=V[key]
        # Independent direct Bayes formula checks values at every time.
        for t in range(1,101):
            pred=expit(f.features[t]@w);pi=f.pi[f.f];ratio=f.alpha[t]/f.alpha[t-1];obs=f.B
            l1=ratio*obs+(1-ratio)*np.where(obs>0,pi,1-pi);l0=ratio*(1-obs)+(1-ratio)*np.where(obs>0,pi,1-pi);qs=[]
            for c in (0,1):
                pre=f.alpha[t-1]*c+(1-f.alpha[t-1])*pi;qs.append(pre*l1/(pre*l1+(1-pre)*l0))
            q=(1-pred)*qs[0]+pred*qs[1];P=np.prod(np.where(f.B[None,:,:]>0,q[:,None,:],1-q[:,None,:]),axis=-1)
            np.testing.assert_allclose(vv[t],P@vv[t-1],atol=1e-12,rtol=1e-12)
        X=data[key+'_states'];S=data[key+'_scores'];Y=data[key+'_endpoints']
        for rep in (0,31):
            # Independent scalar-path scores on first32 trajectories; no cache score reuse.
            contributions=[]
            for j in range(32):
                v=np.zeros((5,10));cost=C[Y[rep,j]]
                for t in range(100,0,-1):
                    x=int(X[rep,j,100-t]);y=int(X[rep,j,101-t]);sc=m.old.score(f,w,t,x,y);v+=(cost-vv[t,x])[:,None]*sc
                contributions.append(v)
            got=np.mean(contributions,axis=0);row=next(z for z in rows if z['seed']==seed and z['update']==step and z['repeat']==rep and z['K']==32 and z['method']=='exact_conditional_value')
            e=float(abs(got-np.array(row['gradient'])).max());err=max(err,e);assert e<1e-11;nchecks+=1
for path,digest in json.loads((p/'protocol.json').read_text())['bindings'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
result=dict(passed=True,independent_Bayes_value_checks=600,scalar_32path_estimator_checks=nchecks,max_estimator_error=err,bindings_unchanged=True,seconds=time.time()-start,RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,GPU_hours=0)
(p/'audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
