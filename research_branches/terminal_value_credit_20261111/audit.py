"""Independent scalar Bayes / scalar path-score reconstruction."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import importlib.util,json,time,resource,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/terminal_value_credit_20261111'
sp=importlib.util.spec_from_file_location('worker',ROOT/'research_branches/terminal_value_credit_20261111/run.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
start=time.time();f,C,*_=m.r.setup();f=m.old.m.a.m.augment(f)
hist=np.load(ROOT/'results/terminal_common_train_20261111/histories.npz');ev=np.load(ROOT/'results/terminal_mc_reliability_20261111/sample_evidence.npz');vs=np.load(ROOT/'results/terminal_state_baseline_20261111/oracle_values.npz');pg=np.load(OUT/'path_gradients.npz');checks=[]
for key in vs.files:
    seed,step=map(int,key.split('_'));w=hist[f'{seed}_common_parameters'][step];V=vs[key];X=ev[key+'_states'];err=0.
    for rep,path in [(0,0),(17,31)]:
        total=np.zeros((5,10))
        for t in range(100,0,-1):
            x,y=map(int,X[rep,path,100-t:102-t]);F=f.features[t,x];logit=F@w;pr=1/(1+np.exp(-logit));q=f.k[t][0,x]+pr*(f.k[t][1,x]-f.k[t][0,x]);sc=np.zeros(10)
            for edge in range(6):
                dy=(f.k[t][1,x,edge]-f.k[t][0,x,edge])*pr[edge]*(1-pr[edge])*F[edge]
                sc+=dy/q[edge] if f.B[y,edge] else -dy/(1-q[edge])
            for cost in range(5):total[cost]+=(V[t-1,y,cost]-V[t,x,cost])*sc
        err=max(err,float(abs(total-pg[key][rep,path]).max()))
        np.testing.assert_allclose(total,pg[key][rep,path],atol=1e-12,rtol=1e-12)
    checks.append(dict(seed=seed,step=step,path_error=err))
for row in json.loads((OUT/'estimates.json').read_text()):
    key=f"{row['seed']}_{row['update']}";np.testing.assert_allclose(pg[key][row['repeat'],:row['K']].mean(0),row['gradient'],atol=1e-14,rtol=1e-14)
protocol=json.loads((OUT/'protocol.json').read_text())
for name,h in protocol['bindings'].items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==h
res=dict(passed=True,scalar_path_checks=12,mean_checks=576,details=checks,seconds=time.time()-start,RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,limitations='Exact expectation check in worker shares value tables; independent scalar audit checks sampled score and accumulation, not an independent production model.')
(OUT/'audit.json').write_text(json.dumps(res,indent=2));print(json.dumps(res))
