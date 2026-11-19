"""Independent CPU matrix exponential checks of32 reconstructed hard inputs.

No model calls or fits. Original failed inputs and scores are never overwritten.
"""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import json,time,importlib.util,signal
from pathlib import Path
import numpy as np
import scipy
from scipy.linalg import expm
from heat_solver_candidate import heat,TIMES
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/three_arm_learning_pilot_20261111T211641/failure_audit'
spec=importlib.util.spec_from_file_location('heat_old',ROOT/'research_branches/typed_heat_audit_20261111/run.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('120second CPU regression')));signal.alarm(120)
start=time.time();rows=[]
p=json.loads((OUT.parent/'protocol.json').read_text())
for ds in ('IMDB','PubMed'):
    data=np.load(p['parents'][ds]['data']);types=data['types'];z=np.load(OUT/(ds+'_CPU_reconstructed.npz'))
    for k,b in enumerate(z['bits']):
        a=old.adj(len(types),z['query'][:,b.astype(bool)].T);h=heat(a,types)
        try:before=old.heat(a,types);difference=float(abs(before['feat']-h['feat']).max());passed=True
        except AssertionError:difference=None;passed=False
        d=a.sum(1);inv=np.divide(1,np.sqrt(d),out=np.zeros(len(d)),where=d>0)
        lap=np.diag((d>0).astype(float))-inv[:,None]*a*inv[None,:]
        ind=np.column_stack([types==t for t in np.unique(types)]).astype(float);err=0.
        for j,t in enumerate(TIMES):
            H=expm(-t*lap);r=inv[:,None]*(H@(np.sqrt(d)[:,None]*ind));r[d==0]=ind[d==0]
            err=max(err,float(abs(r-h['reach'][:,j]).max()),float(abs(np.diag(H)-h['ret'][:,j]).max()))
        assert err<1e-10
        perm=np.random.default_rng(2026111518+k).permutation(len(types));hp=heat(a[np.ix_(perm,perm)],types[perm])
        perr=float(abs(hp['feat']-h['feat'][perm]).max());assert perr<1e-10
        rows.append(dict(dataset=ds,sample=k,original_passed=passed,feature_difference_from_original=difference,
            expm_max_error=err,type_preserving_relabelled_features_error=perr,mass_error=float(abs(h['reach'].sum(2)-1).max()),seconds=h['seconds']))
result=dict(passed=True,graphs=32,independent_expm_matrices=288,rows=rows,seconds=time.time()-start,model_calls=0,GPU=0,
    numpy=np.__version__,scipy=scipy.__version__,candidate_not_wired_or_trained=True,
    note='Node permutation moves types along with nodes; tests label-equivariance. Original tolerances unchanged.')
(OUT/'solver_regression.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='rows'}))
