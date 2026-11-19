"""Independent explicit production-kernel propagation and CPU reporting."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import json,csv,time,hashlib
from pathlib import Path
import numpy as np
from scipy.special import expit
import run as r
OUT=r.OUT
def main():
    start=time.time();f,C,tri,cfg,kernel=r.setup();saved=np.load(OUT/'states_and_paths.npz');index=json.loads((OUT/'state_index.json').read_text());finals=json.loads((OUT/'finals.json').read_text())
    uniforms=np.load(OUT/'sampling_uniforms.npz');checks=[];flat=[]
    for j,meta in enumerate(index):
        theta=saved['parameters'][j];dist=f.prior.copy();u=uniforms[str(meta['seed'])]
        states=np.searchsorted(np.cumsum(dist),u[0],side='right');assert np.array_equal(states,saved['paths'][j,0])
        for t in range(100,0,-1):
            p=expit(f.features[t]@theta)
            q=f.kernel.reverse(p.ravel(),f.B.ravel(),np.tile(f.f,f.S),t-1,t).reshape(f.S,f.M)
            P=np.prod(np.where(f.B[None,:,:]>0,q[:,None,:],1-q[:,None,:]),axis=2)
            assert abs(P.sum(1)-1).max()<1e-12
            c=np.cumsum(P[states],axis=1);c[:,-1]=1.;states=(u[101-t,:,None]>=c).sum(1)
            np.testing.assert_array_equal(states,saved['paths'][j,101-t])
            dist=dist@P
        err=float(abs(dist-saved['distributions'][j]).max());assert err<1e-12
        metric=r.metrics(f,C,tri,dist)
        for k,v in metric.items():assert abs(v-finals[j]['terminal'][k])<1e-11
        empirical=np.bincount(states,minlength=f.S)/len(states)
        max_mc_ratio=0.
        for values in [f.B.sum(1),f.tri,(f.components==1).astype(float),*C.T]:
            mean=dist@values;var=dist@(values-mean)**2;se=np.sqrt(var/len(states));difference=abs(empirical@values-mean)
            ratio=float(difference/max(se,1e-15));max_mc_ratio=max(max_mc_ratio,ratio)
        checks.append(dict(**meta,max_terminal_probability_error=err,all_1024_paths_reproduced=True,max_sample_error_in_exact_SE=max_mc_ratio))
        flat.append(dict(**meta,**metric))
    updates=json.loads((OUT/'updates.json').read_text());banks=np.load(OUT/'input_banks.npz');bank_report=[]
    for seed in r.SEEDS:
        ts=banks[str(seed)+'_t'];xx=banks[str(seed)+'_x'];both=[u for u in updates if u['seed']==seed]
        for step in range(20):
            pair=[u for u in both if u['update']==step+1];assert len(pair)==2
            assert pair[0]['times']==pair[1]['times']==ts[step].tolist()
            assert pair[0]['states']==pair[1]['states']==xx[step].tolist()
        bank_report.append(dict(seed=seed,inputs=ts.size,low_1_33=int((ts<=33).sum()),mid_34_66=int(((ts>33)&(ts<=66)).sum()),high_67_100=int((ts>66).sum()),all_arms_paired=True))
    aggregate=[];metrics=list(finals[0]['terminal'])
    for arm in ['initial','reference_CE','hard_five_project_mean']:
        rows=[x for x in flat if x['arm']==arm];up=[x for x in updates if x['arm']==arm]
        a=dict(arm=arm,toy_initializations=len(rows))
        for k in metrics:
            v=np.array([x[k] for x in rows]);a[k]=float(v.mean());a[k+'_initialization_SD']=float(v.std(ddof=1))
        a['mean_update_norm']=float(np.mean([x['displacement'] for x in up])) if up else 0
        a['zero_updates']=sum(x['zero_direction'] for x in up)
        aggregate.append(a)
    comparisons=[]
    for seed in r.SEEDS:
        rows={x['arm']:x for x in flat if x['seed']==seed}
        comparisons.append(dict(seed=seed,**{k:rows['hard_five_project_mean'][k]-rows['initial'][k] for k in metrics}))
    r.table('terminal_metrics.csv',flat);r.table('aggregate.csv',aggregate);r.table('hard_minus_initial.csv',comparisons)
    protocol=json.loads((OUT/'protocol.json').read_text())
    for p,h in protocol['bindings'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
    r.dump('independent_audit.json',dict(passed=True,checks=checks,input_banks=bank_report,seconds=time.time()-start,
        no_updates_or_new_random_paths=True,all_original_bindings_unchanged=True))
    r.dump('summary.json',dict(aggregate=aggregate,per_initialization=flat,hard_minus_initial=comparisons,MC_audit=checks,input_banks=bank_report,
        scope='4-node toy 100-step exact full chain; not real O/A, no production training or sampler change',
        training_updates=120,number_of_models=6,initial_reference_predictors=3,generated_toy_paths=9216))
    print(json.dumps(aggregate,indent=2))
if __name__=='__main__':main()
