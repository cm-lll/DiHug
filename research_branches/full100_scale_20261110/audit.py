"""Independent reverse-kernel/path recomputation and complete result summaries."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import json,csv,time,hashlib
from pathlib import Path
import numpy as np
from scipy.special import expit
import run as q
r=q.r;OUT=q.OUT
def read(n):return json.loads((OUT/n).read_text())
def main():
    start=time.time();assert read('receipt.json')['status']=='complete'
    f,C,tri,cfg,kernel=r.setup();saved=np.load(OUT/'states_and_paths.npz');index=read('state_index.json');finals=read('finals.json')
    ufile=np.load(q.OLD/'sampling_uniforms.npz');checks=[]
    for j,meta in enumerate(index):
        theta=saved['parameters'][j];dist=f.prior.copy();u=ufile[str(meta['seed'])]
        states=np.searchsorted(np.cumsum(dist),u[0],side='right');np.testing.assert_array_equal(states,saved['paths'][j,0])
        for t in range(100,0,-1):
            p=expit(f.features[t]@theta)
            z=f.kernel.reverse(p.ravel(),f.B.ravel(),np.tile(f.f,f.S),t-1,t).reshape(f.S,f.M)
            P=np.prod(np.where(f.B[None,:,:]>0,z[:,None,:],1-z[:,None,:]),axis=2)
            assert abs(P.sum(1)-1).max()<1e-12
            cp=np.cumsum(P[states],axis=1);cp[:,-1]=1.;states=(u[101-t,:,None]>=cp).sum(1)
            np.testing.assert_array_equal(states,saved['paths'][j,101-t]);dist=dist@P
        error=float(abs(dist-saved['distributions'][j]).max());assert error<1e-12
        found=next(x for x in finals if all(x[k]==v for k,v in meta.items()))
        for k,v in r.metrics(f,C,tri,dist).items():assert abs(v-found['terminal'][k])<1e-12
        checks.append(dict(**meta,exact_distribution_error=error,all_paths_identical=True))
    cal=read('calibration.json');scale=cal['scale'];frozen=read('freeze.json')
    assert hashlib.sha256((OUT/'calibration.json').read_bytes()).hexdigest()==frozen['calibration_SHA256']
    assert hashlib.sha256((OUT/'protocol.json').read_bytes()).hexdigest()==frozen['protocol_SHA256']
    assert scale==float(np.clip(np.median([x['ratio'] for x in cal['records']]),1,10))
    # Independent input regeneration and complete update replay, including fixed validation.
    banks=np.load(OUT/'calibration_and_eval_inputs.npz');oldbanks=np.load(q.OLD/'input_banks.npz')
    oldidx=json.loads((q.OLD/'state_index.json').read_text());old=np.load(q.OLD/'states_and_paths.npz')
    updates=read('updates.json');replay=[]
    for s in r.SEEDS:
        for stream,shape,key in [(q.CAL_STREAM,(24,8),'cal'),(q.EVAL_STREAM,(96,),'eval')]:
            ts,xs=q.bank(f,s,stream,shape)
            np.testing.assert_array_equal(ts,banks[f'{s}_{key}_t']);np.testing.assert_array_equal(xs,banks[f'{s}_{key}_x'])
        for arm,factor in [('hard_original',1.),('hard_scaled',scale)]:
            theta=old['parameters'][next(i for i,x in enumerate(oldidx) if x['seed']==s and x['arm']=='initial')].copy()
            for step in range(20):
                ts=oldbanks[f'{s}_t'][step];xs=oldbanks[f'{s}_x'][step]
                before,G,ce,_=r.atomic(f,C,theta,ts,xs);co,sol=r.project_mean(G@G.T);delta=-r.LR*factor*(G.T@co)
                row=next(x for x in updates if x['seed']==s and x['arm']==arm and x['update']==step+1)
                assert row['times']==ts.tolist() and row['states']==xs.tolist()
                assert abs(row['displacement']-np.linalg.norm(delta))<1e-14
                np.testing.assert_allclose(row['first_order'],G@delta,atol=1e-14,rtol=0)
                theta+=delta
            if arm=='hard_original':target=old['parameters'][next(i for i,x in enumerate(oldidx) if x['seed']==s and x['arm']=='hard_five_project_mean')]
            else:target=saved['parameters'][next(i for i,x in enumerate(index) if x['seed']==s and x['arm']==arm and x['updates']==20)]
            np.testing.assert_array_equal(theta,target);replay.append(dict(seed=s,arm=arm,all_updates_replayed=True,final_parameters_bitwise=True))
    flat=[dict(seed=x['seed'],arm=x['arm'],updates=x['updates'],**x['terminal']) for x in finals]
    for x in json.loads((q.OLD/'finals.json').read_text()):
        if x['arm'] in ('initial','reference_CE'):flat.append(dict(seed=x['seed'],arm=x['arm'],updates=x['updates'],**x['terminal']))
    keys=list(finals[0]['terminal']);aggregate=[]
    for arm,step in sorted({(x['arm'],x['updates']) for x in flat}):
        subset=[x for x in flat if x['arm']==arm and x['updates']==step];a=dict(arm=arm,updates=step,n_initializations=len(subset))
        for k in keys:
            vals=np.array([x[k] for x in subset]);a[k]=float(vals.mean());a[k+'_SD']=float(vals.std(ddof=1))
        aggregate.append(a)
    differences=[]
    for s in r.SEEDS:
        a=next(x for x in flat if x['seed']==s and x['arm']=='hard_original' and x['updates']==20)
        b=next(x for x in flat if x['seed']==s and x['arm']=='hard_scaled' and x['updates']==20)
        differences.append(dict(seed=s,**{k:b[k]-a[k] for k in keys}))
    fixed=list(csv.DictReader((OUT/'fixed_validation.csv').open()));trend=[]
    for s in r.SEEDS:
        for arm in ('hard_original','hard_scaled'):
            rows=[x for x in fixed if int(x['seed'])==s and x['arm']==arm];steps=[x for x in updates if x['seed']==s and x['arm']==arm]
            x=dict(seed=s,arm=arm,mean_displacement=float(np.mean([z['displacement'] for z in steps])),
                mean_projected_to_unprojected=float(np.mean([z['hard_norm']/z['mean_norm'] for z in steps])),
                batches_with_any_finite_cost_increase=sum(any(b>a+1e-12 for a,b in zip(z['before_costs'],z['after_costs'])) for z in steps))
            for key in ('relation_count','joint_degree','triangle001','triangle011','component_cost','mean_five_cost'):
                v=np.array([float(z[key]) for z in rows]);x[key+'_increases']=int((np.diff(v)>1e-12).sum());x[key+'_start']=float(v[0]);x[key+'_end']=float(v[-1])
            trend.append(x)
    for p,h in read('protocol.json')['bindings'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
    r.table('terminal_metrics.csv',flat);r.table('aggregate.csv',aggregate);r.table('scaled_minus_original.csv',differences);r.table('validation_trends.csv',trend)
    r.dump('summary.json',dict(scale=scale,aggregate=aggregate,per_initialization=flat,scaled_minus_original=differences,validation_trends=trend))
    r.dump('independent_audit.json',dict(passed=True,checks=checks,replay=replay,calibration_frozen=True,source_bindings_unchanged=True,seconds=time.time()-start,GPU_hours=0))
    print(json.dumps(dict(scale=scale,aggregate=aggregate,validation_trends=trend),indent=2))
if __name__=='__main__':main()
