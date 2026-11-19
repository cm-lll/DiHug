"""Freeze a reference loss scale from saved parent predictions only.

No model forward/backward. Gradients, if needed later, flow through errors, not
through these frozen scale constants. Reuses the existing exact joint-degree
risk; new closure/propagation distribution targets are explicitly soft plug-ins.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import hashlib
import importlib.util
import json
import math
import resource
import signal
import time
from pathlib import Path
import numpy as np
import torch
from objective import make_plan,descriptors,distribution_errors

torch.set_num_threads(1)
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/hetero_micro_macro_calibration_20261116'
FEATURE=ROOT/'results/hetero_micro_macro_cpu_20261116_v2'
PRED=ROOT/'results/analytic_joint_model_check_20261113'


def combine(ce,errors,scale,dataset):
    """Original CE coefficient stays one; no per-input renormalization."""
    macro=torch.stack([x['error']/(4*scale['sigma_squared'][dataset][str(x['type'])][x['block']]) for x in errors]).mean()
    return ce+scale['lambda_macro']*macro


def main():
    start=time.time(); cpu=time.process_time();OUT.mkdir(exist_ok=False)
    def alarm(*_):raise TimeoutError('600 second CPU calibration cap')
    signal.signal(signal.SIGALRM,alarm);signal.alarm(600)
    bindings={}
    def bind(path,expected=None):
        p=Path(path).resolve();h=hashlib.sha256(p.read_bytes()).hexdigest()
        if expected:assert h==expected,str(p)
        bindings[str(p)]=h;return p
    def read(path):return json.loads(bind(path).read_text())
    def save(name,x):(OUT/name).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False))
    bind(__file__);bind(Path(__file__).with_name('objective.py'))
    prior_check=read(FEATURE/'completion.json');assert prior_check['status']=='complete'
    science=read(ROOT/'results/analytic_joint_generation_run_20261113T225800/protocol.json')
    old=read(PRED/'protocol.json')
    jobs=old['jobs'];assert len(jobs)==4
    save('protocol.json',dict(
        purpose='Freeze one CPU reference objective, no training or generation',
        parent_jobs=jobs,times=[1,75,250,550,800,1000],
        calibration_inputs='Saved update split only, four parents x six inputs=24, all previously viewed',
        descriptive_readout='Saved independent split, 24 inputs; historically viewed, NOT a new holdout',
        degree='Existing exact independent-slot expected hard joint-degree CF risk; frozen original frequencies/scales unchanged.',
        other_blocks='Joint [relation degrees, typed anchored triangles] and [relation degrees, typed simple paths] anonymous Fourier distribution errors on expected descriptors. Soft plug-ins.',
        variance='Per dataset/type/block mean calibration complex squared error /2, floor exp(-12); then frozen. No update from candidate predictions.',
        relative_weight='One shared lambda across both datasets: mean original parent CE / mean calibrated macro loss on the 24 inputs; fixed before candidate training.',
        composition='L=original_CE+lambda*mean_groups(error/(4*sigma_squared)); no division of original CE, no division by current t weight.',
        differences_from_GraphVAE_MM=['Frozen parent calibration rather than fitting variance on each current minibatch.',
            'One-reference heterogeneous diffusion, no VAE KL.', 'Anonymous distribution targets and exact degree risk rather than aligned transition matrices.'],
        limits=['Balancing loss values does not balance parameter gradients.',
            'Six deliberately spread times are a calibration convention, not a uniform-time expectation.',
            'No guarantee against compensation, hard-risk discrepancy, unseen-state drift or diversity loss.',
            'Calibration does not assert any generation gain.'],
        model_calls=0,GPU_hours=0,new_graphs=0,wall_cap_seconds=600))
    spec=importlib.util.spec_from_file_location('existing_exact_risk',bind(ROOT/'research_branches/analytic_joint_risk_20261113/core.py'))
    exact=importlib.util.module_from_spec(spec);spec.loader.exec_module(exact)
    contexts={};rows=[]
    for ds in ('IMDB','PubMed'):
        p=next(x for x in science['parents'] if x['dataset']==ds)
        z=np.load(bind(p['data'],old['bindings'][p['data']]))
        ty=z['types'];schema=z['schema'];pl=make_plan(schema,len(np.unique(ty)))
        e=z['edges'];e=e if e.shape[0]==2 else e.T
        ref=np.zeros((len(ty),len(ty)));ref[e[0],e[1]]=ref[e[1],e[0]]=1
        f=read(FEATURE/(ds+'_frozen.json'))
        for t,groups in f.items():
            for g,b in groups.items():
                groups[g]=dict(scale=torch.tensor(b['scale'],dtype=torch.float64),omega=torch.tensor(b['omega'],dtype=torch.float64),
                    target=torch.tensor(np.array(b['target_real'])+1j*np.array(b['target_imag']),dtype=torch.complex128))
        f={int(t):g for t,g in f.items()}
        frozen_path=ROOT/'results/analytic_joint_risk_20261113'/(ds+'_frozen_target.json')
        bind(frozen_path,old['bindings'][str(frozen_path)])
        contexts[ds]=(ty,schema,pl,ref,f,read(frozen_path))
    for job in jobs:
        ds=job.split('_')[0];ty,schema,pl,ref,f,frozen=contexts[ds]
        record=read(PRED/(job+'.json'))
        z=np.load(bind(PRED/(job+'_baseline_probabilities.npz')))
        q=z['query'];families=z['family']
        assert q.shape[0]==2 and np.all(q[0]<q[1])
        assert len(set(map(tuple,q.T)))==len(q.T)
        np.testing.assert_array_equal(ty,z['types'])
        np.testing.assert_array_equal(ref[q[0],q[1]],z['reference'])
        assert int(ref.sum()/2)==int(z['reference'].sum()),'reference slots missing'
        for r in record['baseline']:
            t=r['t'];split=r['split'];key=f'{split}_t{t}';probs=z[key]
            p=torch.tensor(probs,dtype=torch.float64)
            P=torch.zeros((len(schema),len(ty),len(ty)),dtype=torch.float64)
            P[families,q[0],q[1]]=P[families,q[1],q[0]]=p
            tick=time.perf_counter();d=descriptors(P,ty,pl)
            errors=distribution_errors(d,ty,f)
            risks,parts=exact.risks(p,ty,q,families,frozen)
            np.testing.assert_allclose(float(risks['risk']),r['loss']['risk'],rtol=1e-9,atol=1e-12)
            errors=[x for x in errors if x['block']!='degree']
            errors += [dict(type=x['type'],block='degree',error=x['risk']) for x in parts]
            assert len(errors)==12
            for x in errors:
                rows.append(dict(job=job,dataset=ds,split=split,t=t,input_hash=r['input_hash'],
                    CE=r['CE'],type=x['type'],block=x['block'],error=float(x['error']),CPU_seconds=time.perf_counter()-tick))
    sigma={}
    for ds in contexts:
        sigma[ds]={}
        for typ in range(4):
            sigma[ds][str(typ)]={}
            for block in ('degree','closure','propagation'):
                values=[x['error']/2 for x in rows if x['dataset']==ds and x['type']==typ and x['block']==block and x['split']=='update']
                assert len(values)==12
                sigma[ds][str(typ)][block]=max(float(np.mean(values)),math.exp(-12))
    for r in rows:r['scaled_macro']=r['error']/(4*sigma[r['dataset']][str(r['type'])][r['block']])
    calibration=[r for r in rows if r['split']=='update']
    # Every input contributes 12 rows, hence means have identical input weighting.
    scale=dict(sigma_squared=sigma,lambda_macro=float(np.mean([x['CE'] for x in calibration])/np.mean([x['scaled_macro'] for x in calibration])),
        original_CE_coefficient=1.,frozen_from_parent_only=True,gradient_balance_claim=False)
    ce=torch.tensor(.23,requires_grad=True,dtype=torch.float64)
    test=[dict(type=0,block='degree',error=torch.tensor(.01,dtype=torch.float64))]
    out=combine(ce,test,scale,'IMDB')
    assert float(torch.autograd.grad(out,ce)[0])==1.
    summary=[]
    for job in jobs:
        for split in ('update','independent'):
            select=[x for x in rows if x['job']==job and x['split']==split]
            summary.append(dict(job=job,split=split,CE=float(np.mean([x['CE'] for x in select])),
                macro_weighted=float(scale['lambda_macro']*np.mean([x['scaled_macro'] for x in select]))))
    save('frozen_scales.json',scale);save('per_input_type_block.json',rows);save('summary.json',summary)
    for p,h in bindings.items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
    save('asset_bindings.json',bindings)
    save('completion.json',dict(status='complete',wall_seconds=time.time()-start,CPU_seconds=time.process_time()-cpu,
        RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,model_calls=0,GPU_hours=0,new_graphs=0,
        calibration_inputs=24,descriptive_inputs=24,exact_degree_reproduced=True,CE_coefficient_checked=True,
        saved_probability_historical_hash='No separate original NPZ digest was found here; now bound. Data/reference/query and saved risk values cross-checked.',
        original_assets_unchanged=True))
    print(json.dumps(scale));print(json.dumps(summary));signal.alarm(0)


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        if OUT.exists():(OUT/'failure.json').write_text(json.dumps(dict(type=type(exc).__name__,error=str(exc))))
        raise
