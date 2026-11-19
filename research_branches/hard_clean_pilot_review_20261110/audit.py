"""Read-only post-run audit of both estimator versions; no model construction."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,hashlib,csv
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'research_branches/hard_clean_pilot_v2_20261110'))
from objective import solve_gram

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def main():
    out=ROOT/'results/hard_clean_pilot_review_20261110';out.mkdir(exist_ok=False)
    versions=['hard_clean_pilot_20261110','hard_clean_pilot_v2_20261110'];bindings={};stops=[];states=[]
    for name in versions:
        p=ROOT/'results'/name;cfg=json.loads((p/'protocol.json').read_text())
        for file,h in json.loads((p/'bindings.json').read_text()).items():
            if file not in bindings:bindings[file]=dict(expected=h,actual=sha(file))
            assert bindings[file]['actual']==h,file
        for f in p.glob('*/zero_gradient_*.json'):
            z=json.loads(f.read_text());task=f.parent.name.removeprefix('gate_');row=next(r for r in cfg['tasks'] if r['task']==task)
            u=int(f.stem.split('_u')[1]);history=[json.loads(x) for x in Path(row['history']).read_text().splitlines()]
            score=json.loads((f.parent/'score_definition.json').read_text());costs=np.array(z['costs']);norms=np.array(z['gradient_norms']);lam=np.array(z['coefficients'])
            j=int(lam.argmax());assert lam[j]>1-1e-8 and norms[j]<=1e-12
            # The raw cost column is zero/constant; the zero column is a valid
            # minimum-norm simplex vertex without demonstrating other objectives optimal.
            col=score['active'][j];span=float(np.ptp(costs[:,col]));assert span<1e-14
            stops.append(dict(version=name,task=task,gate=f.parent.name.startswith('gate_'),update=u,t=history[u-1]['t'],
                completed_updates=u-2001,objective=score['names'][col],cost_min=float(costs[:,col].min()),cost_max=float(costs[:,col].max()),
                gradient_norm=float(norms[j]),other_norms=norms.tolist(),reference_components=score['reference']['components'],hard_draws=len(costs)))
        for f in p.glob('*/stopped_complete.pt'):
            z=torch.load(f,map_location='cpu');s=z['state'];rc=s['recovery']
            assert rc['exact_resume_allowed'] and rc['phase']=='committed' and rc['completed_updates']==s['updates']==s['ema']['updates']
            assert z['auxiliary_rng'] is not None
            states.append(dict(path=str(f),sha256=sha(f),updates=s['updates'],complete_boundary=True))
    h=np.array([[1.,0.],[0.,0.]]);lam,res=solve_gram(h)
    assert np.array_equal(lam,np.array([0.,1.]))
    witness=dict(gradient_vectors=[[1.,0.],[0.,0.]],MGDA_coefficients=lam.tolist(),MGDA_direction=[0.,0.],
                 alternative_parameter_displacement=[-1.,0.],first_order_cost_changes=[-1.,0.],
                 limitation='first order only; zero estimated gradient is not proof of zero true gradient or finite-step harmlessness')
    for n,v in [('bindings',bindings),('stops',stops),('saved_boundaries',states),('zero_gradient_witness',witness)]:
        (out/(n+'.json')).write_text(json.dumps(v,ensure_ascii=False,indent=2))
    with (out/'stops.csv').open('w') as f:
        w=csv.DictWriter(f,list(stops[0]));w.writeheader();w.writerows(stops)
    result=dict(status='complete',all_original_bindings_unchanged=True,bound_files=len(bindings),stops=len(stops),saved_boundaries=len(states),GPU_hours=0,model_calls=0)
    (out/'receipt.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
