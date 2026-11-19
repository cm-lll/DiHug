"""Reconstruct AdamW proposals from saved tensors, without a model or autograd.

Uses float64 equations then FP32 parameter quantization. It is a numerical
cross-check, not a demand for CPU/GPU bitwise equality or a counterfactual run.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='1'
import json,time
from pathlib import Path
import torch

torch.set_num_threads(1)
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/real192_update_audit_20261111'
started=time.time();rows=[]
for folder in sorted(OUT.iterdir()):
    if not folder.is_dir():continue
    for path in sorted(folder.glob('vectors_u*.pt')):
        u=int(path.stem.split('u')[-1]);bp=folder/f'boundary_u{u-1}.pt'
        if not bp.exists():continue
        vec=torch.load(path,map_location='cpu',weights_only=False)
        state=torch.load(bp,map_location='cpu',weights_only=False)['state']
        optimizer=state['optimizer'];ids=[i for pg in optimizer['param_groups'] for i in pg['params']]
        assert len(ids)==len(vec['names'])
        configs={i:pg for pg in optimizer['param_groups'] for i in pg['params']}
        errors=[];sq=0.;actualsq=0.;absolute=0.;missing=[];offset=0;largest=None
        for name,size,pid in zip(vec['names'],vec['sizes'],ids):
            theta=state['model'][name].flatten();grad=vec['gradient'][offset:offset+size].double()
            actual=vec['displacement'][offset:offset+size].double();offset+=size
            old=optimizer['state'].get(pid);pg=configs[pid]
            if old is None:
                assert not torch.count_nonzero(grad) and not torch.count_nonzero(actual)
                missing.append(name);continue
            assert not pg['amsgrad'] and not pg['maximize']
            beta1,beta2=pg['betas'];step=float(old['step'])+1
            m=beta1*old['exp_avg'].flatten().double()+(1-beta1)*grad
            v=beta2*old['exp_avg_sq'].flatten().double()+(1-beta2)*grad.square()
            proposed=theta.double()*(1-pg['lr']*pg['weight_decay'])-pg['lr']*(m/(1-beta1**step))/(torch.sqrt(v/(1-beta2**step))+pg['eps'])
            predicted=proposed.to(theta.dtype).double()-theta.double()
            error=predicted-actual;maximum=float(error.abs().max())
            # Diagnostic ULP-scaled error; not a production acceptance tolerance.
            spacing=torch.nextafter(theta.abs(),torch.full_like(theta,float('inf')))-theta.abs()
            normalized=float((error.abs()/spacing.double().clamp_min(torch.finfo(theta.dtype).tiny)).max())
            if maximum>absolute:absolute=maximum;largest=name
            sq+=float(error.square().sum());actualsq+=float(actual.square().sum())
            errors.append(normalized)
        rows.append(dict(task=folder.name,update=u,max_absolute_displacement_error=absolute,
                         relative_L2_error=(sq/actualsq)**.5 if actualsq else None,
                         max_error_in_parameter_ULPs=max(errors),largest_error_parameter=largest,
                         unchanged_parameters_without_optimizer_state=missing))
        del vec,state,optimizer
(OUT/'adam_equation_check.json').write_text(json.dumps(dict(rows=rows,seconds=time.time()-started,
    model_calls=0,autograd_calls=0,GPU_hours=0,
    limitation='FP64 reconstruction then FP32 quantization; rounding differs from CUDA AdamW. No optimizer state is modified.'),indent=2))
print('checked',len(rows),'max_abs',max((r['max_absolute_displacement_error'] for r in rows),default=None))
