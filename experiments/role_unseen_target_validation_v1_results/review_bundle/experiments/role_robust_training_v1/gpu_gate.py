"""Bounded full-size engineering parity gate, no scientific training/sampling."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
os.environ.update(OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1")
import argparse
import json
from pathlib import Path
import resource
import time
import numpy as np
import torch
from core import *

def main():
    p=argparse.ArgumentParser()
    for key in ("source","panel","output"):p.add_argument("--"+key,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);start=time.monotonic();setup()
    status="failed";reason=None;record={}
    try:
        assert torch.cuda.is_available() and torch.cuda.device_count()==1
        cfg=config();records,targets=load_panel(a.panel);case=next(r["case"] for r in records if r["condition"]=="extreme")
        models,init=pair_models(case,cfg,cfg["train_seeds"][0],a.source)
        cpu=models["normal2"];gpu=models["robust"].to("cuda")
        initial={k:v.detach().cpu().clone() for k,v in cpu.state_dict().items()}
        normal,robust,mask,_=paired_inputs(case,cfg,2026102199,0)
        torch.use_deterministic_algorithms(True,warn_only=False)
        differences=[]
        cpu.eval();gpu.eval();cpu.set_target(targets["extreme"]);gpu.set_target(targets["extreme"])
        with torch.no_grad():
            for inp in robust:
                outputs=[]
                for m in (cpu,gpu):
                    l,q=predict(m,case["space"].types,inp["edges"],inp["edge_families"],inp["query"],inp["f"],inp["t"]/64,details=True)
                    outputs.append((l.cpu(),q.cpu()))
                for name,left,right in (("logits",outputs[0][0],outputs[1][0]),("p0",outputs[0][1],outputs[1][1])):
                    torch.testing.assert_close(left,right,atol=2e-6,rtol=1e-5)
                    differences.append(dict(metric=name,max_abs=float((left-right).abs().max())))
        gd={};cd={};co=opt(cpu,cfg);go=opt(gpu,cfg)
        double_update(cpu,co,case,robust,64,mask,diagnostics=cd)
        double_update(gpu,go,case,robust,64,mask,diagnostics=gd)
        max_gradient=0.
        for name,g in cd["gradients"].items():
            other=gd["gradients"][name].cpu();torch.testing.assert_close(g,other,atol=2e-6,rtol=1e-5)
            max_gradient=max(max_gradient,float((g-other).abs().max()))
        updated={k:v.detach().cpu().clone() for k,v in gpu.state_dict().items()}
        cross_parameter=max(float((cpu.state_dict()[k]-updated[k]).abs().max()) for k in updated)
        gpu.load_state_dict(initial,strict=True);go=opt(gpu,cfg)
        double_update(gpu,go,case,robust,64,mask)
        repeat=max(float((gpu.state_dict()[k].detach().cpu()-updated[k]).abs().max()) for k in updated)
        assert repeat==0.,"strict same-device update repeat not exact"
        record=dict(initialization=init,CPU_GPU_differences=differences,gradient_max_abs=max_gradient,
            CPU_GPU_updated_parameter_max_abs=cross_parameter,CPU_GPU_parameters_required_identical=False,
            strict_GPU_repeat_max_abs=repeat,tolerances=dict(atol=2e-6,rtol=1e-5),
            note="cross-backend near-zero gradients can yield Adam update differences; strict same-device repeat is checked independently",
            device_name=torch.cuda.get_device_name(0),visibility=os.environ.get("CUDA_VISIBLE_DEVICES"),
            peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,peak_reserved_mib=torch.cuda.max_memory_reserved()/1024**2)
        status="passed"
    except BaseException as exc:reason=repr(exc);raise
    finally:
        save(a.output/"receipt.json",dict(status=status,reason=reason,wall_seconds=time.monotonic()-start,
            peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
            scientific_updates=0,complete_chains=0,**record))
if __name__=="__main__":main()
