"""Bounded CPU cost only: original scale small-graph forward/backward, no step."""
import os
os.environ["CUDA_VISIBLE_DEVICES"]=""
os.environ["OMP_NUM_THREADS"]="1"
os.environ["OPENBLAS_NUM_THREADS"]="1"
import argparse
import json
from pathlib import Path
import sys
import time
import resource
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).parent/"baseline"))
from hsgd.bw_preflight import common as c
from hsgd.computational import FrozenTypedContext
from role_condition.model import ConditionalDenoiser
from role_condition.experiment import fixed_validation
from diagnose import load_panel,dump
from late_fusion import LateConditionalDenoiser


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--run",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True);a=ap.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    cfg=json.loads((a.run/"resolved_config.json").read_text())
    records,targets=load_panel(a.run);_,_,case,inp,index=fixed_validation(records,cfg)[1]
    torch.set_num_threads(1);torch.set_float32_matmul_precision("highest")
    wall=time.perf_counter();cpu=time.process_time();rows=[]
    for label,cls in [("original_C",ConditionalDenoiser),("late_C",LateConditionalDenoiser)]:
        c.seed_all(2026102041)
        m=cls(case["n_types"],case["n_families"],family_pairs=case["space"].pairs,
              prior=case["prior"],mode="C",**cfg["model"]);m.set_target(targets["balanced"])
        if label=="original_C":shared={k:v.clone() for k,v in m.state_dict().items()}
        else:
            st=m.state_dict()
            for k,v in shared.items():
                if k in st:st[k]=v.clone()
            m.load_state_dict(st,strict=True)
        opt=torch.optim.AdamW(m.parameters(),**{k:v for k,v in cfg["optimizer"].items() if k!="type"})
        backwards=[];sweeps=[]
        for i in range(4):
            tick=time.perf_counter();r=c.update(m,opt,case,inp,64,"cpu",do_step=False)
            backwards.append(time.perf_counter()-tick)
            m.eval();plan=c.plan_for(case,[2026102041,32,61],4)
            tick=time.perf_counter()
            ctx=FrozenTypedContext(m,case["space"].types,inp["edges"],inp["edge_families"],"cost")
            with torch.no_grad():
                for j in range(4):
                    _,f,q=plan.block(j);ctx.predict(q,f,.5,"cost",details=True)
            ctx.close();sweeps.append(time.perf_counter()-tick)
        rows.append(dict(model=label,parameters=sum(p.numel() for p in m.parameters()),
            forward_backward_seconds=backwards,warmup_repetitions=1,steady_repetitions=3,
            median_forward_backward_seconds=float(np.median(backwards[1:])),
            p90_forward_backward_seconds=float(np.quantile(backwards[1:],.9)),
            complete_four_block_prediction_seconds=sweeps,
            median_four_block_prediction_seconds=float(np.median(sweeps[1:])),
            excludes="optimizer step, posterior/sampling, input construction and evaluation"))
    dump(a.output/"cost.json",dict(rows=rows,wall_seconds=time.perf_counter()-wall,
        cpu_seconds=time.process_time()-cpu,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        updates=0,full_chains=0,device="CPU",configuration="same full8layer FP32 model; original34node fixed state"))
if __name__=="__main__":main()
