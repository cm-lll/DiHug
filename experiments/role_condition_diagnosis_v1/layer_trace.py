"""One original block at each t: observe all eight layers, frozen C weights."""
import argparse
import json
from pathlib import Path
import time
import resource
import numpy as np
import torch
from diagnose import c, sha, dump, load_panel, diff, BASE
from role_condition.experiment import fixed_validation
from role_condition.model import paired_models
from hsgd.computational import FrozenTypedContext
from test_late_fusion import directional_witness


def capture(m, case, inp, target, cfg):
    m.set_target(target);m.eval();saved={};handles=[]
    for i,layer in enumerate(m.backbone.tf_layers):
        def hook(module,args,out,i=i):
            # CN context separately calls layer zero; retain both invocations.
            saved.setdefault(str(i),[]).append([v.detach().numpy().copy() for v in out])
        handles.append(layer.register_forward_hook(hook))
    ctx=FrozenTypedContext(m,case["space"].types,inp["edges"],inp["edge_families"],"layers")
    try:
        with torch.no_grad():
            _,logits,p=ctx.predict(inp["query"],inp["f"],inp["t"]/cfg["T"],"layers",details=True)
        return saved,logits.numpy(),p.numpy()
    finally:
        ctx.close()
        for h in handles:h.remove()


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--run",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True);a=ap.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter();cpu=time.process_time();torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    cfg=json.loads((a.run/"resolved_config.json").read_text())
    records,targets=load_panel(a.run);values=fixed_validation(records,cfg)
    hashes=json.loads((a.run/"analysis/weight_hashes.json").read_text());rows=[]
    for seed in cfg["train_seeds"]:
        models=paired_models(records[0]["case"],cfg,seed);m=models["C"]
        del models["A"],models["B"]
        name=f"seed{seed}/C_final_1000.pt"
        assert sha(a.run/name)==hashes[name]
        m.load_state_dict(torch.load(a.run/name,map_location="cpu",weights_only=True),strict=True)
        for identifier,condition,case,inp,index in values[:3]:
            left,ll,lp=capture(m,case,inp,targets["balanced"],cfg)
            right,rl,rp=capture(m,case,inp,targets["extreme"],cfg)
            layers={}
            for i in left:
                assert len(left[i])==len(right[i])
                layers[i]=[{name:diff(b,a) for name,a,b in zip(("node","edge","global"),aa,bb)}
                           for aa,bb in zip(left[i],right[i])]
            rows.append(dict(seed=seed,t=inp["t"],graph=identifier,query_block=inp["round_id"],
                             layers=layers,logit_gap=diff(rl[:,1]-rl[:,0],ll[:,1]-ll[:,0]),
                             p0=diff(rp,lp)))
    dump(a.output/"layers.json",rows)
    dump(a.output/"directional_witness.json",directional_witness())
    dump(a.output/"receipt.json",dict(status="completed",wall_seconds=time.perf_counter()-start,
         cpu_seconds=time.process_time()-cpu,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
         updates=0,full_chains=0,device="CPU",states=9,prediction_calls=18,
         note="Layer magnitudes are observational, not a causal attribution to a particular normalization"))
if __name__=="__main__":main()
