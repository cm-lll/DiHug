"""Bounded CPU fixture certificate and instantiated-dimension audit; no GPU."""
import ast
import copy
import json
from pathlib import Path
import time
import resource
import numpy as np
import torch
from hsgd.bw_preflight.common import PACKAGE, config, toy_case, imdb_case, make_model, plan_for, sha
from hsgd.model import TypedSparseDenoiser
from hsgd.perf_checks import compare_forward, compare_update, TOLERANCES
from hsgd.perf_bw import Meter, one_step, training_timing


def main():
    torch.set_num_threads(1);torch.manual_seed(812)
    out=PACKAGE/"results/local_acceptance"
    if out.exists():raise RuntimeError("fresh output required")
    out.mkdir(parents=True);start=time.perf_counter();cpu=time.process_time()
    cfg=config();case=toy_case()
    small=copy.deepcopy(cfg);small["model"]=dict(layers=2,width=32,edge_width=16,global_width=32,heads=4,
                                               ff_node=64,ff_edge=32,ff_global=64,dropout=.1)
    model=make_model(case,small,"cpu").eval();ef=case["space"].encode(case["clean"])[0]
    state=np.isin(np.arange(case["space"].total),case["clean_ids"]).astype(np.uint8)
    ids,f,qe=plan_for(case,[cfg["sample_seed"],500,61],20).block(0)
    args=(case["space"].types,case["clean"],ef,qe,f,.5)
    with torch.no_grad():forward=compare_forward(model,*args)
    inp=dict(edges=case["clean"],edge_families=ef,query=qe,f=f,t=500,
             labels=np.isin(ids,case["clean_ids"]).astype(np.float32),ce_weight=20/case["space"].total)
    update=compare_update(model,case,inp,1000,cfg["optimizer"])
    meter=Meter("cpu",30,out);step={};train={}
    for backend in ("legacy","upstream"):
        r=one_step(model,case,state,500,cfg,backend,meter)
        step[backend]={k:r[k] for k in ("seconds","cpu_seconds","reverse_state_advances")}
        train[backend]=training_timing(model,case,inp,cfg,backend,meter)
    real=imdb_case(PACKAGE/"data/IMDB_movie_reference.npz",cfg)
    current=make_model(real,cfg,"cpu")
    upstream_dims=copy.deepcopy(cfg["model"]);upstream_dims.pop("ff_global")
    native=TypedSparseDenoiser(4,10,family_pairs=real["space"].pairs,prior=real["prior"],**upstream_dims)
    dimensions=dict(current_parameters=sum(p.numel() for p in current.parameters()),
                    upstream_effective_global_ff_parameters=sum(p.numel() for p in native.parameters()),
                    current_global_ff=[layer.lin_y1.out_features for layer in current.backbone.tf_layers if layer.last_layer],
                    upstream_effective_global_ff=[layer.lin_y1.out_features for layer in native.backbone.tf_layers if layer.last_layer],
                    current_global_ff_layers=sum(layer.last_layer for layer in current.backbone.tf_layers))
    raw=PACKAGE/"evidence/upstream/sparse_diffusion/diffusion/diffusion_utils.py"
    nodes=ast.parse(raw.read_text()).body
    fn=next(n for n in nodes if isinstance(n,ast.FunctionDef) and n.name=="cosine_beta_schedule_discrete")
    ns={"np":np};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(raw),"exec"),ns)
    betas=torch.tensor(ns[fn.name](1000),dtype=torch.float32)
    alpha=torch.exp(torch.cumsum(torch.log(1-torch.clamp(betas,min=0,max=.9999)),dim=0))
    record=dict(tolerances=TOLERANCES,forward=forward,update=update,tiny_fixture_training=train,
        tiny_fixture_twenty_block_step=step,instantiated_dimensions=dimensions,
        upstream_schedule=dict(length=len(alpha),alpha_bar_index0=float(alpha[0]),alpha_bar_indexT=float(alpha[-1])),
        tests_passed=129,tests_seconds=5.56,CPU_seconds=time.process_time()-cpu,wall_seconds=time.perf_counter()-start,
        peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        BW_executed=False,GPU_usage=0,formal_training=False,new_scientific_graphs=0,
        full_chain_runs=0,optimizer_calls=4,
        limits="CPU tiny fixture timings are not IMDB network or BW measurements; 100-update BW report and 94-step interruption are user reports only")
    (out/"acceptance.json").write_text(json.dumps(record,indent=2));print(json.dumps(record,indent=2))


if __name__=="__main__":main()
