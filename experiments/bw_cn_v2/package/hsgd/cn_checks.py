"""Small semantic gate + bounded full-size BW cost measurement. No formal fit."""
import copy
import json
from pathlib import Path
import time
import numpy as np
import torch
from hsgd.bw_preflight import common as c
from hsgd.computational import FrozenTypedContext
from hsgd.kernel import MarginalKernel
from hsgd.runtime import reverse_step
from hsgd.learning_mode import configure
from hsgd.perf_checks import TOLERANCES


def small_model(device="cpu"):
    case=c.make_case(np.array([0,0,1,1,2,2,2,0]),np.array([[0,1,2,2,3,0,1],[2,2,4,5,6,7,6]]),
        tuple(__import__('itertools').combinations_with_replacement(range(3),2)))
    cfg=c.config();cfg["model"].update(layers=2,width=16,edge_width=8,global_width=8,heads=2,
        ff_node=16,ff_edge=8,ff_global=8,cn_hidden=32)
    c.seed_all(1601);m=c.make_model(case,cfg,device)
    return m,case,cfg


def mechanism(device="cpu"):
    m,case,cfg=small_model(device);m.eval()
    query=np.array([[0,0],[1,4]]);fam=case["space"].encode(query)[0]
    ctx=FrozenTypedContext(m,case["space"].types,case["clean"],case["space"].encode(case["clean"])[0],"fixture")
    qt=torch.tensor(query,device=device);qf=torch.tensor(fam,device=device)
    features=m.common_neighbor.aggregate(ctx,qt,qf,.5)
    # Query0 shares node2. Query1 shares node2 as well, with a different target relation.
    if not features.isfinite().all() or features[0,-3:].sum()!=1:raise AssertionError("shared intermediate extraction")
    swapped=m.common_neighbor.aggregate(ctx,qt.flip(0),qf,.5)
    if not torch.equal(features,swapped):raise AssertionError("endpoint swap changed CN features")
    with torch.no_grad():m.common_neighbor.readout.weight.fill_(.01)
    # New context required after a parameter update, including this test assignment.
    ctx.close();ctx=FrozenTypedContext(m,case["space"].types,case["clean"],case["space"].encode(case["clean"])[0],"new")
    graph,logits,p=ctx.predict(query,fam,.5,"new",details=True)
    loss=torch.nn.functional.binary_cross_entropy(p,torch.zeros_like(p));loss.backward()
    grads=[v.grad for v in m.parameters() if v.grad is not None]
    if not grads or not all(torch.isfinite(v).all() for v in grads):raise AssertionError("nonfinite gradient")
    if not any(p.grad is not None and p.grad.abs().sum()>0 for p in m.common_neighbor.phi.parameters()):raise AssertionError("no feature learning gradient")
    ctx.close()
    m.eval();T=4;R=3;kernel=MarginalKernel(case["prior"],c.schedule(T));rng=np.random.default_rng(1602)
    state=(rng.random(case["space"].total)<np.repeat(case["prior"],case["space"].counts)).astype(np.uint8)
    with torch.no_grad():
        for t in range(T,0,-1):state,rec=reverse_step(m,case,state,kernel,t,R,1602,rng,device)
    return dict(status="passed",device=device,checks=["shared intermediate", "symmetric endpoint exchange",
        "finite forward/backward and nonzero phi gradient","complete 4-step/3-block engineering chain"],
        small_reverse_steps=4,not_scientific_output=True,tolerances=TOLERANCES)


def run_check(out,device="cuda",fixture=False):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter();cpu=time.process_time()
    torch.set_num_threads(1);torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;configure(False)
    env=c.environment(device);c.verify_manifest()
    if not fixture and (env["device_name"]!="BW" or not env["hip"]):raise RuntimeError("check requires actual BW/HIP")
    receipt=dict(status="running",engineering_fixture=fixture,environment=env,code=c.code_hashes(),files={})
    c.json_save(out/"check.json",receipt)
    try:
        semantic=mechanism(device);c.json_save(out/"mechanism.json",semantic)
        arms={}
        for arm in ("baseline","cn"):
            c.set_arm(arm);cfg=c.config()
            if fixture:
                model,case,cfg=small_model(device)
                cfg["model"]["cn_hidden"]=32 if arm=="cn" else 0
                model=c.make_model(case,cfg,device)
            else:case=c.imdb_case(c.PACKAGE/"data/IMDB_movie_reference.npz",cfg)
            c.seed_all(cfg["train_seed"]);model=c.make_model(case,cfg,device)
            opt=torch.optim.AdamW(model.parameters(),**cfg["optimizer"])
            rng=np.random.default_rng(1603)
            inp=c.train_input(case,cfg["T"],cfg["rounds"],1603,1,rng,force_t=500)
            if device=="cuda":torch.cuda.reset_peak_memory_stats()
            records=[]
            for i in range(3):
                print(json.dumps(dict(event="cost_update",arm=arm,index=i,phase="warmup" if i==0 else "measured")),flush=True)
                records.append(c.update(model,opt,case,inp,cfg["T"],device,require_parameter_change=False))
            from hsgd.slots import UniformQueryPlan
            state=np.zeros(case["space"].total,dtype=np.uint8)
            f,loc=case["space"].encode(inp["edges"]);state[case["space"].offsets[f]+loc]=1
            model.eval();sampler_rng=np.random.default_rng(1604)
            with torch.no_grad():
                _,step=reverse_step(model,case,state,MarginalKernel(case["prior"],c.schedule(cfg["T"])),
                    500,cfg["rounds"],1604,sampler_rng,device,
                    event=lambda phase,**kw:print(json.dumps(dict(event=phase,arm=arm,**kw)),flush=True) if phase=="reverse_block" else None)
                ctx=FrozenTypedContext(model,case["space"].types,inp["edges"],inp["edge_families"],"cost")
                topology=dict(raw_wedges=ctx.cn_wedges.raw_wedges,index_bytes=ctx.cn_wedges.bytes) if ctx.cn_wedges else None
                ctx.close()
                del ctx
            arms[arm]=dict(dimensions=c.dimensions(model),updates=records,warmup_updates=1,measured_updates=2,
                full_20_block_step=step,cn_topology=topology,**c.memory(device),
                reference=dict(nodes=len(case["space"].types),edges=len(case["clean_ids"]),slots=case["space"].total),
                scope="CPU tiny fixture" if fixture else "random engineering weights, fixed true-scale forward-noised input; not formal learning or a scientific chain")
            c.json_save(out/(arm+"_cost.json"),arms[arm])
            del model,opt
            if device=="cuda":torch.cuda.empty_cache()
        receipt.update(status="passed",arms=arms,checks=["small mechanism forward/backward/short chain",
            "complete query coverage and finite updates (fixture only)" if fixture else "real-size update and complete20-block step finite"],data_sha256=c.config()["imdb_sha256"] if not fixture else "engineering",
            wall_seconds=time.perf_counter()-started,cpu_seconds=time.process_time()-cpu,**c.memory(device))
        receipt["files"]={n:c.sha(out/n) for n in ("mechanism.json","baseline_cost.json","cn_cost.json")}
    except BaseException as exc:
        import traceback
        receipt.update(status="failed",error=type(exc).__name__+": "+str(exc),traceback=traceback.format_exc())
        raise
    finally:c.json_save(out/"check.json",receipt)
    return receipt
