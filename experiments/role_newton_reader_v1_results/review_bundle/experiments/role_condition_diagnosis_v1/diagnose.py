"""Frozen C weights, original validation inputs; hooks only, no optimization."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import argparse
import hashlib
import json
import math
from pathlib import Path
import resource
import sys
import time
import numpy as np
import torch

BASE = Path(__file__).parent / "baseline"
sys.path.insert(0, str(BASE))
from hsgd.bw_preflight import common as c
from hsgd.computational import FrozenTypedContext
from role_condition.model import paired_models, RoleTarget
from role_condition.experiment import fixed_validation, validate


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dump(p, x):
    Path(p).write_text(json.dumps(x, indent=2, allow_nan=False) + "\n")


def load_panel(run):
    p = json.loads((run / "panel.json").read_text())
    records = [dict(id=r["id"], condition=r["condition"], split=r["split"],
                    case=c.make_case(np.array(r["types"]), np.array(r["edges"]).T, tuple(map(tuple, p["schema"]))))
               for r in p["graphs"]]
    targets = {k: RoleTarget(tuple(tuple((tuple(v), int(n)) for v,n in rows) for rows in x["roles"]),
                             tuple(x["type_counts"]), tuple(map(tuple,x["schema"])))
               for k,x in p["anonymous_targets"].items()}
    return records, targets


def diff(a, b):
    d = np.asarray(a,dtype=np.float64)-np.asarray(b,dtype=np.float64)
    return dict(max_abs=float(np.max(np.abs(d))) if d.size else 0.,
                rms=float(np.sqrt(np.mean(d*d))) if d.size else 0.,
                mean_signed=float(d.mean()) if d.size else 0.)


def trace(model, case, inp, plan, target, cfg):
    """Observe actual branch operations without replacing its forward."""
    model.set_target(target); model.eval()
    parts = {k: [] for k in ["role_encoder","query","key","value","role_out","degree_out"]}
    handles = []
    for name in parts:
        def hook(mod, args, out, name=name):
            parts[name].append(out.detach().cpu().numpy().copy())
        handles.append(getattr(model.branch,name).register_forward_hook(hook))
    logits, probs, families, ids, queries = [], [], [], [], []
    token = ("diagnostic",int(inp["t"]))
    ctx = FrozenTypedContext(model,case["space"].types,inp["edges"],inp["edge_families"],token)
    try:
        with torch.no_grad():
            for k in range(cfg["rounds"]):
                ii, ff, qq = plan.block(k)
                _, ll, pp = ctx.predict(qq,ff,inp["t"]/cfg["T"],token,details=True)
                logits.append(ll.numpy().copy());probs.append(pp.numpy().copy())
                ids.append(ii);families.append(ff);queries.append(qq)
            context = ctx.condition_diagnostic["role_context"].numpy().copy()
            delta = ctx.node_residual.numpy().copy()
            degree = ctx.condition_diagnostic["degree"].copy()
    finally:
        ctx.close()
        for h in handles:h.remove()
    order=np.argsort(np.concatenate(ids))
    result=dict(logits=np.concatenate(logits)[order],p0=np.concatenate(probs)[order],
                ids=np.concatenate(ids)[order],families=np.concatenate(families)[order],
                query=np.concatenate(queries,axis=1)[:,order],
                role_context=context,residual=delta,degree=degree,
                role_projection=parts["role_out"][0],degree_projection=parts["degree_out"][0])
    for typ, rows in enumerate(target.roles):
        q,k,v = (parts[x][typ] for x in ["query","key","value"])
        freq=np.array([n/target.type_counts[typ] for _,n in rows])
        # Recompute FP32 softmax for observing attention; require its actual
        # aggregate to agree with the context captured from original forward.
        att=(torch.from_numpy(q)@torch.from_numpy(k).T/math.sqrt(32)+torch.tensor(np.log(freq),dtype=torch.float32)).softmax(-1).numpy()
        ix=case["space"].types==typ
        np.testing.assert_allclose(att@v,context[ix],atol=2e-7,rtol=2e-6)
        entropy=-(att*np.log(att)).sum(1)
        for name,arr in dict(encoded=parts["role_encoder"][typ],query=q,key=k,value=v,
                             attention=att,entropy=entropy,frequency=freq,
                             roles=np.array([vec for vec,_ in rows])).items():
            result[f"type{typ}_{name}"]=arr
    result["logit_gap"]=result["logits"][:,1]-result["logits"][:,0]
    return result


def aligned_tokens(a,b,typ):
    # Distinct role sets have different rows. Compare per-channel weighted
    # first/second moments, save all raw token arrays; never subtract unlike roles.
    out={}
    for field in ["encoded","key","value"]:
        moments=[]
        for x in [a,b]:
            v=x[f"type{typ}_{field}"];f=x[f"type{typ}_frequency"]
            moments.append(((f[:,None]*v).sum(0),(f[:,None]*v*v).sum(0)))
        out[field+"_weighted_mean"]=diff(moments[1][0],moments[0][0])
        out[field+"_weighted_second"]=diff(moments[1][1],moments[0][1])
    return out


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--run",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    args=ap.parse_args();out=args.output;out.mkdir(exist_ok=False,parents=True)
    tic=time.perf_counter();cpu=time.process_time()
    torch.set_num_threads(1);torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(False)
    cfg=json.loads((args.run/"resolved_config.json").read_text())
    assert cfg==json.loads((BASE/"role_condition/config.json").read_text())
    records,targets=load_panel(args.run);values=fixed_validation(records,cfg)
    expected=json.loads((args.run/"analysis/weight_hashes.json").read_text())
    sources={str(args.run/"panel.json"):sha(args.run/"panel.json"),
             str(args.run/"resolved_config.json"):sha(args.run/"resolved_config.json"),
             str(args.run/"analysis/weight_hashes.json"):sha(args.run/"analysis/weight_hashes.json")}
    # Baseline manifest is review-phase frozen, including nested source files.
    manifest=json.loads((BASE/"manifest.json").read_text())
    for name,h in manifest.items():
        if isinstance(h,dict):h=h["sha256"]
        assert sha(BASE/name)==h
    rows=[];changes=[];calls=0
    for seed in cfg["train_seeds"]:
        sd=args.run/f"seed{seed}";path=sd/"C_final_1000.pt"
        assert path.is_file(),"Existing C weights missing; no replacement training allowed"
        assert sha(path)==expected[f"seed{seed}/C_final_1000.pt"]
        sources[str(path)]=sha(path)
        models=paired_models(records[0]["case"],cfg,seed);model=models["C"]
        del models["A"],models["B"]
        initial={k:v.clone() for k,v in model.state_dict().items()}
        # Initial weights were not saved. Deterministic reconstruction is checked
        # against the stored whole initial validation result, not called original bytes.
        v=validate(model,values,cfg,targets,lambda:None)
        old=json.loads((sd/"C_validation_0000.json").read_text())
        assert abs(v["mean_ce"]-old["mean_ce"])<2e-8
        weights=torch.load(path,map_location="cpu",weights_only=True)
        model.load_state_dict(weights,strict=True);model.eval()
        per_parameter=[]
        for name,p in model.named_parameters():
            d=p.detach()-initial[name]
            per_parameter.append(dict(name=name,changed=bool(torch.any(d!=0)),
                                      max_abs=float(d.abs().max()),l2=float(d.norm()),
                                      initial_l2=float(initial[name].norm()),final_l2=float(p.detach().norm())))
        changes.append(dict(seed=seed,initial_source="reconstructed original paired_models using original seed and deps; not saved initial bytes",
                            initial_validation_abs_error=abs(v["mean_ce"]-old["mean_ce"]),parameters=per_parameter))
        for index,(identifier,condition,case,inp,panel_index) in enumerate(values):
            plan=c.plan_for(case,[cfg["validation_seed"],panel_index,21],cfg["rounds"])
            traces=[]
            for cond in ["balanced","extreme","balanced"]:
                traces.append(trace(model,case,inp,plan,targets[cond],cfg));calls+=cfg["rounds"]
            a,b,repeat=traces;mg=a["families"]==case["space"].pairs.index((2,3))
            stats={k:diff(b[k],a[k]) for k in ["role_context","role_projection","degree_projection","residual","logits","logit_gap","p0"]}
            stats["MG_logit_gap"]=diff(b["logit_gap"][mg],a["logit_gap"][mg])
            stats["MG_p0"]=diff(b["p0"][mg],a["p0"][mg])
            by_type={}
            for typ in range(case["n_types"]):
                ix=case["space"].types==typ
                item=aligned_tokens(a,b,typ)
                for k in ["role_context","role_projection","residual"]:
                    item[k]=diff(b[k][ix],a[k][ix])
                item["attention"]={}
                for cond,x in zip(["balanced","extreme"],[a,b]):
                    att=x[f"type{typ}_attention"];freq=x[f"type{typ}_frequency"]
                    item["attention"][cond]=dict(tokens=len(freq),entropy_mean=float(x[f"type{typ}_entropy"].mean()),
                        entropy_min=float(x[f"type{typ}_entropy"].min()),
                        deviation_from_frequency_max=float(np.abs(att-freq[None]).max()),
                        node_variation_max=float(np.max(np.ptp(att,axis=0))))
                by_type[str(typ)]=item
            name=f"seed{seed}_input{index:02d}"
            np.savez_compressed(out/(name+".npz"),**{cond+"__"+k:val for cond,x in zip(["balanced","extreme","repeat"],traces) for k,val in x.items()},
                                Xt_edges=inp["edges"],Xt_families=inp["edge_families"],node_types=case["space"].types)
            row=dict(seed=seed,input_id=identifier,input_condition=condition,t=inp["t"],panel_index=panel_index,
                     target_difference=stats,by_type=by_type,
                     repeat_noise={k:diff(repeat[k],a[k]) for k in ["role_context","role_projection","logits","logit_gap","p0"]},
                     MG_expected_edges={cond:float(x["p0"][mg].sum()) for cond,x in zip(["balanced","extreme"],[a,b])},
                     trace_file=name+".npz")
            rows.append(row)
        print("seed",seed,"12 fixed validation states traced",flush=True)
        del model,models,weights,initial
    assert all(sha(p)==h for p,h in sources.items())
    dump(out/"parameter_changes.json",changes);dump(out/"records.json",rows)
    dump(out/"sources.json",sources)
    dump(out/"receipt.json",dict(status="completed",seeds=cfg["train_seeds"],fixed_states_per_seed=len(values),
        probability_forward_calls=calls,initial_validation_forward_calls=3*len(values)*cfg["rounds"],
        neural_backbone_calls_note="CN may use an additional context backbone pass; above counts are prediction API calls",
        training_updates=0,complete_generation_chains=0,GPU=0,
        wall_seconds=time.perf_counter()-tic,cpu_seconds=time.process_time()-cpu,
        peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,environment=c.environment("cpu")))


if __name__=="__main__":
    main()
