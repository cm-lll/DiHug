"""Read frozen paired weights; no learning or reverse-chain generation."""
import os
os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="")
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

ROOT=Path(__file__).resolve().parent
PREV=ROOT.parent/"role_condition_diagnosis_v1"
sys.path[:0]=[str(ROOT.parent/"role_late_paired_v1"),str(PREV),str(PREV/"baseline")]
from pairing import construct
from diagnose import load_panel
from hsgd.bw_preflight import common as c
from hsgd.computational import FrozenTypedContext
from role_condition.model import degrees

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--run",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True);args=ap.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.set_float32_matmul_precision("highest")
    started=time.monotonic();deadline=started+300;cpu=time.process_time()
    cfg=json.loads((args.run/"resolved_config.json").read_text())
    records,targets=load_panel(args.run)
    hashes=json.loads((args.run/"weight_hashes.json").read_text())
    inputs={str(args.run/n):digest(args.run/n) for n in ["panel.json","resolved_config.json","weight_hashes.json"]}
    rows=[]
    for seed in cfg["train_seeds"]:
        models,_=construct(records[0]["case"],cfg,seed)
        for arm,model in models.items():
            path=args.run/f"seed{seed}/{arm}_final_1000.pt"
            assert digest(path)==hashes[str(path.relative_to(args.run))]
            inputs[str(path)]=digest(path)
            model.load_state_dict(torch.load(path,map_location="cpu",weights_only=True),strict=True)
            model.eval()
            for condition in ["balanced","extreme"]:
                rec=next(r for r in records if r["condition"]==condition and r["split"]=="validation")
                case=rec["case"];mg=case["space"].pairs.index((2,3))
                for t in [1,8,32,64]:
                    assert time.monotonic()<deadline
                    rng=np.random.default_rng([2026102077,seed,t,0 if condition=="balanced" else 1])
                    inp=c.train_input(case,64,4,2026102077,t,rng,force_t=t)
                    plan=c.plan_for(case,[2026102077,t,21],4)
                    for target_name in ["balanced","extreme"]:
                        model.set_target(targets[target_name]);captured={k:[] for k in ["query","key","value"]}
                        hooks=[]
                        for name in captured:
                            def hook(mod,args,out,name=name):captured[name].append(out.detach().numpy().copy())
                            hooks.append(getattr(model.branch,name).register_forward_hook(hook))
                        ctx=FrozenTypedContext(model,case["space"].types,inp["edges"],inp["edge_families"],t)
                        p0=[];queries=[];families=[]
                        try:
                            with torch.no_grad():
                                for block in range(4):
                                    ids,f,q=plan.block(block)
                                    _,ll,pp=ctx.predict(q,f,t/64,t,details=True)
                                    p0.append(c.check_prob(pp,f,case));queries.append(q);families.append(f)
                            context=ctx.condition_diagnostic["role_context"].numpy()
                        finally:
                            ctx.close()
                            for h in hooks:h.remove()
                        p=np.concatenate(p0);q=np.concatenate(queries,axis=1);f=np.concatenate(families)
                        mu=np.zeros((len(case["space"].types),case["n_families"]))
                        np.add.at(mu,(q[0],f),p);np.add.at(mu,(q[1],f),p)
                        current=degrees(case["space"].types,inp["edges"],inp["edge_families"],case["n_families"])
                        clean=degrees(case["space"].types,case["clean"],case["space"].encode(case["clean"])[0],case["n_families"])
                        ix=np.flatnonzero(case["space"].types==3)
                        freq=np.array([n/6 for v,n in targets[target_name].roles[3]])
                        aq,ak,av=(captured[n][3] for n in ["query","key","value"])
                        att=torch.softmax(torch.from_numpy(aq)@torch.from_numpy(ak).T/math.sqrt(32)+torch.tensor(np.log(freq),dtype=torch.float32),-1).numpy()
                        np.testing.assert_allclose(att@av,context[ix],atol=2e-7,rtol=2e-6)
                        high_role=[v[mg] for v,n in targets[target_name].roles[3]]
                        row=dict(seed=seed,arm=arm,input_condition=condition,target=target_name,t=t,
                            Xt_G_degree=current[ix,mg].tolist(),clean_G_degree=clean[ix,mg].tolist(),
                            predicted_G_degree=mu[ix,mg].tolist(),expected_MG_edges=float(mu[ix,mg].sum()),
                            expected_degree_of_true_hub=float(mu[ix[np.argmax(clean[ix,mg])],mg]) if condition=="extreme" else None,
                            attention=att.tolist(),role_degrees=high_role,attention_node_range=float(np.ptp(att,axis=0).max()),
                            attention_deviation_from_frequency=float(np.abs(att-freq).max()),
                            role_context_node_range=float(np.ptp(context[ix],axis=0).max()),
                            attention_entropy=float(-(att*np.log(att)).sum(1).mean()))
                        rows.append(row)
            print("completed frozen diagnosis",seed,arm,flush=True)
    assert all(digest(p)==h for p,h in inputs.items())
    (args.output/"records.json").write_text(json.dumps(rows,indent=2)+"\n")
    (args.output/"sources.json").write_text(json.dumps(inputs,indent=2)+"\n")
    (args.output/"receipt.json").write_text(json.dumps(dict(status="completed",records=len(rows),
        weights=6,training_updates=0,generated_chains=0,GPU=0,
        state_scope="reference validation graphs forward-noised at t=1,8,32,64; NOT reverse-chain intermediates",
        wall_seconds=time.monotonic()-started,CPU_seconds=time.process_time()-cpu,
        peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024),indent=2)+"\n")
if __name__=="__main__":main()
