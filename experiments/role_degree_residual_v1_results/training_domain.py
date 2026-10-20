"""Reconstruct paired training inputs, not optimization; Genre low-t support."""
import os
os.environ.update(OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",CUDA_VISIBLE_DEVICES="")
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"role_specialization_v1"))
from panel import load_panel,c,input_digest,augment,fingerprint,save
from capacity_readout import allocation_marginals,degree_logpmf
from role_condition.model import degrees

def main():
    ap=argparse.ArgumentParser()
    for k in ("baseline","panel","run","output"):ap.add_argument("--"+k,type=Path,required=True)
    a=ap.parse_args();start=time.monotonic();a.output.mkdir(exist_ok=False)
    cfg=json.loads((a.baseline/"resolved_config.json").read_text())
    ledger=json.loads((a.run/"activation.json").read_text())
    records,targets=load_panel(a.panel);train=[r for r in records if r["split"]=="train"]
    logs={(r["seed"],r["update"]):r for r in map(json.loads,(a.baseline/"updates.jsonl").read_text().splitlines())}
    alpha=c.schedule(64);rows=[];verified=0
    for seed in cfg["train_seeds"]:
        rng=np.random.default_rng(seed)
        for u in range(1000):
            assert time.time()<ledger["deadline_epoch"] and not (a.run/"STOP").exists()
            if time.monotonic()-start>60:raise TimeoutError("input-only audit 60s bound")
            rec=train[(u%2)*4+int(rng.integers(4))];case=augment(rec["case"],rng)
            inp=c.train_input(case,64,4,seed,u,rng)
            assert input_digest(case,inp)==logs[seed,u+1]["input_sha256"];verified+=1
            if inp["t"]>8:continue
            typ=case["space"].types;mg=case["space"].pairs.index((2,3))
            d=degrees(typ,inp["edges"],inp["edge_families"],case["n_families"])[typ==3,mg]
            roles=targets[rec["condition"]].roles[3]
            vec=np.array([v[mg] for v,count in roles]);counts=np.array([count for v,count in roles])
            N=int((typ==2).sum());pi=case["prior"][mg]
            ll=np.column_stack([degree_logpmf(N,int(v),alpha[inp["t"]],pi)[d] for v in vec])
            att,_=allocation_marginals(ll,counts);soft=att@vec
            rows.append(dict(seed=seed,update=u+1,t=inp["t"],condition=rec["condition"],
                current_G_degree=d.tolist(),soft_G_degree=soft.tolist()))
    summary=[]
    for condition in ("balanced","extreme"):
        rr=[r for r in rows if r["condition"]==condition]
        all_degree=np.concatenate([r["current_G_degree"] for r in rr])
        soft=np.concatenate([r["soft_G_degree"] for r in rr])
        selected=soft<3 if condition=="extreme" else np.ones(len(soft),bool)
        summary.append(dict(condition=condition,input_graphs=len(rr),nodes=int(selected.sum()),
            selection="soft target<3 (extreme), all nodes (balanced); descriptive, not quality filter",
            degree_histogram=np.bincount(all_degree[selected],minlength=19).tolist(),
            selected_max=int(all_degree[selected].max())))
    save(a.output/"records.json",rows);save(a.output/"summary.json",summary)
    save(a.output/"receipt.json",dict(status="passed",inputs_verified=verified,t_max=8,
        source_updates_sha256=fingerprint(a.baseline/"updates.jsonl"),
        panel_sha256=fingerprint(a.panel/"panel.json"),model_calls=0,training_updates=0,
        scope="exact replay of training input RNG only; node observations are correlated",
        wall_seconds=time.monotonic()-start))
    print(json.dumps(summary))

if __name__=="__main__":main()
