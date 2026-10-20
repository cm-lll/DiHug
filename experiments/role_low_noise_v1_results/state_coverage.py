"""Read-only reconstruction of actual old/new training states; no NN calls."""
import os
os.environ.update(OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",CUDA_VISIBLE_DEVICES="")
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"role_low_noise_v1"))
from learning import load_panel,c,input_digest,augment,save,fingerprint
from time_proposal import paired_input
from capacity_readout import allocation_marginals,degree_logpmf
from role_condition.model import degrees

def main():
    ap=argparse.ArgumentParser()
    for flag in ("baseline","panel","run","output"):ap.add_argument("--"+flag,type=Path,required=True)
    a=ap.parse_args();start=time.monotonic();a.output.mkdir(exist_ok=False)
    ledger=json.loads((a.run/"activation.json").read_text())
    records,targets=load_panel(a.panel);train=[r for r in records if r["split"]=="train"]
    old={(r["seed"],r["update"]):r for r in map(json.loads,(a.baseline/"updates.jsonl").read_text().splitlines())
         if r["arm"]=="capacity_C"}
    new={(r["seed"],r["update"]):r for r in map(json.loads,(a.run/"updates.jsonl").read_text().splitlines())}
    alpha=c.schedule(64);rows=[];verified=0
    for seed in (2026102011,2026102012,2026102013):
        rng=np.random.default_rng(seed)
        for u in range(1000):
            assert time.time()<ledger["deadline_epoch"] and not (a.run/"STOP").exists()
            if time.monotonic()-start>90:raise TimeoutError("read-only state coverage audit90s")
            rec=train[(u%2)*4+int(rng.integers(4))];case=augment(rec["case"],rng)
            candidate,baseline,info=paired_input(case,seed,u,rng)
            assert input_digest(case,baseline)==old[seed,u+1]["input_sha256"]
            assert input_digest(case,candidate)==new[seed,u+1]["input_sha256"];verified+=1
            for arm,inp,weight in (("capacity_C",baseline,1.),("importance_C",candidate,info["importance_weight"])):
                if inp["t"]>8:continue
                typ=case["space"].types;mg=case["space"].pairs.index((2,3))
                d=degrees(typ,inp["edges"],inp["edge_families"],case["n_families"])[typ==3,mg]
                # Independently count both endpoints as a diagnostic consistency check.
                raw=np.zeros(len(typ),int)
                for (left,right),family in zip(inp["edges"].T,inp["edge_families"]):
                    if family==mg:raw[left]+=1;raw[right]+=1
                np.testing.assert_array_equal(d,raw[typ==3])
                roles=targets[rec["condition"]].roles[3]
                vec=np.array([v[mg] for v,count in roles]);counts=np.array([count for v,count in roles])
                n=int((typ==2).sum());pi=case["prior"][mg]
                likelihood=np.column_stack([degree_logpmf(n,int(v),alpha[inp["t"]],pi)[d] for v in vec])
                attention,_=allocation_marginals(likelihood,counts);soft=attention@vec
                rows.append(dict(seed=seed,update=u+1,arm=arm,t=inp["t"],condition=rec["condition"],
                    importance_weight=weight,current_G_degree=d.tolist(),soft_G_degree=soft.tolist()))
    summary=[]
    bins=[("t1",1,1),("t2",2,2),("t3_4",3,4),("t5_8",5,8),("t1_8",1,8)]
    for arm in ("capacity_C","importance_C"):
        for condition in ("balanced","extreme"):
            for seed in (None,2026102011,2026102012,2026102013):
                for label,low,high in bins:
                    rr=[r for r in rows if r["arm"]==arm and r["condition"]==condition and low<=r["t"]<=high
                        and (seed is None or r["seed"]==seed)]
                    hist=np.zeros(19,int);weighted=np.zeros(19,float);allhist=np.zeros(19,int)
                    for r in rr:
                        d=np.array(r["current_G_degree"]);soft=np.array(r["soft_G_degree"])
                        selected=soft<3 if condition=="extreme" else np.ones(len(d),bool)
                        h=np.bincount(d[selected],minlength=19);hist+=h
                        weighted+=h*r["importance_weight"];allhist+=np.bincount(d,minlength=19)
                    summary.append(dict(arm=arm,condition=condition,seed=seed,time_bin=label,input_graphs=len(rr),
                        selected_nodes=int(hist.sum()),selected_degree_histogram=hist.tolist(),
                        selected_degree5_6=int(hist[5:7].sum()),selected_degree_ge5=int(hist[5:].sum()),
                        importance_weighted_histogram=weighted.tolist(),all_nodes_histogram=allhist.tolist()))
    save(a.output/"records.json",rows);save(a.output/"summary.json",summary)
    save(a.output/"receipt.json",dict(status="passed",paired_updates_verified=verified,model_calls=0,training_updates=0,
        low_t_range=[1,8],source_updates_sha256=fingerprint(a.baseline/"updates.jsonl"),
        candidate_updates_sha256=fingerprint(a.run/"updates.jsonl"),panel_sha256=fingerprint(a.panel/"panel.json"),
        selection="Extreme soft role mean<3; balanced all Genre. Soft assignments are diagnostic, not true roles.",
        caution="Correlated observations, not independent nodes/graphs; occurrence counts do not prove learning effect.",
        wall_seconds=time.monotonic()-start))
    print(json.dumps([r for r in summary if r["seed"] is None]))
if __name__=="__main__":main()
