"""Reconstruct all inputs without network calls; audit proposal and coupling."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from learning import load_panel, augment, paired_input, input_digest

def main():
    ap=argparse.ArgumentParser()
    for flag in ("run","source","panel"): ap.add_argument("--"+flag,type=Path,required=True)
    a=ap.parse_args();started=time.monotonic()
    ledger=json.loads((a.run/"activation.json").read_text())
    logs=[json.loads(x) for x in (a.run/"updates.jsonl").read_text().splitlines()]
    old={(r["seed"],r["update"]):r for r in map(json.loads,(a.source/"updates.jsonl").read_text().splitlines())
         if r["arm"]=="capacity_C"}
    records,_=load_panel(a.panel);train=[r for r in records if r["split"]=="train"]
    rows=[]
    for seed in (2026102011,2026102012,2026102013):
        rng=np.random.default_rng(seed)
        hist=np.zeros(64,int);base=np.zeros(64,int);weighted=np.zeros(64,float)
        for u,row in enumerate(r for r in logs if r["seed"]==seed):
            assert time.time()<ledger["deadline_epoch"] and not (a.run/"STOP").exists()
            rec=train[(u%2)*4+int(rng.integers(4))]
            case=augment(rec["case"],rng)
            inp,original,info=paired_input(case,seed,u,rng)
            assert input_digest(case,inp)==row["input_sha256"]
            assert input_digest(case,original)==row["baseline_input_sha256"]==old[seed,u+1]["input_sha256"]
            assert info==row["proposal"] and rec["condition"]==row["condition"]
            assert row["t"]==inp["t"] and row["query_block"]==inp["round_id"]
            assert row["positive_count"]==int(inp["labels"].sum())
            assert row["negative_count"]==len(inp["labels"])-int(inp["labels"].sum())
            hist[info["t"]-1]+=1;base[original["t"]-1]+=1
            weighted[info["t"]-1]+=info["importance_weight"]
        assert hist.sum()==1000
        rows.append(dict(seed=seed,new_histogram=hist.tolist(),old_histogram=base.tolist(),
                         weighted_histogram=weighted.tolist(),new_low_t=int(hist[:8].sum()),old_low_t=int(base[:8].sum())))
    result=dict(status="passed",reconstructed_inputs=3000,baseline_inputs=3000,
                exact_query_and_label_pairing=True,identical_training_Xt=False,seeds=rows,
                wall_seconds=time.monotonic()-started)
    (a.run/"input_audit.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result))
if __name__=="__main__": main()
