"""Independent integer-degree/role-frequency audit; no forward/model inference."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--run",type=Path,required=True);ap.add_argument("--source",type=Path,required=True)
    a=ap.parse_args();begin=time.monotonic()
    ledger=json.loads((a.run/"activation.json").read_text())
    def check():
        assert time.time()<ledger["deadline_epoch"] and not (a.run/"STOP").exists()
    panel=json.loads((a.source/"panel.json").read_text());targets=panel["anonymous_targets"]
    max_tv=max_w1=0.
    for r in json.loads((a.run/"generation.json").read_text()):
        check()
        with np.load(a.run/r["graph_file"],allow_pickle=False) as z:
            types=z["types"];edge=z["edges"];family=z["relations"]-1;R=len(z["schema"])
        d=np.zeros((len(types),R),int)
        for (u,v),f in zip(edge,family):d[u,f]+=1;d[v,f]+=1
        target=targets[r["condition"]]
        for typ,name in enumerate(["Director","Actor","Movie","Genre"]):
            observed=d[types==typ];n=len(observed)
            ref=np.array([vec for vec,count in target["roles"][typ] for _ in range(count)])
            assert len(ref)==n
            left=Counter(map(tuple,observed));right=Counter(map(tuple,ref))
            tv=sum(abs(left[v]-right[v]) for v in set(left)|set(right))/(2*n)
            err=abs(tv-r["comparison"]["joint_relation_degree_TV"][name]);max_tv=max(max_tv,err)
            assert err<2e-12
            distances=np.abs(np.sort(observed,axis=0)-np.sort(ref,axis=0)).mean(0)
            err=float(abs(distances-np.array(r["comparison"]["typed_relation_degree_W1"][name])).max())
            max_w1=max(max_w1,err);assert err<2e-12
    tensors=0;initial_ce_error=0.
    for seed in [2026102011,2026102012,2026102013]:
        check();p=a.run/f"seed{seed}/initialization.json";init=json.loads(p.read_text())
        oldpath=Path(init["source"]);assert hashlib.sha256(oldpath.read_bytes()).hexdigest()==init["sha256"]
        old=torch.load(oldpath,map_location="cpu",weights_only=True)
        new=torch.load(a.run/f"seed{seed}/capacity_C_initial.pt",map_location="cpu",weights_only=True)
        assert set(old)-set(new)==set(init["removed_tensors"]) and not set(new)-set(old)
        for k in new:assert torch.equal(new[k],old[k]);tensors+=1
        before=json.loads((a.source/f"seed{seed}/late_C_validation_0000.json").read_text())["mean_ce"]
        after=json.loads((a.run/f"seed{seed}/capacity_C_validation_0000.json").read_text())["mean_ce"]
        initial_ce_error=max(initial_ce_error,abs(before-after));assert abs(before-after)<2e-12
        del old,new
    check()
    result=dict(status="passed",graphs=48,all_types_joint_TV_max_error=max_tv,all_types_relation_W1_max_error=max_w1,
        initial_common_tensors_independently_equal=tensors,wall_seconds=time.monotonic()-begin,
        initial_fixed_validation_CE_max_error=initial_ce_error,
        cumulative_seconds=time.time()-ledger["activation_epoch"],no_reset=True)
    (a.run/"joint_independent_audit.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result))
if __name__=="__main__":main()
