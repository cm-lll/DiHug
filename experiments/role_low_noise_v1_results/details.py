"""All predeclared relations/conditions/times; no ranking or selection."""
import argparse
import json
from pathlib import Path
import statistics
import time
import numpy as np

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--run",type=Path,required=True);ap.add_argument("--source",type=Path,required=True)
    a=ap.parse_args();start=time.monotonic()
    ledger=json.loads((a.run/"activation.json").read_text())
    assert time.time()<ledger["deadline_epoch"] and not (a.run/"STOP").exists()
    validation=[];families=[];training=[]
    for arm,root in (("capacity_C",a.source),("importance_C",a.run)):
        generated=[r for r in json.loads((root/"generation.json").read_text()) if r["arm"]==arm]
        updates=[json.loads(x) for x in (root/"updates.jsonl").read_text().splitlines() if json.loads(x)["arm"]==arm]
        for seed in (2026102011,2026102012,2026102013):
            uu=[r for r in updates if r["seed"]==seed]
            training.append(dict(arm=arm,seed=seed,updates=len(uu),
                low_t_count=sum(r["t"]<=8 for r in uu),high_t_count=sum(r["t"]>8 for r in uu),
                stable_update_median=float(np.median([r["seconds"] for r in uu[10:]])),
                stable_update_P90=float(np.quantile([r["seconds"] for r in uu[10:]],.9)),
                update_timer_sum=sum(r["seconds"] for r in uu),
                loss_scope="weighted optimizer CE" if arm=="importance_C" else "uniform-t optimizer CE"))
            for update in (0,250,500,1000):
                doc=json.loads((root/f"seed{seed}/{arm}_validation_{update:04d}.json").read_text())
                for condition in ("balanced","extreme"):
                    rr=[r for r in doc["rows"] if r["condition"]==condition]
                    for t in sorted(set(r["t"] for r in rr)):
                        ss=[r for r in rr if r["t"]==t]
                        sums=np.array([r["ce_sum_by_family_label"] for r in ss]).sum(0)
                        counts=np.array([r["counts_by_family_label"] for r in ss]).sum(0)
                        means=[[float(sums[i,j]/counts[i,j]) if counts[i,j] else None for j in range(2)] for i in range(len(sums))]
                        validation.append(dict(arm=arm,seed=seed,update=update,condition=condition,t=t,
                            original_objective_CE=statistics.mean(r["ce"] for r in ss),
                            family_label_count=counts.tolist(),family_label_mean_CE=means,
                            expected_family_edges=np.mean([r["expected_edges"] for r in ss],0).tolist(),
                            expected_family_edge_bias=np.mean([r["expected_edge_bias"] for r in ss],0).tolist(),
                            normalization="unweighted fixed original full-slot validation"))
            for condition in ("balanced","extreme"):
                rr=[r for r in generated if r["seed"]==seed and r["condition"]==condition]
                families.append(dict(arm=arm,seed=seed,condition=condition,n=len(rr),
                    family_edges=np.mean([r["metrics"]["family_edges"] for r in rr],0).tolist(),
                    all_types_joint_TV={typ:statistics.mean(r["comparison"]["joint_relation_degree_TV"][typ] for r in rr)
                        for typ in ("Director","Actor","Movie","Genre")},
                    all_types_relation_degree_W1={typ:np.mean([r["comparison"]["typed_relation_degree_W1"][typ] for r in rr],0).tolist()
                        for typ in ("Director","Actor","Movie","Genre")}))
    result=dict(validation=validation,families=families,training=training,wall_seconds=time.monotonic()-start,
        caution="Training loss scopes are distinct realized sampling panels; only original fixed validation is directly comparable.")
    (a.run/"details.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(training))
if __name__=="__main__":main()
