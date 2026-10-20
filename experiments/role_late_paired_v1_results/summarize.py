"""Read-only aggregation of the completed paired panel; no model calls."""
import json
from pathlib import Path
import statistics

root=Path(__file__).resolve().parent
e=root/"evidence"
graphs=json.loads((e/"generation.json").read_text())
seed_rows=json.loads((e/"seed_summary.json").read_text())
out={"scope":"3 paired training seeds; 8 graphs per arm/condition/seed; no inferential significance claim",
     "cells":[],"validation":[],"condition_response":[]}
for condition in ["balanced","extreme"]:
    for arm in ["original_C","late_C"]:
        rows=[x for x in seed_rows if x["condition"]==condition and x["arm"]==arm]
        gs=[x for x in graphs if x["condition"]==condition and x["arm"]==arm]
        cell=dict(condition=condition,arm=arm,training_seeds=3,graphs=24)
        for key in ["G_W1","G_joint_TV","G_max","G_high_ge12","G_multiple_hubs","G_zero",
                    "G_low_le1","edges","lcc","isolates","triangles","closure_mean"]:
            cell[key]=statistics.mean(x[key] for x in rows)
        cell["family_edges"]=[statistics.mean(x["metrics"]["family_edges"][r] for x in gs) for r in range(10)]
        cell["joint_TV"]={t:statistics.mean(x["comparison"]["joint_relation_degree_TV"][t] for x in gs)
                          for t in ["Director","Actor","Movie","Genre"]}
        cell["connected_count"]=sum(x["metrics"]["connected"] for x in gs)
        cell["edge_MAE"]=statistics.mean(abs(x["metrics"]["edges"]-88) for x in gs)
        out["cells"].append(cell)
for seed in [2026102011,2026102012,2026102013]:
    for arm in ["original_C","late_C"]:
        out["validation"].append(dict(seed=seed,arm=arm,
            mean_CE={u:json.loads((e/f"seed{seed}/{arm}_validation_{u:04d}.json").read_text())["mean_ce"]
                     for u in [0,250,500,1000]}))
        responses=[x for x in json.loads((e/f"seed{seed}/condition_response.json").read_text()) if x["arm"]==arm]
        out["condition_response"].append(dict(seed=seed,arm=arm,
            max_MG_p0_difference=max(x["MG_p0_max_abs"] for x in responses),
            max_repeat_difference=max(x["repeat_max_abs"] for x in responses)))
out["maximum_G_degree_across_all_96"]=max(x["G_max"] for x in graphs)
(root/"summary.json").write_text(json.dumps(out,indent=2)+"\n")
print(json.dumps(out,indent=2))
