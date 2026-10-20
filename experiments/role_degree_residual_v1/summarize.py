"""Read-only aggregate of the frozen candidate; no fitting or sample selection."""
import argparse
import json
from pathlib import Path
import statistics
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--run",type=Path,required=True);ap.add_argument("--source",type=Path,required=True)
    a=ap.parse_args()
    rows=json.loads((a.run/"seed_summary.json").read_text())
    summary={"macro":[],"paired":[],"validation":[]}
    for condition in ["balanced","extreme"]:
        for arm in ["capacity_C","degree_C"]:
            rr=[x for x in rows if x["condition"]==condition and x["arm"]==arm]
            assert len(rr)==3
            cell=dict(condition=condition,arm=arm,training_seeds=3)
            for key in ["G_W1","G_joint_TV","G_max","G_high_ge12","G_multiple_hubs","G_zero","G_low_le1",
                        "edges","lcc","isolates","triangles","closure","edge_MAE"]:
                cell[key]=None if any(x[key] is None for x in rr) else statistics.mean(x[key] for x in rr)
            cell["connected_count"]=sum(x["connected"] for x in rr)
            summary["macro"].append(cell)
        for seed in [2026102011,2026102012,2026102013]:
            old=next(x for x in rows if x["condition"]==condition and x["seed"]==seed and x["arm"]=="capacity_C")
            new=next(x for x in rows if x["condition"]==condition and x["seed"]==seed and x["arm"]=="degree_C")
            summary["paired"].append(dict(seed=seed,condition=condition,new_minus_baseline={
                k:new[k]-old[k] for k in ["G_W1","G_joint_TV","G_max","G_high_ge12","G_multiple_hubs","G_low_le1","edges","edge_MAE","closure","lcc"]}))
    updates={}
    for label,root in [("capacity_C",a.source),("degree_C",a.run)]:
        updates[label]=[json.loads(x) for x in (root/"updates.jsonl").read_text().splitlines() if json.loads(x)["arm"]==label]
    fig,axes=plt.subplots(3,2,figsize=(11,10))
    for i,seed in enumerate([2026102011,2026102012,2026102013]):
        for label,root in [("capacity_C",a.source),("degree_C",a.run)]:
            rr=[x for x in updates[label] if x["seed"]==seed];assert len(rr)==1000
            loss=np.array([x["loss"] for x in rr]).reshape(40,25).mean(1)
            axes[i,0].plot(np.arange(25,1001,25),loss,label=label)
            vals=[json.loads((root/f"seed{seed}/{label}_validation_{u:04d}.json").read_text())["mean_ce"] for u in [0,250,500,1000]]
            axes[i,1].plot([0,250,500,1000],vals,marker="o",label=label)
            summary["validation"].append(dict(seed=seed,arm=label,updates=[0,250,500,1000],mean_ce=vals))
        axes[i,0].set_title(f"seed {seed}: training CE (25-update mean)")
        axes[i,1].set_title(f"seed {seed}: fixed validation CE")
        for ax in axes[i]:ax.legend();ax.set_xlabel("effective updates")
    fig.tight_layout();fig.savefig(a.run/"learning_curves.png",dpi=140);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,3.6))
    for i,condition in enumerate(["balanced","extreme"]):
        for j,seed in enumerate([2026102011,2026102012,2026102013]):
            yy=[next(r for r in rows if r["condition"]==condition and r["arm"]==arm and r["seed"]==seed)["G_W1"]
                for arm in ["capacity_C","degree_C"]]
            axes[i].plot([0,1],yy,marker="o",label=str(seed))
        axes[i].set_xticks([0,1],["capacity C","degree C"]);axes[i].set_ylabel("actual Genre degree W1")
        axes[i].set_title(condition);axes[i].legend(fontsize=8)
    for arm,root in [("capacity_C",a.source),("degree_C",a.run)]:
        rr=[r for r in json.loads((root/"generation.json").read_text()) if r["condition"]=="extreme" and r["arm"]==arm]
        axes[2].hist([r["G_max"] for r in rr],bins=np.arange(-.5,19.5,1),alpha=.6,label=arm)
    axes[2].axvline(18,color="black",linestyle="--",label="target max=18")
    axes[2].set_title("extreme: all24 graphs per arm");axes[2].set_xlabel("Genre maximum degree");axes[2].legend(fontsize=8)
    fig.tight_layout();fig.savefig(a.run/"actual_graph_effects.png",dpi=140);plt.close(fig)
    summary["statistics_scope"]="Means computed within each trained seed first; 3 paired training repeats. No significance or generalization claim."
    (a.run/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))
if __name__=="__main__":main()
