"""Export archived all-seed curves. Read-only; no inference or training."""
import argparse
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser();p.add_argument("--analysis",type=Path,required=True);a=p.parse_args()
    info=json.loads((a.analysis/"analysis.json").read_text())
    seeds=(2026102011,2026102012,2026102013);colors={"normal2":"#2768a0","robust":"#d95f02"}
    fig,axes=plt.subplots(2,3,figsize=(13,7),sharex=True)
    for col,seed in enumerate(seeds):
        for row,condition in enumerate(("balanced","extreme")):
            ax=axes[row,col]
            for arm in colors:
                r=sorted((r for r in info["validation"] if (r["seed"],r["arm"],r["condition"],r["t"])==(seed,arm,condition,1)),key=lambda r:r["update"])
                ax.plot([r["update"] for r in r],[r["ce"] for r in r],label=arm,color=colors[arm])
            ax.set(title=f"{condition}, train seed {seed}",ylabel="Fixed ordinary t=1 CE",xlabel="Effective updates")
            ax.grid(alpha=.2)
    axes[0,0].legend();fig.tight_layout();fig.savefig(a.analysis/"fixed_validation_t1.png",dpi=150);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(13,7),sharex=True)
    fields=("G_W1","MG_signed_bias","G_low_le1")
    for row,condition in enumerate(("balanced","extreme")):
        for col,metric in enumerate(fields):
            ax=axes[row,col]
            for arm in colors:
                for seed in seeds:
                    r=sorted((r for r in info["per_seed"] if (r["seed"],r["arm"],r["condition"])==(seed,arm,condition)),key=lambda r:r["checkpoint_update"])
                    ax.plot([x["checkpoint_update"] for x in r],[x[metric] for x in r],color=colors[arm],alpha=.4,marker=".")
                r=sorted((r for r in info["macro"] if (r["arm"],r["condition"])==(arm,condition)),key=lambda r:r["checkpoint_update"])
                ax.plot([x["checkpoint_update"] for x in r],[x[metric] for x in r],color=colors[arm],lw=2.5,marker="o",label=arm)
            ax.set(title=condition,ylabel=metric,xlabel="Evaluated checkpoint");ax.grid(alpha=.2)
    axes[0,0].legend();fig.tight_layout();fig.savefig(a.analysis/"generation_all_seeds.png",dpi=150);plt.close(fig)
    train=list(csv.DictReader((a.analysis/"training_curves.csv").open()))
    fig,axes=plt.subplots(1,3,figsize=(13,3.6))
    for ax,seed in zip(axes,seeds):
        for arm in colors:
            r=[r for r in train if int(r["seed"])==seed and r["arm"]==arm]
            ax.plot([int(x["end_update"]) for x in r],[float(x["mixed_objective_loss"]) for x in r],color=colors[arm],label=arm)
        ax.set(title=f"Train seed {seed}",xlabel="Effective updates",ylabel="100-update mean, arm's own objective");ax.grid(alpha=.2)
    axes[0].legend();fig.suptitle("Objectives differ: ordinary+ordinary vs ordinary+perturbed; compare fixed validation separately")
    fig.tight_layout();fig.savefig(a.analysis/"training_objectives.png",dpi=150);plt.close(fig)
    print("Three archived figures exported; no model calls")

if __name__=="__main__":main()
