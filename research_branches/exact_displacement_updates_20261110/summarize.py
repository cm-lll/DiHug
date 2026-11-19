"""Descriptive post-run report. No candidate/threshold selection."""
import csv,json,time,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/exact_displacement_updates_20261110'
def main():
    rows=list(csv.DictReader((OUT/'model_summary.csv').open()));summary=[];allsteps=[]
    for ds,seed in sorted(set((r['dataset'],r['seed']) for r in rows)):
        table={r['arm']:r for r in rows if (r['dataset'],r['seed'])==(ds,seed)}
        if 'displacement' not in table:continue
        folder=OUT/f'{ds}_{seed}';u=json.loads((folder/'updates.json').read_text());allsteps+=u
        costs={arm:float(r['mean_five']) for arm,r in table.items()};pair_errors={}
        for arm in ('before','hybrid','common'):
            old=ROOT/'results'/('exact_common_updates_20261110' if arm=='common' else 'exact_feedback_updates_20261110')/folder.name;variances=[]
            for t in (1,75,250,550,800,1000):
                a=np.load(folder/f'evaluation_scored_displacement_t{t}.npz')['costs_remaining'];b=np.load(old/f'evaluation_scored_{arm}_t{t}.npz')['costs_remaining'];delta=(a-b).sum(1)/5;variances.append(delta.var(ddof=1)/len(delta))
            pair_errors[arm]=float(np.sqrt(sum(variances))/6)
        summary.append(dict(dataset=ds,seed=int(seed),mean_five=costs,relative_percent={a:100*(costs['displacement']/costs[a]-1) for a in ('before','hybrid','common')},paired_MC_SE=pair_errors,min_retained_norm_fraction=min(x['retained_norm_fraction'] for x in u),mean_step_seconds=float(np.mean([x['seconds'] for x in u])),max_written_constraint_cosine=max(max(x['actual_update_cosines']) for x in u),steps=u))
    result=dict(models=summary,updates=len(allsteps),max_written_cosine=max(max(u['actual_update_cosines']) for u in allsteps),zero_displacements=sum(u['zero_displacement'] for u in allsteps),min_retained_norm_fraction=min(u['retained_norm_fraction'] for u in allsteps),improved_vs_before=sum(s['relative_percent']['before']<0 for s in summary),improved_vs_mean=sum(s['relative_percent']['hybrid']<0 for s in summary),improved_vs_common=sum(s['relative_percent']['common']<0 for s in summary),limitations=['Four updates per model; conditional clean predictions, not complete reverse-chain generation','Development evaluation reused; only two sampled risk terms have the reported paired MC SE','EMA was maintained but quality readout is raw only','Adam moments retain unprojected mean-gradient proposal semantics'])
    (OUT/'decision_summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='models'},indent=2))
    for x in summary:print(json.dumps({k:v for k,v in x.items() if k!='steps'},indent=2))
if __name__=='__main__':main()
