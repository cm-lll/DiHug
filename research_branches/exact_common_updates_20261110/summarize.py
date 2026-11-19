"""Descriptive summaries only, no new model computation or selection."""
import csv,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/exact_common_updates_20261110'
def main():
    rows=list(csv.DictReader((OUT/'model_summary.csv').open()));mech=list(csv.DictReader((OUT/'mechanism.csv').open()));summaries=[]
    for ds,seed in sorted(set((x['dataset'],x['seed']) for x in rows)):
        rr={x['arm']:x for x in rows if (x['dataset'],x['seed'])==(ds,seed)}
        if 'common' not in rr:continue
        folder=OUT/f'{ds}_{seed}';updates=json.loads((folder/'updates.json').read_text());directions=[]
        for u in updates:
            harmful=[name for name,cos in zip(['count','joint_degree','present_triangle','absent_triangle','components'],u['actual_update_cosines']) if cos>1e-5]
            directions.append(dict(update=u['update'],t=u['t'],coefficient=u['coefficients'],correction_relative_norm=u['correction_relative_norm'],harmful_first_order_groups=harmful,combined_gradient_min_cos=min(u['combined_cosines']),zero_gradient=u['zero_combined_gradient']))
        vals={a:float(rr[a]['mean_five']) for a in rr};old=ROOT/'results/exact_feedback_updates_20261110'/folder.name
        variances=[]
        for t in [1,75,250,550,800,1000]:
            a=np.load(folder/f'evaluation_scored_common_t{t}.npz')['costs_remaining'];b=np.load(old/f'evaluation_scored_hybrid_t{t}.npz')['costs_remaining'];d=(a-b).sum(1)/5;variances.append(d.var(ddof=1)/len(d))
        summaries.append(dict(dataset=ds,seed=int(seed),risk=vals,common_vs_mean_percent=100*(vals['common']/vals['hybrid']-1),common_vs_before_percent=100*(vals['common']/vals['before']-1),common_minus_mean_paired_sampling_SE=float(np.sqrt(sum(variances))/6),directions=directions,mean_step_seconds=float(np.mean([u['seconds'] for u in updates])),mean_solver_seconds=float(np.mean([u['solver_seconds'] for u in updates])),mean_gradient_norm=float(np.mean([u['gradient_norm'] for u in updates])),mean_displacement_norm=float(np.mean([u['displacement_norm'] for u in updates]))))
    summary=dict(models=summaries,steps=sum(len(s['directions']) for s in summaries),steps_with_actual_first_order_harm=sum(bool(d['harmful_first_order_groups']) for s in summaries for d in s['directions']),steps_with_projection=sum(d['correction_relative_norm']>1e-6 for s in summaries for d in s['directions']),zero_gradient_steps=sum(d['zero_gradient'] for s in summaries for d in s['directions']),minimum_combined_gradient_cos=min(d['combined_gradient_min_cos'] for s in summaries for d in s['directions']))
    (OUT/'decision_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
