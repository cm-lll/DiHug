"""CPU reporting only, all 24 conditions retained."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import csv,json,sys,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/typed_score_path_cpu_20261110'
rows=list(csv.DictReader((OUT/'risk_path.csv').open()))
tasks=list(dict.fromkeys(r['task'] for r in rows)); summary=[]; aggregates=[]
alphas=[0.,.25,.5,.75,1.]
for task in tasks:
    old=json.loads((ROOT/'results/hard_clean_zero_safe_20261110'/task/'result.json').read_text())
    files=[np.load(p) for p in sorted(OUT.glob(task+'_t*.npz'))]
    item=dict(task=task,gradient_norm=old['new_gradient_norm'],actual_AdamW_norm=old['actual_AdamW_norm'],
        individual_gradient_norms=old['individual_norms'],max_probability_shift=max(float(abs(x['p_after']-x['p_before']).max()) for x in files),
        average_probability_L1_shift=float(np.mean([abs(x['p_after']-x['p_before']).sum() for x in files])),curves={})
    for name in ['degree_at_joint','closed','eligible_open','unclosable','component','joint']:
        curve=[]
        for alpha in alphas:
            rr=[r for r in rows if r['task']==task and r['metric']==name and float(r['alpha'])==alpha]
            val=float(np.mean([float(r['mean']) for r in rr]));curve.append(val)
            aggregates.append(dict(task=task,metric=name,alpha=alpha,mean_over_six_fixed_times=val,
                times_worse_than_before=sum(float(r['delta_from_before'])>1e-12 for r in rr),times_better_than_before=sum(float(r['delta_from_before'])< -1e-12 for r in rr)))
        item['curves'][name]=curve
    zero=[r for r in rows if r['task']==task and r['metric']=='component' and float(r['alpha'])==0]
    item['initial_component_derivative']=dict(negative=sum(float(r['path_derivative'])<0 for r in zero),
        SE_at_least_magnitude=sum(float(r['derivative_jackknife_SE'])>=abs(float(r['path_derivative'])) for r in zero),
        warning='K32 jackknife is an uncertainty diagnostic, not proof of gradient sign; rare events may be unobserved')
    item['structure_curve']=np.array([f['structure'].mean(1) for f in files]).mean(0).tolist()
    summary.append(item)
with (OUT/'aggregate_curves.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(aggregates[0]));w.writeheader();w.writerows(aggregates)
(OUT/'summary.json').write_text(json.dumps(dict(tasks=summary,alphas=alphas,structure_columns=['edges','triangles','components','LCC','isolates'],
    no_parameter_path_claim=True,no_adoption=True,next='read-only isolate relation-count feedback scale and AdamW state effects before changing loss; no automatic training'),indent=2))
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(4,3,figsize=(12,12))
    for i,task in enumerate(tasks):
        for j,name in enumerate(['closed','joint','component']):
            ax=axs[i,j]
            for t in [1,75,250,550,800,1000]:
                rr=sorted([r for r in rows if r['task']==task and r['metric']==name and int(r['t'])==t],key=lambda r:float(r['alpha']))
                ax.plot(alphas,[float(r['mean']) for r in rr],alpha=.3,lw=1)
            ax.plot(alphas,summary[i]['curves'][name],color='black',marker='o',lw=2,label='mean of 6 fixed times')
            ax.set_title(task.replace('_HARD_MGDA','')+' / '+name,fontsize=9);ax.set_xlabel('probability interpolation alpha');ax.grid(alpha=.2)
    fig.suptitle('Stored-update path audit: conditional hard predictions, not parameter interpolation\nThin: each t, black: mean; lower costs better; no new model calls',fontsize=11)
    fig.tight_layout(rect=(0,0,1,.95));fig.savefig(OUT/'risk_curves.png',dpi=140);plt.close(fig)
except ImportError as exc:
    (OUT/'plot_unavailable.txt').write_text(str(exc))
(OUT/'manifest.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.iterdir()) if p.is_file() and p.name!='manifest.json'},indent=2))
print(json.dumps(dict(models=len(tasks),conditions=24,aggregate_rows=len(aggregates),GPU=0)))
