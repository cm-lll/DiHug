"""Plot saved CPU summary, without invoking models."""
import os
os.environ['MPLBACKEND']='Agg'
import json
from pathlib import Path
import matplotlib.pyplot as plt
OUT=Path(__file__).resolve().parents[2]/'results/analytic_joint_short_20261113'
rows=json.loads((OUT/'summary.json').read_text())
models=sorted({(r['dataset'],r['train_seed']) for r in rows})
metrics=[('joint_expected_degree_proxy','Joint expected-degree matching'),('low_fraction_error','Low-degree fraction error'),
         ('hub_fraction_error','Hub fraction error'),('expected_triangle_abs_error','Expected-triangle count error')]
fig,axes=plt.subplots(4,4,figsize=(16,11),squeeze=False)
for i,(ds,seed) in enumerate(models):
    for j,(metric,label) in enumerate(metrics):
        ax=axes[i,j];ax.axhline(0,color='0.7',lw=.7)
        for arm,color in [('bias','tab:blue'),('risk','tab:orange')]:
            selected=sorted([r for r in rows if r['dataset']==ds and r['train_seed']==seed and r['arm']==arm and r['readout']=='raw'],key=lambda r:r['step'])
            ax.plot([0]+[r['step'] for r in selected],[0]+[r[metric+'_change_percent'] for r in selected],marker='o',color=color,label=arm+' raw',lw=1.4)
            end=[r for r in rows if r['dataset']==ds and r['train_seed']==seed and r['arm']==arm and r['readout']=='ema']
            if end:ax.scatter([16],[end[0][metric+'_change_percent']],marker='x',s=55,color=color,label=arm+' EMA')
        ax.set_title(f'{ds} {seed}\n{label}',fontsize=9)
        ax.set_xlabel('Diagnostic updates');ax.set_ylabel('Error change vs parent (%)')
        ax.set_xticks([0,1,4,8,16]);ax.grid(alpha=.15)
        if i==0 and j==0:ax.legend(fontsize=7)
fig.suptitle('Conditional clean-prediction proxies; lower is better. Not generated-graph metrics.',fontsize=12)
fig.tight_layout(rect=(0,0,1,.96));fig.savefig(OUT/'trajectories.png',dpi=140);plt.close(fig)
