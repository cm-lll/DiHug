import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['MPLCONFIGDIR']='/tmp/dihug_full100_mpl'
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path(__file__).resolve().parents[2]/'results/full100_toy_20261110'
rows=list(csv.DictReader((OUT/'trajectory_metrics.csv').open()))
fig,axes=plt.subplots(2,2,figsize=(10,7))
for ax,key,title in zip(axes.flat,['state_edges','state_connected','state_triangle_MAE','clean_onpolicy_mean_five_cost'],['Current graph: expected edges','Current graph: connected probability','Current graph: triangle count MAE','One-shot clean risk on visited states']):
    for arm,label in [('initial','Initial'),('reference_CE','CE, 20 updates'),('hard_five_project_mean','Hard structure, 20 updates')]:
        times=sorted({int(x['t']) for x in rows if x['arm']==arm and (key.startswith('state_') or int(x['t'])>0)},reverse=True)
        values=np.array([[float(x[key]) for x in rows if x['arm']==arm and int(x['t'])==t] for t in times])
        mean=values.mean(1);sd=values.std(1,ddof=1)
        line=ax.plot(times,mean,marker='.',label=label)[0];ax.fill_between(times,mean-sd,mean+sd,alpha=.12,color=line.get_color())
    ax.set_title(title);ax.set_xlabel('Reverse time (100 to 0)');ax.invert_xaxis();ax.grid(alpha=.2)
axes[0,0].axhline(4,color='black',ls=':',lw=1,label='Reference')
axes[0,0].legend(fontsize=8)
fig.suptitle('4-node toy: full 100-step chain; mean +/- SD of 3 initializations\nExact distributions at registered checkpoints; no hard count constraint',fontsize=11)
fig.tight_layout();fig.savefig(OUT/'full_chain.png',dpi=160)
