"""Plot saved CPU results; no model or training calls."""
import os
os.environ['MPLBACKEND']='Agg'
import csv,json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[2];out=root/'results/full100_scale_20261110'
s=json.loads((out/'summary.json').read_text());fixed=list(csv.DictReader((out/'fixed_validation.csv').open()))
fig,axes=plt.subplots(2,3,figsize=(13,7.5))
for col,seed in enumerate([2026110901,2026110902,2026110903]):
    for arm,label,color in [('hard_original','Original hard','#777777'),('hard_scaled','Calibrated hard','#1976d2')]:
        rows=[x for x in fixed if int(x['seed'])==seed and x['arm']==arm]
        axes[0,col].plot([int(x['update']) for x in rows],[float(x['mean_five_cost']) for x in rows],label=label,color=color)
        initial=next(x for x in s['per_initialization'] if x['seed']==seed and x['arm']=='initial')
        rows=[x for x in s['per_initialization'] if x['seed']==seed and x['arm']==arm]
        rows=sorted(rows,key=lambda x:x['updates'])
        axes[1,col].plot([0]+[x['updates'] for x in rows],[initial['mean_five_cost']]+[x['mean_five_cost'] for x in rows],marker='o',color=color,label=label)
    ce=next(x for x in s['per_initialization'] if x['seed']==seed and x['arm']=='reference_CE')
    axes[1,col].scatter([20],[ce['mean_five_cost']],marker='x',s=70,color='#d97706',label='CE endpoint')
    axes[0,col].set_title(f'Toy initialization {seed}')
    for row in range(2):axes[row,col].set_xlabel('Toy optimizer updates');axes[row,col].grid(alpha=.2)
axes[0,0].set_ylabel('Fixed noised-input mean risk')
axes[1,0].set_ylabel('Full 100-step terminal mean risk')
axes[0,0].legend();axes[1,0].legend()
fig.suptitle('4-node CPU diagnostic: fixed scale 7.4061; five costs shown separately in CSV\nMean risk is descriptive; lower does not imply uniform superiority to CE')
fig.tight_layout();fig.savefig(out/'scale_comparison.png',dpi=170);plt.close(fig)
