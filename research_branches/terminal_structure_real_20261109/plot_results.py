"""CPU descriptive figures only; sampling SD is not a confidence interval."""
import os
os.environ['MPLBACKEND']='Agg';os.environ['OPENBLAS_NUM_THREADS']='1'
import sys,json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
out=Path(sys.argv[1]);P=json.loads((out/'protocol.json').read_text());rows=json.loads((out/'terminal_costs.json').read_text());gr=json.loads((out/'gradient_diagnostics.json').read_text())
terms=P['score_terms'];labels=['Joint degree','Marginal degree','Low degree','Top-2 hubs','Node triangles','Typed triangles','Components']
fig,axes=plt.subplots(2,2,figsize=(13,8),constrained_layout=True)
for col,ds in enumerate(P['models']):
    rr=[r for r in rows if r['dataset']==ds]
    for r in rr:axes[0,col].plot(range(len(terms)),[r[t] for t in terms],'o-',alpha=.65,label=str(r['sample_seed']))
    axes[0,col].set(title=f'{ds}: terminal errors of frozen A (no update)',ylabel='Frozen normalized cost (lower is better)',xticks=range(len(terms)),xticklabels=labels)
    axes[0,col].tick_params(axis='x',rotation=35);axes[0,col].legend(fontsize=8)
    gg=[r for r in gr if r['dataset']==ds];yy=[next((r['two_disjoint_pairs_cosine'] for r in gg if r['term']==t),None) for t in terms]
    for i,v in enumerate(yy):
        if v is not None:axes[1,col].bar(i,v,color='#3976a9' if v>=0 else '#b45e4c')
        else:axes[1,col].text(i,0,'undefined',rotation=90,ha='center',va='center',fontsize=8)
    axes[1,col].axhline(0,color='black',linewidth=.6);axes[1,col].set(ylim=(-1,1),ylabel='Cosine between two disjoint pair estimates',title='Direction repeatability: only two 2-path estimates',xticks=range(len(terms)),xticklabels=labels);axes[1,col].tick_params(axis='x',rotation=35)
fig.savefig(out/'terminal_feedback_diagnostic.png',dpi=180);plt.close(fig)
