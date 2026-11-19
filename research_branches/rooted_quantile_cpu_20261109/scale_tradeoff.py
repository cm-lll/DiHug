"""Explain the remaining PubMed example; no alternative scorer is selected."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,json,csv
from pathlib import Path
import numpy as np
import torch
torch.set_num_threads(1)
from quantile import QuantileScore
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'research_branches/rooted_connection_20261109'))
from score import RootedScore,RootedIncremental
RUN=ROOT/'results/rooted_connection_20261109T015142';OUT=ROOT/'results/rooted_quantile_cpu_20261109'
def main():
    assert not (OUT/'scale_tradeoff.json').exists()
    ex=next(x for x in json.loads((OUT/'known_examples.json').read_text()) if x['dataset']=='PubMed')
    with np.load(RUN/'PubMed_cpu_cost_panel.npz') as z:core=RootedScore(z['types'],z['query'],z['families'],z['reference'],10)
    jobs=json.loads((RUN/'evaluation_jobs.json').read_text());j=next(j for j in jobs if all(j[k]==ex[k] for k in ('dataset','train_seed','arm','sample_seed')))
    with np.load(Path(j['folder'])/'graph.npz') as z:
        a=np.zeros((core.n,core.n));a[z['edges'][0],z['edges'][1]]=1;a+=a.T
    q=core.query.numpy();inc=RootedIncremental(core,a[q[0],q[1]]);raw=inc.raw;new=inc.raw_flips([ex['slot']])[0][0];ref=core.features(core.reference).numpy();rows=[]
    for t,ids in core.nodes.items():
        ids=ids.numpy();mask=core.masks[t].numpy();cols=np.flatnonzero(mask)
        r=np.sort(ref[ids][:,mask],axis=0);x=np.sort(raw[ids][:,mask],axis=0);xx=np.sort(new[ids][:,mask],axis=0)
        old_abs=abs(x-r).mean(0);new_abs=abs(xx-r).mean(0);old_sq=np.square(x-r).mean(0);new_sq=np.square(xx-r).mean(0)
        old_log=np.square(np.log1p(x)-np.log1p(r)).mean(0);new_log=np.square(np.log1p(xx)-np.log1p(r)).mean(0)
        for k,c in enumerate(cols):
            if np.any(xx[:,k]!=x[:,k]):rows.append(dict(type=int(t),feature=int(c),block='degree' if c<core.r else 'triangle' if c<core.base_width else 'path',
                raw_W1_change=float(new_abs[k]-old_abs[k]),raw_MSE_change=float(new_sq[k]-old_sq[k]),log_MSE_change=float(new_log[k]-old_log[k]),
                weighted_axis_change=float((new_log[k]-old_log[k])/(core.scales[t][k].item()**2)*.5/len(cols)/len(core.nodes))))
    y=np.array([1.,100.]);before=np.array([0.,100.]);after=np.array([1.,120.])
    toy=dict(reference=y.tolist(),before=before.tolist(),after=after.tolist(),
        raw_W1_before=float(abs(before-y).mean()),raw_W1_after=float(abs(after-y).mean()),
        log_MSE_before=float(np.square(np.log1p(before)-np.log1p(y)).mean()),log_MSE_after=float(np.square(np.log1p(after)-np.log1p(y)).mean()),
        limitation='abstract one-dimensional nonnegative feature example, not asserted graph-realizable and not evidence of actual training cause')
    summary=dict(changed_coordinates=len(rows),log_improves_raw_W1_worsens=sum(x['log_MSE_change']<-1e-12 and x['raw_W1_change']>1e-12 for x in rows),
        weighted_log_conflict_improvement=sum(x['weighted_axis_change'] for x in rows if x['log_MSE_change']<-1e-12 and x['raw_W1_change']>1e-12),toy=toy)
    (OUT/'scale_tradeoff.json').write_text(json.dumps(dict(summary=summary,coordinates=rows),indent=2))
    with (OUT/'scale_tradeoff.csv').open('w') as f:
        w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
