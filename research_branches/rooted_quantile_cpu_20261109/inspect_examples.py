"""Decompose the two pre-existing examples without changing the candidate."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,json,time,hashlib,csv
from pathlib import Path
import numpy as np
import torch
torch.set_num_threads(1)
from quantile import QuantileScore
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'research_branches/rooted_connection_20261109'))
from score import RootedScore,RootedIncremental
RUN=ROOT/'results/rooted_connection_20261109T015142';OUT=ROOT/'results/rooted_quantile_cpu_20261109'
def main():
    start=time.time();rows=[];detail=[];bindings={}
    assert not (OUT/'example_decomposition.json').exists()
    jobs=json.loads((RUN/'evaluation_jobs.json').read_text())
    for ex in json.loads((OUT/'known_examples.json').read_text()):
        ds=ex['dataset']
        with np.load(RUN/f'{ds}_cpu_cost_panel.npz') as z:core=RootedScore(z['types'],z['query'],z['families'],z['reference'],10)
        qs=QuantileScore(core);job=next(j for j in jobs if all(j[k]==ex[k] for k in ('dataset','train_seed','arm','sample_seed')))
        path=Path(job['folder'])/'graph.npz';binding=json.loads((path.parent/'binding.json').read_text());bindings[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest();assert bindings[str(path)]==binding['graph_sha256']
        with np.load(path) as z:
            a=np.zeros((core.n,core.n));a[z['edges'][0],z['edges'][1]]=1;a+=a.T
        q=core.query.numpy();inc=RootedIncremental(core,a[q[0],q[1]]);newraw=inc.raw_flips([ex['slot']])[0][0];raw=inc.raw
        old,aa,mm=qs.raw_score(raw);new,bb,nn=qs.raw_score(newraw)
        rows.append(dict(dataset=ds,slot=ex['slot'],weighted_axis_change=float(.5*(bb-aa)[0]),weighted_mixed_change=float(.5*(nn-mm)[0]),score_change=new-old))
        for t,c in qs.cache.items():
            if c is None:continue
            oldx=np.log1p(raw[c['nodes']][:,c['mask']])/c['scale'];newx=np.log1p(newraw[c['nodes']][:,c['mask']])/c['scale']
            delta=(np.square(np.sort(newx,axis=0)-c['target_axis'])-np.square(np.sort(oldx,axis=0)-c['target_axis'])).mean(0)*.5/len(c['scale'])/len(qs.cache)
            columns=np.flatnonzero(c['mask'])
            for group,sel in [('degree',columns<core.r),('triangle',(columns>=core.r)&(columns<core.base_width)),('path',columns>=core.base_width)]:
                detail.append(dict(dataset=ds,slot=ex['slot'],type=int(t),block=group,weighted_axis_change=float(delta[sel].sum())))
    for name,data in [('example_decomposition',rows),('example_axis_blocks',detail)]:
        (OUT/(name+'.json')).write_text(json.dumps(data,indent=2))
        with (OUT/(name+'.csv')).open('w') as f:
            w=csv.DictWriter(f,list(data[0]));w.writeheader();w.writerows(data)
    print(json.dumps(rows,indent=2));print('seconds',time.time()-start)
if __name__=='__main__':main()
