"""Explain one posthoc largest-score-improvement/no-checked-benefit edge per dataset."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import json,hashlib,time
import numpy as np
import torch
from score import RootedScore,RootedIncremental,kernel
from feedback_audit_cpu import ROOT,RUN,OUT
torch.set_num_threads(1)
def terms(inc):
    rows=[]
    for t,c in inc.cache.items():
        rows.append(dict(type=int(t),self=float(c['xx'].mean()),cross=float(-2*kernel(c['x'],c['y']).mean())))
    return rows
def main():
    start=time.time();assert not (OUT/'score_decomposition_examples.json').exists()
    edges=json.loads((OUT/'final_edge_metrics.json').read_text());jobs=json.loads((RUN/'evaluation_jobs.json').read_text());out=[]
    for ds in ['IMDB','PubMed']:
        eligible=[x for x in edges if x['dataset']==ds and x['score_change']<-1e-12 and all(v>=-1e-12 for k,v in x.items() if k.endswith('_change') and k!='score_change') and any(v>1e-12 for k,v in x.items() if k.endswith('_change') and k!='score_change')]
        ex=min(eligible,key=lambda x:x['score_change'])
        with np.load(RUN/f'{ds}_cpu_cost_panel.npz') as z:core=RootedScore(z['types'],z['query'],z['families'],z['reference'],10)
        job=next(j for j in jobs if all(j[k]==ex[k] for k in ('dataset','train_seed','arm','sample_seed')))
        path=ROOT/job['folder']/'graph.npz';binding=json.loads((path.parent/'binding.json').read_text());assert hashlib.sha256(path.read_bytes()).hexdigest()==binding['graph_sha256']
        with np.load(path) as z:
            a=np.zeros((core.n,core.n));a[z['edges'][0],z['edges'][1]]=1;a+=a.T
        q=core.query.numpy();bits=a[q[0],q[1]];before=RootedIncremental(core,bits);bits[ex['slot']]=1-bits[ex['slot']];after=RootedIncremental(core,bits)
        tb,ta=terms(before),terms(after)
        changes=[dict(type=x['type'],self_change=y['self']-x['self'],cross_change=y['cross']-x['cross']) for x,y in zip(tb,ta)]
        xx=float(np.mean([x['self_change'] for x in changes]));xy=float(np.mean([x['cross_change'] for x in changes]))
        assert abs(xx+xy-ex['score_change'])<1e-10
        out.append(dict(**ex,node_u=int(q[0,ex['slot']]),node_v=int(q[1,ex['slot']]),self_change=xx,cross_change=xy,per_type=changes,graph_sha256=binding['graph_sha256'],selection='posthoc largest score decrease among no-checked-benefit selected flips, illustrative not rate'))
    (OUT/'score_decomposition_examples.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2));print('seconds',time.time()-start)
if __name__=='__main__':main()
