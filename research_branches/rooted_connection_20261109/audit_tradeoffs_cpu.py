"""Account for additional structural tradeoffs in ALL three-metric-worse edges.

Exploratory follow-up to the frozen CPU panel, not a new training objective.
"""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import json,time,csv,hashlib,resource
from pathlib import Path
import numpy as np
import torch
from score import RootedScore,RootedIncremental
from feedback_audit_cpu import metrics,ROOT,OLD,EDGE,OUT,RUN
torch.set_num_threads(1)
def main():
    start=time.time();rows=[];summary=[];bindings={}
    assert not (OUT/'conditional_tradeoffs.json').exists()
    def load(p):
        bindings[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest();return np.load(p)
    protocol=dict(selection='ALL edges rewarded by RC that worsen total-degree W1, joint-degree TV, triangle absolute error in the already frozen24-graph panel',
        purpose='Check whether additional marginal relation-degree, node triangles, path counts, hub or edge-count metrics show compensation; not exhaustive structure utility',
        CPU_seconds_limit=180,GPU=0,models=0,threshold=1e-12)
    (OUT/'tradeoff_protocol.json').write_text(json.dumps(protocol,indent=2))
    panels=json.loads((OUT/'conditional_panel.json').read_text())
    for ds in ['IMDB','PubMed']:
        with load(RUN/f'{ds}_cpu_cost_panel.npz') as z:types=z['types'].copy()
        fixed=torch.load(OLD/f'{ds}_fixed_inputs.pt',map_location='cpu',weights_only=False)['inputs']
        bindings[str(OLD/f'{ds}_fixed_inputs.pt')]=hashlib.sha256((OLD/f'{ds}_fixed_inputs.pt').read_bytes()).hexdigest()
        for point in [r for r in panels if r['dataset']==ds and r['three_worse']]:
            assert time.time()-start<180
            t=point['t'];inp=next(x for tt,ss,x in fixed if int(tt)==t)
            q,f,y=map(np.asarray,(inp['query'],inp['f'],inp['labels']));core=RootedScore(types,q,f,y,10)
            tag=f"{ds}_{point['parent_history']}_CE_t{t}";draw=point['draw']
            with load(EDGE/f'{tag}_draw{draw}_flips.npz') as z:
                bits=(1-z['flip_sign'])/2
                bad=np.logical_and.reduce([z[k+'_flip']>1e-12 for k in ('total_degree_W1','joint_degree_TV','triangle_abs_error')])
            with load(OUT/f'{tag}_draw{draw}_full.npz') as z:score_flip=z['conditional_difference']*(1-2*bits)
            ix=np.flatnonzero(bad&(score_flip<-1e-12));assert len(ix)==point['three_worse']
            inc=RootedIncremental(core,bits);ref=core.features(core.reference).numpy();base=metrics(core,inc.raw,ref)
            gain=0;no_gain=0
            for offset in range(0,len(ix),64):
                ids=ix[offset:offset+64];raw,_=inc.raw_flips(ids);values=metrics(core,raw,ref)
                for j,e in enumerate(ids):
                    diff={k:float(values[k][j]-base[k][0]) for k in values}
                    assert all(diff[k]>1e-12 for k in ('total_degree_W1','joint_degree_TV','triangle_error'))
                    any_gain=any(x<-1e-12 for x in diff.values());gain+=int(any_gain);no_gain+=int(not any_gain)
                    rows.append(dict(dataset=ds,parent_history=point['parent_history'],t=t,draw=draw,slot=int(e),relation=int(f[e]),present=int(bits[e]),score_change=float(score_flip[e]),any_checked_gain=any_gain,**{k+'_change':v for k,v in diff.items()}))
            summary.append(dict(dataset=ds,parent_history=point['parent_history'],t=t,draw=draw,three_worse=len(ix),other_checked_gain=gain,no_checked_gain=no_gain))
    for name,data in [('conditional_tradeoffs',rows),('conditional_tradeoff_summary',summary)]:
        (OUT/(name+'.json')).write_text(json.dumps(data,indent=2,allow_nan=False))
        if data:
            with (OUT/(name+'.csv')).open('w') as f:
                w=csv.DictWriter(f,list(data[0]));w.writeheader();w.writerows(data)
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in bindings.items())
    receipt=dict(seconds=time.time()-start,peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,edges=len(rows),other_checked_gain=sum(r['other_checked_gain'] for r in summary),no_checked_gain=sum(r['no_checked_gain'] for r in summary),GPU=0,model_calls=0,sources_unchanged=True)
    (OUT/'tradeoff_bindings.json').write_text(json.dumps(bindings,indent=2))
    (OUT/'tradeoff_receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
if __name__=='__main__':main()
