"""Post-audit score attribution only; frozen definitions, no revised scoring."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ[key]='1'
import json,time,signal,resource,csv
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from audit import ROOT,SOURCE,Descriptor,sha,dump,table

def main():
    torch.set_num_threads(1); start=time.time()
    signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('120s attribution bound'))); signal.alarm(120)
    out=ROOT/'results/typed_score_cpu_20261110_attempt2'; original=json.loads((SOURCE/'protocol.json').read_text())
    bindings=json.loads((out/'protocol.json').read_text())['bindings']
    records=[]; rows=[]
    for task in original['tasks']:
        inp=torch.load(task['evaluation_file'],map_location='cpu',weights_only=False)['inputs'][0]
        order=np.argsort(inp['ids']); q=inp['query'][:,order]; f=inp['f'][order]; ref=inp['labels'][order]
        d=Descriptor(np.load(task['data'])['types'],q,f,ref)
        for state in ('before','after'):
            for t in original['times']:
                saved=np.load(SOURCE/task['task']/f'{state}_t{t}.npz')
                batch=[]
                for k,bits in enumerate(saved['bits']):
                    ft=d.features(bits); parts=[]
                    for typ in d.type_ids:
                        ix=d.types==typ; mask=d.masks[int(typ)]; scale=d.scales[int(typ)]
                        left=ft['local'][ix][:,mask]/scale; right=d.ref['local'][ix][:,mask]/scale
                        deg=cdist(ft['degree'][ix],d.ref['degree'][ix],'cityblock')/max(1,d.n-1)
                        loc=cdist(left,right,'cityblock')/max(1,mask.sum())
                        i,j=linear_sum_assignment(deg+loc)
                        dif=np.abs(left[i]-right[j]); modes=np.flatnonzero(mask)//d.r**2
                        val=dict(J_at_joint=float(deg[i,j].mean()))
                        for mode,name in enumerate(('closed','eligible_open','unclosable')):
                            val[name]=float(dif[:,modes==mode].sum()/max(1,mask.sum())/len(i))
                            val[name+'_channels']=int(np.sum(modes==mode))
                        records.append(dict(task=task['task'],state=state,t=t,sample=k,node_type=int(typ),**val)); parts.append(val)
                    value={name:float(np.mean([p[name] for p in parts])) for name in ('J_at_joint','closed','eligible_open','unclosable')}
                    value['JL_reconstructed']=sum(value.values()); batch.append(value)
                rows.append(dict(task=task['task'],state=state,t=t,**{name:float(np.mean([b[name] for b in batch])) for name in batch[0]}))
        print(task['task'],flush=True)
    old=list(csv.DictReader((out/'batches.csv').open()))
    for row in rows:
        match=next(b for b in old if b['task']==row['task'] and b['state']==row['state'] and int(b['t'])==row['t'])
        assert abs(row['JL_reconstructed']-float(match['JL']))<1e-12
    for p,h in bindings.items(): assert sha(p)==h
    table(out/'joint_attribution_types.csv',records); table(out/'joint_attribution_batches.csv',rows)
    dump(out/'attribution_receipt.json',dict(seconds=time.time()-start,peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        rows=len(records),batches=len(rows),exact_decomposition=True,unchanged_bindings=len(bindings),GPU=0,model_calls=0,sha256=sha(__file__)))

if __name__=='__main__': main()
