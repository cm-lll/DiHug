"""CPU weight adapter equivalence; no denoiser invocation."""
import json,hashlib,sys,os
from pathlib import Path
import numpy as np
import torch
from runtime import science as g
from runtime.objective import make_objective
ROOT=Path(__file__).resolve().parents[2]
def main():
    torch.set_num_threads(1)
    rows=[]
    tasks=json.loads((ROOT/'results/pb_degree_formal_20261106T170600/tasks.json').read_text())
    manifest=json.loads((ROOT/'results/closure_context_gpu_20261107/inputs.json').read_text())
    for ds in ['IMDB','PubMed']:
        cfg=next(x['config'] for x in tasks if x['dataset']==ds)
        data=ROOT/g.SCI.relative_to(g.ROOT)/'package'/cfg['datasets'][ds]['file']
        with np.load(data) as z:case=g.common.make_case(z['types'],z['edges'].T,tuple(map(tuple,z['schema'])))
        obj=make_objective(case,cfg,ds)
        for t in [1,100,500,800]:
            p=Path(manifest[ds][str(t)]['path']);assert hashlib.sha256(p.read_bytes()).hexdigest()==manifest[ds][str(t)]['sha256']
            inp=dict(np.load(p));w=obj.weights(inp['query'],inp['f'],inp,device='cpu',dtype=torch.float64).numpy()
            old=np.load(ROOT/f'results/closure_context_gpu_20261107/{ds}_initial/t{t}_outputs.npz')
            assert np.array_equal(w,old['weights'])
            perm=np.arange(len(w))[::-1].copy()
            assert np.array_equal(obj.weights(inp['query'][:,perm],inp['f'][perm],inp,device='cpu',dtype=torch.float64).numpy(),w[perm])
            changed=dict(inp,labels=1-inp['labels'])
            assert np.array_equal(obj.weights(inp['query'],inp['f'],changed,device='cpu',dtype=torch.float64).numpy(),w)
            rejected=False
            try:obj.weights(inp['query'][:,:-1],inp['f'][:-1],inp,device='cpu',dtype=torch.float64)
            except ValueError:rejected=True
            assert rejected
            rows.append(dict(dataset=ds,t=t,exact_old_weights=True,permutation=True,label_independent=True,subset_rejected=True,**obj.last))
    target=Path(os.environ.get('STRUCTURE_RUN','/tmp/closure_context_precheck'))
    target.mkdir(exist_ok=True,parents=True)
    (target/'weight_cpu_test.json').write_text(json.dumps(dict(passed=True,model_calls=0,rows=rows),indent=2))
    print('eight state exact weight and query/label invariance checks passed')
if __name__=='__main__':main()

