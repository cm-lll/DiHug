"""One fixed degree-profile assignment, diagnostic only; not a new label rule.

Uses archived probabilities and reference labels. No search over assignments,
no optimized edge matching, no model invocation, no generated graph samples.
"""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import csv, hashlib, json, resource, time
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
torch.set_num_threads(1)
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/identity_structure_audit_20261109/alignment_verified'
OLD=ROOT/'results/hard_structure_response_20261108T182154'

def main():
    OUT.mkdir(parents=True,exist_ok=False)
    start=time.time();cpu=time.process_time();binding={}
    def bind(p):
        binding[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest();return p
    protocol=dict(rule='Within each type minimize sum mean |expected relation degree - reference relation degree| / max(1,type reference mean), active relations only. One scipy assignment, no CE optimization.',
        purpose='Check whether simple consistent role reassignment explains current edge-label mismatch; failure cannot rule out other assignments.',
        ties='scipy deterministic default; no outcome-dependent tie search',
        probability_loss='exact logs of saved probabilities, infinite CE counted and not clipped; not guaranteed identical to saved-logit stable CE',
        budget_seconds=120,model_calls=0,GPU=0,training_updates=0)
    (OUT/'alignment_protocol.json').write_text(json.dumps(protocol,indent=2))
    tasks=json.loads(bind(ROOT/'results/output_boundary_replay_20261108T081654/tasks.json').read_text());rows=[]
    def ce(p,y):
        actual=np.where(y==1,p,1-p); zeros=int(np.sum(actual==0))
        return (None if zeros else float(-np.log(actual).mean())),zeros
    for ds in ['IMDB','PubMed']:
        task=next(x for x in tasks if x['dataset']==ds);cfg=task['row']['config']['datasets'][ds]
        path=ROOT/'research_branches/typed_structure_early_cn_expansion_review/experiments/typed_structure_early_cn_expansion_v1/package'/cfg['file']
        bind(path);assert binding[str(path)]==cfg['sha256']
        with np.load(path) as z:types=z['types'].copy()
        fixed=torch.load(bind(OLD/f'{ds}_fixed_inputs.pt'),map_location='cpu',weights_only=False)['inputs']
        for t,seed,inp in fixed:
            q,f,y=map(np.asarray,(inp['query'],inp['f'],inp['labels']));n=len(types);r=int(f.max())+1
            y=y.astype(np.float64)
            ref=np.zeros((n,n),dtype=int);ref[q[0],q[1]]=y;ref+=ref.T
            target=np.zeros((n,r))
            for ends in q:np.add.at(target,(ends,f),y)
            for hist,arm in [('CE','CE'),('CE','CE_STRUCT'),('PS','CE'),('PS','CE_STRUCT')]:
                assert time.time()-start<120
                with np.load(bind(OLD/f'{ds}_{hist}_{arm}_t{t}.npz')) as z:p=z['probability'].astype(float)
                pred=np.zeros_like(target)
                for ends in q:np.add.at(pred,(ends,f),p)
                perm=np.arange(n);before=[];after=[]
                for ty in np.unique(types):
                    ix=np.flatnonzero(types==ty);a=pred[ix];b=target[ix];mask=(a.sum(0)+b.sum(0))>0
                    scale=np.maximum(1,b.mean(0));cost=(abs(a[:,None,mask]-b[None,:,mask])/scale[mask]).mean(-1)
                    ii,jj=linear_sum_assignment(cost);perm[ix[ii]]=ix[jj]
                    before.append(float(np.trace(cost)/len(ix)));after.append(float(cost[ii,jj].mean()))
                assert np.mean(after)<=np.mean(before)+1e-12
                other=ref[np.ix_(perm,perm)][q[0],q[1]];assert other.sum()==y.sum()
                ce0,inf0=ce(p,y);ce1,inf1=ce(p,other)
                count=np.bincount(f);prior=np.divide(np.bincount(f,weights=y),count,out=np.zeros(len(count)),where=count>0)
                retention=np.cos((t/1000+.008)/1.008*np.pi/2)**2/np.cos(.008/1.008*np.pi/2)**2 if t<1000 else 0.
                p0=retention*y+(1-retention)*prior[f];p1=retention*other+(1-retention)*prior[f]
                xt=np.zeros((n,n),dtype=bool);e=np.asarray(inp['edges']);xt[e[0],e[1]]=xt[e[1],e[0]]=True;observed=xt[q[0],q[1]]
                ratio=float(np.where(observed,np.log(p1/p0),np.log1p(-p1)-np.log1p(-p0)).sum())
                if np.array_equal(y,other):assert ratio==0., 'identical source likelihood ratio'
                rows.append(dict(dataset=ds,history=hist,arm=arm,t=t,seed=task['seed'],
                    degree_profile_cost_before=float(np.mean(before)),degree_profile_cost_after=float(np.mean(after)),
                    edge_CE_original=ce0,edge_CE_reassigned=ce1,CE_infinite_slots_original=inf0,CE_infinite_slots_reassigned=inf1,
                    changed_labels=int((y!=other).sum()),forward_log_likelihood_ratio=ratio))
    with (OUT/'degree_profile_alignment.csv').open('w') as f:
        w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
    binding[str(Path(__file__))]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in binding.items())
    (OUT/'alignment_binding.json').write_text(json.dumps(binding,indent=2))
    receipt=dict(status='complete',panels=len(rows),seconds=time.time()-start,cpu_seconds=time.process_time()-cpu,
        peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,model_calls=0,GPU_hours=0,training_updates=0,
        finite_pairs=sum(r['edge_CE_original'] is not None and r['edge_CE_reassigned'] is not None for r in rows),
        CE_improved=sum(r['edge_CE_original'] is not None and r['edge_CE_reassigned'] is not None and r['edge_CE_reassigned']<r['edge_CE_original']-1e-12 for r in rows))
    assert not torch.cuda.is_initialized()
    (OUT/'alignment_receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt,indent=2));print(rows)

if __name__=='__main__':main()
