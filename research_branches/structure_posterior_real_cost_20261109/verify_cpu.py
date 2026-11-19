"""Independent numpy triangle enumeration and full score; no model imports."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import csv,json,hashlib,time,resource,itertools,platform
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/structure_posterior_real_cost_20261109'
def save(n,o):
    (OUT/(n+'.json')).write_text(json.dumps(o,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    if isinstance(o,list) and o:
        with (OUT/(n+'.csv')).open('w',newline='') as f:
            w=csv.DictWriter(f,list(o[0]));w.writeheader();w.writerows(o)
def main():
    tick=time.monotonic();binding=json.loads((OUT/'binding.json').read_text())
    for p,h in binding.items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
    with np.load(ROOT/'results/rooted_connection_20261109T015142/PubMed_cpu_cost_panel.npz') as z:
        types=z['types'].copy();q=z['query'].copy();f=z['families'].copy();y=z['reference'].copy()
    # Archive states use the saved input's query order. Reconstruct that order from labels+slot deltas is not allowed.
    # The original audit serialized selection by its exact ordered query; use the bound fixed input on CPU only.
    import torch
    torch.set_num_threads(1)
    inputs=torch.load(ROOT/'results/hard_structure_response_20261108T182154/PubMed_fixed_inputs.pt',map_location='cpu',weights_only=False)
    inp=next(x for t,s,x in inputs['inputs'] if t==1000);q,f,y=map(np.asarray,(inp['query'],inp['f'],inp['labels']))
    classes=list(itertools.combinations_with_replacement(range(4),3));pairs=list(itertools.combinations_with_replacement(range(4),2))
    ci={c:i for i,c in enumerate(classes)};pi={p:i for i,p in enumerate(pairs)}
    def features(b):
        a=np.zeros((192,192),bool);a[q[0],q[1]]=b.astype(bool);a|=a.T
        degree=np.zeros((192,10));np.add.at(degree,(q[0],f),b);np.add.at(degree,(q[1],f),b)
        tri=np.zeros((192,10));tc=np.zeros(20)
        for u in range(192):
            for v in np.flatnonzero(a[u]&(np.arange(192)>u)):
                for w in np.flatnonzero(a[u]&a[v]&(np.arange(192)>v)):
                    tc[ci[tuple(sorted(map(int,types[[u,v,w]])))]]+=1
                    for node,i,j in [(u,v,w),(v,u,w),(w,u,v)]:tri[node,pi[tuple(sorted(map(int,types[[i,j]])))]]+=1
        return degree,tri,tc
    rd,rt,rtc=features(y)
    def score(b):
        d,tr,tc=features(b);val=float(abs(tc-rtc).sum())/max(1.,rtc.sum())/8
        for t in range(4):
            ids=np.flatnonzero(types==t);cols=np.unique(f[np.isin(q[0],ids)|np.isin(q[1],ids)])
            r=rd[ids][:,cols];x=d[ids][:,cols];scale=np.maximum(1.,r.mean(0))
            costs=cdist(x/scale,r/scale,metric='cityblock')/len(cols);i,j=linear_sum_assignment(costs)
            joint=costs[i,j].mean();marg=(abs(np.sort(x,axis=0)-np.sort(r,axis=0))/scale).mean()
            sorted_ref=np.sort(rd[ids].sum(1));diff=abs(np.sort(d[ids].sum(1))-sorted_ref);k=max(1,len(ids)//2)
            low=diff[:k].mean()/max(1.,sorted_ref[:k].mean());hub=diff[-2:].mean()/max(1.,sorted_ref[-2:].mean())
            node_tri=(abs(np.sort(tr[ids],axis=0)-np.sort(rt[ids],axis=0))/np.maximum(1.,rt[ids].mean(0))).mean()
            val+=(joint/8+marg/8+low/4+hub/4+node_tri/8)/4
        return float(val)
    plans=json.loads((OUT/'frozen_slots.json').read_text());rows=json.loads((OUT/'checks.json').read_text());tests=[]
    for tag in dict.fromkeys(p['state'] for p in plans):
        with np.load(ROOT/'results/joint_degree_full_signal_20261109'/f'{tag}.npz') as z:b=z['bits'].copy()
        base=score(b)
        for relation in range(10):
            plan=next(p for p in plans if p['state']==tag and p['relation']==relation and len(p['slots']))
            slot=plan['slots'][0];bb=b.copy();bb[slot]=1-bb[slot];delta=score(bb)-base
            row=next(r for r in rows if r['state']==tag and r['slot']==slot)
            err=abs(delta-row['score_change']);assert err<1e-10
            tests.append(dict(state=tag,relation=relation,slot=slot,independent_score_delta=delta,max_error=err))
    save('independent_checks',tests)
    # Algebraic consequence only: NOT a chosen real posterior, nor estimated graph density.
    d01=np.array([(1-2*r['before'])*r['score_change'] for r in rows]);h=1/(1+np.exp(d01))
    save('base_measure_warning',dict(hypothesis='IF old real score were used with uniform base,tau1 at alpha0, without adoption',
        selected_slots=len(rows),conditional_probability_min=float(h.min()),conditional_probability_max=float(h.max()),
        max_abs_score_delta=float(abs(d01).max()),
        meaning='terminal likelihood cancels; these are one-edge conditionals on four old backgrounds, not posterior marginals or predicted final graph density',
        consequence='Toy uniform-base construction cannot be assumed to inherit real sparse corruption marginals. Base measure and structural scale need explicit design; do not silently choose temperature.'))
    bench=json.loads((OUT/'benchmarks.json').read_text());summary=[]
    for tag in dict.fromkeys(r['state'] for r in bench):
        by={m:[r['seconds_per_flip'] for r in bench if r['state']==tag and r['mode']==m] for m in ['full','incremental']}
        summary.append(dict(state=tag,full_ms_mean=1000*float(np.mean(by['full'])),incremental_ms_mean=1000*float(np.mean(by['incremental'])),
            incremental_ms_min=1000*min(by['incremental']),incremental_ms_max=1000*max(by['incremental']),speed_ratio=float(np.mean(by['full'])/np.mean(by['incremental']))))
    save('timing_summary',summary)
    save('verification',dict(status='passed',independent_full_checks=len(tests),max_error=max(r['max_error'] for r in tests),
        seconds=time.monotonic()-tick,RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        environment=dict(python=platform.python_version(),numpy=np.__version__,torch=torch.__version__,CPU_threads=torch.get_num_threads()),
        old_assets_unchanged=True,external_review=False,limitation='Same assistant separate implementation; no GPU or model calls'))
if __name__=='__main__':main()
