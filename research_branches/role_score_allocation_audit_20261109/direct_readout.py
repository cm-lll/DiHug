"""Re-evaluate ONLY old offset-64 conditional samples, with original RNG.

No new sample seeds, no denoiser; validate every reconstructed sample against
archived edges, triangle count, degree W1 and joint-degree TV.
"""
import os
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import csv
import argparse
import hashlib
import json
import resource
import time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from run import AllocationMetrics, JointLocalScore, empirical_tv, sha, table

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / 'results/direct_medium_run_20261108T230502'
OUT = ROOT / 'results/role_score_allocation_audit_20261109/direct64_role_readout_v2'


def main():
    OUT.mkdir(exist_ok=False)
    start = time.time(); bound = {}
    def bind(p):
        bound[str(p)] = sha(p)
        return p
    def read(p):
        return json.loads(bind(p).read_text())
    def save(n, x):
        (OUT/n).write_text(json.dumps(x, indent=2, allow_nan=False))
    save('protocol.json', dict(offset=64, panel='new', conditions=4, times_per_dataset=20,
        arms=['CE', 'CE_STRUCT'], draws=32, expected_samples=5120,
        role_metrics='same definitions frozen in parent audit', wall_seconds_limit=300,
        GPU=0, model_calls=0, new_samples=0, warning='posthoc conditional clean-graph sample re-evaluation, two historical training seeds'))
    old = list(csv.DictReader(bind(SRC/'conditional_samples.csv').open()))
    old = {(x['condition'],x['arm'],int(x['index']),int(x['sample'])):x
           for x in old if x['offset']=='64' and x['panel']=='new'}
    assert len(old)==5120
    probes = [p for p in read(SRC/'probes.json') if p['offset']==64 and p['panel']=='new']
    assert len(probes)==160
    stored_bind = read(SRC/'probability_binding.json')
    # Historical manifests used repository-relative paths. Canonicalize keys;
    # retain every hash check and reject ambiguous aliases.
    old_bind = {str((ROOT/Path(k)).resolve()): v for k,v in stored_bind.items()}
    assert len(old_bind) == len(stored_bind)
    cache={}
    for ds in ['IMDB','PubMed']:
        p=bind(SRC/f'{ds}_inputs.pt'); assert bound[str(p)]==old_bind[str(p)]
        bundle=torch.load(p,map_location='cpu')
        for item in bundle['fixed']:
            inp=item['input']
            core=JointLocalScore(bundle['types'],inp['query'],inp['f'],inp['labels'],bundle['relations'])
            cache[ds,item['index']]=(item,core,AllocationMetrics(core))
    rows=[]; maxerr=0.
    for n,p in enumerate(probes):
        if time.time()-start>300 or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024>2048:
            raise RuntimeError('CPU readout limit')
        path=bind(SRC/p['path']); assert bound[str(path)]==old_bind[str(path)]
        with np.load(path) as z: prob=z['probability'].copy()
        item,core,metric=cache[p['dataset'],p['index']]
        rng=np.random.default_rng(item['conditional_graph_seed'])
        bits=(rng.random((32,len(prob)))<prob).astype(float)
        with torch.no_grad(): raw=core.features(torch.from_numpy(bits)).numpy()
        for j, feature in enumerate(raw):
            historic=old[p['condition'],p['arm'],p['index'],j]
            assert p['t']==int(historic['t']) and p['seed']==int(historic['seed'])
            w1=[]; tv=[]
            for ty,ix in metric.nodes.items():
                a=feature[ix,:core.r]; b=metric.ref[ix,:core.r]
                w1.append(float(abs(np.sort(a.sum(1))-np.sort(b.sum(1))).mean()))
                tv.append(empirical_tv(a,b))
            checks=dict(sampled_edges=float(bits[j].sum()),triangles=float(feature[:,core.r:].sum()/3),
                total_degree_W1=float(np.mean(w1)),joint_TV=float(np.mean(tv)))
            for k,v in checks.items():
                err=abs(v-float(historic[k])); maxerr=max(maxerr,err); assert err<1e-10,(p,k,err)
            values,_=metric.evaluate(feature)
            ids={k:p[k] for k in ['condition','dataset','seed','arm','offset','index','t']}
            rows.append(dict(**ids,sample=j,**values))
        if (n+1)%20==0: print(json.dumps(dict(probes=n+1,seconds=time.time()-start)),flush=True)
    table(OUT/'samples.csv',rows)
    grouped=defaultdict(list)
    for r in rows: grouped[r['condition'],r['arm'],r['index']].append(r)
    fields=list(values)
    paired=[]
    for condition in sorted({r['condition'] for r in rows}):
        for index in range(20):
            a=sorted(grouped[condition,'CE',index],key=lambda x:x['sample'])
            b=sorted(grouped[condition,'CE_STRUCT',index],key=lambda x:x['sample'])
            assert len(a)==len(b)==32
            row=dict(condition=condition,dataset=a[0]['dataset'],seed=a[0]['seed'],index=index,t=a[0]['t'],draws=32)
            for k in fields:
                row[k+'_CE']=float(np.mean([x[k] for x in a])) if all(x[k] is not None for x in a) else None
                row[k+'_DIRECT']=float(np.mean([x[k] for x in b])) if all(x[k] is not None for x in b) else None
                d=[y[k]-x[k] for x,y in zip(a,b) if x[k] is not None and y[k] is not None]
                row[k+'_valid_pairs']=len(d)
                row[k+'_delta']=float(np.mean(d)) if len(d)==32 else None
                row[k+'_paired_MC_SE']=float(np.std(d,ddof=1)/np.sqrt(len(d))) if len(d)==32 else None
            paired.append(row)
    table(OUT/'paired_inputs.csv',paired)
    summary=[]
    for condition in sorted({r['condition'] for r in rows}):
        ps=[r for r in paired if r['condition']==condition]
        row=dict(condition=condition,dataset=ps[0]['dataset'],seed=ps[0]['seed'],inputs=len(ps))
        for k in fields:
            ds=[x[k+'_delta'] for x in ps]
            row[k+'_delta']=float(np.mean(ds)) if all(x is not None for x in ds) else None
            row[k+'_improved_inputs']=sum(x is not None and x < -1e-12 for x in ds)
            row[k+'_worse_inputs']=sum(x is not None and x > 1e-12 for x in ds)
        summary.append(row)
    table(OUT/'summary.csv',summary); save('summary.json',summary)
    save('asset_binding.json',bound)
    assert all(sha(p)==h for p,h in bound.items())
    assert not torch.cuda.is_initialized()
    save('receipt.json',dict(status='complete',samples=len(rows),probability_files=len(probes),
        old_metric_checks=len(rows)*4,max_abs_error=maxerr,assets_unchanged=True,
        seconds=time.time()-start,peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        GPU=0,model_calls=0,new_samples=0,updates=0))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=OUT)
    OUT=parser.parse_args().output
    main()
