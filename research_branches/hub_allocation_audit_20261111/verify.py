"""Independent reconstruction of recorded CPU moves; does not call scoring/model."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ[name]='1'
import json,hashlib,time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/hub_allocation_audit_20261111'


def main():
    start=time.time()
    rows=json.loads((OUT/'per_graph.json').read_text())
    ops=json.loads((OUT/'operations.json').read_text())
    cache=np.load(OUT/'graphs.npz')
    tasks=json.loads((ROOT/'results/real192_update_audit_20261111/protocol.json').read_text())['tasks']
    meta={}
    for ds in ['IMDB','PubMed']:
        task=next(t for t in tasks if t['parent']['dataset']==ds)
        data=np.load(task['parent']['data'])
        z=np.load(ROOT/'results/real192_update_audit_20261111'/task['id']/'validation_u2000_raw_t1.npz')
        a=np.zeros((len(data['types']),len(data['types'])),dtype=np.int64)
        for u,v in data['edges']: a[u,v]=a[v,u]=1
        meta[ds]=(a,data['types'],z['query'],z['families'])
    verified=0; changed=0
    for ds,t,seed,mode in sorted({(r['dataset'],r['focal_type'],r['seed'],r['mode']) for r in rows if r['focal_type'] is not None}):
        original,types,q,f=meta[ds];a=original.copy();count=0
        selected=sorted([v for v in ops if (v['dataset'],v['focal_type'],v['seed'],v['mode'])==(ds,t,seed,mode)],key=lambda x:x['index'])
        targets=sorted([r for r in rows if (r['dataset'],r['focal_type'],r['seed'],r['mode'])==(ds,t,seed,mode)],key=lambda x:x['fraction'])
        for row in targets:
            while count<row['moves']:
                op=selected[count];u,v,w=[op[k] for k in ['donor','recipient','neighbor']]
                assert types[u]==types[v]==t and len({u,v,w})==3
                assert a[u,w]==original[u,w]==1 and a[v,w]==original[v,w]==0
                before=a.sum(1);a[u,w]=a[w,u]=0;a[v,w]=a[w,v]=1;after=a.sum(1)
                delta=int(np.sum(after**2)-np.sum(before**2));assert delta==op['degree_square_delta']
                assert mode!='spread' or delta<0
                assert mode!='concentrate' or delta>0
                count+=1;changed+=1
            np.testing.assert_array_equal(a[q[0],q[1]],cache[row['key']+'_bits'])
            np.testing.assert_array_equal(np.bincount(f,weights=a[q[0],q[1]]),np.bincount(f,weights=original[q[0],q[1]]))
            assert int(np.trace(a@a@a)//6)==row['triangles']
            # Independent reachability, without the scorer's component routine.
            unseen=set(range(len(a)));sizes=[]
            while unseen:
                stack=[unseen.pop()];size=0
                while stack:
                    u=stack.pop();size+=1
                    new=unseen.intersection(np.flatnonzero(a[u]).tolist())
                    unseen.difference_update(new);stack.extend(new)
                sizes.append(size)
            assert len(sizes)==row['components'] and max(sizes)==row['lcc']
            verified+=1
    result=dict(status='passed',snapshots_reconstructed=verified,moves_reconstructed=changed,
        unique_graph_bitvectors=len({hashlib.sha256(cache[k].tobytes()).hexdigest() for k in cache.files}),
        total_saved_rows=len(rows),seconds=time.time()-start,model_calls=0,GPU_hours=0,
        checks=['recorded moves', 'same type endpoints', 'no repeated or self edges','fixed relation counts','triangle enumeration by trace A^3','independent reachability'])
    (OUT/'independent_checks.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
