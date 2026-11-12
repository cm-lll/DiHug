"""CPU data/support and memory estimates; no model import or dense graph allocation."""
import argparse,hashlib,json,math
from pathlib import Path
import numpy as np

def inspect(path,rounds):
    with np.load(path,allow_pickle=False) as z:
        t=z['types'];e=z['edges'];sc=z['schema']
        if e.shape[0]!=2:e=e.T
        assert t.ndim==1 and np.issubdtype(t.dtype,np.integer) and (t>=0).all()
        assert np.array_equal(np.unique(t),np.arange(int(t.max())+1))
        assert sc.ndim==2 and sc.shape[1]==2 and len(set(map(tuple,sc)))==len(sc)
        assert (sc[:,0]<=sc[:,1]).all() and sc.min()>=0 and sc.max()<=t.max()
        assert e.shape[0]==2 and np.issubdtype(e.dtype,np.integer) and (e>=0).all() and (e<len(t)).all()
        assert (e[0]<e[1]).all() and len(set(map(tuple,e.T)))==e.shape[1]
        inventory=np.bincount(t);lookup={tuple(map(int,p)):i for i,p in enumerate(sc)}
        f=np.array([lookup[tuple(sorted((int(t[u]),int(t[v]))))] for u,v in e.T],int)
        if 'families' in z:assert np.array_equal(z['families'],f)
        if 'relations' in z:assert np.array_equal(z['relations'],f+1)
        slots=[int(inventory[a]*(inventory[a]-1)//2 if a==b else inventory[a]*inventory[b]) for a,b in sc]
        counts=np.bincount(f,minlength=len(sc));S=sum(slots);active=sum(n for n,c in zip(slots,counts) if c)
        return dict(sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),nodes=len(t),edges=e.shape[1],type_counts=inventory.tolist(),ordered_schema=sc.tolist(),schema_slots=S,active_slots=active,relation_edges=counts.tolist(),prior=[c/n if n else 0 for c,n in zip(counts.tolist(),slots)],rounds=rounds,max_active_queries_per_block=math.ceil(active/rounds),one_dense_float64_matrix_GiB=8*len(t)**2/2**30,one_float64_full_slot_buffer_GiB=8*S/2**30,one_int64_query_pair_block_GiB=16*math.ceil(active/rounds)/2**30,limits=['estimates are individual buffers, NOT peak CPU/GPU memory','dense heat eigendecomposition O(N^3), storage O(N^2)','per-type MMD O(sum n_type^2)','reverse state, random uniforms and coverage retain O(S) host memory','block count affects context and quality; not guaranteed equivalent to R1','fixed type inventory, undirected simple graph, one relation per endpoint type pair','prior=0 relations remain fixed absent; do not omit intended schema pairs'])
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True);p.add_argument('--rounds',type=int,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    assert a.rounds>0 and not a.out.exists()
    result=inspect(a.data,a.rounds);a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
