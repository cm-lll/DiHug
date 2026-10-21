"""Same544 matrices, original scaling first+repeat; no model or new state."""
import time
import numpy as np
from prepare import OUT,read,save,sha,check
import scaling as scaling_reference
def main():
    rows=[];start=time.perf_counter()
    for meta in read(OUT/'input_manifest.json'):
        check();path=OUT/meta['path'];assert sha(path)==meta['sha256']
        with np.load(path) as z:L=z['L'].copy();c=z['counts'].copy()
        p,r=scaling_reference.scale(L,c,check);q,s=scaling_reference.scale(L,c,check)
        for key in ('status','iterations','row_residual','column_residual','underflow_count'):assert r[key]==s[key]==meta['old_scaling'][key]
        if p is not None:np.testing.assert_array_equal(p,q)
        rows.append(dict(index=meta['index'],scope=meta['scope'],first=r,repeat=s))
        if len(rows)%50==0:save(OUT/'reference_cost_partial.json',rows)
    save(OUT/'reference_cost.json',dict(status='completed',rows=rows,seconds=time.perf_counter()-start,new_states=0))
if __name__=='__main__':main()
