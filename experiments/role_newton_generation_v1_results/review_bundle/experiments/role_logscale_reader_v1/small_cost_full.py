"""All44 seed11 states, including rejected allocations; preload arrays before tracing."""
import time,tracemalloc
import numpy as np
from diagnose import OUT,c,cap,check,CFG,ORIGINAL
from scaling import scale

def main():
    records,targets=c.load_data();case=c.metadata_case(c.base,c.PANEL,targets);rows=[];mem=[]
    for name in c.read(c.RESULT/'conditions.json')['final_targets']:
        for t in CFG['diagnostic_times']:
            check();p=OUT/f'seed2026102011/{name}_t{t}.npz'
            if not p.exists():p=OUT/f'seed2026102011/{name}_t{t}_failure_evidence.npz'
            with np.load(p) as z:data={k:z[k].copy() for k in z.files}
            old,fam=c.base.c.state_edges(case,data['state']);deg=cap.degrees(case['space'].types,old,fam,10)
            capacity=cap.capacities(case['space'].types,case['space'].pairs);tables={}
            def construct():
                for typ,roles in enumerate(targets[name].roles):
                    ix=case['space'].types==typ;mat=np.zeros((int(ix.sum()),len(roles)))
                    for j,(v,n) in enumerate(roles):
                        for r,N in enumerate(capacity[typ]):
                            if not N:continue
                            key=(int(N),v[r],t,float(case['prior'][r]))
                            if key not in tables:tables[key]=cap.degree_logpmf(int(N),v[r],cap.schedule(64)[t],float(case['prior'][r]))
                            mat[:,j]+=tables[key][deg[ix,r]]
                    np.testing.assert_array_equal(mat,data[f'type{typ}_LL'])
            tick=time.perf_counter();construct();cold=time.perf_counter()-tick;tick=time.perf_counter();construct();warm=time.perf_counter()-tick
            rows.append(dict(condition=name,t=t,cold_likelihood_seconds=cold,warm_likelihood_seconds=warm,one_time_cache_bytes=sum(v.nbytes for v in tables.values())))
            if name in ('balanced','dev_double','value_0') and t==8:
                for typ in range(4):
                    L=data[f'type{typ}_LL'];counts=data[f'type{typ}_counts']
                    for label,fn in [('exact',ORIGINAL),('scaled',scale)]:
                        check();tracemalloc.start();tick=time.perf_counter();a,r=fn(L,counts);elapsed=time.perf_counter()-tick
                        current,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
                        mem.append(dict(condition=name,type=typ,n=L.shape[0],k=L.shape[1],method=label,
                            traced_peak_bytes=peak,seconds_with_tracing=elapsed,status=r.get('status','exact_DP')))
    c.save(OUT/'small_cost_full.json',dict(likelihood=rows,memory=mem,
        input_decompression_excluded_from_tracing=True,
        memory_scope='traced solver allocation incl output; NOT RSS; tracing time separate from low-interference timing',
        original_partial_cost_retained=True,model_calls=0))
if __name__=='__main__':main()
