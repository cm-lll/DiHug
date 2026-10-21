"""Certify delivery nonfinite-guard amendment on all528 stored LLs, no NN."""
import hashlib,json,time
import numpy as np
from diagnose import OUT,HERE,c,CFG,check
from scaling import scale

def main():
    start=time.perf_counter();old=OUT/'numerical_guard_original/scaling.py'
    assert c.sha(old)==c.read(OUT/'freeze.json')['source']['scaling.py']
    count=0;times=[]
    for seed in (2026102011,2026102012,2026102013):
        for name in c.read(c.RESULT/'conditions.json')['final_targets']:
            for t in CFG['diagnostic_times']:
                stem=OUT/f'seed{seed}/{name}_t{t}';r=c.read(stem.with_suffix('.json'))
                success=r['status']=='completed';path=stem.with_suffix('.npz') if success else stem.parent/(stem.name+'_failure_evidence.npz')
                aa=r['allocation'] if success else c.read(stem.parent/(stem.name+'_failure_evidence.json'))['allocations']
                with np.load(path) as z:
                    for item in aa:
                        check();typ=item['type'];L=z[f'type{typ}_LL'];counts=z[f'type{typ}_counts'];oldp=z[f'type{typ}_'+('scaled' if success else 'unaccepted_scaled')]
                        p,receipt=scale(L,counts,check);assert receipt['status']==item['scaled']['status']
                        assert receipt['iterations']==item['scaled']['iterations'];np.testing.assert_array_equal(p,oldp)
                        count+=1;times.append(receipt['seconds'])
    assert count==528
    c.save(OUT/'numerical_guard_equivalence.json',dict(status='passed',matrices=count,all_probability_arrays_bitwise_equal=True,
        all_status_and_iteration_counts_equal=True,source_before_sha256=c.sha(old),source_after_sha256=c.sha(HERE/'scaling.py'),
        amendment='reject nonfinite potentials/logP or objective immediately; no math/tolerance/settings change',
        algorithm_seconds=times,wall_seconds=time.perf_counter()-start,model_calls=0,training=0,chain_replay=0))
if __name__=='__main__':main()
