"""Independent full panel KKT/constraints/rounding and cost summary."""
import json,time,resource,tracemalloc
from collections import Counter
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
from prepare import OUT,HERE,read,sha,save,check
from solver import solve

def stats(v):
    a=np.array(v,float);return dict(n=len(a),median=float(np.median(a)),P90=float(np.quantile(a,.9)),max=float(a.max()),total=float(a.sum()))
def main():
    start=time.perf_counter();rows=read(OUT/'small_partial.json')+read(OUT/'large_partial.json');assert len(rows)==544
    status=Counter();old=Counter();comparisons=[];bounds=[];independent=[];classification=Counter()
    for row in rows:
        check();meta=row['input'];r=row['result'];status[r['status']]+=1;old[meta['old_scaling']['status']]+=1
        assert sha(OUT/meta['path'])==meta['sha256']
        if not r.get('new_contract_pass'):continue
        assert sha(OUT/row['output_path'])==row['output_sha256']
        with np.load(OUT/meta['path']) as z:L=z['L'];c=z['counts']
        with np.load(OUT/row['output_path']) as z:P=z['P'];LP=z['logP']
        assert np.isfinite(P).all() and np.all(P>=0);assert abs(P.sum(1)-1).max()<=1e-8
        assert np.max(abs(P.sum(0)-c)/np.maximum(1,c))<=1e-8
        finite=np.isfinite(LP);mathematical=np.isneginf(L);forced=np.isfinite(L)&~finite
        rounded=finite&(P==0);assert np.all(LP[rounded]<-700)
        assert rounded.sum()==r['rounded_positive_count'] and forced.sum()==r['constraint_forced_zeros']
        np.testing.assert_array_equal(P[~finite],0)
        np.testing.assert_array_equal(P[finite&~rounded],np.exp(LP[finite&~rounded]))
        mass=rounded.sum(1)*np.exp(-700);np.testing.assert_array_equal(mass,np.array(r['row_omitted_mass_upper']))
        assert mass.max(initial=0)<=1e-280 and mass.sum()<=1e-276
        # Independent additive row/column-potential KKT recovery, not solver's bias.
        assert finite.all(),'actual panel expected full support; sparse support handled by exhaustive tests'
        D=LP-L;residual=float(abs(D-D[:,:1]-D[:1,:]+D[0,0]).max())
        assert residual<=2e-10
        normalized=float(abs(logsumexp(LP,axis=1)).max());assert normalized<=2e-12
        independent.append(dict(index=meta['index'],stationarity=residual,log_row_normalization=normalized))
        bounds.append(float(mass.sum()))
        if row.get('same_target_comparison_pass') is not None:comparisons.append(dict(index=meta['index'],passed=row['same_target_comparison_pass'],max_abs=row['old_scaled_max_abs'],objective_abs=row['objective_abs_difference']))
        inputLP=L-logsumexp(L,axis=1)[:,None]
        classification['full_support_concentrated' if np.max(inputLP,axis=1).max()>np.log1p(-1e-12) else 'full_support_other']+=1
    assert all(x['passed'] for x in comparisons),'same entropy target mismatch; stop before optional NN'
    memory=[]
    for index in (0,527,537,541):
        row=next(x for x in rows if x['input']['index']==index)
        with np.load(OUT/row['input']['path']) as z:L=z['L'].copy();c=z['counts'].copy()
        check();tracemalloc.start();tick=time.perf_counter();p,lp,r=solve(L,c,check);current,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
        memory.append(dict(index=index,traced_peak_bytes=peak,seconds_with_tracing=time.perf_counter()-tick,status=r['status']))
    costs={}
    for scope in ('small','large'):
        rr=[x for x in rows if x['input']['scope']==scope]
        costs[scope]=dict(first=stats([x['result']['seconds'] for x in rr]),repeat=stats([x['repeat']['seconds'] for x in rr]),
            support=stats([x['result']['support_seconds'] for x in rr]),solver=stats([x['result']['solver_seconds'] for x in rr]),
            iterations=stats([x['result']['iterations'] for x in rr]),evaluations=stats([x['result']['function_evaluations'] for x in rr]))
    save(OUT/'independent_KKT.json',independent);save(OUT/'same_target_comparisons.json',comparisons)
    save(OUT/'audit.json',dict(status='passed',matrices=len(rows),accepted=sum(r['result'].get('new_contract_pass',False) for r in rows),
        statuses=dict(status),original_scaling=dict(old),classification=dict(classification),same_target_pairs=len(comparisons),
        max_same_target_P_difference=max(x['max_abs'] for x in comparisons),
        maximum_independent_stationarity=max(x['stationarity'] for x in independent),
        maximum_global_tail_mass_bound=max(bounds),costs=costs,memory=memory,
        worst_process_rss_mib=max(x['peak_rss_mib'] for x in rows),seconds=time.perf_counter()-start,
        contract='new bounded-tail contract explicitly distinguished from original strict output'))
    print('independent audit',dict(status), 'same-target pairs',len(comparisons),flush=True)
if __name__=='__main__':main()
