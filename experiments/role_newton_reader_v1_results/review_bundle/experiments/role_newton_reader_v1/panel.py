"""All544 LLs, first + repeat, independent residuals and original target comparisons."""
import argparse,json,resource,sys,time,traceback
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
from solver import solve
from prepare import OUT,HERE,SOURCE,read,sha,save,check
import scaling as scaling_reference

def objective(P,LP,L):
    ix=P>0;return float(np.sum(P[ix]*(L[ix]-LP[ix])))
def run(scope):
    start=time.perf_counter();rows=[];status='failed';reason=None
    try:
        frozen=read(OUT/'freeze.json')
        for n,h in frozen['source'].items():assert sha(HERE/n)==h,n
        assert sha(OUT/'input_manifest.json')==frozen['input_manifest_sha256']
        cases=[x for x in read(OUT/'input_manifest.json') if x['scope']==scope]
        for meta in cases:
            check();path=OUT/meta['path'];assert sha(path)==meta['sha256']
            with np.load(path) as z:L=z['L'].copy();c=z['counts'].copy()
            p,lp,r=solve(L,c,check);again,logs,s=solve(L,c,check)
            assert s['status']==r['status']
            if p is not None:np.testing.assert_array_equal(p,again);np.testing.assert_array_equal(lp,logs)
            row=dict(input=meta,result=r,repeat=s,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
            if p is not None:
                assert np.isfinite(p).all() and (p>=0).all();assert abs(p.sum(1)-1).max()<=1e-8
                assert (abs(p.sum(0)-c)/np.maximum(1,c)).max()<=1e-8
                # Independent primal values include declared tiny-tail loss bounds.
                row['objective']=objective(p,lp,L)
                reference=None
                if scope=='small':
                    src=HERE.parents[0]/meta['source_payload']
                    with np.load(src) as z:reference=z[meta['old_scaling_P_key']].copy()
                else:
                    reference,oldr=scaling_reference.scale(L,c,check)
                    row['old_scaling_recomputed']=oldr
                    for key in ('status','iterations','row_residual','column_residual','underflow_count'):
                        assert oldr[key]==meta['old_scaling'][key],(meta['index'],key)
                if meta['old_scaling']['status']=='converged':
                    row['old_scaled_max_abs']=float(abs(p-reference).max())
                    ix=reference>0;Jold=float(np.sum(reference[ix]*(L[ix]-np.log(reference[ix]))))
                    row['old_scaled_objective']=Jold;row['objective_abs_difference']=abs(row['objective']-Jold)
                    row['same_target_comparison_pass']=row['old_scaled_max_abs']<=1e-6 and row['objective_abs_difference']<=1e-8*max(1,abs(Jold))
                dst=OUT/f'outputs/matrix{meta["index"]:04d}.npz';dst.parent.mkdir(exist_ok=True)
                np.savez_compressed(dst,P=p,logP=lp);row['output_path']=str(dst.relative_to(OUT));row['output_sha256']=sha(dst)
            rows.append(row);save(OUT/f'{scope}_partial.json',rows)
            if len(rows)%20==0 or scope=='large':print(scope,len(rows),meta['index'],r['status'],r['iterations'],r.get('column_residual'),r['seconds'],flush=True)
        status='completed'
    except BaseException as exc:reason=repr(exc);traceback.print_exc()
    finally:save(OUT/f'{scope}_receipt.json',dict(status=status,reason=reason,rows=len(rows),seconds=time.perf_counter()-start,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024))
    return int(status!='completed')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('scope',choices=['small','large']);sys.exit(run(p.parse_args().scope))
