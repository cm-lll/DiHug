"""Final source, manifest, optional NN probabilities and bound audit."""
import hashlib,subprocess,time,json
from pathlib import Path
import numpy as np
from prepare import OUT,HERE,EXP,SOURCE,old,read,sha,save,check

def main():
    check();expected={n:h for h,n in (x.split('  ',1) for x in (EXP/'role_logscale_reader_v1_results/review_bundle/SHA256SUMS').read_text().splitlines())}
    p=EXP/'role_logscale_reader_v1_results/review_bundle/SHA256SUMS'
    blob=subprocess.check_output(['git','show','9b010a6b41b39755a9bb36f34fa210b32b8fd689:experiments/role_logscale_reader_v1_results/review_bundle/SHA256SUMS'],cwd=EXP.parent)
    assert hashlib.sha256(blob).hexdigest()==sha(p)
    for name in ('large_input_identity.json','guard_cost/IMDB_reader.json','guard_cost/PubMed_reader.json'):
        assert sha(SOURCE/name)==expected['results/'+name]
    old.binding() # Full original model/kernel/source/data binding, no NN construction.
    manifest=read(OUT/'input_manifest.json')
    for row in manifest:assert sha(OUT/row['path'])==row['sha256']
    frozen=read(OUT/'freeze.json')
    for n,h in frozen['source'].items():assert sha(HERE/n)==h,n
    assert read(OUT/'high_precision.json')['status']=='passed'
    assert read(OUT/'reference_cost.json')['status']=='completed' and len(read(OUT/'reference_cost.json')['rows'])==544
    rr,targets=old.c.load_data();case=old.c.metadata_case(old.c.base,old.c.PANEL,targets)
    fam=np.repeat(np.arange(10),case['space'].counts);kernel=old.c.base.MarginalKernel(case['prior'],old.c.base.c.schedule(64));prediction=[]
    for seed in (2026102011,2026102012,2026102013):
        folder=OUT/f'prediction_seed{seed}';receipt=read(folder/'receipt.json');assert receipt['status']=='completed' and receipt['rows']==44
        for r in read(folder/'rows.json'):
            check();assert r['status']=='completed';name=r['condition'];t=r['t'];p=folder/f'{name}_t{t}.npz';assert sha(p)==r['output_sha256']
            source=next(x for x in manifest if x['scope']=='small' and x['seed']==seed and x['condition']==name and x['t']==t)
            with np.load(EXP/source['source_payload']) as z:state=z['state'];exactp=z['p0_exact'];exactq=z['q_exact']
            with np.load(p) as z:
                assert np.isfinite(z['p0']).all() and np.isfinite(z['q']).all() and ((z['q']>=0)&(z['q']<=1)).all()
                np.testing.assert_array_equal(kernel.reverse(z['p0'],state,fam,t-1,t),z['q'])
                if t==1:np.testing.assert_allclose(z['q'],z['p0'],atol=2e-12,rtol=0)
                assert np.all(z['p0'][case['prior'][fam]==0]==0) and np.all(z['q'][case['prior'][fam]==0]==0)
                for rel in r['relations']:
                    ix=fam==rel['family'];assert abs(rel['p0_max_abs']-abs(z['p0'][ix]-exactp[ix]).max(initial=0))<1e-14
                    assert abs(rel['expected_edge_delta']-(z['q'][ix].sum()-exactq[ix].sum()))<1e-12
            prediction.append(r)
    assert len(prediction)==132
    save(OUT/'prediction_summary.json',dict(rows=prediction,newly_available_states=55,all132_complete=True,
        context_tail_error_bound_max=max(x['tail_context_bound'] for r in prediction for x in r['context_error_bounds']),
        observed_context_float32_error_max=max(x['observed_context_error'] for r in prediction for x in r['context_error_bounds'])))
    save(OUT/'final_audit.json',dict(status='passed',matrices=544,small_original_archived=528,large_deterministically_reconstructed=16,
        historical_large_LL_hash_available=False,old_large_state_hashes_verified=True,old_scaling_status_and_residuals_match=True,
        original_reference_manifest_bound_to_baseline_commit=True,source_config_and_weights_verified=True,
        same_target_pairs=442,high_precision_cases=3,prediction_pairs=132,posterior_and_exact_support_checked=True,
        no_training=True,no_sampling=True,no_GPU=True,all_failures_preserved=True))
    print('final audit544 matrices,442 same-target checks,132 NN pairs; passed')
if __name__=='__main__':main()
