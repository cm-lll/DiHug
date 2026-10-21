"""Independent receipt/array/source audit. No NN, fitting, or sampling."""
import json,time
from collections import Counter
from pathlib import Path
import numpy as np
from diagnose import c,OUT,HERE,CFG,check

def main():
    tick=time.perf_counter();freeze=c.read(OUT/'freeze.json');supp=c.read(OUT/'supplement_freeze.json')
    guard=c.read(OUT/'numerical_guard_equivalence.json')
    assert guard['status']=='passed' and guard['matrices']==528 and guard['all_probability_arrays_bitwise_equal']
    assert c.sha(HERE/'scaling.py')==guard['source_after_sha256']
    for n,h in freeze['source'].items():
        p=OUT/'numerical_guard_original/scaling.py' if n=='scaling.py' else HERE/n
        assert c.sha(p)==h,n
    for n,h in supp['source'].items():assert c.sha(HERE/n)==h,n
    assert CFG['row_absolute_tolerance']==CFG['column_relative_tolerance']==1e-8 and CFG['entropy_coefficient']==1
    assert c.read(OUT/'supplement_receipt.json')['status']=='completed'
    summ=c.read(OUT/'summary.json');complete=failed=0;types=Counter();failed_rows=[];large_states=[]
    records,targets=c.load_data();small_case=c.metadata_case(c.base,c.PANEL,targets)
    kernel=c.base.MarginalKernel(small_case['prior'],c.base.c.schedule(64))
    families=np.repeat(np.arange(10),small_case['space'].counts)
    for seed in (2026102011,2026102012,2026102013):
        assert c.read(OUT/f'seed{seed}/validation.json')['maximum_difference']==0
        assert c.read(OUT/f'seed{seed}/receipt.json')['status']=='completed'
        for name in c.read(c.RESULT/'conditions.json')['final_targets']:
            assert c.read(OUT/f'seed{seed}/{name}_replay.json')['all64_hashes_match']
            for t in CFG['diagnostic_times']:
                check();stem=OUT/f'seed{seed}/{name}_t{t}';r=c.read(stem.with_suffix('.json'))
                if r['status']=='completed':
                    complete+=1;path=stem.with_suffix('.npz');aa=r['allocation'];ak='scaled'
                else:
                    failed+=1;rr=c.read(Path(str(stem)+'_failure_evidence.json'));aa=rr['allocations'];path=Path(str(stem)+'_failure_evidence.npz');ak='unaccepted_scaled'
                    assert rr['downstream_scaled_prediction_not_executed']
                with np.load(path,allow_pickle=False) as z:
                    for label in ('exact','scaled'):
                        if 'p0_'+label not in z:continue
                        q=kernel.reverse(z['p0_'+label],z['state'],families,t-1,t)
                        np.testing.assert_array_equal(q,z['q_'+label])
                        if t==1:np.testing.assert_allclose(q,z['p0_'+label],atol=2e-12,rtol=0)
                    for a in aa:
                        typ=a['type'];L=z[f'type{typ}_LL'];counts=z[f'type{typ}_counts'];p=z[f'type{typ}_'+ak];rc=a['scaled']
                        row=float(abs(p.sum(1)-1).max());col=float((abs(p.sum(0)-counts)/np.maximum(1,counts)).max())
                        assert abs(row-rc['row_residual'])<1e-14 and abs(col-rc['column_residual'])<1e-14
                        assert np.all(p[np.isneginf(L)]==0) and rc['support_feasible']
                        types[(typ,rc['status'])]+=1
                        if rc['status']=='converged':assert row<=1e-8 and col<=1e-8
                        else:
                            assert rc['iterations']==1000 and max(row,col)>1e-8
                            failed_rows.append(dict(seed=seed,condition=name,t=t,type=typ,**rc))
    assert (complete,failed)==(summ['completed_states'],len(summ['failed_states']))==(77,55)
    # Reconstruct only the pre-frozen large forward-noise inputs for identity receipts.
    for name in ('IMDB','PubMed'):
        inv=c.read(c.EXP/'role_target_scalability_v1/evidence/large_graph_inventory.json')[name]
        path=c.EXP/f'joint_cn_dual_rounds_v1_1/package/data/{name}.npz';assert c.sha(path)==inv['sha256']
        with np.load(path) as z:case=c.base.c.make_case(z['types'],z['edges'].T,tuple(map(tuple,z['schema'])))
        report=c.read(OUT/f'{name}_reader.json');assert report['status']=='completed' and len(report['rows'])==8
        for t in CFG['large_times']:
            check();ei,ef,_=c.base.c.global_noisy_edges(case['space'],case['clean'],case['prior'],c.base.c.schedule(1000)[t],np.random.default_rng([CFG['large_noise_seed'],t]))
            large_states.append(dict(dataset=name,t=t,edge_array_sha256=__import__('hashlib').sha256(ei.tobytes()+ef.tobytes()).hexdigest(),
                source_sha256=inv['sha256'],seed=[CFG['large_noise_seed'],t],actual_edges=ei.shape[1],
                state_source='original kernel forward noise from verified reference; NOT generated chain state'))
    c.save(OUT/'failed_allocations.json',failed_rows);c.save(OUT/'large_input_identity.json',large_states)
    c.save(OUT/'audit_guard.json',dict(status='passed',complete_prediction_pairs=complete,rejected_prediction_states=failed,
        all132_state_payloads_present=True,allocation_status_counts=[dict(type=k[0],status=k[1],count=v) for k,v in sorted(types.items())],
        original_source_preserved=True,only_guard_amendment_verified_bitwise=True,large_reader_cases=16,large_networks=0,
        no_unconverged_allocation_used_for_network=True,zero_training_updates=True,new_independent_samples=0,
        failed_inputs_not_falsely_passed=True,posterior_audited_all132=True,seconds=time.perf_counter()-tick))
    print('audit passed; 77 accepted,55 rejected,all132 payloads;16 large reader cases')
if __name__=='__main__':
    main()
