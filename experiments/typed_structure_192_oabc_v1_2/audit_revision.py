"""Audit existing refits/engineering states only; never fit or step."""
import pathlib,json,sys,hashlib,time
sys.path.insert(0,str(pathlib.Path(__file__).parent/'package'))
import numpy as np
import torch
from role_dictionary import weighted_rows,masked_score
from admission import assignment
from probe import equal
from restore_checks import restoration_passes

def main(root):
    root=pathlib.Path(root);report=dict(status='passed',dictionaries={},engineering={},source_scope='GPU source snapshots retained; later additions are launcher/summary/tests, not neural code',formal_updates=0,generated_graphs=0);cost=dict(complete_eight_cells=True,cells={},concurrency_margin=1.25,margin_meaning='UNMEASURED conservative planning factor; not eight-card throughput evidence',schedule='fixed immutable per-device FIFO queues from admission.assignment')
    for dataset in ['IMDB','PubMed']:
        d=root/('dictionary_'+dataset);r=json.loads((d/'selection.json').read_text());assert r['status']=='passed';panel=json.loads((d/'panel.json').read_text());splits=json.loads((d/'frozen_splits.json').read_text());arrays=[np.load(d/x['file'])['features'] for x in panel];types=np.load(d/panel[0]['file'])['types'];sides=[np.array(x) for x in splits['sides']];score_error=0.;reconstruction_error=0.;fits=0
        for t,row in r['types'].items():
            folds=[np.array(x) for x in splits['node_folds'][t]['folds']]
            for c in row['candidates']:
                assert c['valid']
                for fold,f in enumerate(c['fits']):
                    assert f['converged'];fits+=1;name='candidate_type%s_K%d_fold%d'%(t,c['K'],fold);z=np.load(d/(name+'_raw.npz'));matrix=weighted_rows(arrays,folds[fold]);assert str(z['input_sha256'])==hashlib.sha256(np.ascontiguousarray(matrix).tobytes()).hexdigest();assert hashlib.sha256((d/(name+'_raw.npz')).read_bytes()).hexdigest()==f['artifact_sha256']
                    h=z['H']/np.linalg.norm(z['H'],axis=1)[:,None];np.testing.assert_array_equal(h,z['normalized_basis']);np.testing.assert_array_equal(h,np.load(d/(name+'.npz'))['basis']);err=abs(np.linalg.norm(matrix-z['W']@z['H'])-f['reconstruction_error']);assert err<1e-8;reconstruction_error=max(reconstruction_error,err)
                    scores=masked_score(arrays,folds[1-fold],h,sides)
                    for a,b in zip(scores,c['scores'][fold]):score_error=max(score_error,abs(a['weighted_mse']-b['weighted_mse']))
            assert row['final_fit']['converged'];z=np.load(d/('final_type%s_raw.npz'%t));matrix=weighted_rows(arrays,np.flatnonzero(types==int(t)));assert str(z['input_sha256'])==hashlib.sha256(np.ascontiguousarray(matrix).tobytes()).hexdigest()
        assert score_error<=1e-14
        report['dictionaries'][dataset]=dict(candidate_fold_fits=fits,K=r['K'],offsets=r['offsets'],boundary_hit=[v['boundary_hit'] for v in r['types'].values()],recomputed_score_max_abs=score_error,reconstruction_error_max_abs=reconstruction_error,old_converged_comparisons=len(r['old_converged_comparison']),old_basis_max_abs=max(v['basis_max_abs'] for v in r['old_converged_comparison']),old_score_max_abs=max(v['score_max_abs'] for v in r['old_converged_comparison']))
        e=root/('engineering_'+dataset);receipt=json.loads((e/'summary.json').read_text());assert receipt['status']=='engineering_completed_no_formal';rng_proofs=[];models={}
        for arm in 'OABC':
            label=dataset+'_'+arm;row=receipt['cells'][label];assert row['passed'] and row['actual_steps']==3;initial=torch.load(e/arm/'initial.pt',map_location='cpu');models[arm]=initial['model'];assert initial['updates']==0;rng_proofs.append(initial['rng']);proof=json.loads((e/arm/'initialization.json').read_text());assert all(proof['copied_exact'].values())
            same=json.loads((e/arm/'same_gpu.json').read_text());rest=json.loads((e/arm/'restore.json').read_text());assert same['passes'] and rest['passes'] and rest['optimizer_exact'] is True and restoration_passes(rest) and row['load_exact']
            report['engineering'][label]=dict(steps=3,load_exact=True,same_gpu_passed=True,same_gpu_gradient_max_abs=max(v['max_abs'] for v in same['gradients'].values()),continuation_passed=True,continuation_parameter_max_abs=max(v['max_abs'] for v in rest['parameters'].values()),optimizer_exact=rest['optimizer_exact'],rng_exact=rest['rng_exact'],parameters=row['parameters'],normal2_seconds=row['normal2_seconds'],sampling_step_seconds=row['sampling_step_seconds'],peak_allocated_mib=row['peak_allocated_mib'],peak_reserved_mib=row['peak_reserved_mib'])
            # Nine fixed validation forwards + three checkpoint writes + explicit 10s per-cell unmeasured I/O reserve.
            allowance=3*sum(x['total_seconds'] for x in row['full_step'])+3*row['save_seconds']+10
            cost['cells'][label]=dict(passed=True,normal2_seconds=row['normal2_seconds'],sampling_step_seconds=row['sampling_step_seconds'],validation_save_seconds=allowance,allowance_meaning='component estimate, includes extra 10s/cell; not an actual formal validation-save measurement')
        assert all(equal(rng_proofs[0],x) for x in rng_proofs)
        for arm in 'OBC':
            for key,value in models[arm].items():
                if key in models['A'] and models['A'][key].shape==value.shape:assert torch.equal(value,models['A'][key])
                elif key=='joint_head.0.weight':assert torch.equal(value,models['A'][key][:,:value.shape[1]])
        report.setdefault('paired_initial_state_and_rng',{})[dataset]=True
    cost['tiers']=[];remaining=1793135169-time.time()
    for cards in [7,8]:
        for updates in [2000,1000,500]:
            queues,loads=assignment(cost,updates,list(range(cards)));seconds=max(loads)*cost['concurrency_margin']+2100;cost['tiers'].append(dict(cards=cards,updates=updates,queues=queues,loads_seconds=loads,seconds_including_20min_delivery_15min_fault=seconds,remaining_seconds_at_estimate=remaining,fits_at_estimate=seconds<=remaining,actual_card_availability_confirmed=cards==7))
    cost['selected_updates']=None;cost['authorization']=False;cost['measured_concurrency']='two single-GPU processes, each four sequential cells; no eight-card timing';cost['formal_risk']='independent forward-noise engineering states do not bound all free-generation CN/NNLS costs'
    return report,cost

if __name__=='__main__':
    p=pathlib.Path(sys.argv[1]);report,cost=main(p)
    for name,r in [('revision_audit.json',report),('cost_proposal.json',cost)]:
        with (p/name).open('x') as f:json.dump(r,f,indent=2)
