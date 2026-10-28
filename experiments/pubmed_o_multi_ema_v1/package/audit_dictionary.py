"""Read-only audit of the actual offline evidence; no NMF fitting or model calls."""
import argparse,json,pathlib,time,hashlib,resource
import numpy as np
from prepare import load_case,sha
from features import extract,digest
from role_dictionary import feature_groups,node_folds,masked_score,seed32
from hsgd.bw_preflight.common import schedule
from hsgd.noising import global_noisy_edges

def audit(directory):
    start=time.perf_counter();directory=pathlib.Path(directory);rows={}
    for dataset in ['IMDB','PubMed']:
        root=directory/('dictionary_'+dataset);selection=json.loads((root/'selection.json').read_text());panel=json.loads((root/'panel.json').read_text());splits=json.loads((root/'frozen_splits.json').read_text())
        case,cfg=load_case(dataset);types=case['space'].types;groups,sides,perm=feature_groups();folds=node_folds(types,dataset)
        assert splits['groups']==groups and splits['sides']==[x.tolist() for x in sides]
        assert splits['permutation']==perm.tolist()
        values=[]
        for state in panel:
            file=root/state['file'];assert sha(file)==state['sha256'];z=np.load(file,allow_pickle=False)
            np.testing.assert_array_equal(z['types'],types);assert int(z['t'])==state['t']
            if state['t']==0:edges=case['clean'];families=case['space'].encode(edges)[0]
            else:
                seed=seed32(state['root_seed'],dataset=dataset,time=state['t'],purpose='dictionary_forward_noise');assert seed==state['derived_seed']
                edges,families,_=global_noisy_edges(case['space'],case['clean'],case['prior'],schedule(1000)[state['t']],np.random.default_rng(seed))
            np.testing.assert_array_equal(edges,z['edges']);np.testing.assert_array_equal(families,z['families'])
            computed=extract(types,z['edges'],z['families'],case['space'].pairs)
            np.testing.assert_array_equal(computed,z['features']);assert digest(computed)==state['feature_sha256'];values.append(computed)
        failures=[];fits=valid=score_count=0;max_error=0.;max_kkt=0.
        for typ,v in folds.items():
            saved=splits['node_folds'][str(typ)];assert saved['folds']==[x.tolist() for x in v['folds']] and saved['Kmax']==v['Kmax']
            row=selection['types'][str(typ)];assert len(row['candidates'])==v['Kmax'];assert 'selected_K' not in row
            for c in row['candidates']:
                valid+=int(c['valid']);score_index=0;all_scores=[]
                assert len(c['fits'])==2
                for fold,fit in enumerate(c['fits']):
                    fits+=1;assert fit['seed']==seed32(2026102861,dataset=dataset,type=typ,fold=fold,K=c['K'],purpose='candidate_fit')
                    if not fit['converged']:
                        failures.append(dict(type=typ,K=c['K'],fold=fold,iterations=fit['iterations'],warnings=fit['warnings']));assert not c['valid'];continue
                    z=np.load(root/('candidate_type%d_K%d_fold%d.npz'%(typ,c['K'],fold)),allow_pickle=False);h=z['basis']
                    np.testing.assert_allclose(np.linalg.norm(h,axis=1),1,atol=1e-12,rtol=1e-12)
                    scores=masked_score(values,v['folds'][1-fold],h,sides);old=c['scores'][score_index];score_index+=1
                    for a,b in zip(scores,old):
                        error=max(abs(a['weighted_mse']-b['weighted_mse']),max(abs(x-y) for x,y in zip(a['state_mse'],b['state_mse'])))
                        assert error<=1e-14;max_error=max(max_error,error);max_kkt=max(max_kkt,a['kkt_max']);score_count+=1
                    all_scores+=scores
                if c['valid']:assert abs(np.mean([s['weighted_mse'] for s in all_scores])-c['mean_mse'])<=1e-14
        assert selection['status']=='blocked_incomplete_candidate_comparison' and failures and not (root/'final_basis.npz').exists()
        rows[dataset]=dict(status='audit_passed_selection_blocked',nodes=len(types),edges=case['clean'].shape[1],type_counts=np.bincount(types).tolist(),S=case['space'].total,active_slots=int(np.sum(case['space'].counts[case['active']])),candidate_count=sum(v['Kmax'] for v in folds.values()),fold_fits=fits,valid_candidates=valid,nonconverged=failures,recomputed_direction_scores=score_count,score_max_abs_error=max_error,NNLS_max_KKT=max_kkt,panel_states_exact=7,selected_K=None,final_basis_exists=False,original_seconds=selection['seconds'],original_peak_rss_mib=selection['peak_rss_mib'])
    return dict(status='passed',datasets=rows,seconds=time.perf_counter()-start,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,model_calls=0,optimizer_steps=0,NMF_refits=0)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--evidence',required=True);p.add_argument('--output',required=True);a=p.parse_args();result=audit(a.evidence)
    with pathlib.Path(a.output).open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))
