"""Bounded offline preparation; failure never drops a candidate or changes K."""
import argparse,json,pathlib,time,signal,traceback,resource
import numpy as np
from prepare import load_case,save,sha,ROOT
from features import extract,digest
from role_dictionary import seed32,feature_groups,node_folds,weighted_rows,fit,masked_score,choose,NMF_OPTIONS
from hsgd.bw_preflight.common import schedule
from hsgd.noising import global_noisy_edges

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',choices=['IMDB','PubMed'],required=True);ap.add_argument('--output',required=True);ap.add_argument('--ledger',required=True);a=ap.parse_args()
    out=pathlib.Path(a.output);out.mkdir(exist_ok=False);ledger=json.loads(pathlib.Path(a.ledger).read_text());deadline=ledger['preparation_deadline_epoch']-300
    def check():
        if time.time()>=deadline:raise TimeoutError('preparation delivery reserve')
        if (out/'STOP').exists():raise InterruptedError('STOP')
    def interrupted(s,f):raise InterruptedError('signal '+str(s))
    for sig in [signal.SIGTERM,signal.SIGINT,signal.SIGHUP]:signal.signal(sig,interrupted)
    signal.signal(signal.SIGALRM,lambda s,f:(_ for _ in ()).throw(TimeoutError('preparation limit')));signal.alarm(max(1,int(deadline-time.time())))
    result=dict(status='running',dataset=a.dataset,types={},formal_authorized=False,started_epoch=time.time(),model_calls=0)
    try:
        check();case,cfg=load_case(a.dataset);types=case['space'].types;groups,sides,perm=feature_groups();folds=node_folds(types,a.dataset)
        split={str(t):dict(seed=v['seed'],folds=[x.tolist() for x in v['folds']],Kmax=v['Kmax']) for t,v in folds.items()}
        save(out/'frozen_splits.json',dict(groups=groups,group_names=[kind+'_family_'+str(r) for kind in ['degree','egonet_internal','egonet_boundary'] for r in range(10)],permutation=perm.tolist(),sides=[x.tolist() for x in sides],node_folds=split,NMF=NMF_OPTIONS,seed_derivation='SHA256 sorted compact JSON of root and labels; first 4 bytes little-endian uint32',score='mean of 2 folds x 2 feature directions; clean .25 and each noisy state .125',data=cfg['datasets'][a.dataset],S=case['space'].total,prior=case['prior'].tolist()))
        features=[];states=[]
        for t,root in [(0,None)]+[(t,root) for t in [100,500,1000] for root in [2026102831,2026102832]]:
            check();seed=None if root is None else seed32(root,dataset=a.dataset,time=t,purpose='dictionary_forward_noise')
            if t==0:e=case['clean'];f=case['space'].encode(e)[0]
            else:e,f,_=global_noisy_edges(case['space'],case['clean'],case['prior'],schedule(1000)[t],np.random.default_rng(seed))
            begin=time.perf_counter();value=extract(types,e,f,case['space'].pairs);elapsed=time.perf_counter()-begin
            name='state_'+str(t)+'_'+str(root);np.savez_compressed(out/(name+'.npz'),types=types,edges=e,families=f,t=np.array(t),features=value)
            states.append(dict(file=name+'.npz',sha256=sha(out/(name+'.npz')),t=t,root_seed=root,derived_seed=seed,feature_sha256=digest(value),feature_seconds=elapsed,edges=len(f)))
            features.append(value)
        save(out/'panel.json',states)
        # Freeze splits and actual Xt before ANY candidate fit.
        for typ,v in folds.items():
            ids=np.flatnonzero(types==typ)
            for side in sides:
                if not any(np.any(f[ids][:,side]>0) for f in features):raise ValueError('empty effective feature side for type '+str(typ))
        all_valid=True
        for typ,v in folds.items():
            row=dict(Kmax=v['Kmax'],candidates=[]);result['types'][str(typ)]=row
            for k in range(1,v['Kmax']+1):
                candidate=dict(K=k,fits=[],scores=[],valid=True)
                for fold in range(2):
                    check();seed=seed32(2026102861,dataset=a.dataset,type=typ,fold=fold,K=k,purpose='candidate_fit')
                    matrix=weighted_rows(features,v['folds'][fold]);basis,rec=fit(matrix,k,seed);candidate['fits'].append(rec)
                    if basis is None:candidate['valid']=False;continue
                    np.savez_compressed(out/('candidate_type%d_K%d_fold%d.npz'%(typ,k,fold)),basis=basis)
                    try:candidate['scores'].append(masked_score(features,v['folds'][1-fold],basis,sides))
                    except Exception as exc:candidate['valid']=False;candidate.setdefault('errors',[]).append(repr(exc))
                candidate['mean_mse']=float(np.mean([direction['weighted_mse'] for fold in candidate['scores'] for direction in fold])) if candidate['valid'] and len(candidate['scores'])==2 else None
                row['candidates'].append(candidate);all_valid &= candidate['valid'];save(out/'selection.json',result)
                print(a.dataset,'type',typ,'K',k,'valid',candidate['valid'],'iterations',[r['iterations'] for r in candidate['fits']],flush=True)
        if not all_valid:
            result['status']='blocked_incomplete_candidate_comparison';result['reason']='At least one nonconverged/invalid candidate. No candidate discarded; no K selected; no final basis; no downstream release.'
        else:
            matrices=[];offsets=[0];ks=[];final_ok=True
            for typ,v in folds.items():
                check();row=result['types'][str(typ)];k=choose(row['candidates']);row.update(selected_K=k,boundary_hit=k==v['Kmax']);ks.append(k)
                seed=seed32(2026102861,dataset=a.dataset,type=typ,fold='all',K=k,purpose='final_fit');basis,rec=fit(weighted_rows(features,np.flatnonzero(types==typ)),k,seed);row['final_fit']=rec;final_ok &= basis is not None
                if basis is not None:matrices.append(basis);offsets.append(offsets[-1]+k)
                save(out/'selection.json',result)
            if final_ok:
                np.savez_compressed(out/'final_basis.npz',basis=np.concatenate(matrices),offsets=np.array(offsets,dtype=np.int64),K=np.array(ks,dtype=np.int64))
                result.update(status='passed',basis_sha256=sha(out/'final_basis.npz'),K=ks,offsets=offsets,panel_sha256=sha(out/'panel.json'),splits_sha256=sha(out/'frozen_splits.json'))
            else:result.update(status='blocked_final_fit',reason='No release of partial final dictionary')
    except BaseException as exc:result.update(status='failed_or_incomplete',error=repr(exc),traceback=traceback.format_exc())
    finally:
        result.update(finished_epoch=time.time(),seconds=time.time()-result['started_epoch'],peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024);save(out/'selection.json',result);print(result['status'],flush=True)
    return 0 if result['status']=='passed' else 2
if __name__=='__main__':raise SystemExit(main())
