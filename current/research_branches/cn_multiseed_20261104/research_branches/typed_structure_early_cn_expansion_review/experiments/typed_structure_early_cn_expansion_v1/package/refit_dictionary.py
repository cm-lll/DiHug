"""Authorized 50000-iteration refit, using exactly the archived matrices/splits."""
import argparse,pathlib,json,shutil,time,signal,traceback,resource
import numpy as np
from prepare import sha,save
from role_dictionary import fit,weighted_rows,masked_score,choose,seed32,NMF_OPTIONS

def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True,choices=['IMDB','PubMed']);p.add_argument('--source',required=True);p.add_argument('--manifest',required=True);p.add_argument('--manifest-sha256',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    source=pathlib.Path(a.source).resolve();manifest=pathlib.Path(a.manifest).resolve();out=pathlib.Path(a.output);out.mkdir(exist_ok=False);started=time.time()
    result=dict(status='running',dataset=a.dataset,types={},started_epoch=started,refit_not_resume=True,options=NMF_OPTIONS,original_manifest_sha256=a.manifest_sha256,original_status_sha256=sha(source/'selection.json'))
    def check():
        if time.time()>=1793126769:raise TimeoutError('fixed preparation deadline with 5-minute delivery reserve')
        if (out/'STOP').exists():raise InterruptedError('STOP')
    def stop(s,f):raise InterruptedError('signal '+str(s))
    for s in [signal.SIGTERM,signal.SIGINT,signal.SIGHUP]:signal.signal(s,stop)
    signal.signal(signal.SIGALRM,stop);signal.alarm(max(1,int(1793126769-time.time())))
    try:
        check();assert sha(manifest)==a.manifest_sha256;inventory=json.loads(manifest.read_text())['files'];anchor=manifest.parent
        for file in source.iterdir():
            if file.is_file():assert sha(file)==inventory[str(file.relative_to(anchor))]['sha256']
        old=json.loads((source/'selection.json').read_text());splits=json.loads((source/'frozen_splits.json').read_text());panel=json.loads((source/'panel.json').read_text())
        # Preserve original split/panel bytes, including historical NMF=5000 metadata.
        # This revision's only fit override is explicit in refit_config.json.
        for name in ['panel.json','frozen_splits.json']+[s['file'] for s in panel]:shutil.copy2(source/name,out/name)
        save(out/'refit_config.json',dict(options=NMF_OPTIONS,historical_splits_metadata_not_effective_solver=True,source_files={p.name:sha(p) for p in source.iterdir() if p.is_file()},comparison_atol=1e-14,comparison_rtol=0))
        arrays=[np.load(source/s['file'],allow_pickle=False)['features'] for s in panel];types=np.load(source/panel[0]['file'],allow_pickle=False)['types'];sides=[np.array(x) for x in splits['sides']];valid=True;comparison=[]
        for typ_string,split in splits['node_folds'].items():
            typ=int(typ_string);row=dict(Kmax=split['Kmax'],candidates=[]);result['types'][typ_string]=row;folds=[np.array(x) for x in split['folds']]
            for k in range(1,split['Kmax']+1):
                c=dict(K=k,fits=[],scores=[],valid=True);old_c=old['types'][typ_string]['candidates'][k-1];old_score=0
                for fold in range(2):
                    check();seed=seed32(2026102861,dataset=a.dataset,type=typ,fold=fold,K=k,purpose='candidate_fit');matrix=weighted_rows(arrays,folds[fold]);name='candidate_type%d_K%d_fold%d'%(typ,k,fold)
                    h,receipt=fit(matrix,k,seed,out/(name+'_raw.npz'));c['fits'].append(receipt)
                    if h is None:c['valid']=False;continue
                    np.savez_compressed(out/(name+'.npz'),basis=h);score=masked_score(arrays,folds[1-fold],h,sides);c['scores'].append(score)
                    if old_c['fits'][fold]['converged']:
                        previous=np.load(source/(name+'.npz'),allow_pickle=False)['basis'];basis_error=float(np.max(np.abs(h-previous)));score_error=max(abs(x['weighted_mse']-y['weighted_mse']) for x,y in zip(score,old_c['scores'][old_score]));old_score+=1
                        comparison.append(dict(type=typ,K=k,fold=fold,basis_max_abs=basis_error,score_max_abs=score_error,basis_exact=bool(np.array_equal(h,previous))))
                        if basis_error>1e-14 or score_error>1e-14:raise AssertionError('unexpected old/new converged fit mismatch')
                c['mean_mse']=float(np.mean([s['weighted_mse'] for scores in c['scores'] for s in scores])) if c['valid'] and len(c['scores'])==2 else None
                row['candidates'].append(c);valid &= c['valid'];result['old_converged_comparison']=comparison;save(out/'selection.json',result);print(a.dataset,typ,k,c['valid'],[f['iterations'] for f in c['fits']],flush=True)
        if not valid:result.update(status='blocked_incomplete_candidate_comparison',reason='mandatory candidate failed; no selection or final fit')
        else:
            matrices=[];offsets=[0];ks=[]
            for typ_string,split in splits['node_folds'].items():
                check();typ=int(typ_string);row=result['types'][typ_string];k=choose(row['candidates']);row.update(selected_K=k,boundary_hit=k==row['Kmax']);seed=seed32(2026102861,dataset=a.dataset,type=typ,fold='all',K=k,purpose='final_fit')
                matrix=weighted_rows(arrays,np.flatnonzero(types==typ));h,rec=fit(matrix,k,seed,out/('final_type%d_raw.npz'%typ));row['final_fit']=rec;save(out/'selection.json',result)
                if h is None:raise RuntimeError('final fit failed; no final dictionary released')
                matrices.append(h);ks.append(k);offsets.append(offsets[-1]+k)
            np.savez_compressed(out/'final_basis.npz',basis=np.concatenate(matrices),offsets=np.array(offsets),K=np.array(ks))
            result.update(status='passed',K=ks,offsets=offsets,basis_sha256=sha(out/'final_basis.npz'),panel_sha256=sha(out/'panel.json'),splits_sha256=sha(out/'frozen_splits.json'))
    except BaseException as exc:result.update(status='failed_or_incomplete',error=repr(exc),traceback=traceback.format_exc())
    finally:
        result.update(seconds=time.time()-started,finished_epoch=time.time(),peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024);save(out/'selection.json',result)
    return 0 if result['status']=='passed' else 2
if __name__=='__main__':raise SystemExit(main())
