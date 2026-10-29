"""CPU-only complete/incomplete stage500 optimization panel, no graph metrics."""
import argparse,pathlib,time,json,csv,statistics
from runtime_contract import ROOT,jobs,read,save,sha,PROTOCOL

def maybe(path):
    return read(path) if path.exists() else None
def relation_names(dataset):
    names=read(ROOT/'package/config.json')['datasets'][dataset]['type_names']
    return [names[i][0]+names[j][0] for i in range(len(names)) for j in range(i,len(names))]
def write_csv(path,rows):
    if not rows:return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def build(run,deadline):
    cells=[];validation=[];family=[];gradients=[];curves=[];missing=[];pairing=[]
    for job in jobs()['training']:
        if time.time()>=deadline:raise TimeoutError('summary deadline')
        path=run/job['directory'];status=maybe(path/'summary.json') or maybe(path/'progress.json')
        initial=maybe(path/'binding.json');values=maybe(path/'validation.json') or []
        log=[]
        if (path/'training.jsonl').exists():
            for line in (path/'training.jsonl').read_text().splitlines():
                try:log.append(json.loads(line))
                except json.JSONDecodeError:missing.append(dict(task=job['id'],reason='truncated training row'))
        ident={k:job[k] for k in ['dataset','train_seed','arm']}
        cell=dict(**ident,id=job['id'],status=status or {'status':'not_started'},completed_updates=len(log),
                  initial_tensor_sha256=initial['initial']['tensor_sha256'] if initial else None,
                  reset_rng=initial['initial']['rng_reset'] if initial else None,
                  input_hashes={str(r['update']):r['input_sha256'] for r in log},
                  checkpoint_files={str(u):maybe(path/('initial.sha256.json' if u==0 else f'checkpoint_u{u}.sha256.json')) for u in PROTOCOL['checkpoints']},
                  update_seconds_median=statistics.median([r['total_update_seconds'] for r in log]) if log else None)
        cells.append(cell)
        if not status or status.get('status')!='complete' or len(log)!=500:missing.append(dict(task=job['id'],reason='not complete at500',status=status))
        for r in log:
            curves.append(dict(**ident,update=r['update'],ce=r['ce'],auxiliary=r['auxiliary'],weighted_auxiliary=r['weighted_auxiliary'],loss=r['loss'],gradient_norm=r['gradient_norm'],seconds=r['total_update_seconds'],t0=r['t'][0],t1=r['t'][1]))
            if r.get('gradient_observation'):
                for group,st in {'ALL':r['gradient_observation']['all_parameters'],**r['gradient_observation']['disjoint_groups']}.items():
                    gradients.append(dict(**ident,update=r['update'],group=group,**st))
        names=relation_names(job['dataset'])
        seen=set()
        for v in values:
            for kind in ['raw','ema_0.99']:
                for row in v.get(kind,{}).get('rows',[]):
                    key=(v['updates'],kind,row['t']);seen.add(key)
                    sums=sum(r['expected_edges'] for r in row['family']);target=sum(r['true_edges'] for r in row['family'])
                    validation.append(dict(**ident,update=v['updates'],readout=kind,t=row['t'],input_sha256=row['input_sha256'],
                                           ce=row['ce_sum_over_S'],auxiliary=row['auxiliary'],weighted_auxiliary=row['weighted_auxiliary'],
                                           shadow_weighted_auxiliary=row.get('shadow_weighted_auxiliary'),expected_edges=sums,target_edges=target,
                                           signed_error=sums-target,absolute_error=abs(sums-target),
                                           sum_relation_absolute_error=sum(abs(r['signed_count_error']) for r in row['family'])))
                    counts=row.get('counts',{});den=counts.get('relation_denominator',0)
                    for rel in row['family']:
                        f=rel['family'];rho=counts.get('normalized_penalty',[None]*len(names))[f];incl=counts.get('included',[False]*len(names))[f]
                        shadow=.001*rho/den if rho is not None and incl and den else 0.
                        family.append(dict(**ident,update=v['updates'],readout=kind,t=row['t'],relation=names[f],**rel,
                                           absolute_count_error=abs(rel['signed_count_error']),penalty=rho,
                                           shadow_aux_contribution=shadow,actual_aux_contribution=shadow if job['arm']=='CE_COUNT' else 0.,
                                           input_sha256=row['input_sha256']))
        for u in PROTOCOL['checkpoints']:
            for kind in ['raw','ema_0.99']:
                for t in PROTOCOL['validation_times']:
                    if (u,kind,t) not in seen:missing.append(dict(task=job['id'],reason='missing validation',update=u,readout=kind,t=t))
        for u in PROTOCOL['gradient_observation_updates']:
            if not (path/f'gradient_u{u}.json').exists():missing.append(dict(task=job['id'],reason='missing gradient observation',update=u))
    for ds,seeds in PROTOCOL['train_seeds'].items():
        for seed in seeds:
            a=next(c for c in cells if (c['dataset'],c['train_seed'],c['arm'])==(ds,seed,'CE'))
            b=next(c for c in cells if (c['dataset'],c['train_seed'],c['arm'])==(ds,seed,'CE_COUNT'))
            pairing.append(dict(dataset=ds,train_seed=seed,
                initial_exact=a['initial_tensor_sha256']==b['initial_tensor_sha256'] if a['initial_tensor_sha256'] and b['initial_tensor_sha256'] else None,
                reset_rng_exact=a['reset_rng']==b['reset_rng'] if a['reset_rng'] and b['reset_rng'] else None,
                input_pairing_complete=a['input_hashes']==b['input_hashes'] and len(a['input_hashes'])==500,
                common_inputs_matching=all(a['input_hashes'][k]==b['input_hashes'][k] for k in a['input_hashes'].keys()&b['input_hashes'].keys()),
                matched_updates=len(a['input_hashes'].keys()&b['input_hashes'].keys())))
    pairs=paired_rows(validation,['ce','absolute_error','sum_relation_absolute_error','expected_edges'],False)
    family_pairs=paired_rows(family,['absolute_count_error','expected_edges','positive_ce','negative_ce'],True)
    return dict(status='complete' if not missing and all(p['initial_exact'] and p['reset_rng_exact'] and p['input_pairing_complete'] for p in pairing) else 'incomplete',
                scope='optimization only; generated graphs=0; no generation-quality inference',
                cells=cells,missing=missing,pairing=pairing,validation=validation,family=family,gradients=gradients,training_curve=curves,
                paired_validation=pairs,paired_family=family_pairs)
def paired_rows(rows,metrics,rel):
    keys=['dataset','train_seed','update','readout','t']+(['relation'] if rel else [])
    lookup={tuple(r[k] for k in keys)+(r['arm'],):r for r in rows}
    out=[]
    for ds,seeds in PROTOCOL['train_seeds'].items():
        for seed in seeds:
            for u in PROTOCOL['checkpoints']:
                for kind in ['raw','ema_0.99']:
                    for t in PROTOCOL['validation_times']:
                        for relation in (relation_names(ds) if rel else [None]):
                            key=(ds,seed,u,kind,t)+((relation,) if rel else ())
                            a=lookup.get(key+('CE',));b=lookup.get(key+('CE_COUNT',))
                            same=bool(a and b and a['input_sha256']==b['input_sha256'])
                            row=dict(zip(keys,key));row.update(comparison='CE_COUNT minus CE',same_input=same,reason=None if same else 'missing or mismatched input')
                            for metric in metrics:
                                av=a.get(metric) if a else None;bv=b.get(metric) if b else None
                                row[metric+'_CE']=av;row[metric+'_COUNT']=bv
                                row[metric+'_difference']=bv-av if same and av is not None and bv is not None else None
                            out.append(row)
    return out
def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--output',required=True);p.add_argument('--deadline',type=float,required=True);a=p.parse_args()
    run=pathlib.Path(a.run);r=build(run,a.deadline);save(a.output,r)
    for key in ['validation','family','gradients','training_curve','paired_validation','paired_family']:
        write_csv(run/(key+'.csv'),r[key])
    return 0 if r['status']=='complete' else 2
if __name__=='__main__':raise SystemExit(main())
