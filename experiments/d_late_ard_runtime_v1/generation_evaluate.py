"""Full48logical-graph evaluation; original/LCC and missing-aware nesting.

Uses the inherited metric implementations unchanged. No model or Torch loading.
Every metric is computed per graph before sampling/training-seed aggregation.
"""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse
import csv
import itertools
import json
import math
import resource
import statistics
import time
import traceback
from pathlib import Path
from common_run import ROOT, SCI, OUT, TASKOUT, read, save, sha, now
import numpy as np
from lcc_eval import metrics as M
import evaluation_fields as E
from threadpoolctl import threadpool_info

ROUTES=('baseline','A','D','ard')
VERSIONS=('original','lcc')


def valid(value):
    return value is not None and isinstance(value,(int,float,np.integer,np.floating)) and math.isfinite(value)


def avg(values):
    return statistics.mean(values) if values and all(valid(v) for v in values) else None


def sd(values):
    return statistics.stdev(values) if len(values)>1 and all(valid(v) for v in values) else None


def csvwrite(path, rows):
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('x',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in row.items()})


def panel_shape(tasks):
    datasets=sorted({j['dataset'] for j in tasks})
    if datasets!=['PubMed']:
        raise ValueError('frozen generation panel is PubMed only')
    seeds=sorted({j['train_seed'] for j in tasks})
    samples=list(tasks[0]['sample_seeds']) if tasks else []
    actual=[(j['dataset'],j['train_seed'],j['method']) for j in tasks]
    expected={('PubMed',seed,method) for seed in seeds for method in ROUTES}
    if len(seeds)!=3 or len(samples)!=4 or len(set(samples))!=4 or len(actual)!=12 or len(set(actual))!=12 or set(actual)!=expected:
        raise ValueError('expected4routes x3trainingseeds x4samplingseeds')
    for job in tasks:
        if (job['sample_seeds']!=samples or job['updates']!=3000 or job['EMA_decay']!=.99
                or job['sampling_T']!=1000 or job['query_R']!=1
                or job['arm'] not in ('A','D','E') or job['configuration']!=job['arm']+'_late'):
            raise ValueError('mismatched model panel configuration')
    return seeds,samples


def aggregate(rows, fields, seeds, samples):
    """No available-only means or route/baseline intersection restriction."""
    lookup={(r['dataset'],r['train_seed'],r['method'],r['sample_seed'],r['version'],r['kind'],r['metric']):r for r in rows}
    means,pairs,across=[],[],[]
    for dataset in sorted(fields):
        for version in VERSIONS:
            for kind,metric in sorted(fields[dataset]):
                by_seed={}
                for seed in seeds:
                    for method in ROUTES:
                        records=[lookup.get((dataset,seed,method,s,version,kind,metric),{}) for s in samples]
                        values=[r.get('value') for r in records]
                        value=avg(values)
                        by_seed[seed,method]=value
                        reasons=sorted({r.get('reason') or 'undefined_metric' for r in records if not valid(r.get('value'))})
                        means.append(dict(dataset=dataset,train_seed=seed,method=method,version=version,
                            kind=kind,metric=metric,value=value,sample_std=sd(values),
                            sample_values=values,valid_samples=sum(valid(v) for v in values),
                            expected_samples=len(samples),reason=';'.join(reasons) if reasons else None))
                    for method in ROUTES[1:]:
                        for sample in samples+['sample_mean']:
                            if sample=='sample_mean':
                                left,right=by_seed[seed,method],by_seed[seed,'baseline']
                            else:
                                left=lookup.get((dataset,seed,method,sample,version,kind,metric),{}).get('value')
                                right=lookup.get((dataset,seed,'baseline',sample,version,kind,metric),{}).get('value')
                            pairs.append(dict(dataset=dataset,train_seed=seed,sample_seed=sample,
                                comparison=method+'-baseline',method=method,version=version,kind=kind,metric=metric,
                                left=left,right=right,difference=left-right if valid(left) and valid(right) else None,
                                reason=None if valid(left) and valid(right) else 'one_or_both_predeclared_sides_unavailable'))
                for method in ROUTES:
                    values=[by_seed[seed,method] for seed in seeds]
                    across.append(dict(dataset=dataset,method=method,version=version,kind=kind,metric=metric,
                        training_seeds=seeds,training_seed_means=values,mean=avg(values),
                        training_seed_std=sd(values),valid_training_seeds=sum(valid(v) for v in values),
                        expected_training_seeds=len(seeds),
                        reason=None if all(valid(v) for v in values) else 'incomplete_or_undefined_training_seed_mean'))
                for method in ROUTES[1:]:
                    differences=[by_seed[seed,method]-by_seed[seed,'baseline']
                        if valid(by_seed[seed,method]) and valid(by_seed[seed,'baseline']) else None for seed in seeds]
                    across.append(dict(dataset=dataset,method=method+'-baseline',version=version,kind=kind,metric=metric,
                        training_seeds=seeds,training_seed_means=differences,mean=avg(differences),
                        training_seed_std=sd(differences),valid_training_seeds=sum(valid(v) for v in differences),
                        expected_training_seeds=len(seeds),
                        reason=None if all(valid(v) for v in differences) else 'incomplete_predeclared_pair'))
    return means,pairs,across


def detail_leaves(value, prefix=''):
    """Literal inherited statistic fields, not new metric definitions."""
    if isinstance(value,dict):
        for key,item in value.items():
            yield from detail_leaves(item,prefix+'/'+str(key))
    elif isinstance(value,list):
        for index,item in enumerate(value):
            yield from detail_leaves(item,prefix+'/'+str(index))
    else:
        yield prefix,value


def queue_reasons():
    path=OUT/'queue_receipt.json'
    if not path.exists():return {}
    q=read(path);reasons={r['task_id']:r.get('reason','not_started') for r in q.get('unstarted',[])}
    reasons.update({task:'queue_interrupted' for task in q.get('interrupted',[])})
    return reasons


def model_audit(job,dictionary,input_contracts):
    """Read checkpoint hashes and complete input logs, without loading tensors."""
    old=job.get('reuse_completed_baseline') is True
    folder=Path(job['folder'])
    binding=read(folder/'binding.json')
    if (binding['dataset']!=job['dataset'] or binding['train_seed']!=job['train_seed']
            or binding['configuration']!=job['configuration'] or binding['config']!=job['config']
            or binding['updates']!=3000 or binding['inputs_per_update']!=1):
        raise ValueError('training identity/configuration not strictly matched')
    if not old and (binding.get('method')!=job['method'] or binding.get('dictionary_record')!=dictionary):
        raise ValueError('new training dictionary route binding mismatch')
    artifact=Path(dictionary.get('artifact_absolute',dictionary.get('artifact',''))) if dictionary.get('artifact') else None
    if artifact is not None and sha(artifact)!=dictionary['artifact_sha256']:
        raise ValueError('frozen role dictionary changed')
    key=job['dataset']+'_'+str(job['train_seed']);contract_record=input_contracts[key]
    cp=Path(contract_record['file'])
    if sha(cp)!=contract_record['sha256']:
        raise ValueError('training input contract hash changed')
    contract=read(cp)
    if len(contract)!=3000 or contract_record['count']!=3000:
        raise ValueError('training input contract incomplete')
    log=folder/'training.jsonl';count=0
    with log.open() as source:
        for count,line in enumerate(source,1):
            if count>3000:raise ValueError('extra training updates')
            row=json.loads(line);expected=contract[count-1]
            if (row['update']!=count or row['input_sha256']!=expected['sha256']
                    or row['t']!=expected['t'] or row.get('paired_input_exact') is not True):
                raise ValueError('training update/input mismatch: '+str(count))
    if count!=3000:raise ValueError('training log does not contain3000updates')
    weight=folder/'readout_ema_0.99.pt';weight_sha=sha(weight)
    if read(weight.with_suffix('.sha256.json'))['sha256']!=weight_sha:
        raise ValueError('EMA readout sidecar mismatch')
    return dict(status='complete',legacy_baseline_reuse=old,model_folder=str(folder),
        weight_file=str(weight),weight_sha256=weight_sha,dictionary=dictionary,
        dictionary_artifact_sha256=sha(artifact) if artifact else None,input_contract=contract_record,
        input_contract_sha256=sha(cp),training_updates=count,training_log_sha256=sha(log),
        model_binding_sha256=sha(folder/'binding.json'),tensor_payload_loaded=False)


def graph_audit(job,sample,reused,model,contracts):
    key=(job['dataset'],job['train_seed'],job['method'],sample)
    source=reused.get(key)
    if source is not None:
        if source.get('strict_match') is not True:raise ValueError('reuse not authenticated')
        graph=Path(source['graph']);summary_path=Path(source['summary'])
        if sha(graph)!=source['graph_sha256'] or sha(summary_path)!=source['summary_sha256']:
            raise ValueError('reused graph/summary changed')
    else:
        folder=Path(job['folder'])/'ema_0.99'/f'sample_{sample}'
        graph=folder/'graph.npz';summary_path=folder/'summary.json'
    summary=read(summary_path)
    expected=dict(dataset=job['dataset'],train_seed=job['train_seed'],sample_seed=sample,
        updates=3000,readout='EMA0.99',configuration=job['configuration'],T=1000,R=1,status='complete',steps=1000)
    if any(summary.get(k)!=v for k,v in expected.items()):
        raise ValueError('graph sampling identity, length or readout mismatch')
    if summary.get('canonical_all1000_plans_and_draws') is not True:
        raise ValueError('canonical sampling receipt absent')
    if sha(graph)!=summary['graph_sha256'] or model['weight_sha256']!=summary['model_sha256']:
        raise ValueError('sample graph/weight hash binding mismatch')
    contract=contracts[job['dataset']+'_'+str(sample)]
    if summary['initial_state_sha256']!=contract['initial_state_sha256']:
        raise ValueError('initial-noise pairing differs')
    proof='archived_summary_and_frozen_reuse_manifest'
    if source is None:
        expected_steps={r['t']:r for r in contract['steps']};count=0
        with (graph.parent/'steps.jsonl').open() as source_file:
            for count,line in enumerate(source_file,1):
                row=json.loads(line);t=1001-count
                if t not in expected_steps:raise ValueError('extra sampling steps')
                target=expected_steps[t]
                if (row['time']!=t or row.get('canonical_pairing') is not True
                        or row.get('exact_coverage') is not True
                        or row['uniform_sha256']!=target['uniform_sha256']
                        or row['query_plan_sha256']!=target['query_plan_sha256']):
                    raise ValueError('sampling plan/draw/coverage mismatch at'+str(t))
        if count!=1000:raise ValueError('partial sampling step log')
        proof='all1000_new_step_plans_draws_and_coverage_rechecked'
    return graph,summary,dict(graph_file=str(graph),graph_sha256=sha(graph),
        summary_file=str(summary_path),summary_sha256=sha(summary_path),
        weight_sha256=model['weight_sha256'],dictionary_artifact_sha256=model['dictionary_artifact_sha256'],
        input_contract_sha256=model['input_contract_sha256'],sampling_pairing_proof=proof,
        initial_state_sha256=summary['initial_state_sha256'],reused=source is not None)


def main(output=None):
    started=time.time();dest=Path(output) if output else OUT/'evaluation'
    dest.mkdir(parents=True,exist_ok=False)
    ledger=read(TASKOUT/'ledger.json');approval=read(OUT/'approval.json')
    for path,expected in approval['source_files'].items():
        if sha(path)!=expected:raise ValueError('frozen generation source changed: '+path)
    reference_binding=read(TASKOUT/'reviews/reference_data_binding.json')
    for item in reference_binding['files']:
        if sha(item['file'])!=item['legacy_approval_sha256']:
            raise ValueError('reference data/config differs from archived approval: '+item['file'])
    def check():
        if time.time()>=ledger['final_deadline_epoch']-90:
            raise TimeoutError('final delivery reserve')
    tasks=read(OUT/'tasks.json');seeds,samples=panel_shape(tasks)
    if Path(M.__file__).resolve()!=(SCI/'lcc_eval/metrics.py').resolve() or Path(E.__file__).resolve()!=(SCI/'evaluation_fields.py').resolve():
        raise ValueError('metric modules resolved outside inherited frozen science')
    dictionaries=read(OUT/'dictionaries.json');inputs=read(OUT/'input_contracts.json')
    contracts=read(OUT/'prepared/sampling_contracts.json')
    reused={(r['dataset'],r['train_seed'],r['method'],r['sample_seed']):r for r in read(OUT/'reused_samples.json')}
    reasons=queue_reasons()
    F=M.frozen(SCI/'lcc_eval/vendor/frozen_metrics.py',SCI/'lcc_eval/vendor/aligned_evaluate.py',SCI/'lcc_eval/vendor/hgen_utils.py')
    metric_sources=[Path(__file__),SCI/'lcc_eval/metrics.py',SCI/'evaluation_fields.py',
        SCI/'lcc_eval/vendor/frozen_metrics.py',SCI/'lcc_eval/vendor/aligned_evaluate.py',SCI/'lcc_eval/vendor/hgen_utils.py']
    save(dest/'evaluation_binding.json',dict(at=now(),source_files={str(p):sha(p) for p in metric_sources},
        approval_sha256=sha(OUT/'approval.json'),tasks_sha256=sha(OUT/'tasks.json'),
        dictionaries_sha256=sha(OUT/'dictionaries.json'),input_contracts_sha256=sha(OUT/'input_contracts.json'),
        sampling_contracts_sha256=sha(OUT/'prepared/sampling_contracts.json'),
        reused_samples_sha256=sha(OUT/'reused_samples.json'),quality_gate='none; technically valid frozen methods reported'))
    z=np.load(SCI/'package/assets/PubMed.npz',allow_pickle=False)
    ref=M.make_graph(z['types'],z['edges'].T,z['schema'])
    names=read(SCI/'package/config.json')['datasets']['PubMed']['type_names']
    rs=M.stats(ref,F,names,check);rc=M.compare(ref,rs,ref,rs,F,names);reference,reference_dist=E.scalar(rs,rs,rc,names)
    fields={'PubMed':{(kind,metric) for kind,values in [('raw',reference),('absolute_error',reference),('distance',reference_dist)] for metric in values}}
    save(dest/'reference_PubMed.json',dict(statistics=rs,comparison=rc,raw=reference,distance=reference_dist))
    rows,inventory,errors,model_receipts,diversity=[],[],[],[],[]
    graphs={};detail_file=(dest/'structural_detail.csv').open('x',newline='')
    detail_writer=csv.DictWriter(detail_file,fieldnames=['dataset','train_seed','method','sample_seed','version','section','path','value']);detail_writer.writeheader()
    try:
        for job in tasks:
            method=job['method'];model=None;model_error=None;model_error_kind=None
            try:model=model_audit(job,dictionaries[method],inputs)
            except Exception as exc:
                model_error=repr(exc)
                model_error_kind='missing' if isinstance(exc,FileNotFoundError) else 'failed'
            model_receipts.append(dict(task_id=job['task_id'],method=method,train_seed=job['train_seed'],
                status='complete' if model else model_error_kind,error=model_error,receipt=model,
                dictionary_record=dictionaries.get(method),
                input_contract=inputs.get('PubMed_'+str(job['train_seed']))))
            for sample in samples:
                identity=dict(dataset='PubMed',train_seed=job['train_seed'],method=method,arm=job['arm'],
                    sample_seed=sample,updates=3000,readout='EMA0.99',task_id=job['task_id'])
                reused_source=reused.get(('PubMed',job['train_seed'],method,sample))
                expected_graph=(Path(reused_source['graph']) if reused_source else
                    Path(job['folder'])/'ema_0.99'/f'sample_{sample}'/'graph.npz')
                inv=dict(identity,status='missing',reason=None,graph_file=str(expected_graph),versions_complete=[])
                try:
                    check()
                    if model is None:
                        if model_error_kind=='failed':
                            raise ValueError('model identity/integrity audit failed: '+model_error)
                        raise FileNotFoundError(reasons.get(job['task_id'],model_error or 'model_unavailable'))
                    path,binding,proof=graph_audit(job,sample,reused,model,contracts)
                    with np.load(path,allow_pickle=False) as zz:
                        graph=M.make_graph(zz['types'],zz['edges'],zz['schema'])
                    lcc,mapping=M.lcc(graph);folder=dest/job['task_id']/str(sample);folder.mkdir(parents=True)
                    if not set(map(tuple,lcc['original_ids'][lcc['edges']].T))<=set(map(tuple,graph['edges'].T)):
                        raise ValueError('LCC introduced an edge')
                    np.savez_compressed(folder/'lcc.npz',types=lcc['types'],edges=lcc['edges'],schema=lcc['schema'],original_ids=lcc['original_ids'])
                    save(folder/'mapping.json',dict(source=str(path),source_sha256=sha(path),mapping=mapping,
                        node_retention=len(lcc['types'])/len(graph['types']),
                        edge_retention=lcc['edges'].shape[1]/graph['edges'].shape[1] if graph['edges'].shape[1] else None,
                        type_original=np.bincount(graph['types'],minlength=4).tolist(),
                        type_retained=np.bincount(lcc['types'],minlength=4).tolist()))
                    for version,g in [('original',graph),('lcc',lcc)]:
                        check();stats=M.stats(g,F,names,check);comp=M.compare(g,stats,ref,rs,F,names)
                        raw,dist=E.scalar(stats,rs,comp,names)
                        ae={k:abs(v-reference[k]) if valid(v) and valid(reference.get(k)) else None for k,v in raw.items()}
                        if dist['main_incident4_W1'] is not None and not math.isclose(dist['all10_compat_W1'],.4*dist['main_incident4_W1'],abs_tol=1e-10,rel_tol=1e-10):
                            raise ValueError('W1 convention changed')
                        save(folder/(version+'.json'),dict(identity,version=version,statistics=stats,
                            comparison=comp,raw=raw,distance=dist,absolute_error=ae,reference=reference,
                            graph_binding=binding,identity_audit=proof,dictionary_record=dictionaries[method]))
                        graphs[('PubMed',job['train_seed'],method,sample,version)]=(g,stats)
                        for kind,values in [('raw',raw),('absolute_error',ae),('distance',dist)]:
                            for metric,value in values.items():
                                fields['PubMed'].add((kind,metric))
                                rows.append(dict(identity,version=version,kind=kind,metric=metric,value=value,
                                    reason=None if valid(value) else 'undefined_metric; fullJSON retains source reason'))
                        for section,value in [('statistics',stats),('comparison',comp)]:
                            for metric_path,value in detail_leaves(value):
                                detail_writer.writerow(dict(dataset='PubMed',train_seed=job['train_seed'],method=method,
                                    sample_seed=sample,version=version,section=section,path=metric_path,value=value))
                        inv['versions_complete'].append(version)
                    inv.update(status='complete',reason=None,**proof)
                except Exception as exc:
                    reason=reasons.get(job['task_id'],repr(exc))
                    inv.update(status='missing' if isinstance(exc,FileNotFoundError) else 'failed',reason=reason)
                    errors.append(dict(identity,error=repr(exc),reason=reason,traceback=traceback.format_exc()))
                inventory.append(inv);save(dest/'inventory.json',inventory);save(dest/'errors.json',errors)
    finally:
        detail_file.close()
    lookup={(r['dataset'],r['train_seed'],r['method'],r['sample_seed'],r['version'],r['kind'],r['metric']):r for r in rows}
    for inv in inventory:
        for version in VERSIONS:
            for kind,metric in sorted(fields['PubMed']):
                key=(inv['dataset'],inv['train_seed'],inv['method'],inv['sample_seed'],version,kind,metric)
                if key not in lookup:
                    row={k:inv[k] for k in ['dataset','train_seed','method','arm','sample_seed','updates','readout','task_id']}
                    row.update(version=version,kind=kind,metric=metric,value=None,
                        reason=inv.get('reason') or 'metric_unavailable')
                    rows.append(row);lookup[key]=row
    means,pairs,across=aggregate(rows,fields,seeds,samples)
    for job in tasks:
        for version in VERSIONS:
            for sa,sb in itertools.combinations(samples,2):
                a=('PubMed',job['train_seed'],job['method'],sa,version)
                b=('PubMed',job['train_seed'],job['method'],sb,version)
                result=dict(dataset='PubMed',train_seed=job['train_seed'],method=job['method'],version=version,
                    sample_a=sa,sample_b=sb,status='unavailable',metrics=None,
                    reason='one_or_both_predeclared_samples_unavailable',
                    limitation='six dependent pairs from four samples of one model; not independent training repeats')
                if a in graphs and b in graphs:
                    try:
                        check();(ga,sta),(gb,stb)=graphs[a],graphs[b]
                        comp=M.compare(ga,sta,gb,stb,F,names)
                        ea=set(map(tuple,ga['original_ids'][ga['edges']].T));eb=set(map(tuple,gb['original_ids'][gb['edges']].T))
                        result.update(status='complete',reason=None,metrics=dict(comparison=comp,
                            labeled_edge_Jaccard=len(ea&eb)/len(ea|eb) if ea|eb else None,
                            labeled_edge_difference=len(ea^eb),
                            exact_labeled_duplicate=ea==eb and set(ga['original_ids'])==set(gb['original_ids']),
                            degree_W1=comp['total_degree_W1'],triangle_composition_TV=comp['triangle_composition_TV'],
                            MP2_composition_TV=comp['MP_composition_TV'][1],MP3_composition_TV=comp['MP_composition_TV'][2]))
                    except Exception as exc:
                        result.update(status='failed',reason=repr(exc))
                diversity.append(result)
    for name,values in [('per_graph_all_metrics',rows),('per_training_seed',means),
                        ('paired_route_minus_baseline',pairs),('across_training_seeds',across),
                        ('sample_diversity',diversity),('inventory',inventory)]:
        save(dest/(name+'.json'),values)
        csvwrite(dest/(name+'.csv'),values)
    save(dest/'model_identity_audit.json',model_receipts)
    core={('absolute_error','edges'),('absolute_error','triangles'),('distance','main_incident4_W1'),
        ('distance','main_joint_degree_TV'),('distance','triangle_composition_TV'),
        ('distance','MP2_composition_TV'),('distance','MP3_composition_TV'),
        ('raw','lcc'),('raw','isolates'),('raw','type_Chemical_low_le1_fraction'),
        ('raw','type_Chemical_maximum'),('raw','type_Chemical_second')}
    decision=[r for r in means if r['version']=='original' and (r['kind'],r['metric']) in core]
    save(dest/'decision_table.json',decision);csvwrite(dest/'decision_table.csv',decision)
    save(dest/'receipt.json',dict(at=now(),expected_models=12,expected_original_graphs=48,
        complete_original_graphs=sum('original' in r['versions_complete'] for r in inventory),
        complete_LCC_graphs=sum('lcc' in r['versions_complete'] for r in inventory),
        missing_graph_records=sum(r['status']=='missing' for r in inventory),
        failed_graph_records=sum(r['status']=='failed' for r in inventory),
        budget_unstarted_graph_records=sum(r.get('reason')=='full3seedARDarm_exceeds_measured_budget_with_reserve' for r in inventory),
        routes=list(ROUTES),training_seeds=seeds,sample_seeds=samples,
        expected_sample_pairs=144,actual_pair_rows=len(diversity),
        all_metrics_exported=True,errors=errors,no_available_only_or_intersection_means=True,
        graph_error_before_aggregation=True,original_primary_LCC_separate=True,
        sampling_pairs_are_independent_training_repeats=False,seconds=time.time()-started,
        max_rss_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        CPU_user_seconds=resource.getrusage(resource.RUSAGE_SELF).ru_utime,
        CPU_system_seconds=resource.getrusage(resource.RUSAGE_SELF).ru_stime,
        threadpools=threadpool_info(),GPU=0,torch_imported='torch' in __import__('sys').modules))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    main(args.output)
