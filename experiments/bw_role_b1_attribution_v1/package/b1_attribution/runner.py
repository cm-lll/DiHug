"""Strict B1-only read/replay/attribution; imports no neural model entry."""
import json
from pathlib import Path
import resource
import time
import numpy as np
from multiscale_role.runtime_guard import Budget,atomic_json,checked_sha
from multiscale_role.short import POLICY,read_pinned,resolve,validate_record,verify_original_q,classify
from multiscale_role.legacy import atomic_npz,extract_frequency,baseline_alignment
from multiscale_role.evaluation import validate_spec,evaluate,ForwardPMFCache
from b1_attribution.math import (TOL,check_close,posterior_chain,components,sum_certificate,replay,
                                alignment,independent_cf_gradient,first_order,genre_laws)


def old_case_binding(summary,old):
    if summary.get('status')!='complete' or summary.get('completed_records')!=6 or summary.get('input_hash_verification_complete') is not True or summary.get('input_hashes_unchanged') is not True:
        raise ValueError('original short run is not a verified completed panel')
    if (old.get('arm'),old.get('t'),old.get('index'))!=('B',1,5):raise ValueError('requires ORIGINAL case_05 B,t=1 only')
    if len(summary.get('records',[]))!=6 or old!=summary['records'][5]:raise ValueError('case_05 differs from pinned old summary record')


def compare_metrics(current,old,path='metrics'):
    rows=[]
    if isinstance(old,dict):
        if set(current)!=set(old):raise ValueError(path+' metric keys differ')
        for k in old:rows+=compare_metrics(current[k],old[k],path+'.'+k)
    elif isinstance(old,list):
        if len(current)!=len(old):raise ValueError(path+' metric lengths differ')
        for i,(a,b) in enumerate(zip(current,old)):rows+=compare_metrics(a,b,path+'['+str(i)+']')
    elif isinstance(old,(float,int)) and not isinstance(old,bool):
        if isinstance(old,int):
            if current!=old:raise ValueError(path+' integer metric differs')
        else:
            atol=TOL['metric_W1_atol'] if any(k in path for k in ('W1','degree','edges')) else TOL['metric_CF_atol']
            rows.append(dict(metric=path,**check_close(current,old,atol,TOL['metric_rtol'],path)))
    elif current!=old:raise ValueError(path+' definition differs')
    return rows


def run(args,out,deadline,cleanup_deadline):
    out=Path(out);budget=Budget(deadline,out/'STOP',cleanup_deadline);budget.install();check=budget.check
    started,cpu=time.monotonic(),time.process_time();hashes={};timings={}
    summary=dict(status='incomplete',scope='B1 single recorded step only',real_B1_executed=False,
                 model_calls=0,training_updates=0,scientific_graphs=0,GPU_use=0,new_candidate_searches=0,
                 input_hash_verification_complete=False,tolerances=TOL)
    def stage(name):
        check();print(json.dumps(dict(stage=name,event='start')),flush=True);return time.monotonic()
    def done(name,tick):
        check();timings[name]=time.monotonic()-tick;print(json.dumps(dict(stage=name,event='done',seconds=timings[name])),flush=True)
    def pin(path,expected=None):
        path=Path(path).resolve();h=checked_sha(path,check)
        if expected is not None and h!=expected:raise ValueError('input binding differs: '+str(path))
        hashes[str(path)]=h;return h
    def write(name,value):atomic_json(out/name,value,check)
    try:
        tick=stage('input_binding');short=Path(args.short_run).resolve();export=Path(args.export).resolve()
        pin(short/'summary.json',args.summary_sha256);pin(short/'case_05.json',args.case_sha256)
        old_summary=read_pinned(short/'summary.json',args.summary_sha256,check);old=read_pinned(short/'case_05.json',args.case_sha256,check)
        old_case_binding(old_summary,old)
        # Expected input hashes come from the PINNED ORIGINAL run receipt,
        # not freshly computed and promoted to a source certificate.
        expected={}
        for path,h in old_summary['input_hashes'].items():
            canonical=str(Path(path).resolve())
            if canonical in expected and expected[canonical]!=h:raise ValueError('conflicting original symlink/path receipts')
            expected[canonical]=h
        def original_pin(path,extra=None):
            path=Path(path).resolve()
            if str(path) not in expected:raise ValueError('input not recorded in original short summary: '+str(path))
            if extra is not None and expected[str(path)]!=extra:raise ValueError('original receipt/producer hash disagreement')
            return pin(path,expected[str(path)])
        manifest_path=export/'six_states.json';spec_path=export/'original_evaluation.json'
        original_pin(manifest_path);original_pin(spec_path)
        manifest=json.loads(manifest_path.read_text());spec=json.loads(spec_path.read_text())
        if manifest.get('format')!='legacy_export_v1_1' or manifest.get('base_commit')!='acbbddb5cdbcbaa2ae129bcd59e434e6a298b130':raise ValueError('original export receipt differs')
        rows=[r for r in manifest['records'] if (r['arm'],r['t'])==('B',1)]
        if len(rows)!=1:raise ValueError('missing/duplicate B1 input')
        row=rows[0];meta_path=resolve(export,row['metadata']);original_pin(meta_path,row['metadata_sha256'])
        meta=json.loads(meta_path.read_text())
        for key in ('npz','checkpoint','source_manifest','config'):original_pin(resolve(export,row[key]),row[key+'_sha256'])
        for key in ('legacy_collection','legacy_report'):original_pin(meta[key],meta[key+'_sha256'])
        # Read the original sealed ledger, and B1 origin only. Do NOT rehash
        # other five probability arrays or rerun their computations.
        from multiscale_role.entry_guard import SEALED_LEDGER_SHA256
        ledgers=[p for p,h in meta['original_input_hashes'].items() if Path(p).name=='evidence_hashes.json']
        if len(ledgers)!=1:raise ValueError('missing/ambiguous sealed diagnostic ledger')
        original_pin(ledgers[0],SEALED_LEDGER_SHA256)
        source_root=Path(ledgers[0]).parent
        for p in (Path(meta['legacy_collection']),Path(meta['legacy_report']),source_root/Path(meta['legacy_collection']).parent.relative_to(source_root)/'baseline.npz'):
            if str(p.resolve()) in meta['original_input_hashes']:original_pin(p,meta['original_input_hashes'][str(p.resolve())])
        from hsgd.bw_preflight import common as c
        from hsgd.kernel import MarginalKernel
        from stepwise_role.core import Layout,transition_kl
        from role_degree.role_statistics import joint_profile
        from multiscale_role.core import freeze,MultiScaleGuide
        summary['environment']=c.environment('cpu')
        reference=c.PACKAGE/'data/IMDB_movie_reference.npz';pin(reference,c.config()['imdb_sha256'])
        case=c.imdb_case(reference,c.config());layout=Layout.canonical(case['space'].types,case['space'].pairs,case['prior'])
        type_file=c.PACKAGE/'data/IMDB_preprocessing_provenance.json';pin(type_file)
        type_names=json.loads(type_file.read_text())['source_meta']['node_types']
        data,meta=validate_record(row,export,case,check,{})
        profile=joint_profile(layout.types,case['clean'],case['space'].encode(case['clean'])[0],layout.pairs)
        guide=MultiScaleGuide(layout,freeze(profile,layout,c.schedule(1000)),check);kernel=MarginalKernel(layout.prior,guide.alpha)
        pin(short/'frozen_objective.json');pin(short/'policy.json')
        if json.loads((short/'frozen_objective.json').read_text())!=guide.frozen or old_summary['objective_sha256']!=guide.frozen['sha256']:
            raise ValueError('original frozen mathematical target differs')
        if json.loads((short/'policy.json').read_text())!=POLICY:raise ValueError('original scientific policy differs')
        if guide.channels['3']!=[dict(families=[8],p=1/51,reference_D90=153,kind='relation')]:raise ValueError('IMDB Genre D90/probe contract differs')
        validate_spec(spec,guide,c.config()['imdb_sha256'])
        origin=resolve(export,spec['original_artifact_source']);original_pin(origin,spec['original_artifact_sha256'])
        omega,path=extract_frequency(json.loads(origin.read_text()),layout)
        if not np.array_equal(omega,spec['omega64']) or path!=spec['frequency_field_path'] or spec['CF_normalizer']!=1:
            raise ValueError('original independent Genre64 bank/normalization differs')
        p,q,bridge=verify_original_q(data,kernel,layout,1)
        check_close(q,p,5e-14,0.,'t1 actual q=p0')
        done('input_binding',tick)
        tick=stage('original_gradient_and_recorded_step')
        summary['real_B1_executed']=True
        nq,loss,total,detail,affine=guide.posterior_gradient(p,data['state'],kernel,1)
        if detail['underflow_node_frequencies']!=old['guidance']['underflow_node_frequencies']:
            raise ValueError('original underflow/controller trust condition differs')
        before_parts,parts=components(q,guide,0,True,posterior_chain(p,data['state'],kernel,guide,1)[1],check)
        certificate,summed=sum_certificate(parts,total)
        check_close(before_parts['total_loss'],loss,TOL['loss_atol'],TOL['loss_rtol'],'weighted component loss sum')
        check_close(loss,old['guidance']['loss_before'],TOL['loss_atol'],TOL['loss_rtol'],'old loss before')
        pp,qq,dz,replay_record=replay(p,data['state'],kernel,guide,1,old['guidance'],total)
        after_parts,_=components(qq,guide,0,False,check=check)
        check_close(after_parts['total_loss'],old['guidance']['loss_after'],TOL['loss_atol'],TOL['loss_rtol'],'old loss after')
        original_kind_proofs=[]
        for label,parts_info in [('before',before_parts),('after',after_parts)]:
            for kind,field in [('CF','raw_CF_global_normalized'),('tail','raw_tail_global_normalized')]:
                value=sum(v[field] for v in parts_info['details'].values())
                original_kind_proofs.append(dict(stage=label,component=kind,**check_close(value,
                    old['guidance']['components_'+label][kind+'_loss'],TOL['loss_atol'],TOL['loss_rtol'],'old '+label+' '+kind+' loss')))
        check_close(qq,pp,5e-14,0.,'t1 adjusted q=p0')
        done('original_gradient_and_recorded_step',tick)
        tick=stage('old_metrics_reproduction')
        cache=ForwardPMFCache();before=evaluate(q,guide,0,spec,check,cache);after=evaluate(qq,guide,0,spec,check,cache)
        proofs=compare_metrics(before,old['before'],'before')+compare_metrics(after,old['after'],'after')
        original_report=json.loads(Path(meta['legacy_report']).read_text());baseline_alignment(before,original_report)
        # Label equality is required; numeric tolerances are the old metric contract.
        if classify(before,after,spec)['label']!=old['decision']['label']:raise ValueError('old per-case decision label differs')
        for key,value in [('clean_probability_l1',float(abs(pp-p).sum())),('clean_probability_max',float(abs(pp-p).max(initial=0))),
                          ('transition_probability_l1',float(abs(qq-q).sum())),('transition_probability_max',float(abs(qq-q).max(initial=0)))]:
            check_close(value,old['guidance'][key],TOL['loss_atol'],TOL['metric_rtol'],'old '+key)
        done('old_metrics_reproduction',tick)
        tick=stage('gradient_alignment_and_Genre64_first_order')
        _,metric_q,metric_proof=independent_cf_gradient(q,guide,0,spec,check)
        metric_g=metric_q*posterior_chain(p,data['state'],kernel,guide,1)[1]
        by_type={c:parts[c+'/CF']+parts[c+'/tail'] for c in guide.omega}
        other=sum((v for c,v in by_type.items() if c!='3'),np.zeros_like(total))
        align=dict(Genre_CF_vs_tail=alignment(parts['3/CF'],parts['3/tail']),
                   Genre_vs_other_types={c:alignment(by_type['3'],v) for c,v in by_type.items() if c!='3'},
                   Genre_vs_all_other_types=alignment(by_type['3'],other),component_pairs={})
        keys=list(parts)
        for i,a in enumerate(keys):
            for b in keys[i:]:check();align['component_pairs'][a+'|'+b]=alignment(parts[a],parts[b])
        attribution=first_order(parts,metric_g,replay_record['accepted_delta'],replay_record['original_gradient_infinity_norm'],
                                after['Genre64']['error']-before['Genre64']['error'],dz)
        done('gradient_alignment_and_Genre64_first_order',tick)
        tick=stage('Genre_full_PMF_CDF_and_W1_intervals')
        laws,law_arrays=genre_laws(q,qq,guide,0,check=check)
        for name,metric in [('before',before),('after',after)]:
            old_w1=metric['expected_degree_PMF_W1']['3'][0]
            check_close(laws['states'][name]['W1'],old_w1['W1'],TOL['metric_W1_atol'],TOL['metric_rtol'],'PMF W1 reproduction')
            check_close(laws['states'][name]['original_tail_probe'],(before_parts if name=='before' else after_parts)['details']['3']['probes'][0]['value'],TOL['PMF_atol'],0.,'PMF/original tail probe')
        check_close(laws['states']['target']['original_tail_probe'],guide.target(0)[1]['3'][0],TOL['PMF_atol'],0.,'target probe')
        done('Genre_full_PMF_CDF_and_W1_intervals',tick)
        tick=stage('save_and_final_integrity')
        atomic_npz(out/'genre_pmf_cdf.npz',law_arrays,check)
        arrays={('gradient_'+k.replace('/','_')):g for k,g in parts.items()}
        arrays.update(original_total_logit_gradient=total,summed_component_logit_gradient=summed,
                      independent_Genre64_logit_gradient=metric_g,original_logit_delta=dz,
                      p0_before=p,p0_after=pp,q_before=q,q_after=qq,canonical_slots=layout.slots,families=layout.families)
        atomic_npz(out/'B1_gradient_evidence.npz',arrays,check)
        write('attribution.json',dict(component_losses_before=before_parts,component_losses_after=after_parts,
            node_type_names=type_names,relation_families=layout.pairs.tolist(),
            component_loss_changes={k:dict(before=v,after=after_parts['losses'][k],change=after_parts['losses'][k]-v)
                                    for k,v in before_parts['losses'].items()},
            gradient_sum=certificate,alignment=align,independent_Genre64=attribution,Genre_laws=laws,
            independent_Genre64_value_check=metric_proof,original_posterior_affine_error=affine,
            original_CF_tail_loss_comparisons=original_kind_proofs,
            metrics_before=before,metrics_after=after,old_metric_comparisons=proofs,replay=replay_record,
            gradient_coordinate='collected clean p0 logit; original Bayes slope and exact-support mask',
            approximation='first-order only; residual is not allocated as causal component effects',
            old_case_sha256=args.case_sha256,old_summary_sha256=args.summary_sha256))
        for path,h in hashes.items():
            if checked_sha(path,check)!=h:raise ValueError('input changed during diagnostic: '+path)
        outputs={n:checked_sha(out/n,check) for n in ('attribution.json','genre_pmf_cdf.npz','B1_gradient_evidence.npz')}
        done('save_and_final_integrity',tick)
        summary.update(status='complete',real_B1_executed=True,original_step_reproduced=True,gradient_sum_verified=True,
                       original_metrics_reproduced=True,output_hashes=outputs,input_hash_verification_complete=True,
                       replay=replay_record,independent_Genre64=attribution,Genre_laws=laws,
                       provenance_boundary='caller pins must come from sealed copies; metadata/hash cross-check is not execution proof',
                       source_scope='only B1 arrays/weight/reference/receipts read; other five scientific arrays NOT rehashed or evaluated')
    except BaseException as exc:
        summary.update(status='incomplete' if isinstance(exc,(TimeoutError,InterruptedError,KeyboardInterrupt)) else 'protocol_failed',
                       reason=type(exc).__name__+': '+str(exc))
    finally:
        summary.update(timings=timings,input_hashes=hashes,worker_wall_seconds=time.monotonic()-started,
                       worker_CPU_seconds=time.process_time()-cpu,peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
        try:atomic_json(out/'summary.json',summary,budget.cleanup_check)
        finally:budget.restore()
    return 0 if summary['status']=='complete' else 2
