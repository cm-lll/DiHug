"""Read-only adapter for the hash-pinned ACTUAL diagnostic source archive.

Never imports/executes worker.py, loads model tensors or advances a chain.
Discover result folders from collection/provenance files, not folder names.
"""
import ast
import hashlib
import json
import os
from pathlib import Path
import resource
import time
from zipfile import ZipFile
import numpy as np
from stepwise_role.core import Layout,signature
from multiscale_role.runtime_guard import Budget,checked_sha,atomic_json
from multiscale_role.evaluation import genre64,degree_w1

ZIP_SHA='c0fc3a8dade80a66f550a0506e405bf62bf80b3c1fb9f902d6a8baebf861194b'
SEED=2026101701
PREFIX='joint_role_frozen_diag_v1/'
ROOT=Path(__file__).resolve().parents[1]
ALIGN=dict(CF_atol=2e-10,CF_rtol=1e-9,W1_atol=1e-8,W1_rtol=1e-9,
           note='numeric replay tolerances frozen before reading real six-state arrays; no quality threshold change')


def read_json(path):return json.loads(Path(path).read_text())


def atomic_npz(path,values,check):
    path=Path(path);tmp=path.with_name(path.name+'.tmp')
    if path.exists() or tmp.exists():raise ValueError('refusing to overwrite export')
    try:
        check()
        with tmp.open('xb') as f:
            np.savez_compressed(f,**values);check();f.flush();os.fsync(f.fileno())
        check();os.replace(tmp,path)
    finally:
        if tmp.exists():tmp.unlink()


def source_archive(path,check=lambda:None):
    if checked_sha(path,check)!=ZIP_SHA:raise ValueError('wrong original diagnostic source ZIP')
    with ZipFile(path) as z:
        files={}
        for n in z.namelist():
            check()
            if n.startswith(PREFIX) and not n.endswith('/'):files[n[len(PREFIX):]]=z.read(n)
    release=json.loads(files['release.json'])
    for name,h in release.items():
        check()
        if hashlib.sha256(files[name]).hexdigest()!=h:raise ValueError('original release file differs: '+name)
    contract=read_json(ROOT/'LEGACY_SOURCE_CONTRACT.json')
    for name,h in contract['files'].items():
        if hashlib.sha256(files[name]).hexdigest()!=h:raise ValueError('audited original source differs: '+name)
    for name in ('hsgd/slots.py','hsgd/kernel.py'):
        if files['source/'+name]!=(ROOT/name).read_bytes():raise ValueError('canonical slot/kernel source not identical')
    def function(data,name):
        tree=ast.parse(data.decode())
        return ast.dump(next(v for v in tree.body if isinstance(v,ast.FunctionDef) and v.name==name),include_attributes=False)
    for name in ('plan_for','schedule'):
        if function(files['source/hsgd/bw_preflight/common.py'],name)!=function((ROOT/'hsgd/bw_preflight/common.py').read_bytes(),name):
            raise ValueError('plan/schedule wrapper differs from original source')
    return files,json.loads(files['original_pins.json']),json.loads(files['input_pins.json'])


def discover(root,ledger,check=lambda:None):
    root=Path(root).resolve();found={}
    for collection in root.rglob('collection.json'):
        check();folder=collection.parent
        if not (folder/'baseline.npz').is_file():raise ValueError('collection without original baseline')
        ancestors=[p for p in folder.parents if p==root or root in p.parents]
        provenance=next((p/'provenance.json' for p in ancestors if (p/'provenance.json').is_file()),None)
        if provenance is None:raise ValueError('collection missing actual arm provenance')
        cp=read_json(collection);pr=read_json(provenance);key=(pr['arm'],cp['time'])
        if key in found:raise ValueError('duplicate discovered arm/time')
        if key[0] not in ('A','B') or key[1] not in (500,20,1):raise ValueError('unexpected collection')
        for f in (collection,folder/'baseline.npz',folder/'collected.npz',folder/'genre_baseline.json',folder/'completed.json',provenance):
            name=str(f.relative_to(root))
            if name not in ledger or checked_sha(f,check)!=ledger[name]:raise ValueError('missing/changed original ledger evidence: '+name)
        found[key]=(folder,collection,provenance)
    expected={(a,t) for a in 'AB' for t in (500,20,1)}
    if set(found)!=expected:raise ValueError('actual directory does not contain exactly six unique cases')
    return found


def original_query(case,t,collection,check=lambda:None):
    from hsgd.bw_preflight.common import plan_for
    if collection['query_seed']!=[SEED,t,61] or collection['time']!=t or collection['next_time']!=t-1 or collection['query_rounds']!=20:
        raise ValueError('original seed/time/query metadata differs')
    digest=hashlib.sha256();block=np.full(case['space'].total,-1,dtype=np.int64)
    coverage=np.zeros(case['space'].total,dtype=np.uint8)
    plan=plan_for(case,[SEED,t,61],20)
    for r in range(20):
        check();ids,f,query=plan.block(r)
        # EXACT worker.py bytes/order, including empty blocks. No new framing.
        digest.update(ids.tobytes());digest.update(f.tobytes());digest.update(query.tobytes())
        if coverage[ids].any():raise ValueError('duplicate original query coverage')
        block[ids]=r;coverage[ids]+=1
    expected=np.repeat(case['active'].astype(np.uint8),case['space'].counts)
    if not np.array_equal(coverage,expected) or digest.hexdigest()!=collection['query_sha256']:
        raise ValueError('complete ids/families/query digest differs from actual collection')
    return block,coverage,digest.hexdigest()


def old_digest(a):
    return dict(shape=list(a.shape),dtype=str(a.dtype),sha256=hashlib.sha256(a.tobytes()).hexdigest())


def convert_arrays(case,baseline,collected,collection,t,check=lambda:None):
    from hsgd.bw_preflight.common import schedule
    from hsgd.kernel import MarginalKernel
    for k in ('p0','state','q0','types','pairs','prior','alpha'):
        if k not in baseline:raise ValueError('original baseline missing '+k)
    if not np.array_equal(baseline['types'],case['space'].types) or not np.array_equal(baseline['pairs'],case['space'].pairs):
        raise ValueError('original types/relation order differs')
    if not np.array_equal(baseline['prior'],case['prior']) or not np.array_equal(baseline['alpha'],schedule(1000)):
        raise ValueError('original prior/timetable differs')
    layout=Layout.canonical(baseline['types'],baseline['pairs'],baseline['prior'])
    for k in ('p0','state'):
        if not np.array_equal(baseline[k],collected[k]) or old_digest(baseline[k])!=collection[k]:
            raise ValueError('original baseline/collection array differs: '+k)
    if not np.isin(baseline['state'],[0,1]).all():raise ValueError('nonbinary original Xt')
    layout.probability(baseline['p0']);layout.probability(baseline['state']);layout.probability(baseline['q0'])
    blocks,coverage,qhash=original_query(case,t,collection,check)
    if not np.array_equal(coverage,collected['coverage']):raise ValueError('original coverage differs')
    q=MarginalKernel(layout.prior,baseline['alpha']).reverse(baseline['p0'],baseline['state'],layout.families,t-1,t)
    error=float(abs(q-baseline['q0']).max(initial=0))
    if error>5e-14:raise ValueError('q0 is not the saved next-state Bayes posterior')
    values=dict(state=baseline['state'],p0=baseline['p0'],q=baseline['q0'],types=baseline['types'],
                prior=baseline['prior'],alpha=baseline['alpha'],slots=layout.slots,families=layout.families,
                t=np.array(t,dtype=np.int64),query_block=blocks)
    origin={k:dict(kind='original',source_field='q0' if k=='q' else k) for k in ('state','p0','q','types','prior','alpha')}
    origin.update({k:dict(kind='deterministically_reconstructed',recipe={'slots':'TypedSlots canonical from original types/pairs',
        'families':'canonical family IDs','t':'validated collection.time','query_block':'actual worker query recipe'}[k])
        for k in ('slots','families','t','query_block')})
    return values,origin,error,qhash


def nested(data,path):
    for key in path:data=data[key]
    return data


def extract_frequency(report,layout):
    path=['independent_diagnostic_frequencies','omega'];w=np.asarray(nested(report,path),float)
    if w.shape!=(64,) or not np.isfinite(w).all():raise ValueError('original nested Genre64 must be 64 finite scalar frequencies')
    if report.get('movie_genre_family')!=8 or len(layout.pairs)<=8 or not np.array_equal(layout.pairs[8],[2,3]):
        raise ValueError('Movie(type2)-Genre(type3) relation8 mapping differs')
    result=np.zeros((64,len(layout.pairs)));result[:,8]=w
    return result,path


def baseline_alignment(metrics,report):
    # Explicitly exclude conditional variance and mean_expected_squared... .
    old_cf=float(report['independent_diagnostic_frequencies']['mean_squared_mean_cf_distance'])
    old_w1=float(report['w1_between_expected_empirical_laws'])
    new_cf=float(metrics['Genre64']['error'])
    channels=metrics['expected_degree_PMF_W1']['3']
    matching=[r for r in channels if r['families']==[8]]
    if len(matching)!=1:raise ValueError('missing exact Genre MG PMF-W1 channel')
    new_w1=matching[0]['W1']
    ct=ALIGN['CF_atol']+ALIGN['CF_rtol']*abs(old_cf)
    wt=ALIGN['W1_atol']+ALIGN['W1_rtol']*abs(old_w1)+matching[0].get('numerical_allowance',0.)
    result=dict(CF_error=abs(new_cf-old_cf),W1_error=abs(new_w1-old_w1),CF_tolerance=ct,W1_tolerance=wt,
                old_CF=old_cf,recomputed_CF=new_cf,old_W1=old_w1,recomputed_W1=new_w1,
                passed=abs(new_cf-old_cf)<=ct and abs(new_w1-old_w1)<=wt)
    if not result['passed']:raise ValueError('old baseline CF/W1 alignment failed: '+str(result))
    return result


def validate_provenance(prov,arm,pins,inputs,files,layout,case,release_sha):
    """Cross-check original receipts; never deserialize/execute private model."""
    from hsgd.bw_preflight import common as c
    names=json.loads(files['source/data/IMDB_preprocessing_provenance.json'])['source_meta']['node_types']
    if prov['arm']!=arm or prov['model_checkpoint_sha256']!=pins['checkpoints'][arm]['sha256'] or prov['release_sha256']!=release_sha:
        raise ValueError('actual model/release provenance differs')
    if not np.array_equal(prov['family_pairs'],layout.pairs) or prov['type_names']!=names:
        raise ValueError('original type names/relation mapping differs')
    if prov['inputs']!=inputs or c.runtime_binding(prov['runtime'])!=pins['runtime']:
        raise ValueError('original input/runtime provenance differs')
    binding=prov['scientific_binding']
    def digest(a):return signature(old_digest(np.asarray(a)))
    code={n[len('source/'):]:hashlib.sha256(v).hexdigest() for n,v in files.items()
          if n.endswith('.py') and n.startswith(('source/hsgd/','source/role_degree/','source/stepwise_role/'))}
    receipt=dict(formal_training=True,completed_updates=6000,experiment_arm='cn' if arm=='B' else 'baseline',
        final=True,checkpoint_sha256=pins['checkpoints'][arm]['sha256'],source_sha256=signature(pins['source_code']),
        config_sha256=signature(pins['config']),data_sha256=pins['config']['imdb_sha256'])
    expected=dict(arm=arm,seed=SEED,rounds=20,time_steps=1000,slots=case['space'].total,backend='upstream',
                  data_sha256=digest(case['clean']),types_sha256=digest(layout.types),prior_sha256=digest(layout.prior),
                  alpha_sha256=digest(c.schedule(1000)),code_sha256=signature(code),receipt_sha256=signature(receipt))
    for k,v in expected.items():
        if binding.get(k)!=v:raise ValueError('original scientific binding differs: '+k)
    return binding


def export_run(zip_path,run_root,ledger_sha,out,deadline,cleanup_deadline):
    from multiscale_role.short import array_sha
    out=Path(out);budget=Budget(deadline,out/'STOP',cleanup_deadline);budget.install();check=budget.check
    summary=dict(status='incomplete',records=[],model_calls=0,training_updates=0,scientific_graphs=0,input_hash_verification_complete=False)
    hashes={};start=time.monotonic()
    def pin(path,expected=None):
        path=Path(path).resolve();value=checked_sha(path,check)
        if expected is not None and value!=expected:raise ValueError('original source/evidence hash differs: '+str(path))
        hashes[str(path)]=value;return value
    def emit(name,value):atomic_json(out/name,value,check)
    try:
        root=Path(run_root).resolve()
        if root==out.resolve() or root in out.resolve().parents:raise ValueError('export directory must be outside original results tree')
        from multiscale_role.entry_guard import verify_sealed_ledger
        verify_sealed_ledger(root,ledger_sha,check)
        files,pins,inputs=source_archive(zip_path,check);pin(zip_path,ZIP_SHA)
        ledger=read_json(root/'evidence_hashes.json');pin(root/'evidence_hashes.json',ledger_sha)
        for name,h in ledger.items():pin(root/name,h)
        activation=read_json(root/'activation.json');completed=read_json(root/'completed.json')
        release_sha=hashlib.sha256(files['release.json']).hexdigest()
        if activation['release_sha256']!=release_sha or activation['seed']!=SEED or activation['R']!=20 or activation['T']!=1000:
            raise ValueError('actual diagnostic activation/source differs')
        if completed.get('status')!='completed' or completed.get('frozen_states')!=6:raise ValueError('old diagnostic not completed')
        found=discover(root,ledger,check)
        # Paths come from real activation metadata, not guessed timestamps.
        states=Path(activation['states']);weights=Path(activation['checkpoints'])
        for name,h in inputs['files'].items():pin(states/name,h)
        for arm in 'AB':pin(weights/pins['checkpoints'][arm]['file'],pins['checkpoints'][arm]['sha256'])
        from hsgd.bw_preflight import common as c
        from role_degree.role_statistics import joint_profile
        from multiscale_role.core import MultiScaleGuide,freeze
        case=c.imdb_case(ROOT/'data/IMDB_movie_reference.npz',c.config())
        layout=Layout.canonical(case['space'].types,case['space'].pairs,case['prior'])
        profile=joint_profile(layout.types,case['clean'],case['space'].encode(case['clean'])[0],layout.pairs)
        guide=MultiScaleGuide(layout,freeze(profile,layout,c.schedule(1000)),check)
        source_record=dict(source_zip_sha256=ZIP_SHA,release_sha256=release_sha,conversion_code_sha256=checked_sha(__file__,check),
                           origin='deterministically reconstructed receipt from hash-pinned original archive; NOT an original producer manifest',
                           source_file_hashes={n:hashlib.sha256(v).hexdigest() for n,v in files.items() if n in ('worker.py','genre_diagnostics.py','original_pins.json','input_pins.json')})
        emit('legacy_source_contract.json',source_record);emit('producer_config.json',pins['config'])
        emit('discovered_layout.json',{a+':'+str(t):dict(folder=str(v[0]),collection=str(v[1]),provenance=str(v[2])) for (a,t),v in found.items()})
        manifest=dict(format='legacy_export_v1_1',base_commit='acbbddb5cdbcbaa2ae129bcd59e434e6a298b130',reference_sha256=c.config()['imdb_sha256'],records=[])
        spec=None
        for arm in 'AB':
            for t in (500,20,1):
                check();folder,collection_path,prov_path=found[arm,t]
                cp=read_json(collection_path);prov=read_json(prov_path)
                binding=validate_provenance(prov,arm,pins,inputs,files,layout,case,release_sha)
                # Completed graph provenance and record authenticate original Xt
                # without deserializing a checkpoint or loading a model.
                graph=read_json(states/arm/'graph.json')['binding']
                if signature(graph)!=prov['completed_binding_sha256'] or any(graph.get(k)!=v for k,v in binding.items()):
                    raise ValueError('original completed trajectory lineage differs')
                if cp['state_file_sha256']!=inputs['files'][arm+'/step_%04d/state.pt'%t]:raise ValueError('original Xt file differs')
                with np.load(folder/'baseline.npz',allow_pickle=False) as z:baseline={k:z[k].copy() for k in z.files}
                with np.load(folder/'collected.npz',allow_pickle=False) as z:collected={k:z[k].copy() for k in z.files}
                values,origins,error,qhash=convert_arrays(case,baseline,collected,cp,t,check)
                if read_json(states/arm/('step_%04d'%t)/'record.json')['state']!=old_digest(values['state']):raise ValueError('original committed Xt record differs')
                report_path=folder/'genre_baseline.json';report=read_json(report_path);omega,path=extract_frequency(report,layout)
                if report['next_state_time']!=t-1 or report['next_state_alpha']!=guide.alpha[t-1]:raise ValueError('old Genre target time differs')
                if spec is None:
                    spec=dict(reference_sha256=c.config()['imdb_sha256'],profile_sha256=signature(profile),Genre_type=3,omega64=omega.tolist(),CF_normalizer=1,
                              CF_contract='squared_distance_of_expected_empirical_joint_CF_to_forward_target',W1_contract='W1_of_expected_empirical_integer_degree_PMF_unscaled',
                              original_artifact_source=str(report_path),original_artifact_sha256=pin(report_path),frequency_field_path=path,
                              embedding=dict(family_index=8,pair=[2,3]),format='legacy_nested_scalar_Genre64')
                elif not np.array_equal(omega,spec['omega64']):raise ValueError('old independent frequency bank changes across cases')
                metrics=dict(Genre64=genre64(values['q'],guide,t-1,spec,check),
                             expected_degree_PMF_W1=degree_w1(values['q'],guide,t-1,check,only_types={'3'}))
                alignment=baseline_alignment(metrics,report)
                name=arm+'_t%04d'%t
                target=out/(name+'.npz')
                atomic_npz(target,values,check)
                check();output_sha=checked_sha(target,check)
                meta=dict(arm=arm,model_arm=arm,t=t,T=1000,query_rounds=20,sampling_seed=SEED,
                          reference_sha256=c.config()['imdb_sha256'],checkpoint_sha256=pins['checkpoints'][arm]['sha256'],
                          source_manifest_sha256=checked_sha(out/'legacy_source_contract.json',check),config_sha256=checked_sha(out/'producer_config.json',check),
                          completed_updates=6000,final_weight=True,array_sha256={k:array_sha(a) for k,a in values.items()},
                          field_origins=origins,legacy_query_sha256=qhash,legacy_q0_bridge_error=error,
                          legacy_collection=str(collection_path),legacy_collection_sha256=pin(collection_path),
                          legacy_report=str(report_path),legacy_report_sha256=pin(report_path),baseline_alignment=alignment,
                          conversion=source_record,original_baseline_sha256=pin(folder/'baseline.npz'),export_npz_sha256=output_sha)
                meta['producer_config_origin']=dict(kind='deterministically_reconstructed',source='original_pins.json.config',source_zip_sha256=ZIP_SHA)
                meta['original_input_hashes']=dict(hashes)
                emit(name+'.json',meta)
                row=dict(arm=arm,t=t,reference_sha256=meta['reference_sha256'],npz=target.name,npz_sha256=output_sha,
                         metadata=name+'.json',metadata_sha256=checked_sha(out/(name+'.json'),check),
                         checkpoint=str(weights/pins['checkpoints'][arm]['file']),checkpoint_sha256=meta['checkpoint_sha256'],
                         source_manifest='legacy_source_contract.json',source_manifest_sha256=meta['source_manifest_sha256'],
                         config='producer_config.json',config_sha256=meta['config_sha256'])
                manifest['records'].append(row);summary['records'].append(dict(arm=arm,t=t,baseline_alignment=alignment))
                emit('progress.json',dict(exported=len(manifest['records']),total=6))
        emit('six_states.json',manifest);emit('original_evaluation.json',spec)
        for p,h in hashes.items():
            if checked_sha(p,check)!=h:raise ValueError('source changed during conversion')
        summary.update(status='exported',input_hash_verification_complete=True,input_hashes_unchanged=True,
                       manifest_sha256=checked_sha(out/'six_states.json',check),evaluation_sha256=checked_sha(out/'original_evaluation.json',check))
    except BaseException as exc:
        summary.update(status='incomplete' if isinstance(exc,(InterruptedError,TimeoutError,KeyboardInterrupt)) else 'protocol_failed',reason=type(exc).__name__+': '+str(exc))
    finally:
        summary.update(wall_seconds=time.monotonic()-start,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                       source_input_hashes=hashes,real_array_execution_claim='only if this invocation exported all six; packaging is not execution')
        try:atomic_json(out/'summary.json',summary,budget.cleanup_check)
        finally:budget.restore()
    return 0 if summary['status']=='exported' else 2
