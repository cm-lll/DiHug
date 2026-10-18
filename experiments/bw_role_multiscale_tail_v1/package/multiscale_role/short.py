"""Six SAVED states only; strict input binding, CPU reference, no model import.

Full scientific input is deliberately unavailable locally. This module never
creates a state, probability, evaluation frequency, or substitute request.
"""
import hashlib
import json
import signal
import sys
import time
from pathlib import Path
import numpy as np
from stepwise_role.core import Layout,signature
from role_degree.role_statistics import joint_profile
from multiscale_role.core import freeze,MultiScaleGuide
from multiscale_role.evaluation import evaluate,validate_spec

BASE='acbbddb5cdbcbaa2ae129bcd59e434e6a298b130'
POLICY=dict(version='six_state_development_v1',fixed_cases=[[a,t] for a in ('A','B') for t in (500,20,1)],
            q_bridge_atol=5e-14,CF_atol=2e-10,W1_atol=1e-8,metric_rtol=1e-9,
            W1_allowance_rule='add before+after 64*eps*capacity*(capacity+nodes+role_rows+1)',
            full_panel_rule='all six complete; neither Genre64 nor any Genre W1 channel worsens beyond numerical tolerance; at least one state improves BOTH Genre64 and a Genre W1 channel',
            mixed_rule='any independent metric deterioration => tradeoff, not overall success',
            unresolved_rule='no resolved independent gain => no evidence of gain; not equivalence',
            failure_rule='missing/binding/numerical error or timeout => incomplete/protocol failure; never declare success from subset',
            meaning='descriptive frozen-state evidence only, no generated-graph quality claim')


def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def array_sha(a):
    a=np.asarray(a)
    return signature(dict(shape=list(a.shape),dtype=str(a.dtype),raw_sha256=hashlib.sha256(a.tobytes()).hexdigest()))


def resolve(parent,name):
    path=Path(name)
    return path if path.is_absolute() else parent/path


def read_pinned(path,expected):
    if file_sha(path)!=expected:raise ValueError('pinned file differs: '+str(path))
    return json.loads(Path(path).read_text())


def validate_record(record,parent,case,check,hashed):
    from hsgd.bw_preflight.common import plan_for,schedule
    check();path=resolve(parent,record['npz']);source=resolve(parent,record['metadata'])
    if file_sha(path)!=record['npz_sha256']:raise ValueError('saved state/probability NPZ hash differs')
    meta=read_pinned(source,record['metadata_sha256'])
    if meta.get('arm')!=record['arm'] or meta.get('t')!=record['t'] or meta.get('T')!=1000 or meta.get('query_rounds')!=20:
        raise ValueError('saved request arm/time/schedule/query binding differs')
    if meta.get('reference_sha256')!=record['reference_sha256'] or meta.get('model_arm')!=record['arm']:
        raise ValueError('saved reference/native-model arm differs')
    # Compare real files to the producer's pinned metadata. Never print or load
    # checkpoint tensors; these files remain private and are not release input.
    for field in ('checkpoint','source_manifest','config'):
        artifact=resolve(parent,record[field]);expected=record[field+'_sha256']
        if str(artifact) not in hashed:hashed[str(artifact)]=file_sha(artifact)
        if hashed[str(artifact)]!=expected or meta.get(field+'_sha256')!=expected:
            raise ValueError('saved producer artifact binding differs: '+field)
    if meta.get('completed_updates')!=6000 or meta.get('final_weight') is not True:
        raise ValueError('native final6000 model binding required')
    with np.load(path,allow_pickle=False) as z:
        required=('state','p0','q','slots','families','types','prior','alpha','query_block','t')
        if not set(required).issubset(z.files):raise ValueError('missing saved arrays: '+str(set(required)-set(z.files)))
        data={k:z[k].copy() for k in required}
    for name,array in data.items():
        if meta.get('array_sha256',{}).get(name)!=array_sha(array):raise ValueError('saved array binding differs: '+name)
    space=case['space'];f,local=space.encode(data['slots'])
    if (not np.array_equal(data['types'],space.types) or not np.array_equal(data['families'],f) or
        not np.array_equal(space.offsets[f]+local,np.arange(space.total)) or int(data['t'])!=record['t']):
        raise ValueError('node/schema/global slot order differs')
    if not np.array_equal(data['prior'],case['prior']) or not np.array_equal(data['alpha'],schedule(1000)):
        raise ValueError('prior or full T1000 timetable differs')
    if not np.isin(data['state'],[0,1]).all():raise ValueError('invalid frozen state')
    plan=plan_for(case,[meta['sampling_seed'],record['t'],61],20)
    expected=np.full(space.total,-1,dtype=np.int64)
    for r in range(20):
        check();ids,_,_=plan.block(r);expected[ids]=r
    if not np.array_equal(expected,data['query_block']):raise ValueError('saved original query plan/order differs')
    return data,meta


def classify(before,after,spec):
    c=str(spec['Genre_type']);diff=after['Genre64']['error']-before['Genre64']['error']
    ct=POLICY['CF_atol']+POLICY['metric_rtol']*abs(before['Genre64']['error'])
    dd=[]
    for a,b in zip(before['expected_degree_PMF_W1'][c],after['expected_degree_PMF_W1'][c]):
        if a['families']!=b['families']:raise AssertionError('evaluation channel changed')
        dd.append(dict(families=a['families'],difference=b['W1']-a['W1'],
                       tolerance=POLICY['W1_atol']+POLICY['metric_rtol']*abs(a['W1'])+
                       a.get('numerical_allowance',0.)+b.get('numerical_allowance',0.)))
    worse=diff>ct or any(d['difference']>d['tolerance'] for d in dd)
    dual=diff < -ct and any(d['difference'] < -d['tolerance'] for d in dd) and not worse
    return dict(Genre64_difference=diff,Genre64_tolerance=ct,Genre_W1_differences=dd,
                any_deterioration=worse,dual_improvement=dual,
                label='tradeoff_or_deterioration' if worse else 'dual_improvement' if dual else 'partial_or_unresolved')


def verify_original_q(data,kernel,layout,t):
    p0=layout.probability(data['p0']);layout.probability(data['state']);layout.probability(data['q'])
    if not np.isin(data['state'],[0,1]).all():raise ValueError('nonbinary frozen Xt')
    q=kernel.reverse(p0,data['state'],layout.families,t-1,t)
    error=float(abs(q-data['q']).max(initial=0))
    if error>POLICY['q_bridge_atol']:raise ValueError('original saved q does not match original posterior')
    return p0,q,error


def run(manifest,manifest_sha,spec_path,spec_sha,output,deadline):
    # Timer starts in external controller before this worker imports NumPy.
    import resource
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    start,cpu=time.monotonic(),time.process_time();records=[];stop=out/'STOP'
    def check():
        if stop.exists():raise InterruptedError('safe STOP requested; no request replacement')
        if time.monotonic()>=deadline:raise TimeoutError('single-run budget reached')
    def emit(name,value):
        temp=out/(name+'.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False));temp.replace(out/name)
    summary=dict(status='incomplete',records=records,policy=POLICY,model_calls=0,GPU_use=0,
                 training_updates=0,scientific_graphs=0,backend='numpy_float64_CPU_reference',
                 gpu_peak_memory=None,gpu_decisions_verified=False)
    initial_hash={}
    try:
        check();m=read_pinned(manifest,manifest_sha);spec=read_pinned(spec_path,spec_sha)
        if m.get('base_commit')!=BASE or sorted((r['arm'],r['t']) for r in m['records'])!=sorted(map(tuple,POLICY['fixed_cases'])):
            raise ValueError('requires exactly six original A/B times500,20,1; no duplicates/replacements')
        # Import only data/slot/kernel helpers. NO model instantiation or calls.
        from hsgd.bw_preflight import common as c
        from hsgd.kernel import MarginalKernel
        summary['environment']=c.environment('cpu')
        case=c.imdb_case(c.PACKAGE/'data/IMDB_movie_reference.npz',c.config())
        reference_sha=c.config()['imdb_sha256']
        if m.get('reference_sha256')!=reference_sha:raise ValueError('manifest reference differs')
        if spec.get('Genre_type')!=3:raise ValueError('complete IMDB Genre type must be original type3')
        layout=Layout.canonical(case['space'].types,case['space'].pairs,case['prior'])
        profile=joint_profile(layout.types,case['clean'],case['space'].encode(case['clean'])[0],layout.pairs)
        tick=time.monotonic();frozen=freeze(profile,layout,c.schedule(1000));guide=MultiScaleGuide(layout,frozen,check)
        summary['objective_prepare_seconds']=time.monotonic()-tick
        summary['objective_sha256']=frozen['sha256'];summary['evaluation_spec_sha256']=spec_sha
        validate_spec(spec,guide,reference_sha)
        origin=resolve(Path(spec_path).parent,spec['original_artifact_source'])
        if file_sha(origin)!=spec['original_artifact_sha256']:raise ValueError('original independent64 artifact hash differs')
        if origin.suffix=='.npz':
            with np.load(origin,allow_pickle=False) as z:w=z[spec['frequency_array_key']]
        else:w=np.asarray(json.loads(origin.read_text())[spec['frequency_array_key']])
        if not np.array_equal(w,np.asarray(spec['omega64'])):raise ValueError('evaluation frequency array not original artifact')
        emit('frozen_objective.json',frozen);emit('policy.json',POLICY)
        kernel=MarginalKernel(case['prior'],guide.alpha);parent=Path(manifest).parent;hashed={}
        # Freeze complete file hashes before processing; end recheck even after
        # interruption. Never modify any input, original probabilities or graph.
        all_inputs=[Path(manifest),Path(spec_path),origin,c.PACKAGE/'data/IMDB_movie_reference.npz']
        for row in m['records']:
            for key in ('npz','metadata','checkpoint','source_manifest','config'):all_inputs.append(resolve(parent,row[key]))
        initial_hash={str(p):file_sha(p) for p in set(all_inputs)}
        for index,row in enumerate(m['records']):
            check();case_start=time.monotonic();print(json.dumps(dict(event='saved_case_start',arm=row['arm'],t=row['t'])),flush=True)
            if row.get('reference_sha256')!=reference_sha:raise ValueError('per-record reference differs')
            tick=time.monotonic();data,meta=validate_record(row,parent,case,check,hashed)
            verify_seconds=time.monotonic()-tick
            tick=time.monotonic();p0,q,bridge_error=verify_original_q(data,kernel,layout,row['t']);bridge_seconds=time.monotonic()-tick
            tick=time.monotonic();guide.target(row['t']-1);target_seconds=time.monotonic()-tick
            tick=time.monotonic();before=evaluate(q,guide,row['t']-1,spec,check);before_eval=time.monotonic()-tick
            pp,qq,guidance=guide.apply(p0,data['state'],kernel,row['t'],check)
            if guidance['status']=='numerical_failure':raise FloatingPointError(guidance['reason'])
            tick=time.monotonic();after=evaluate(qq,guide,row['t']-1,spec,check);after_eval=time.monotonic()-tick
            result=dict(index=index,arm=row['arm'],t=row['t'],original_q_max_error=bridge_error,guidance=guidance,
                        before=before,after=after,decision=classify(before,after,spec),
                        timings=dict(input_verification=verify_seconds,original_posterior=bridge_seconds,
                                     next_time_target=target_seconds,before_evaluation=before_eval,after_evaluation=after_eval,
                                     total_case=time.monotonic()-case_start),
                        provenance='pinned original file/array/query/producer bindings cross-checked; not an independent reconstruction of producer execution')
            records.append(result);emit('case_%02d.json'%index,result);emit('progress.json',dict(completed=len(records),total=6))
        any_worse=any(r['decision']['any_deterioration'] for r in records)
        summary.update(status='complete',decision='tradeoff' if any_worse else
                       'consistent_fixed_panel_improvement' if any(r['decision']['dual_improvement'] for r in records) else 'no_resolved_dual_gain')
    except BaseException as exc:
        summary.update(status='incomplete' if isinstance(exc,(TimeoutError,InterruptedError,KeyboardInterrupt)) else 'protocol_failed',
                       exception=type(exc).__name__,reason=str(exc))
    finally:
        changed=[p for p,old in initial_hash.items() if not Path(p).is_file() or file_sha(p)!=old]
        if changed:summary.update(status='protocol_failed',modified_inputs=changed)
        summary.update(completed_records=len(records),worker_wall_seconds=time.monotonic()-start,
                       worker_CPU_seconds=time.process_time()-cpu,worker_peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                       input_hashes_unchanged=(not bool(changed)) if initial_hash else None,input_hashes=initial_hash,
                       input_hash_verification_complete=bool(initial_hash))
        emit('summary.json',summary)
    return 0 if summary['status']=='complete' else 2
