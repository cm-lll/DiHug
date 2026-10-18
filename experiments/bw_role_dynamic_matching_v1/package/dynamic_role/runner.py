"""Bounded CPU-only frozen probability diagnostics; no model or new graph."""
import json
import resource
import time
from pathlib import Path
import numpy as np
from .core import MatchingGuide,freeze,Layout,signature,SETTINGS
from .portable import Portable
from multiscale_role.runtime_guard import atomic_json,checked_sha

POLICY=dict(version='dynamic_role_frozen_state_development_v1',
    baseline='7049e64b108f2ef434c40815457c74037c4455eb',
    cases=[[a,t] for a in ('A','B') for t in (500,20,1)],
    CF_atol=2e-10,W1_atol=1e-8,metric_rtol=1e-9,
    W1_allowance='add existing FP64 operation-count allowances; not interval proof',
    matching='fixed-assignment descent >1e-12+1e-9*abs(loss); rematch <= fixed within assignment tolerance',
    all_skipped='ineffective at these states, never successful',
    full_panel='all six valid; no independent Genre64 or Genre W1 worsening beyond tolerance AND at least one state improves both',
    tradeoff='any independent deterioration => tradeoff/deterioration, never overall repair',
    unresolved='no resolved dual gain => evidence insufficient, not equivalence',
    incomplete='missing input, mismatch, timeout or partial panel => incomplete; no subset success',
    no_tuning='B1 engineering check then freeze code/target/controller before six; no frequency, step or state selection',
    scope='development frozen-state evidence, not full-chain quality or generalization')


def fingerprint():
    root=Path(__file__).resolve().parent
    return signature({p.name:checked_sha(p,lambda:None) for p in sorted(root.glob('*.py'))})


def distributions(p,layout):
    p=layout.probability(p);entropy=np.zeros_like(p)
    take=p>0;entropy[take]-=p[take]*np.log(p[take]);take=p<1;entropy[take]-=(1-p[take])*np.log1p(-p[take])
    return dict(mean_entropy=float(entropy.mean()),exact_zero=int((p==0).sum()),exact_one=int((p==1).sum()),
        near_saturation_fraction=float(((p<=1e-6)|(p>=1-1e-6)).mean()),
        expected_relation_edges=np.bincount(layout.families,weights=p,minlength=len(layout.pairs)).tolist())


def genre_law(q,oldguide,s,check):
    """Exact integer-degree PMFs for P(D>153), not W1 of expected degrees."""
    from multiscale_role.evaluation import full_pmf_rows
    l=oldguide.layout;c='3';ids,node=oldguide.tail_layout[c][0];n=l.sizes[3]
    counts=np.bincount(node,minlength=n);cap=int(counts[0])
    if not np.all(counts==cap):raise ValueError('Genre node-specific capacity unsupported')
    start=np.r_[0,np.cumsum(counts)[:-1]];pos=np.arange(len(ids))-np.repeat(start,counts)
    mat=np.zeros((n,cap));mat[node,pos]=q[ids]
    pmf,err=full_pmf_rows(mat,check);pmf=pmf.mean(0)
    g=oldguide.frozen['payload']['profile']['groups'][c];rows=np.array(g['rows']);weights=np.array(g['counts'])/g['nodes']
    a=oldguide.alpha[s];pi=l.prior[8];b=a+(1-a)*pi;z=(1-a)*pi
    targetmat=np.full((len(rows),cap),z)
    for i,d in enumerate(rows[:,8]):targetmat[i,:d]=b
    target,te=full_pmf_rows(targetmat,check);target=weights@target
    return dict(predicted_P_D_gt_153=float(pmf[154:].sum()),target_P_D_gt_153=float(target[154:].sum()),
        predicted_mean=float(pmf@np.arange(cap+1)),target_mean=float(target@np.arange(cap+1)),
        predicted_PMF=pmf.tolist(),target_PMF=target.tolist(),predicted_mass_error=err,target_mass_error=te)


def compare(current,old,path='metrics'):
    """All numeric definitions and channels of old before metrics must agree."""
    if isinstance(old,dict):
        if set(current)!=set(old):raise ValueError('metric definition keys differ: '+path)
        return [x for k in old for x in compare(current[k],old[k],path+'.'+k)]
    if isinstance(old,list):
        if len(current)!=len(old):raise ValueError('metric length differs: '+path)
        return [x for i,(a,b) in enumerate(zip(current,old)) for x in compare(a,b,path+'.'+str(i))]
    if isinstance(old,(int,float)) and not isinstance(old,bool):
        tol=(POLICY['W1_atol'] if any(k in path for k in ('W1','degree','edges')) else POLICY['CF_atol'])+POLICY['metric_rtol']*abs(old)
        if not np.isfinite(current) or abs(current-old)>tol:raise ValueError('old baseline metric failed reproduction: '+path)
        return [dict(field=path,error=abs(current-old),tolerance=tol)]
    if current!=old:raise ValueError('metric definition differs: '+path)
    return []


def run(args,out,check):
    from hsgd.bw_preflight import common as c
    from hsgd.kernel import MarginalKernel
    from role_degree.role_statistics import joint_profile
    from multiscale_role.core import MultiScaleGuide,freeze as oldfreeze
    from multiscale_role.evaluation import validate_spec,evaluate,ForwardPMFCache
    from multiscale_role.legacy import extract_frequency,baseline_alignment
    from multiscale_role.short import classify
    out=Path(out);started=time.monotonic();cpu=time.process_time();records=[]
    result=dict(status='incomplete',scope=args.action,records=records,policy=POLICY,settings=SETTINGS,
        model_calls=0,training_updates=0,scientific_graphs=0,GPU_use=0,
        source_authentication='actual delivered payloads checked; private weights historical receipts only')
    bundle=Portable(args.portable,args.portable_sha256,check)
    case=c.imdb_case(c.PACKAGE/'data/IMDB_movie_reference.npz',c.config());reference_sha=c.config()['imdb_sha256']
    layout=Layout.canonical(case['space'].types,case['space'].pairs,case['prior'])
    profile=joint_profile(layout.types,case['clean'],case['space'].encode(case['clean'])[0],layout.pairs)
    target=freeze(profile,layout,c.schedule(1000),reference_sha);guide=MatchingGuide(layout,target)
    atomic_json(out/'anonymous_target.json',target,check)
    tick=time.monotonic();oldguide=MultiScaleGuide(layout,oldfreeze(profile,layout,c.schedule(1000)),check)
    result['old_evaluation_target_prepare_seconds']=time.monotonic()-tick
    oldobjective=bundle.json(bundle.short/'frozen_objective.json');oldpolicy=bundle.json(bundle.short/'policy.json')
    from multiscale_role.short import POLICY as OLDPOLICY
    if oldobjective!=oldguide.frozen or oldpolicy!=OLDPOLICY or bundle.old['objective_sha256']!=oldguide.frozen['sha256']:
        raise ValueError('old objective/policy/summary not the pinned definition')
    validate_spec(bundle.spec,oldguide,reference_sha)
    artifact=resolve_original(bundle.export,bundle.spec['original_artifact_source'])
    omega,path=extract_frequency(bundle.json(artifact,bundle.spec['original_artifact_sha256']),layout)
    if not np.array_equal(omega,bundle.spec['omega64']) or path!=bundle.spec['frequency_field_path'] or bundle.spec['CF_normalizer']!=1:
        raise ValueError('old independent Genre64 bank differs')
    kernel=MarginalKernel(layout.prior,guide.alpha);cache=ForwardPMFCache()
    import platform,scipy
    numeric_runtime=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__)
    implementation=fingerprint();result.update(implementation_sha256=implementation,target_sha256=target['sha256'],environment=c.environment('cpu'),numeric_runtime=numeric_runtime)
    if args.action=='six':
        if not args.b1_receipt or not args.b1_receipt_sha256:raise ValueError('B1 development receipt required before frozen six-state run')
        receipt=json.loads(Path(args.b1_receipt).read_text())
        if checked_sha(args.b1_receipt,check)!=args.b1_receipt_sha256 or receipt.get('status')!='complete' or receipt.get('scope')!='b1' or receipt.get('implementation_sha256')!=implementation or receipt.get('target_sha256')!=target['sha256'] or receipt.get('portable_sha256')!=args.portable_sha256 or receipt.get('numeric_runtime')!=numeric_runtime:
            raise ValueError('B1 receipt source/target/input not same frozen implementation')
    for index,row in enumerate(bundle.original['records']):
        if args.action=='b1' and (row['arm'],row['t'])!=('B',1):continue
        check();print(json.dumps(dict(event='case_start',arm=row['arm'],t=row['t'],index=index)),flush=True)
        tick=time.monotonic();data,meta,certificate=bundle.record(row,case,kernel,layout)
        if args.action=='check':
            records.append(dict(arm=row['arm'],t=row['t'],binding=certificate));continue
        p=data['p0'];q=data['q'];tick_eval=time.monotonic()
        before=evaluate(q,oldguide,row['t']-1,bundle.spec,check,cache)
        old_record=next(x for x in bundle.old['records'] if (x['arm'],x['t'])==(row['arm'],row['t']))
        proofs=compare(before,old_record['before'])
        baseline_alignment(before,bundle.json(meta['legacy_report'],meta['legacy_report_sha256']))
        baseline_seconds=time.monotonic()-tick_eval
        pp,qq,diagnostic=guide.apply(p,data['state'],kernel,row['t'],check)
        if diagnostic['status']=='numerical_failure':raise FloatingPointError(diagnostic['reason'])
        tick_eval=time.monotonic();after=evaluate(qq,oldguide,row['t']-1,bundle.spec,check,cache)
        laws_before=genre_law(q,oldguide,row['t']-1,check);laws_after=genre_law(qq,oldguide,row['t']-1,check)
        evaluation_seconds=time.monotonic()-tick_eval+baseline_seconds
        rec=dict(index=index,arm=row['arm'],t=row['t'],binding=certificate,guidance=diagnostic,
            no_guidance=before,old_multiscale_tail=old_record['after'],new_role_matching=after,
            old_baseline_reproduction=proofs,Genre_law_before=laws_before,Genre_law_after=laws_after,
            clean_before=distributions(p,layout),clean_after=distributions(pp,layout),
            transition_before=distributions(q,layout),transition_after=distributions(qq,layout),
            decision=classify(before,after,bundle.spec),evaluation_seconds=evaluation_seconds,
            case_wall_seconds=time.monotonic()-tick)
        atomic_json(out/('case_%02d.json'%index),rec,check);records.append(rec)
        check();print(json.dumps(dict(event='case_done',arm=row['arm'],t=row['t'],status=diagnostic['status'],wall_seconds=rec['case_wall_seconds'])),flush=True)
    check();bundle.rehash();check()
    if fingerprint()!=implementation or checked_sha(c.PACKAGE/'data/IMDB_movie_reference.npz',check)!=reference_sha:
        raise ValueError('used implementation/reference changed during computation')
    result.update(status='complete',completed_records=len(records),portable_sha256=args.portable_sha256,
        wall_seconds=time.monotonic()-started,cpu_seconds=time.process_time()-cpu,
        cpu_total_process_seconds=time.process_time(),
        peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,input_hashes_unchanged=True)
    if args.action!='check':
        accepted=sum(r['guidance']['status']=='accepted' for r in records)
        result['accepted']=accepted
        result['conclusion']=('all_skipped_no_effect' if not accepted else
            'tradeoff_or_deterioration' if any(r['decision']['any_deterioration'] for r in records) else
            'frozen_panel_dual_gain' if args.action=='six' and len(records)==6 and any(r['decision']['dual_improvement'] for r in records) else 'partial_or_unresolved')
    return result


def resolve_original(parent,path):
    p=Path(path);return p if p.is_absolute() else parent/p
