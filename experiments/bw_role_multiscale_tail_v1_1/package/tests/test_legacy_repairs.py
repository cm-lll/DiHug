"""Small synthetic engineering arrays; NOT real A/B diagnostic artifacts."""
import copy
import hashlib
import itertools
import json
from pathlib import Path
import signal
import time
import numpy as np
import pytest
from hsgd.bw_preflight import common as c
from hsgd.kernel import MarginalKernel
from role_degree.role_statistics import joint_profile
from stepwise_role.core import Layout
from multiscale_role.core import MultiScaleGuide,freeze
from multiscale_role.legacy import SEED,old_digest,original_query,convert_arrays,extract_frequency,baseline_alignment,atomic_npz,discover
from multiscale_role.evaluation import genre64,degree_w1
from multiscale_role.short import read_pinned
from multiscale_role.runtime_guard import Budget,atomic_json,checked_sha
from run_multiscale import validate_result
import legacy_genre_reference as old


def fixture():
    types=np.repeat(np.arange(4),3);pairs=list(itertools.combinations_with_replacement(range(4),2))
    first=Layout.canonical(types,pairs,np.full(10,.5))
    chosen=[np.flatnonzero(first.families==r)[0] for r in (1,2,4,5,8)]
    case=c.make_case(types,first.slots[:,chosen],pairs)
    l=Layout.canonical(types,pairs,case['prior']);alpha=c.schedule(1000);t=20
    p0=np.repeat(case['prior'],case['space'].counts);state=np.zeros(len(p0),dtype=np.uint8);state[case['clean_ids']]=1
    k=MarginalKernel(l.prior,alpha);q=k.reverse(p0,state,l.families,t-1,t)
    plan=c.plan_for(case,[SEED,t,61],20);h=hashlib.sha256();coverage=np.zeros(len(p0),dtype=np.uint8)
    for block in range(20):
        ids,f,query=plan.block(block)
        for a in (ids,f,query):h.update(a.tobytes())
        coverage[ids]+=1
    cp=dict(time=t,next_time=t-1,query_seed=[SEED,t,61],query_rounds=20,query_sha256=h.hexdigest(),state=old_digest(state),p0=old_digest(p0))
    base=dict(state=state,p0=p0,q0=q,types=types,pairs=np.asarray(pairs),prior=l.prior,alpha=alpha)
    collected=dict(state=state.copy(),p0=p0.copy(),coverage=coverage)
    return case,l,base,collected,cp


def test_old_mapping_canonical_slots_and_full_query_bytes():
    case,l,b,col,cp=fixture();v,origins,error,qhash=convert_arrays(case,b,col,cp,20)
    assert np.array_equal(v['q'],b['q0']) and error==0 and qhash==cp['query_sha256']
    assert np.array_equal(v['slots'],l.slots) and np.array_equal(v['families'],l.families)
    assert origins['q']==dict(kind='original',source_field='q0')
    assert all(origins[k]['kind']=='deterministically_reconstructed' for k in ('slots','families','t','query_block'))
    assert set(v['query_block']).issubset(set(range(-1,20)))


@pytest.mark.parametrize('bad',['seed','digest','family_order','q0','coverage','p0_digest','alpha'])
def test_old_mapping_rejects_tampering(bad):
    case,l,b,col,cp=fixture();b=copy.deepcopy(b);cp=copy.deepcopy(cp)
    if bad=='seed':cp['query_seed'][0]+=1
    elif bad=='digest':cp['query_sha256']='0'*64
    elif bad=='family_order':b['pairs']=b['pairs'][::-1].copy()
    elif bad=='q0':b['q0']=b['q0'].copy();b['q0'][np.flatnonzero(l.prior[l.families]>0)[0]]+=.01
    elif bad=='coverage':col['coverage']=col['coverage'].copy();col['coverage'][0]=255
    elif bad=='p0_digest':cp['p0']['sha256']='0'*64
    else:b['alpha']=b['alpha'].copy();b['alpha'][20]+=.001
    with pytest.raises(ValueError):convert_arrays(case,b,col,cp,20)


def test_nested_scalar_original_CF_and_W1_replay_independent_algorithm():
    case,l,b,col,cp=fixture();pf=joint_profile(l.types,case['clean'],case['space'].encode(case['clean'])[0],l.pairs)
    g=MultiScaleGuide(l,freeze(pf,l,b['alpha']));s=19
    degrees=np.repeat(np.asarray(pf['groups']['3']['rows'])[:,8],pf['groups']['3']['counts'])
    # The old helper's unused guide-bank report requires <=.5. The independent
    # held-out bank is still the actual old source constant, not new guide omega.
    report=old.diagnose_genre_arrays(b['q0'],l.types,l.slots,l.families,l.pairs,l.prior,g.alpha[s],degrees,np.zeros((32,10)))
    w,path=extract_frequency(report,l)
    assert path==['independent_diagnostic_frequencies','omega'] and w.shape==(64,10)
    assert np.count_nonzero(w[:,np.arange(10)!=8])==0
    spec=dict(Genre_type=3,omega64=w.tolist(),CF_normalizer=1)
    metrics=dict(Genre64=genre64(b['q0'],g,s,spec),expected_degree_PMF_W1=degree_w1(b['q0'],g,s,only_types={'3'}))
    evidence=baseline_alignment(metrics,report)
    assert evidence['passed'] and evidence['CF_error']<1e-13 and evidence['W1_error']<1e-12
    # The conditional-variance metric is deliberately not substituted.
    assert report['independent_diagnostic_frequencies']['mean_expected_squared_empirical_cf_distance']>report['independent_diagnostic_frequencies']['mean_squared_mean_cf_distance']
    wrong=copy.deepcopy(report);wrong['movie_genre_family']=7
    with pytest.raises(ValueError):extract_frequency(wrong,l)
    wrong=copy.deepcopy(report);wrong['independent_diagnostic_frequencies']['omega']=[[1.]]*64
    with pytest.raises(ValueError):extract_frequency(wrong,l)
    report['w1_between_expected_empirical_laws']+=.1
    with pytest.raises(ValueError,match='alignment failed'):baseline_alignment(metrics,report)


def test_source_hash_and_final_IO_checks_reject_without_partial_commit(tmp_path):
    p=tmp_path/'source.json';p.write_text('{}')
    with pytest.raises(ValueError,match='pinned file'):read_pinned(p,'0'*64)
    state=dict(calls=0)
    def stop():
        state['calls']+=1
        if state['calls']>2:raise InterruptedError('STOP during save')
    with pytest.raises(InterruptedError):atomic_json(tmp_path/'result.json',{'x':list(range(20))},stop)
    assert not (tmp_path/'result.json').exists() and not (tmp_path/'result.json.tmp').exists()
    p.write_bytes(b'x'*2000000)
    state['calls']=0
    with pytest.raises(InterruptedError):checked_sha(p,stop)
    atomic_npz(tmp_path/'saved.npz',{'x':np.arange(3)},lambda:None)
    with pytest.raises(ValueError,match='overwrite'):atomic_npz(tmp_path/'saved.npz',{'x':np.arange(3)},lambda:None)


def test_budget_restores_handlers_and_stop_precedes_commit(tmp_path):
    prior={s:signal.getsignal(s) for s in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP)}
    b=Budget(time.monotonic()+20,tmp_path/'STOP');b.install();(tmp_path/'STOP').touch()
    try:
        with pytest.raises(InterruptedError):b.check()
    finally:b.restore()
    assert all(signal.getsignal(s)==v for s,v in prior.items())


@pytest.mark.parametrize('action',['short','export-legacy','cost'])
def test_completed_label_alone_cannot_pass_evidence_gate(tmp_path,action):
    status={'short':'complete','export-legacy':'exported','cost':'completed_synthetic_engineering_cost'}[action]
    (tmp_path/'summary.json').write_text(json.dumps(dict(status=status,input_hash_verification_complete=True)))
    with pytest.raises((ValueError,FileNotFoundError)):validate_result(action,tmp_path,lambda:None)


def test_discover_actual_metadata_not_folder_names_and_reject_duplicate(tmp_path):
    ledger={};times=(500,20,1)
    for arm in 'AB':
        p=tmp_path/('unexpected_'+arm);p.mkdir();(p/'provenance.json').write_text(json.dumps(dict(arm=arm)))
        ledger[str((p/'provenance.json').relative_to(tmp_path))]=checked_sha(p/'provenance.json')
        for t in times:
            q=p/('arbitrary_'+str(t));q.mkdir()
            for name in ('baseline.npz','collected.npz','genre_baseline.json','completed.json','collection.json'):
                (q/name).write_text(json.dumps(dict(time=t)))
                ledger[str((q/name).relative_to(tmp_path))]=checked_sha(q/name)
    got=discover(tmp_path,ledger);assert len(got)==6
    p=tmp_path/'unexpected_A'/'extra';p.mkdir()
    for name in ('baseline.npz','collected.npz','genre_baseline.json','completed.json','collection.json'):
        (p/name).write_text('{"time":500}');ledger[str((p/name).relative_to(tmp_path))]=checked_sha(p/name)
    with pytest.raises(ValueError,match='duplicate'):discover(tmp_path,ledger)


def test_short_six_labels_without_case_file_rejected(tmp_path):
    summary=dict(status='complete',input_hash_verification_complete=True,input_hashes_unchanged=True,
                 completed_records=6,records=[dict(legacy_baseline_alignment=dict(passed=True))]*6)
    (tmp_path/'summary.json').write_text(json.dumps(summary))
    with pytest.raises(FileNotFoundError):validate_result('short',tmp_path,lambda:None)


def test_export_complete_manifest_without_array_rejected(tmp_path):
    manifest=dict(records=[dict(npz='absent.npz',npz_sha256='0'*64,metadata='absent.json',metadata_sha256='0'*64)]*6)
    (tmp_path/'six_states.json').write_text(json.dumps(manifest));(tmp_path/'original_evaluation.json').write_text('{}')
    summary=dict(status='exported',input_hash_verification_complete=True,input_hashes_unchanged=True,records=[{}]*6,
                 manifest_sha256=checked_sha(tmp_path/'six_states.json'),evaluation_sha256=checked_sha(tmp_path/'original_evaluation.json'))
    (tmp_path/'summary.json').write_text(json.dumps(summary))
    with pytest.raises(FileNotFoundError):validate_result('export-legacy',tmp_path,lambda:None)
