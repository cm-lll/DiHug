from pathlib import Path
import sys,subprocess,json
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import *
from prepare import choose,ALLOWED,HELDOUT,DEVELOPMENT
from coverage_evaluation import independent

def test_frozen_split_reproducible_and_disjoint():
    wide,combo,num,positive=choose();f=read(RESULT/'conditions.json')
    assert len(wide)==12 and len(combo)==len(num)==3 and len(set(wide+combo+num))==18
    assert set(x for v in wide for x in v)==ALLOWED and not (ALLOWED&HELDOUT)
    assert not set(wide+combo+num)&DEVELOPMENT
    assert all(set(v)<=ALLOWED for v in combo)
    assert all(set(v)-ALLOWED for v in num)
    assert list(positive)==f['vectors'][f['positive_control']]
    assert len(f['final_targets'])==11

def test_clean_supervision_inventory_sources_and_shared_controls():
    records,targets=load_data();f=read(RESULT/'conditions.json')
    for arm in ('narrow','broad'):assert sum(r['arm']==arm and r['split']=='train' for r in records)==48
    assert sum(r['split']=='validation' for r in records)==24
    for r in records:
        assert base.extract_target(r['case'])==targets[r['condition']]
        assert r['case']['counts'].tolist()==[0,10,18,0,0,36,0,0,24,0]
    for name in ('balanced','extreme'):
        a=[r for r in records if r['arm']=='narrow' and r['condition']==name]
        b=[r for r in records if r['arm']=='broad' and r['condition']==name]
        assert len(a)==24 and len(b)==4
        for x,y in zip(a,b):assert np.array_equal(x['case']['clean'],y['case']['clean'])
    for name in f['training']['broad']:
        tr=[r for r in records if r['split']=='train' and r['arm']=='broad' and r['condition']==name]
        assert len(tr)==4
        for i,a in enumerate(tr):
            for b in tr[:i]:assert not base.isomorphic(a['case'],b['case'])
        for v in (r for r in records if r['split']=='validation' and r['condition']==name):
            for t in (r for r in records if r['split']=='train' and r['condition']==name):assert not base.isomorphic(v['case'],t['case'])

def test_paired_full_model_original_initialization():
    rec,tar=load_data();cfg=settings();case=rec[0]['case']
    for seed in cfg['train_seeds']:
        a,ia=make_model(case,cfg,seed);b,ib=make_model(case,cfg,seed)
        assert ia['source_sha256']==ib['source_sha256']
        assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())
        assert all(x.data_ptr()!=y.data_ptr() for x,y in zip(a.parameters(),b.parameters()))
        assert not base.opt(a,cfg).state and not base.opt(b,cfg).state
        del a,b

def test_training_sources_times_and_queries_are_paired():
    rec,tar=load_data();cfg=settings()
    for u in (0,1,12,5900):
        a,ca,ia,ra=inputs_for(rec,tar,'narrow',cfg['train_seeds'][0],u,cfg)
        b,cb,ib,rb=inputs_for(rec,tar,'broad',cfg['train_seeds'][0],u,cfg)
        for k in ('condition_uniform','augmentation_seed','noise_seeds','t','query_sha256'):assert ra[k]==rb[k]
        for inp in ia+ib:assert inp['ce_weight']==4/561

def test_independent_full_MP_pattern_and_metric_audit(tmp_path,monkeypatch):
    import coverage_evaluation
    rec,tar=load_data();r=next(r for r in rec if r['split']=='validation');case=r['case'];edges=case['clean'];f=case['space'].encode(edges)[0]
    path=tmp_path/'fixture.npz';np.savez_compressed(path,types=case['space'].types,edges=edges.T,relations=f+1,schema=np.asarray(case['space'].pairs))
    m=graph(case,edges,NAMES);row=dict(graph_file='fixture.npz',condition=r['condition'],metrics=m,target_errors=error_stats(m,tar[r['condition']].as_dict()))
    monkeypatch.setattr(coverage_evaluation,'RESULT',tmp_path);independent(row,tar)
    assert row['target_errors']['Genre']['W1']==0

def test_boolean_statistics_no_quantile_subtract_error():
    r=stat([False,True,None]);assert r['mean']==.5 and r['P90']==.9 and r['NA']==1

def test_stable_exact_integer_joint_TV():
    rec,tar=load_data();case=rec[0]['case'];m=graph(case,case['clean'],NAMES)
    e=error_stats(m,tar[rec[0]['condition']].as_dict());assert all(e[n]['joint_TV']==0 for n in NAMES)

def test_public_help_never_activates_training():
    before=(RESULT/'formal_launch.json').exists()
    r=subprocess.run([sys.executable,str(Path(__file__).with_name('run.py')),'--help'],capture_output=True,timeout=10)
    assert r.returncode==0 and b'--execute' in r.stdout
    assert (RESULT/'formal_launch.json').exists()==before

def test_no_forbidden_method_change_and_original_optimizer():
    cfg=settings();assert cfg['model']==read(EXP/'role_robust_training_v1/config.json')['model']
    assert cfg['optimizer']==read(EXP/'role_robust_training_v1/config.json')['optimizer']
    assert cfg['T']==64 and cfg['rounds']==4 and cfg['updates']==6000 and cfg['expected_chains']==528
    assert cfg['scheduler'] is None and cfg['gradient_clipping'] is None and not cfg['class_reweighting']

def test_original_correctness_receipt():
    r=read(RESULT/'correctness.json');assert r['status']=='passed' and not r['computation_error_found']
    assert len(r['validation'])==3 and all(x['maximum_validation_difference']==0 for x in r['validation'])
    assert len(r['condition_checks'])==60 and max(x['max_DP_cells'] for x in r['condition_checks'])<=200000
