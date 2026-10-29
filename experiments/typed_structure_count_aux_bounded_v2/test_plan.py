import json,hashlib,math
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def read(name):return json.loads((ROOT/name).read_text())

def test_eight_paired_tasks_no_chain():
    p=read('stage500_plan.json');tasks=p['tasks']
    assert len(tasks)==len({t['id'] for t in tasks})==8
    assert sorted(t['slot'] for t in tasks)==list(range(8))
    pairs={}
    for t in tasks:
        pairs.setdefault(t['paired_initial_id'],set()).add(t['arm'])
        assert t['updates']==500 and t['generation_chains']==0
    assert len(pairs)==4 and all(v=={'CE','CE_COUNT'} for v in pairs.values())
    assert p['stage1_generation_tasks']==[] and not p['formal_authorized'] and not p['resume_authorized']

def test_sparse_observations_and_review_boundaries():
    p=read('stage500_plan.json')
    assert p['gradient_observation_updates']==[1,250,500]
    assert p['fixed_validation_updates']==[0,100,250,500]
    assert p['fixed_validation_times']==[1,500,1000]
    assert p['validation_readouts']==['raw','ema_0.99']
    assert 'one seed temporarily worse' in p['analysis_not_automatic_failures']

def test_budget_arithmetic_queue():
    b=read('budget.json');s=b['stage500']
    assert s['total_normal2_updates']==4000 and s['full_chains']==0
    assert s['compute_wall_seconds']==max(x['serial_seconds'] for x in s['queue'])
    assert s['recommended_budget_seconds']>=s['planned_total_seconds']
    ids={x['id'] for x in s['queue']}
    for count,v in b['alternatives_by_card_count'].items():
        flat=[a for q in v['queue'] for a in q]
        assert len(flat)==8 and set(flat)==ids and len(v['queue'])==int(count)
    f=b['followup_500_to_3000'];assert f['additional_updates']==20000 and f['full_chains']==16 and not f['authorized_now']

def test_source_evidence_and_formula_binding():
    p=read('provenance.json')
    assert p['formula_sha256']==hashlib.sha256((ROOT/'formula.json').read_bytes()).hexdigest()
    assert p['new_loss_sha256']==hashlib.sha256((ROOT/'package/count_aux.py').read_bytes()).hexdigest()
    for n,h in p['source_tables'].items():
        assert hashlib.sha256((ROOT/'evidence'/n).read_bytes()).hexdigest()==h
    assert p['old_scientific_files']==read('frozen_science_bindings.json')
