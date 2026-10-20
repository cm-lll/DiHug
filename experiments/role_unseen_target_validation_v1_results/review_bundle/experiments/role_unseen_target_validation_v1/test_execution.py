"""Engineering gates only; zero scientific chains, no training/GPU."""
from pathlib import Path
from types import SimpleNamespace
import json
import time
import sys
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parent))
from execute import (new_output, save, budget_check, lazy, target_objects, metadata_case,
                     verify_inputs, source_files, check_sources, chain, PLAN, cfg)
from analyse import w1, target_errors, stat, audit, independently_check

RUN=Path('/data2/lyh/gnn_project/DiHug/results/role_robust_training_6000_20261021_v1/learning_run01')
PANEL=Path('/data2/lyh/gnn_project/DiHug/results/role_late_paired_20261020_run01')


def test_protected_output_before_writes(tmp_path):
    old=tmp_path/'input';old.mkdir();(old/'evidence').write_text('unchanged')
    with pytest.raises(ValueError):new_output(old/'bad',[old])
    assert sorted(p.name for p in old.iterdir())==['evidence']


def test_existing_output_no_reactivation(tmp_path):
    with pytest.raises(FileExistsError):new_output(tmp_path,[])


def test_budget_STOP_deadline(tmp_path):
    save(tmp_path/'activation.json',dict(deadline_epoch=time.time()+100))
    check=budget_check(tmp_path);check();(tmp_path/'STOP').touch()
    with pytest.raises(InterruptedError):check()
    (tmp_path/'STOP').unlink()
    save(tmp_path/'activation.json',dict(deadline_epoch=time.time()-1))
    with pytest.raises(TimeoutError):budget_check(tmp_path)()


def test_exact_frozen_metadata_no_clean_edges():
    c=lazy();targets=target_objects(c);case=metadata_case(c,PANEL,targets)
    records,_=c.load_panel(PANEL);old=records[0]['case']
    assert not {'clean','clean_ids'} & case.keys()
    assert np.array_equal(case['space'].types,old['space'].types)
    assert np.array_equal(case['counts'],old['counts']) and np.array_equal(case['prior'],old['prior'])
    for seed in (2026102031,2026102038):
        for t in (64,32,1):
            for b in range(4):
                for a,z in zip(c.c.plan_for(case,[seed,t,61],4).block(b),c.c.plan_for(old,[seed,t,61],4).block(b)):
                    assert np.array_equal(a,z)


def test_frozen_weights_sources_environment_and_strict_load():
    c=lazy();expected,actual=verify_inputs(SimpleNamespace(run=RUN,panel=PANEL),c)
    case=metadata_case(c,PANEL,target_objects(c))
    model=c.CapacityDenoiser(4,10,family_pairs=case['space'].pairs,prior=case['prior'],T=64,**expected['full_resolved_config']['model'])
    for w in expected['weights']:
        state=c.torch.load(w['path'],map_location='cpu',weights_only=True)
        model.load_state_dict(state,strict=True)
        assert all(c.torch.equal(v,model.state_dict()[k]) for k,v in state.items())
    assert not c.torch.cuda.is_initialized()


def test_targets_unchanged_joint_inventory():
    c=lazy();targets=target_objects(c);base=targets['balanced']
    assert list(targets)==cfg()['target_order']
    for name,t in targets.items():
        assert t.roles[:3]==base.roles[:3] and t.type_counts==base.type_counts
        assert sum(v[8]*n for v,n in t.roles[3])==24
    assert sorted(v[8] for v,n in targets['double10'].roles[3] for _ in range(n))==[1,1,1,1,10,10]


def test_full_source_binding_tamper(tmp_path):
    save(tmp_path/'source_manifest.json',source_files());check_sources(tmp_path)
    save(tmp_path/'source_manifest.json',{'role_unseen_target_validation_v1/config.json':'0'*64})
    with pytest.raises(ValueError):check_sources(tmp_path)


@pytest.mark.parametrize('a,b,expected',[([0,2],[0,2],0),([2],[0,0,2],2),([1,1],[0,1,1],1)])
def test_W1(a,b,expected):assert w1(a,b)==expected


def test_statistics_NA_and_P90():
    r=stat([0,1,2,None]);assert r['mean']==1 and r['NA']==1 and r['P90']==1.8


def test_audit_rejects_empty(tmp_path):
    with pytest.raises(AssertionError):audit(tmp_path,lambda:None)


def test_target_statistics_and_independent_raw_graph_audit(tmp_path):
    c=lazy();records,targets=c.load_panel(PANEL)
    from role_condition.metrics import graph
    case=records[0]['case'];edges=case['clean'];fam=case['space'].encode(edges)[0]
    metrics=graph(case,edges,c.NAMES)
    np.savez_compressed(tmp_path/'fixture.npz',types=case['space'].types,edges=edges.T,
                        relations=fam+1,schema=np.asarray(case['space'].pairs))
    target=targets[records[0]['condition']].as_dict();err=target_errors(metrics,target)
    assert all(err[name]['joint_TV']==0 and not any(err[name]['relation_W1']) for name in c.NAMES)
    row=dict(graph_file='fixture.npz',metrics=metrics,target_errors=err)
    independently_check(row,tmp_path,target)
    row['metrics']['family_edges'][8]+=1
    with pytest.raises(AssertionError):independently_check(row,tmp_path,target)


def test_generation_never_reads_proof_graphs_or_old_graphs():
    source=(Path(__file__).parent/'execute.py').read_text()
    assert 'np.load(' not in source and 'load_panel(' not in source
    assert 'normal2_final_6000.pt' in source
    assert 'optimizer=' not in source


def test_driver_complete_steps_and_random_contract_with_engineering_stub():
    c=lazy();case=metadata_case(c,PANEL,target_objects(c));calls=[]
    class Stub:
        def eval(self):pass
        def set_target(self,target):self.target=target
    def reverse(model,case,state,kernel,t,rounds,seed,rng,device,check_time):
        calls.append(t);rng.random(case['space'].total)
        return state.copy(),dict(exact_coverage=True,blocks=4,covered_slots=312,step_seconds=0.)
    fake=SimpleNamespace(**{k:getattr(c,k) for k in ('c','sha','uniform_contract','MarginalKernel','torch')},reverse_step=reverse)
    original=case['prior'].copy()
    state,steps,initial=chain(Stub(),case,target_objects(c)['double10'],2026102031,fake,lambda:None,lambda *args:None)
    assert calls==list(range(64,0,-1)) and len(steps)==64 and np.array_equal(original,case['prior'])
    assert all(s['exact_coverage'] for s in steps)

