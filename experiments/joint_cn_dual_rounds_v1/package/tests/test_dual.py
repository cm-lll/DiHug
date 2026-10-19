import copy
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
from collections import Counter
import numpy as np
import pytest
import torch
from hsgd.bw_preflight import common as c
from hsgd.runtime import reverse_step
from hsgd.kernel import MarginalKernel
from hsgd.fusion_checks import rng_equal
from dual.data import ROOT,load,inventory,risk,digest
from dual.io import configure,optimizer,binding,save_checkpoint,load_checkpoint,protect
from dual.engine import run_train,sample,preflight,validate_fresh,verify_completed_sample
from dual.fixture import fixture
from dual.validation import panel,run as validate
from dual.metrics import paths,graph,compare,canon

@pytest.mark.parametrize('name,n,m,types',[('IMDB',1653,4079,['Director','Actor','Movie','Genre']),('PubMed',1565,12284,['Gene','Disease','Chemical','Species'])])
def test_complete_data_switch(name,n,m,types):
    case,cfg=load(name);i=inventory(case)
    assert i['nodes']==n and i['edges']==m and cfg['data']['type_names']==types
    assert case['counts'].sum()==m and sum(i['slot_capacities'])==n*(n-1)//2
    assert cfg['model']['cn_hidden']==32 and not cfg['role_guidance'] and not cfg['postprocessing']
    assert cfg['max_updates']==12000 and cfg['evaluation_updates']==[3000,6000,9000,10000,12000]
    assert list(range(1000,12001,cfg['checkpoint_every_updates']))==list(range(1000,12001,1000))
    assert cfg['optimizer']['amsgrad'] is False and cfg['scheduler'] is None
    assert risk(case)['forward_max']<cfg['model']['max_wedges']

def test_configs_seeds_and_directories_isolated():
    _,a=load('IMDB');_,b=load('PubMed')
    for k in ['train_seed','validation_seed','sample_seed']:assert a[k]!=b[k]
    assert a['train_seed'] not in [2026101407,1603]
    assert a['data']['sha256']!=b['data']['sha256']
    from launch_pair import validate
    jobs=validate(1,0,Path('/tmp/example_pair'))
    assert jobs[0][0]=='PubMed' and jobs[0][1]==0 and jobs[1][1]==1 and jobs[0][2]!=jobs[1][2]
    with pytest.raises(ValueError):validate(0,0,Path('/tmp/example_pair'))

def test_fresh_rejects_existing_and_source(tmp_path):
    with pytest.raises(ValueError):validate_fresh(tmp_path,True)
    with pytest.raises(ValueError):validate_fresh(tmp_path/'new',False)
    before=list(ROOT.iterdir())
    with pytest.raises(ValueError):validate_fresh(ROOT/'forbidden',True)
    assert list(ROOT.iterdir())==before

def test_checkpoint_atomic_immutable_binding_restore(tmp_path):
    case,cfg=fixture();env=configure('cpu',False);c.seed_all(7)
    model=c.make_model(case,cfg,'cpu');opt=optimizer(model,cfg);rng=np.random.default_rng(7);b=binding(case,cfg,env)
    r=save_checkpoint(tmp_path,model,opt,rng,'cpu',0,b)
    assert r['completed_updates']==0 and (tmp_path/'milestones.json').exists()
    with pytest.raises(ValueError):save_checkpoint(tmp_path,model,opt,rng,'cpu',0,b)
    wrong=dict(b,backend='wrong')
    with pytest.raises(ValueError):load_checkpoint(tmp_path,model,opt,rng,'cpu',wrong)
    load_checkpoint(tmp_path,model,opt,rng,'cpu',b,0)
    assert all(not p.name.endswith('.tmp') for p in tmp_path.rglob('*'))

def test_validation_full_coverage_prior_and_rng():
    case,cfg=fixture();configure('cpu',False);c.seed_all(91);rng=np.random.default_rng(91)
    model=c.make_model(case,cfg,'cpu');inputs=panel(case,cfg);before=c.rng_save(rng,'cpu')
    result=validate(model,case,cfg,'cpu',inputs);rng_equal(before,c.rng_save(rng,'cpu'))
    for r in result['times']:
        assert np.allclose(np.array(r['count_by_relation_label']).sum(1)[case['active']],case['space'].counts[case['active']])
        assert np.isfinite(r['constant_prior_ce']) and len(r['expected_edges'])==case['n_families']

def enumerate_paths(types,edges,fam):
    adj=[{} for _ in types]
    for (a,b),r in zip(edges.T,fam):adj[a][b]=r;adj[b][a]=r
    counters=[Counter(),Counter(),Counter()]
    def dfs(nodes,sequence):
        k=len(nodes)-1
        if k and nodes[0]<nodes[-1]:counters[k-1][canon(sequence)]+=1
        if k==3:return
        for v,r in adj[nodes[-1]].items():
            if v not in nodes:dfs(nodes+[v],sequence+[int(r),int(types[v])])
    for u in range(len(types)):dfs([u],[int(types[u])])
    return counters

@pytest.mark.parametrize('edges',[
    [[0,1,2],[1,2,3]],[[0,0,0],[1,2,3]],[[0,0,1],[1,2,2]],
    [[0,0,1,2,2],[1,2,2,3,4]],[[],[]]])
def test_exact_metapaths_independent_dfs(edges):
    types=np.array([0,1,0,1,2]);pairs=tuple(itertools.combinations_with_replacement(range(3),2))
    case=c.make_case(types,np.asarray(edges,dtype=np.int64),pairs);f=case['space'].encode(case['clean'])[0]
    a,_,_=paths(types,case['clean'],f);b=enumerate_paths(types,case['clean'],f)
    assert a==b
    g=graph(case,case['clean'],['a','b','c']);assert compare(g,g)['mp_frequency_TV']==[0,0,0]

def test_metrics_relabel_and_imdb_closure():
    case=c.make_case(np.array([0,1,2,2,3]),np.array([[0,1,0,1,0],[2,2,3,3,1]]),tuple(itertools.combinations_with_replacement(range(4),2)))
    g=graph(case,case['clean'],['Director','Actor','Movie','Genre'])
    assert g['imdb_closure']['path_numerator']==2 and g['imdb_closure']['path_denominator']==2
    assert g['imdb_closure']['unique_pair_numerator']==1
    perm=np.array([1,4,3,0,2]);types=np.empty(5,int);types[perm]=case['space'].types
    edges=np.sort(perm[case['clean']],axis=0)
    other=c.make_case(types,edges,case['space'].pairs);h=graph(other,edges,['Director','Actor','Movie','Genre'])
    assert g==h

def test_stop_and_resume_preserves_count(tmp_path):
    case,cfg=fixture();out=tmp_path/'training'
    assert run_train(case,cfg,out,'cpu',fresh=True,accept_gap=True)
    before=json.loads((out/'milestones.json').read_text())
    assert set(before)=={'0','1','2','3','4'}
    assert run_train(case,cfg,out,'cpu',resume=True,accept_gap=True)
    assert before==json.loads((out/'milestones.json').read_text())
    (out/'STOP').touch()
    with pytest.raises(ValueError):run_train(case,cfg,out,'cpu',resume=True,accept_gap=True)

def test_real_engineering_entry_and_short_chain(tmp_path):
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(ROOT))
    out=tmp_path/'run';target=tmp_path/'sample'
    for args in [('train',str(out)),('sample',str(target),'--run',str(out))]:
        r=subprocess.run([sys.executable,'-m','dual.fixture',*args],cwd=ROOT,env=env,capture_output=True,text=True,timeout=120)
        assert r.returncode==0,r.stdout+r.stderr
    summary=json.loads((target/'summary.json').read_text());assert summary['status']=='completed' and summary['completed_steps']==4
    evaluation=json.loads((target/'evaluation.json').read_text())
    assert evaluation['graph']['postprocessing'] is False and evaluation['single_graph_exploratory']
    with np.load(target/'graph.npz') as z:assert len(z['types'])==8
    case,cfg=fixture();assert verify_completed_sample(case,cfg,out,target,4)
    (target/'evaluation.json').write_text('{}')
    with pytest.raises((ValueError,KeyError)):verify_completed_sample(case,cfg,out,target,4)

def test_interrupted_training_then_actual_resume_and_sampling_restore(tmp_path,monkeypatch):
    case,cfg=fixture();full=tmp_path/'full';split=tmp_path/'split'
    assert run_train(case,cfg,full,'cpu',fresh=True,accept_gap=True)
    original=c.update;calls=[]
    def partial(*a,**kw):
        result=original(*a,**kw);calls.append(1)
        if len(calls)==2:(split/'STOP').touch()
        return result
    monkeypatch.setattr(c,'update',partial)
    assert not run_train(case,cfg,split,'cpu',fresh=True,accept_gap=True)
    assert json.loads((split/'training_summary.json').read_text())['completed_updates']==2
    (split/'STOP').unlink();monkeypatch.setattr(c,'update',original)
    assert run_train(case,cfg,split,'cpu',resume=True,accept_gap=True)
    a=c.trusted_load(full/'checkpoints'/'u000004.pt');b=c.trusted_load(split/'checkpoints'/'u000004.pt')
    assert c.tree_error(a['model'],b['model'])==0 and c.tree_error(a['optimizer'],b['optimizer'])==0
    rng_equal(a['rng'],b['rng'])
    assert a['binding']['run_id']!=b['binding']['run_id']
    assert sample(case,cfg,full,tmp_path/'sfull',4,'cpu')
    assert not sample(case,cfg,full,tmp_path/'ssplit',4,'cpu',engineering=True,max_steps=2)
    assert sample(case,cfg,full,tmp_path/'ssplit',4,'cpu',resume=True)
    sa=c.trusted_load(tmp_path/'sfull'/'state.pt');sb=c.trusted_load(tmp_path/'ssplit'/'state.pt')
    assert np.array_equal(sa['state'],sb['state']);rng_equal(sa['rng'],sb['rng'])

def test_strict_restore_gate(tmp_path):
    case,cfg=fixture();preflight(case,cfg,tmp_path/'gate','cpu')
    r=json.loads((tmp_path/'gate'/'gate.json').read_text());assert r['status']=='passed' and r['parameter_error']==0

def test_one_step_interruption_rng_and_state():
    case,cfg=fixture();configure('cpu',True);model=c.make_model(case,cfg,'cpu');model.eval()
    rng=np.random.default_rng(20);state=np.zeros(case['space'].total,np.uint8);before=c.rng_save(rng,'cpu')
    with pytest.raises(InterruptedError):reverse_step(model,case,state,MarginalKernel(case['prior'],c.schedule(4)),4,3,23,rng,'cpu',interrupt_block=1)
    rng_equal(before,c.rng_save(rng,'cpu'));assert not state.any()

def test_checkpoint_requires_exact_requested_milestone(tmp_path):
    case,cfg=fixture();configure('cpu',False);model=c.make_model(case,cfg,'cpu')
    with pytest.raises(ValueError):sample(case,cfg,tmp_path,tmp_path/'s',3,'cpu')

def test_no_role_import_in_production():
    for path in (ROOT/'dual').glob('*.py'):
        if path.name=='fixture.py':continue
        text=path.read_text();assert 'import role_' not in text and 'from role_' not in text

def test_frozen_test_set_is_nonempty_and_covers_core():
    ids=json.loads((ROOT/'expected_tests.json').read_text())
    assert len(ids)==59 and len(set(ids))==59
    assert {x.split('::')[0] for x in ids}=={'tests/test_contract.py','tests/test_fusion.py',
        'tests/test_loss_response.py','tests/test_dual.py'}
