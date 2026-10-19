"""Synthetic CPU fixtures only. Device identities below are logic mocks, NOT BW tests."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import time
import os
import numpy as np
import pytest
import torch
from hsgd.bw_preflight import common as c
from dual import io,transactions,migration,rng as owned
from dual.fixture import fixture
from dual.engine import run_train,sample
from dual.supervisor import supervise

def setup():
    case,cfg=fixture();env=io.configure('cpu',False);owned.seed_all(74)
    model=c.make_model(case,cfg,'cpu');opt=io.optimizer(model,cfg);r=np.random.default_rng(74)
    bind=io.binding(case,cfg,env)
    return case,cfg,model,opt,r,bind

@pytest.mark.parametrize('stage',['prepared','latest.json','milestones.json'])
def test_interrupted_publications_repair_and_selected_evaluation_load(tmp_path,monkeypatch,stage):
    case,cfg,model,opt,r,bind=setup();original=io.atomic_json;publish=transactions.publish
    def fail(path,obj):
        if Path(path).name==stage:raise InterruptedError('synthetic publication fault')
        original(path,obj)
    if stage=='prepared':
        monkeypatch.setattr(transactions,'publish',lambda *a,**kw:(_ for _ in ()).throw(InterruptedError('after durable intent')))
    else:monkeypatch.setattr(io,'atomic_json',fail)
    with pytest.raises(InterruptedError):io.save_checkpoint(tmp_path,model,opt,r,'cpu',3000,bind)
    # 3000 is synthetic metadata here, not 3000 real optimizer updates.
    assert (tmp_path/'checkpoints/u003000.pt').exists()==(stage!='prepared')
    assert (tmp_path/'latest.json').exists()==(stage=='milestones.json')
    monkeypatch.setattr(io,'atomic_json',original);monkeypatch.setattr(transactions,'publish',publish)
    ck,receipt=io.load_checkpoint(tmp_path,model,opt,r,'cpu',bind,update=3000)
    assert ck['completed_updates']==3000 and c.sha(tmp_path/receipt['path'])==receipt['sha256']
    assert json.loads((tmp_path/'milestones.json').read_text())['3000']==receipt
    before=(tmp_path/receipt['path']).read_bytes()
    assert io.save_checkpoint(tmp_path,model,opt,r,'cpu',3000,bind)==receipt
    assert before==(tmp_path/receipt['path']).read_bytes()

@pytest.mark.parametrize('alter',['model','optimizer','rng','binding','update','receipt'])
def test_conflicting_content_not_overwritten(tmp_path,alter):
    case,cfg,model,opt,r,bind=setup();receipt=io.save_checkpoint(tmp_path,model,opt,r,'cpu',0,bind)
    path=tmp_path/receipt['path'];ck=c.trusted_load(path)
    if alter=='model':ck['model'][next(iter(ck['model']))].reshape(-1)[0]+=1
    elif alter=='optimizer':ck['optimizer']['param_groups'][0]['eps']=1e-4
    elif alter=='rng':ck['rng']['torch'][0]^=1
    elif alter=='binding':ck['binding']=dict(bind,backend='tampered')
    elif alter=='update':ck['completed_updates']=3
    if alter=='receipt':
        x=json.loads((tmp_path/'latest.json').read_text());x['sha256']='0'*64
        (tmp_path/'latest.json').write_text(json.dumps(x))
    else:
        c.torch_save(path,ck)
        # Even if a byte receipt were changed, full original semantic hash still detects it.
        j=tmp_path/'transactions/u000000.pt.json';intent=json.loads(j.read_text());intent['receipt']['sha256']=c.sha(path)
        j.write_text(json.dumps(intent))
    before=path.read_bytes()
    with pytest.raises(ValueError):io.load_checkpoint(tmp_path,model,opt,r,'cpu',bind)
    assert before==path.read_bytes()

def test_idempotence_rejects_different_in_memory_snapshot(tmp_path):
    _,_,model,opt,r,b=setup();receipt=io.save_checkpoint(tmp_path,model,opt,r,'cpu',0,b)
    with torch.no_grad():next(model.parameters()).add_(.1)
    before=(tmp_path/receipt['path']).read_bytes()
    with pytest.raises(ValueError):io.save_checkpoint(tmp_path,model,opt,r,'cpu',0,b)
    assert (tmp_path/receipt['path']).read_bytes()==before

def test_existing_file_without_intent_needs_complete_known_snapshot(tmp_path):
    _,_,model,opt,r,b=setup();receipt=io.save_checkpoint(tmp_path,model,opt,r,'cpu',0,b)
    (tmp_path/'transactions/u000000.pt.json').unlink()
    (tmp_path/'latest.json').unlink();(tmp_path/'milestones.json').unlink()
    before=(tmp_path/receipt['path']).read_bytes()
    with pytest.raises(ValueError):io.load_checkpoint(tmp_path,model,opt,r,'cpu',b)
    recovered=io.save_checkpoint(tmp_path,model,opt,r,'cpu',0,b)
    assert recovered['sha256']==receipt['sha256'] and before==(tmp_path/receipt['path']).read_bytes()
    io.load_checkpoint(tmp_path,model,opt,r,'cpu',b,update=0)

@pytest.mark.parametrize('u',[0,2])
def test_real_small_training_entry_resumes_publication_failure(tmp_path,monkeypatch,u):
    case,cfg=fixture();out=tmp_path/'run';original=io.atomic_json
    def fault(path,obj):
        if Path(path).name=='milestones.json' and str(u) in obj:raise InterruptedError('after latest')
        original(path,obj)
    monkeypatch.setattr(io,'atomic_json',fault)
    with pytest.raises(InterruptedError):run_train(case,cfg,out,'cpu',fresh=True,accept_gap=True)
    monkeypatch.setattr(io,'atomic_json',original)
    assert run_train(case,cfg,out,'cpu',resume=True,accept_gap=True)
    assert set(json.loads((out/'milestones.json').read_text()))=={'0','1','2','3','4'}
    assert sample(case,cfg,out,tmp_path/'graph',2,'cpu')

def identities(bind):
    a=copy.deepcopy(bind);a['device_identity']=dict(device_index=0,device_uuid='synthetic-A',visibility={'CUDA_VISIBLE_DEVICES':'4'})
    a['runtime']['device_name']='synthetic-A';b=copy.deepcopy(a)
    b['device_identity']=dict(device_index=1,device_uuid='synthetic-B',visibility={'CUDA_VISIBLE_DEVICES':'1,7'})
    b['runtime']['device_name']='synthetic-B'
    return a,b

def test_migration_is_explicit_and_source_preserved(tmp_path):
    _,_,model,opt,r,b=setup();a,d=identities(b)
    io.atomic_json(tmp_path/'binding.json',a);original=(tmp_path/'binding.json').read_bytes()
    receipt=io.save_checkpoint(tmp_path,model,opt,r,'cpu',1,a)
    with pytest.raises(ValueError):io.load_checkpoint(tmp_path,model,opt,r,'cpu',a,execution_binding=d)
    io.load_checkpoint(tmp_path,model,opt,r,'cpu',a,execution_binding=d,allow_device_migration=True)
    assert (tmp_path/'binding.json').read_bytes()==original
    assert c.sha(tmp_path/receipt['path'])==receipt['sha256']
    records=list((tmp_path/'migrations').glob('*.json'));assert len(records)==1
    rec=json.loads(records[0].read_text());assert not rec['bitwise_cross_device_reproducibility'] and rec['rng_restored']
    io.save_checkpoint(tmp_path,model,opt,r,'cpu',2,a,execution_binding=d)
    io.load_checkpoint(tmp_path,model,opt,r,'cpu',a,execution_binding=d) # same migrated device now strict

@pytest.mark.parametrize('field',['source_commit','release_sha256','config_sha256','data_sha256','backend','python_version','torch_runtime_version','packages','deterministic_algorithms'])
def test_migration_never_relaxes_science_or_software(tmp_path,field):
    _,_,model,opt,r,b=setup();a,d=identities(b);io.save_checkpoint(tmp_path,model,opt,r,'cpu',0,a)
    if field in ('python_version','torch_runtime_version','packages','deterministic_algorithms'):d['runtime'][field]='tampered'
    else:d[field]='tampered'
    with pytest.raises(ValueError):io.load_checkpoint(tmp_path,model,opt,r,'cpu',a,execution_binding=d,allow_device_migration=True)
    assert not (tmp_path/'migrations').exists()

def test_fresh_start_cannot_use_migration(tmp_path):
    case,cfg=fixture()
    with pytest.raises(ValueError):run_train(case,cfg,tmp_path/'run','cpu',fresh=True,accept_gap=True,allow_device_migration=True)
    assert not (tmp_path/'run').exists()

def test_selected_card_rng_only_mocked_api(monkeypatch):
    calls=[];state=torch.tensor([1,2,3],dtype=torch.uint8)
    monkeypatch.setattr(torch.cuda,'current_device',lambda:3)
    monkeypatch.setattr(torch.cuda,'manual_seed',lambda seed:calls.append(('seed',seed)))
    monkeypatch.setattr(torch.cuda,'get_rng_state',lambda device:(calls.append(('get',device)) or state.clone()))
    monkeypatch.setattr(torch.cuda,'set_rng_state',lambda value,device:calls.append(('set',device,value.tolist())))
    def forbidden(*a,**kw):raise AssertionError('all-visible RNG API forbidden')
    for name in ('get_rng_state_all','set_rng_state_all','manual_seed_all'):monkeypatch.setattr(torch.cuda,name,forbidden)
    owned.seed_all(77,'cuda');rng=np.random.default_rng(77);saved=owned.rng_save(rng,'cuda')
    # Stored index can differ: restore targets the CURRENT selected device explicitly.
    saved['selected_device_index']=0;owned.rng_restore(saved,rng,'cuda')
    assert calls==[('seed',77),('get',3),('set',3,[1,2,3])]

def test_core_rollback_uses_selected_helpers_and_restores_globals():
    old=c.rng_save,c.rng_restore
    with owned.selected_core_rng():assert c.rng_save is owned.rng_save and c.rng_restore is owned.rng_restore
    assert (c.rng_save,c.rng_restore)==old

def test_selected_reverse_step_rolls_back_partial_step():
    from hsgd.kernel import MarginalKernel
    case,cfg,model,opt,r,bind=setup();model.eval();state=np.zeros(case['space'].total,np.uint8)
    before=owned.rng_save(r,'cpu');old=c.rng_save,c.rng_restore
    with pytest.raises(InterruptedError):owned.reverse_step(model,case,state,MarginalKernel(case['prior'],c.schedule(4)),4,3,2,r,'cpu',interrupt_block=1)
    after=owned.rng_save(r,'cpu')
    assert before['generator']==after['generator'] and torch.equal(before['torch'],after['torch'])
    assert (c.rng_save,c.rng_restore)==old and not state.any()

def test_training_and_sampling_device_migration_logic_end_to_end(tmp_path,monkeypatch):
    import dual.engine as engine
    case,cfg=fixture();out=tmp_path/'run';current=['A'];configure=engine.configure;update=c.update;calls=[]
    def logical_env(device,strict=False):
        env=configure(device,strict);env.update(device_name='logical-'+current[0],device_uuid=current[0],device_index=0,visibility={'fixture':current[0]})
        return env
    def stop_after_two(*a,**kw):
        result=update(*a,**kw);calls.append(1)
        if len(calls)==2:(out/'STOP').touch()
        return result
    monkeypatch.setattr(engine,'configure',logical_env);monkeypatch.setattr(c,'update',stop_after_two)
    assert not run_train(case,cfg,out,'cpu',fresh=True,accept_gap=True)
    original=(out/'binding.json').read_bytes();(out/'STOP').unlink();current[0]='B';monkeypatch.setattr(c,'update',update)
    with pytest.raises(ValueError):run_train(case,cfg,out,'cpu',resume=True,accept_gap=True)
    assert run_train(case,cfg,out,'cpu',resume=True,accept_gap=True,allow_device_migration=True)
    assert run_train(case,cfg,out,'cpu',resume=True,accept_gap=True) # new checkpoint records device B
    assert (out/'binding.json').read_bytes()==original
    target=tmp_path/'sample';assert not sample(case,cfg,out,target,4,'cpu',engineering=True,max_steps=2)
    current[0]='C'
    with pytest.raises(ValueError):sample(case,cfg,out,target,4,'cpu',resume=True)
    assert sample(case,cfg,out,target,4,'cpu',resume=True,allow_device_migration=True)
    assert list((target/'migrations').glob('*.json'))

@pytest.mark.parametrize('phase',['data_load','CUDA_init','device_lock'])
def test_timeout_includes_startup_stages(tmp_path,phase):
    out=tmp_path/phase
    code="import pathlib,time; p=pathlib.Path(%r);p.mkdir();(p/'phase').write_text(%r);time.sleep(10)"%(str(out),phase)
    start=time.monotonic();rc=supervise([sys.executable,'-c',code],.25,out,started=start,cleanup_seconds=.1)
    receipt=json.loads((out/'supervisor.json').read_text())
    assert rc==124 and not receipt['completed'] and (out/'phase').read_text()==phase
    assert .25<=receipt['computation_window_seconds']<1 and receipt['cleanup_seconds']<1

@pytest.mark.parametrize('mode,expected',[('success',0),('failure',7),('STOP',130)])
def test_light_supervisor_subprocess_status(tmp_path,mode,expected):
    out=tmp_path/mode
    code="import pathlib,time,sys;p=pathlib.Path(%r);p.mkdir();"%str(out)
    code+="(p/'STOP').touch();time.sleep(10)" if mode=='STOP' else "sys.exit(%d)"%(7 if mode=='failure' else 0)
    assert supervise([sys.executable,'-c',code],2,out)==expected
    receipt=json.loads((out/'supervisor.json').read_text());assert receipt['completed']==(expected==0)

def test_public_short_help_has_no_numerical_imports():
    code="""import sys,runpy,importlib.abc
class Guard(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in ('torch','numpy','scipy'):raise RuntimeError('heavy import in parent: '+fullname)
sys.meta_path.insert(0,Guard());sys.argv=['dual.cli','short-test','--help'];runpy.run_module('dual.cli',run_name='__main__')
"""
    p=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True,timeout=10)
    assert p.returncode==0,p.stderr
    assert '570s TOTAL' in p.stdout

def test_final_evidence_verification_is_inside_deadline(tmp_path):
    out=tmp_path/'late'
    def slow(out):time.sleep(.3)
    assert supervise([sys.executable,'-c','pass'],.2,out,completion_check=slow)==124
    assert not json.loads((out/'supervisor.json').read_text())['completed']
