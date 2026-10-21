import copy,importlib.util,json,os,signal,subprocess,sys,time,types
from pathlib import Path
import numpy as np
import pytest
import torch
import pair_core as p
import capacity_readout
from pair_model import Reader,paired_initial,create
from role_condition.model import extract_target
from hsgd.computational import FrozenTypedContext
from hsgd.vendor.layers import SparseEtoy
from run import safe_output,supervise

@pytest.fixture(autouse=True)
def cpu_setup():p.setup('cpu',True)

def test_two_patches_logits_loss_gradient_update_equivalent():
    case=p.toy();cfg=p.config();target=extract_target(case);inputs=p.input_pair(case,cfg,12,0)
    module=types.ModuleType('reference_capacity');exec((p.ROOT/'reference/capacity_readout.txt').read_text(),module.__dict__)
    oldlayers=types.ModuleType('reference_layers');exec((p.ROOT/'reference/layers.txt').read_text(),oldlayers.__dict__)
    records={};original_layer=SparseEtoy.forward
    initial=create(case,cfg,11,'role','cpu')
    with torch.no_grad():initial.role_fusion.weight.fill_(.01)
    initial=p.cpu(initial.state_dict())
    try:
        for variant in ('original','prior_patch_only','global_patch_only','both'):
            m=create(case,cfg,13,'role','cpu');m.load_state_dict(initial,strict=True);m.set_target(target)
            if variant in ('original','global_patch_only'):
                m.branch.forward=types.MethodType(module.CapacityNodeCondition.forward,m.branch)
            if variant in ('original','prior_patch_only'):
                def old(self,edge_index,edge_attr,batch,top_triu=False,num_graphs=None):return oldlayers.SparseEtoy.forward(self,edge_index,edge_attr,batch,top_triu)
                SparseEtoy.forward=old
            else:SparseEtoy.forward=original_layer
            reader=Reader().install(m);module.allocation_marginals=reader
            try:
                inp=inputs[0]
                with torch.no_grad():l,q=p.predict(m,case['space'].types,inp['edges'],inp['edge_families'],inp['query'],inp['f'],inp['t']/cfg['T'],details=True)
                result,g=p.update(m,p.optimizer(m,cfg),case,inputs,cfg,gradients=True)
                records[variant]=(l,q,result['loss'],g,p.cpu(m.state_dict()))
            finally:reader.close()
    finally:SparseEtoy.forward=original_layer
    left=records['original']
    for right in records.values():
        torch.testing.assert_close(left[0],right[0],atol=0,rtol=0);torch.testing.assert_close(left[1],right[1],atol=0,rtol=0)
        assert left[2]==right[2] and left[3].keys()==right[3].keys()
        for k in left[3]:torch.testing.assert_close(left[3][k],right[3][k],atol=0,rtol=0)
        for k in left[4]:assert torch.equal(left[4][k],right[4][k])

def test_global_target_allocation_trailing_empty_graph_and_gradients():
    module=types.ModuleType('ref_layers');exec((p.ROOT/'reference/layers.txt').read_text(),module.__dict__)
    original=module.SparseEtoy(5,3);current=SparseEtoy(5,3);current.load_state_dict(original.state_dict());ei=torch.tensor([[0,1,2],[1,0,3]]);batch=torch.tensor([0,0,1,1,2])
    a=torch.randn(3,5,requires_grad=True);b=a.detach().clone().requires_grad_()
    left=original(ei,a,batch,top_triu=True);right=current(ei,b,batch,top_triu=True,num_graphs=3)
    assert right.shape==(3,3);torch.testing.assert_close(left,right,atol=0,rtol=0)
    left.square().sum().backward();right.square().sum().backward();torch.testing.assert_close(a.grad,b.grad,atol=0,rtol=0)
    for x,y in zip(original.parameters(),current.parameters()):torch.testing.assert_close(x.grad,y.grad,atol=0,rtol=0)

def test_paired_initialization_base_no_target_or_role_compute():
    case=p.toy();cfg=p.config();a,b=paired_initial(case,cfg,1)
    assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())
    assert all(x.data_ptr()!=y.data_ptr() for x,y in zip(a.parameters(),b.parameters()))
    with pytest.raises(ValueError):a.set_target(extract_target(case))
    hook=a.branch.role_encoder.register_forward_hook(lambda *args:(_ for _ in ()).throw(AssertionError('base read target')))
    inp=p.input_pair(case,cfg,2,0)
    try:row,grads=p.update(a,p.optimizer(a,cfg),case,inp,cfg,gradients=True)
    finally:hook.remove()
    assert row['parameter_changed'] and any(k.startswith('branch.degree_encoder') for k in grads)
    assert not any(k.startswith('branch.role_encoder') for k in grads)

def test_context_prior_and_learning_cache_invalidation():
    case=p.toy();cfg=p.config();m=create(case,cfg,1,'role','cpu');m.set_target(extract_target(case));reader=Reader().install(m)
    inp=p.input_pair(case,cfg,2,0)[0];ctx=FrozenTypedContext(m,case['space'].types,inp['edges'],inp['edge_families'],'x')
    try:
        assert not ctx.prior.flags.writeable
        with torch.no_grad():ctx.predict(inp['query'],inp['f'],inp['t']/cfg['T'],'x')
        old=ctx.node_residual
        with torch.no_grad():ctx.predict(inp['query'],inp['f'],inp['t']/cfg['T'],'x')
        assert ctx.node_residual is old
        with torch.no_grad():m.prior.add_(0)
        with pytest.raises(ValueError,match='metadata'):ctx.check('x')
    finally:ctx.close();reader.close()
    with pytest.raises(ValueError):ctx.check('x')

def test_prior_object_replacement_rejected_before_cached_reuse():
    case=p.toy();m=create(case,p.config(),1,'base','cpu');ctx=FrozenTypedContext(m,case['space'].types,case['clean'],case['space'].encode(case['clean'])[0],'prior')
    try:
        m.prior=m.prior.clone()
        with pytest.raises(ValueError,match='prior tensor replaced'):ctx.check('prior')
    finally:ctx.close()

def test_selected_card_rng_never_calls_all_visible(monkeypatch):
    calls=[];state=torch.get_rng_state()
    monkeypatch.setattr(torch.cuda,'get_rng_state',lambda d:calls.append(('get',d)) or state.clone())
    monkeypatch.setattr(torch.cuda,'set_rng_state',lambda s,d:calls.append(('set',d)))
    monkeypatch.setattr(torch.cuda,'get_rng_state_all',lambda:(_ for _ in ()).throw(AssertionError('all cards')))
    monkeypatch.setattr(torch.cuda,'set_rng_state_all',lambda s:(_ for _ in ()).throw(AssertionError('all cards')))
    rng=np.random.default_rng(1);saved=p.rng_save(rng,'cuda');p.rng_restore(saved,rng,'cuda');assert calls==[('get',0),('set',0)]

def test_checkpoint_orphan_recovery_conflict_and_binding(tmp_path):
    value={'binding':{'code':'a'},'updates':1000,'model':torch.tensor([1.,2.]),'rng':p.rng_save(np.random.default_rng(1),'cpu')}
    p.checkpoint(tmp_path,value);(tmp_path/'latest.json').unlink();loaded=p.load_checkpoint(tmp_path,{'code':'a'});assert loaded['updates']==1000
    p.checkpoint(tmp_path,value)
    bad=dict(value,model=torch.tensor([9.,9.]))
    with pytest.raises(ValueError):p.checkpoint(tmp_path,bad)
    with pytest.raises(ValueError):p.load_checkpoint(tmp_path,{'code':'b'})

def test_validation_preserves_rng_and_same_twenty_block_panel():
    case=p.toy();cfg=p.config();cfg=dict(cfg,validation_times=[20]);m=create(case,cfg,3,'role','cpu');target=extract_target(case);reader=Reader().install(m)
    rng=np.random.default_rng(5);before=p.digest(p.rng_save(rng,'cpu'))
    try:r=p.validate(m,case,target,cfg,lambda:None)
    finally:reader.close()
    assert p.digest(p.rng_save(rng,'cpu'))==before and r['RNG_restored'] and len(r['rows'])==1

def test_failed_forecast_and_fixture_cannot_enable_formal():
    from formal import forecast_check
    with pytest.raises(ValueError):forecast_check({'status':'passed_engineering_fixture'},p.config(),1000)
    with pytest.raises(ValueError):forecast_check({'status':'passed_target_GPU','target_GPU_gate_passed':True,'full_IMDB_cost_complete':True,'checks':[]},p.config(),1000)

def test_path_rejection_before_any_write(tmp_path):
    before=set(p.ROOT.rglob('*'))
    with pytest.raises(ValueError):safe_output(p.ROOT/'data'/'bad-output')
    assert before==set(p.ROOT.rglob('*'))

@pytest.mark.parametrize('mode',['success','failure','timeout','STOP'])
def test_public_supervisor_exit_contract_real_child(tmp_path,mode):
    out=tmp_path/mode;out.mkdir();worker=tmp_path/(mode+'.py')
    worker.write_text("import time,json,sys\nfrom pathlib import Path\np=Path(sys.argv[1])\n"+
        ("(p/'summary.json').write_text(json.dumps({'status':'completed'}))\n" if mode=='success' else "raise SystemExit(7)\n" if mode=='failure' else "time.sleep(30)\n"))
    if mode=='STOP':(out/'STOP').touch()
    code=supervise([sys.executable,str(worker),str(out)],out,time.time()+(.1 if mode=='timeout' else 3),'summary.json',grace=.1)
    r=json.loads((out/'supervisor.json').read_text());assert r['all_own_children_reaped'];assert (code==0)==(mode=='success')

def test_public_formal_cli_refuses_without_approval(tmp_path):
    out=tmp_path/'not-created';env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    result=subprocess.run([sys.executable,str(p.ROOT/'run.py'),'formal','--output',str(out),'--preflight',str(tmp_path/'missing')],env=env,capture_output=True,timeout=10)
    assert result.returncode!=0 and not out.exists()

def test_full_small_chain_stop_resume_and_finalization(tmp_path):
    from formal import chain
    cfg=dict(p.config(),T=4,rounds=20);case=p.toy();target=extract_target(case)
    m=create(case,cfg,31,'role','cpu');m.set_target(target)
    first=tmp_path/'continuous';a=chain(m,case,cfg,33,first,{'fixture':'not scientific'},lambda:None)
    calls=[0]
    def stop():
        calls[0]+=1
        if calls[0]==19:raise InterruptedError('fixture STOP mid-step')
    second=tmp_path/'resumed'
    with pytest.raises(InterruptedError):chain(m,case,cfg,33,second,{'fixture':'not scientific'},stop)
    b=chain(m,case,cfg,33,second,{'fixture':'not scientific'},lambda:None)
    assert a==b
    with np.load(first/'graph.npz') as z,np.load(second/'graph.npz') as y:np.testing.assert_array_equal(z['edges'],y['edges'])
    # Final graph exists but evaluation/summary is missing: revalidate then finish.
    (second/'summary.json').unlink();again=chain(m,case,cfg,33,second,{'fixture':'not scientific'},lambda:None);assert again==a
    with pytest.raises(ValueError,match='source mismatch'):chain(m,case,cfg,34,second,{'fixture':'not scientific'},lambda:None)

def test_rolling_recovery_keeps_permanent_milestones(tmp_path):
    p.checkpoint(tmp_path,{'binding':{},'updates':0,'model':torch.tensor(1)})
    for u in (10,20,30):p.rolling_recovery(tmp_path,{'binding':{},'updates':u,'model':torch.tensor(u)})
    assert (tmp_path/'pair_u00000000.pt').is_file()
    assert len(list((tmp_path/'recovery').glob('*.pt')))==2
    assert p.load_checkpoint(tmp_path,{})['updates']==30

def test_zero_projection_later_role_encoder_gradients_nonzero():
    case=p.toy();cfg=p.config();m=create(case,cfg,41,'role','cpu');m.set_target(extract_target(case));opt=p.optimizer(m,cfg);reader=Reader().install(m)
    try:
        _,first=p.update(m,opt,case,p.input_pair(case,cfg,43,0),cfg,gradients=True)
        assert all(torch.count_nonzero(g)==0 for k,g in first.items() if k.startswith(('branch.role_encoder','branch.value')))
        _,later=p.update(m,opt,case,p.input_pair(case,cfg,43,1),cfg,gradients=True)
        assert any(torch.count_nonzero(g)>0 for k,g in later.items() if k.startswith('branch.role_encoder'))
        assert any(torch.count_nonzero(g)>0 for k,g in later.items() if k.startswith('branch.value'))
        assert all(torch.isfinite(g).all() for g in later.values())
    finally:reader.close()

def test_target_switch_invalidates_only_explicit_frozen_context():
    case=p.toy();cfg=p.config();m=create(case,cfg,41,'role','cpu');target=extract_target(case);m.set_target(target);inp=p.input_pair(case,cfg,43,0)[0]
    changed=case['clean'].copy();changed[1,-1]=7;alternate=p.c.make_case(case['space'].types,changed,case['space'].pairs);other=extract_target(alternate)
    assert other.fingerprint!=target.fingerprint
    reader=Reader().install(m);ctx=FrozenTypedContext(m,case['space'].types,inp['edges'],inp['edge_families'],'a')
    try:
        with torch.no_grad():ctx.predict(inp['query'],inp['f'],inp['t']/cfg['T'],'a')
        m.set_target(other)
        with pytest.raises(ValueError,match='target/time'):ctx.predict(inp['query'],inp['f'],inp['t']/cfg['T'],'a')
        ctx.close()
        with torch.no_grad():p.predict(m,case['space'].types,inp['edges'],inp['edge_families'],inp['query'],inp['f'],inp['t']/cfg['T'])
    finally:ctx.close();reader.close()

def test_base_matches_original_degree_CN_model_without_role_read():
    from role_condition.model import ConditionalDenoiser
    case=p.toy();cfg=p.config();m=create(case,cfg,51,'base','cpu')
    with torch.no_grad():m.branch.degree_out.weight.fill_(.01)
    b=ConditionalDenoiser(case['n_types'],case['n_families'],family_pairs=case['space'].pairs,prior=case['prior'],mode='B',**cfg['model'])
    state=b.state_dict();shared=m.state_dict()
    for k in state:state[k]=shared[k].clone()
    b.load_state_dict(state,strict=True);inp=p.input_pair(case,cfg,53,0)[0];outputs=[]
    with torch.no_grad():
        for model in (m,b):outputs.append(p.predict(model,case['space'].types,inp['edges'],inp['edge_families'],inp['query'],inp['f'],inp['t']/cfg['T'],details=True))
    for a,b in zip(*outputs):torch.testing.assert_close(a,b,atol=0,rtol=0)

def test_role_sampling_does_not_read_reference_edge_identity():
    case=p.toy();cfg=p.config();m=create(case,cfg,61,'role','cpu');m.set_target(extract_target(case));state=p.state_from_edges(case,case['clean'])
    class Forbidden:
        def __array__(self,*a,**kw):raise AssertionError('reference edge identity leak')
    inference=dict(case,clean=Forbidden(),clean_ids=Forbidden());reader=Reader().install(m)
    try:
        with torch.no_grad():_,info=p.reverse_step(m,inference,state,p.MarginalKernel(case['prior'],p.c.schedule(cfg['T'])),500,20,63,np.random.default_rng(64),'cpu')
        assert info['exact_coverage']
    finally:reader.close()

def test_success_exit_without_evidence_is_rejected(tmp_path):
    with pytest.raises(FileNotFoundError):supervise([sys.executable,'-c','pass'],tmp_path,time.time()+3,'summary.json')
    report=json.loads((tmp_path/'supervisor.json').read_text());assert report['status']=='failed_or_incomplete' and report['child_exit_code']!=0

def test_reference_frozen_tail_thresholds():
    from evaluate import tails
    case=p.toy();names=p.config()['data']['type_names'];ref=p.graph(case,case['clean'],names)
    values=tails(ref,ref)
    assert values['Genre']['sorted_top10_total_degree']==[1,1]
    assert values['Genre']['low_le1']==2 and values['Genre']['zero']==0

def test_metric_export_cli_without_model_loading(tmp_path):
    run=tmp_path/'run';folder=run/'seed1';folder.mkdir(parents=True)
    (folder/'training.jsonl').write_text(json.dumps({'seed':1,'completed_updates':1,'elapsed_seconds':2,'arms':{'base':{'loss':.1,'gradient_norm':.2,'update_seconds':.3}}})+'\n')
    (folder/'validation_00000000.json').write_text(json.dumps({'base':{'rows':[{'t':1,'CE':.2,'constant_prior_CE':.3}]}}))
    out=tmp_path/'export';result=subprocess.run([sys.executable,str(p.ROOT/'export_metrics.py'),'--run',str(run),'--output',str(out)],capture_output=True,timeout=10)
    assert result.returncode==0 and 'base' in (out/'training.csv').read_text() and len((out/'validation.csv').read_text().splitlines())==2
