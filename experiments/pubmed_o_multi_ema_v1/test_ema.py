import copy,io,json,pickle,random,subprocess,sys
import numpy as np
import pytest
import torch
from ema import ParameterEMA,DECAY
from integration import complete_normal2,checkpoint_payload,restore_payload,final_readouts,ema_validation
from review import plan
from legacy_assessment import assess,SEEDS,SAMPLES

class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__();self.w=torch.nn.Parameter(torch.tensor([0.,2.]));self.register_buffer('prior',torch.tensor([0.,1.]));self.register_buffer('count',torch.tensor(0))

def rng():return dict(python=random.getstate(),numpy=np.random.get_state(),generator=copy.deepcopy(gen.bit_generator.state),torch=torch.get_rng_state())
gen=np.random.default_rng(123)
def restore_rng(x):
    random.setstate(x['python']);np.random.set_state(x['numpy']);gen.bit_generator.state=copy.deepcopy(x['generator']);torch.set_rng_state(x['torch'])
def equal(a,b):
    if torch.is_tensor(a):return torch.equal(a,b)
    if isinstance(a,np.ndarray):return np.array_equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b

def test_theta0_and_three_steps_match_explicit_formula():
    m=Tiny();e=ParameterEMA(m);expected=m.w.detach().double().clone()
    for i,v in enumerate([[10.,4.],[-5.,8.],[3.,0.]],1):
        with torch.no_grad():m.w.copy_(torch.tensor(v))
        expected=.99*expected+.01*m.w.detach().double();e.update(m,i)
        torch.testing.assert_close(e.shadow['w'].double(),expected,atol=2e-6,rtol=1e-5)
        assert not e.shadow['w'].requires_grad and e.shadow['w'].grad_fn is None
    assert e.updates==3

def test_torch21_first_call_copy_is_not_our_contract():
    from torch.optim.swa_utils import AveragedModel
    m=Tiny();e=ParameterEMA(m);a=AveragedModel(m,avg_fn=lambda old,new,n:DECAY*old+(1-DECAY)*new)
    with torch.no_grad():m.w.fill_(10)
    a.update_parameters(m);e.update(m,1)
    assert torch.equal(a.module.w,m.w)
    assert abs(e.shadow['w'][0].item()-.1)<2e-6
    assert not torch.equal(e.shadow['w'],a.module.w)

def test_buffers_copied_not_averaged_and_live_model_untouched():
    m=Tiny();e=ParameterEMA(m);m.prior.fill_(.25);m.count.fill_(7);m.w.grad=torch.tensor([1.,-2.]);before=copy.deepcopy(m.state_dict());grad=m.w.grad.clone()
    e.update(m,1);export=e.readout_state_dict(m)
    assert equal(before,m.state_dict()) and torch.equal(grad,m.w.grad)
    assert torch.equal(export['prior'],m.prior) and export['count'].dtype==torch.int64
    assert export['count'].item()==7 and export['prior'][0].item()==.25
    export['w'].zero_();assert not torch.equal(export['w'],e.shadow['w'])

def test_alias_readout_and_snapshot_are_not_raw_aliases():
    m=Tiny();m.other=m.w;e=ParameterEMA(m)
    with torch.no_grad():m.w.add_(3)
    e.update(m,1);state=e.readout_state_dict(m)
    assert torch.equal(state['w'],state['other']) and not torch.equal(state['w'],m.w)
    snap=e.state_dict();snap['shadow']['w'].zero_();assert e.shadow['w'].abs().sum()>0

def test_rng_and_raw_parameter_stream_unchanged():
    before=rng();initial=Tiny()
    def run(use):
        restore_rng(before);m=copy.deepcopy(initial);e=ParameterEMA(m) if use else None;inputs=[]
        for i in range(1,4):
            x=torch.rand(2)+float(random.random()+np.random.random()+gen.random());inputs.append(x.clone())
            with torch.no_grad():m.w.add_(x)
            if use:e.update(m,i);e.state_dict();e.readout_state_dict(m)
        return inputs,m.state_dict(),rng()
    a=run(False);b=run(True);assert equal(a,b);restore_rng(before)

def test_checkpoint_ema_optimizer_rng_exact_load_and_next_ema():
    m=Tiny();e=ParameterEMA(m);opt=torch.optim.AdamW(m.parameters(),lr=.0002)
    # Synthetic optimizer state, NOT an actual optimizer update or model training.
    opt.state[m.w]=dict(step=torch.tensor(2.),exp_avg=torch.tensor([.1,.2]),exp_avg_sq=torch.tensor([.01,.04]))
    for i in [1,2]:
        with torch.no_grad():m.w.add_(i)
        e.update(m,i)
    random_state=rng();payload=checkpoint_payload(m,opt,e,2,random_state,{'fixture':'tiny'})
    buf=io.BytesIO();torch.save(payload,buf);buf.seek(0);saved=torch.load(buf,map_location='cpu')
    m2=Tiny();e2=ParameterEMA(m2);opt2=torch.optim.AdamW(m2.parameters(),lr=.0002)
    assert restore_payload(saved,m2,opt2,e2,{'fixture':'tiny'},restore_rng)==2
    assert equal(m.state_dict(),m2.state_dict()) and equal(opt.state_dict(),opt2.state_dict()) and equal(e.state_dict(),e2.state_dict()) and equal(rng(),random_state)
    for mm,ee in [(m,e),(m2,e2)]:
        with torch.no_grad():mm.w.add_(7)
        ee.update(mm,3)
    assert equal(e.state_dict(),e2.state_dict())

@pytest.mark.parametrize('which',['counter','decay','shape','nan'])
def test_invalid_ema_checkpoint_rejected(which):
    m=Tiny();e=ParameterEMA(m);p=e.state_dict()
    if which=='counter':p['updates']=2
    if which=='decay':p['decay']=.999
    if which=='shape':p['shadow']['w']=torch.zeros(3)
    if which=='nan':p['shadow']['w'][0]=float('nan')
    with pytest.raises(ValueError):e.load_state_dict(p,m,0)

def test_duplicate_skipped_update_and_nonfinite_rejected():
    m=Tiny();e=ParameterEMA(m);e.update(m,1);saved=e.state_dict()
    for step in [1,3]:
        with pytest.raises(ValueError):e.update(m,step)
    with torch.no_grad():m.w[0]=float('inf')
    with pytest.raises(FloatingPointError):e.update(m,2)
    assert equal(saved,e.state_dict())

def test_normal2_hook_only_after_completed_update():
    m=Tiny();e=ParameterEMA(m);calls=[]
    def existing(*args,**kw):
        calls.append(kw);assert e.updates==0 and kw=={'step':True}
        with torch.no_grad():m.w.add_(10)
        return {'optimizer_steps':1,'loss':.5}
    r=complete_normal2(existing,m,None,None,['first','second'],1000,'cpu',e,0)
    assert len(calls)==1 and e.updates==1 and r['loss']==.5
    assert abs(e.shadow['w'][0].item()-.1)<2e-6
    def fail(*a,**k):raise RuntimeError('normal2 incomplete')
    with pytest.raises(RuntimeError):complete_normal2(fail,m,None,None,[],1000,'cpu',e,1)
    assert e.updates==1
    with pytest.raises(RuntimeError):e.state_dict()
    with pytest.raises(RuntimeError):e.readout_state_dict(m)
    with pytest.raises(ValueError):complete_normal2(existing,m,None,None,[],1000,'cpu',e,1)

def test_raw_step_completed_but_ema_fails_cannot_save_completed_checkpoint():
    m=Tiny();e=ParameterEMA(m);opt=torch.optim.AdamW(m.parameters())
    def bad_result(*a,**k):
        with torch.no_grad():m.w.fill_(float('inf'))
        return {'optimizer_steps':1}
    with pytest.raises(FloatingPointError):complete_normal2(bad_result,m,opt,None,[],1000,'cpu',e,0)
    with pytest.raises(RuntimeError):checkpoint_payload(m,opt,e,0,rng(),{})

@pytest.mark.parametrize('fail',[False,True])
def test_validation_swap_restores_raw_and_parameter_identity(fail):
    m=Tiny();e=ParameterEMA(m)
    with torch.no_grad():m.w.add_(10)
    e.update(m,1);raw=copy.deepcopy(m.state_dict());pid=id(m.w);random_state=rng()
    def frozen_validation(model):
        model.eval();assert abs(model.w[0].item()-.1)<2e-6
        if fail:raise RuntimeError('fixture evaluation error')
        return {'readout':'ema','value':model.w[0].item()}
    if fail:
        with pytest.raises(RuntimeError):ema_validation(frozen_validation,m,e)
    else:assert ema_validation(frozen_validation,m,e)['readout']=='ema'
    assert equal(m.state_dict(),raw) and m.training and id(m.w)==pid and equal(rng(),random_state)

def test_final_only_and_review_cli_no_execution():
    m=Tiny();e=ParameterEMA(m)
    with pytest.raises(ValueError):final_readouts(m,e,0)
    p=plan();assert len(p['training'])==8 and len(p['sampling'])==64 and len({s['directory'] for s in p['sampling']})==64
    import pathlib
    cli=pathlib.Path(__file__).with_name('review.py');a=subprocess.run([sys.executable,str(cli),'plan'],capture_output=True,text=True);assert a.returncode==0 and json.loads(a.stdout)==p
    b=subprocess.run([sys.executable,str(cli),'formal'],capture_output=True,text=True);assert b.returncode!=0 and 'not authorized' in b.stderr

def records():
    return [dict(train_seed=s,readout=r,sample_seed=k,status='complete',edges=1000,W1_incident4=1.,LCC=190.,family_edges=[1000]+[0]*9,other_metrics={'triangle':17}) for s in SEEDS for r in ['raw','ema'] for k in SAMPLES]
def change(rs,s,readout,edges=None,**kw):
    for r in rs:
        if r['train_seed']==s and r['readout']==readout:
            r.update(kw)
            if edges is not None:r.update(edges=edges,family_edges=[edges]+[0]*9)

def test_consistent_accuracy_requires_both_seeds_and_no_auto_adoption():
    rs=records()
    for s in SEEDS:change(rs,s,'ema',edges=950)
    a=assess(rs);assert a['accuracy_conclusion']=='consistent_accuracy_gain_in_this_panel' and a['recommend_adoption'] is False
    assert a['original_records'][0]['other_metrics']=={'triangle':17}

def test_pooled_gain_with_one_seed_worse_is_mixed():
    rs=records();change(rs,SEEDS[0],'ema',edges=926);change(rs,SEEDS[1],'ema',edges=1020)
    a=assess(rs);assert a['pooled_error_difference']<0 and a['accuracy_conclusion']=='mixed_or_partial_not_consistent'

def test_equal_wrong_numbers_are_not_accuracy_gain():
    a=assess(records());assert a['accuracy_conclusion']=='accuracy_hypothesis_not_supported_in_this_panel'

def test_redlines_are_per_seed_not_pooled():
    rs=records();change(rs,SEEDS[0],'ema',W1_incident4=1.2,LCC=170.);change(rs,SEEDS[1],'ema',W1_incident4=.5,LCC=192.)
    a=assess(rs);assert all(a['per_seed'][0]['redlines'].values()) and a['non_adoption_redline']

def test_missing_graph_is_not_mean_of_successes():
    a=assess(records()[:-1]);assert a['status']=='incomplete' and a['pooled_error_difference'] is None and a['accuracy_conclusion'] is None

def test_per_graph_errors_precede_means_and_relations_retained():
    rs=records();change(rs,SEEDS[0],'raw',edges=900);rs[1].update(edges=952,family_edges=[952]+[0]*9)
    a=assess(rs);assert a['per_seed'][0]['raw']['signed_edge_bias']==0 and a['per_seed'][0]['raw']['edge_absolute_error']==26
    assert len(a['per_seed'][0]['family_error_differences'])==10
