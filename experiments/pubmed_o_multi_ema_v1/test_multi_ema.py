import copy
import pytest
import torch
from multi_ema import MultiEMA
from runtime_contract import DECAYS,SEEDS,OLD_SEEDS,NEW_SEEDS,SAMPLES
from integration import checkpoint_payload,restore_payload,ema_validation
from assessment import assess,REFERENCE_FAMILY_EDGES

def test_three_recurrences_from_theta0_raw_source_and_rng():
    model=torch.nn.Linear(1,1,bias=False);model.weight.data.fill_(2.)
    rng=torch.get_rng_state().clone();bank=MultiEMA(model)
    assert torch.equal(rng,torch.get_rng_state())
    expected={k:torch.tensor([[2.]]) for k in DECAYS}
    assert len({x.shadow['weight'].data_ptr() for x in bank.members.values()})==3
    for u,value in enumerate([4.,-3.,8.],1):
        model.weight.data.fill_(value)
        bank.update(model,u)
        for k,d in DECAYS.items():
            expected[k].mul_(d).add_(model.weight.detach(),alpha=1-d)
            assert torch.equal(bank.members[k].shadow['weight'],expected[k])
        assert model.weight.item()==value and torch.equal(rng,torch.get_rng_state())
    assert bank.updates==3

def test_bank_save_restore_continue_and_decay_identity():
    model=torch.nn.Linear(1,1);opt=torch.optim.AdamW(model.parameters());bank=MultiEMA(model)
    bank.update(model,1);state=checkpoint_payload(model,opt,bank,1,{'fixture':1},{'source':'test'})
    new=MultiEMA(model);seen=[]
    assert restore_payload(state,model,opt,new,{'source':'test'},seen.append)==1
    assert seen==[{'fixture':1}]
    model.weight.data.add_(4);bank.update(model,2);new.update(model,2)
    for k in DECAYS:assert torch.equal(bank.members[k].shadow['weight'],new.members[k].shadow['weight'])
    bad=copy.deepcopy(state);bad['ema']['members']['ema_0.9']['decay']=.99
    with pytest.raises(ValueError):restore_payload(bad,model,opt,new,{'source':'test'},seen.append)

@pytest.mark.parametrize('name',list(DECAYS))
def test_each_validation_readout_restores_raw_on_exception(name):
    model=torch.nn.Linear(1,1);bank=MultiEMA(model);model.weight.data.add_(2)
    raw={k:v.clone() for k,v in model.state_dict().items()};model.train()
    def validate(m):
        assert torch.equal(m.weight,bank.members[name].shadow['weight'])
        m.eval();raise RuntimeError('test exception')
    with pytest.raises(RuntimeError):ema_validation(validate,model,bank.members[name])
    assert model.training and all(torch.equal(v,raw[k]) for k,v in model.state_dict().items())

def test_partial_bank_update_cannot_export_checkpoint(monkeypatch):
    model=torch.nn.Linear(1,1);bank=MultiEMA(model)
    def fail(*a):raise RuntimeError('partial bank')
    monkeypatch.setattr(bank.members['ema_0.97'],'update',fail)
    with pytest.raises(RuntimeError):bank.update(model,1)
    assert not bank.at_update_boundary
    with pytest.raises(RuntimeError):bank.state_dict()
    with pytest.raises(RuntimeError):bank.readouts(model)

def records():
    rows=[]
    for seed in SEEDS:
        for kind,change in [('raw',20),('ema_0.9',30),('ema_0.97',20),('ema_0.99',10)]:
            for sample in SAMPLES:
                f=list(REFERENCE_FAMILY_EDGES);f[0]+=change
                rows.append(dict(train_seed=seed,readout=kind,sample_seed=sample,status='complete',edges=926+change,family_edges=f,W1_incident4=1.,LCC=192))
    return rows

def test_decay_identity_groups_primary_and_no_best_selection():
    report=assess(records());assert report['status']=='complete'
    for k,delta in [('ema_0.9',10),('ema_0.97',0),('ema_0.99',-10)]:
        d=report['decays'][k]
        for group,n in [('all_eight',8),('old_two',2),('new_six',6)]:
            g=d['groups'][group];assert g['mean_difference']==delta==g['median_difference']
            assert g['expected_seeds']==n
        assert d['primary']==(k=='ema_0.99')
    assert report['decays']['ema_0.9']['groups']['all_eight']['worsened']==8
    assert report['decays']['ema_0.97']['groups']['all_eight']['tied']==8
    assert report['decays']['ema_0.99']['groups']['all_eight']['improved']==8
    assert not report['recommend_adoption']

def test_missing_new_seed_does_not_average_successes_or_poison_old_group():
    rows=records();rows=[r for r in rows if not (r['train_seed']==NEW_SEEDS[0] and r['readout']=='ema_0.99' and r['sample_seed']==SAMPLES[0])]
    d=assess(rows)['decays']['ema_0.99']
    assert d['groups']['all_eight']['mean_difference'] is None
    assert d['groups']['new_six']['mean_difference'] is None
    assert d['groups']['old_two']['mean_difference']==-10

def test_warning_is_per_seed_not_automatic_panel_veto():
    rows=records()
    for r in rows:
        if r['train_seed']==OLD_SEEDS[0] and r['readout']=='ema_0.99':r.update(W1_incident4=1.2,LCC=170)
    d=assess(rows)['decays']['ema_0.99']
    assert all(d['per_seed'][0]['warnings'].values()) and not any(d['per_seed'][1]['warnings'].values())
    assert d['groups']['all_eight']['status']=='complete' and d['groups']['all_eight']['improved']==8
