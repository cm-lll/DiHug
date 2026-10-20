"""Only loss numerics and decision-response regressions; no architecture changes."""
import numpy as np
import json
import pytest
import torch
from torch.nn import functional as F
from hsgd.losses import supported_binary_ce
from hsgd.fusion_checks import (fixture,make,diagnostics,decision_response,
    decision_mixed_response,decision_coordinates,mechanism)
from hsgd.bw_preflight import common as c


def moderate_evidence():
    z=torch.tensor([[.2,-.3],[-1.2,.8],[1.,-.4],[-2.,1.],[2.,-1.]],dtype=torch.float64,requires_grad=True)
    y=torch.tensor([0.,1.,1.,0.,1.],dtype=torch.float64);pi=torch.tensor([.2,.7,.4,0.,1.],dtype=torch.float64)
    old=z.softmax(-1)[:,1];old=torch.where(pi==0,torch.zeros_like(old),torch.where(pi==1,torch.ones_like(old),old))
    before=F.binary_cross_entropy(old,y,reduction='none');after=supported_binary_ce(z,y,pi)
    weight=3/28
    gb=torch.autograd.grad(before.sum()*weight,z,retain_graph=True)[0]
    ga=torch.autograd.grad(after.sum()*weight,z)[0]
    expected=(z.detach().softmax(-1)-F.one_hot(y.long(),2))*((pi>0)&(pi<1))[:,None]*weight
    assert torch.allclose(before,after,atol=1e-14,rtol=1e-13)
    assert torch.allclose(gb,ga,atol=1e-14,rtol=1e-13)
    assert torch.allclose(ga,expected,atol=1e-14,rtol=1e-13)
    assert torch.equal(after[-2:],torch.zeros(2,dtype=torch.float64))
    return dict(loss_max_abs=float((before-after).abs().max()),gradient_max_abs=float((gb-ga).abs().max()),
        analytic_gradient_max_abs=float((ga-expected).abs().max()),weighted_loss=float(after.sum()*weight),ce_weight=weight)


def extreme_evidence():
    z=torch.tensor([[1000.,-1000.],[-1000.,1000.]],requires_grad=True)
    y=torch.tensor([1.,0.]);pi=torch.tensor([.2,.8])
    old=F.binary_cross_entropy(z.softmax(-1)[:,1],y,reduction='sum')
    gold=torch.autograd.grad(old,z,retain_graph=True)[0]
    new=supported_binary_ce(z,y,pi).sum();g=torch.autograd.grad(new,z)[0]
    assert torch.isfinite(new) and torch.isfinite(g).all() and new.item()==4000
    assert torch.equal(g,torch.tensor([[1.,-1.],[-1.,1.]]))
    assert torch.equal(gold,torch.zeros_like(gold))
    # One step in the negative gradient direction reduces BOTH wrong margins.
    candidate=z.detach()-.1*g
    assert supported_binary_ce(candidate,y,pi).sum()<new
    return dict(logits=z.detach().tolist(),labels=y.tolist(),old_probability_BCE=float(old),
        old_gradient=gold.tolist(),stable_CE=float(new),stable_gradient=g.tolist(),
        loss_after_negative_gradient_step=float(supported_binary_ce(candidate,y,pi).sum()))


def translation_evidence():
    z=torch.tensor([[0.,1.],[2.,-1.]],dtype=torch.float64);pi=torch.tensor([.2,.7],dtype=torch.float64)
    direct=decision_response(z+8,z,pi);mixed=decision_mixed_response(z+8,z,z,z,pi)
    for r in [direct,mixed]:
        assert r['interface_logits_max_abs']==8 and r['logit_gap_max_abs']==0 and r['p0_max_abs']==0
    return dict(shared_shift=8.,direct=direct,pure_shared_shift_interaction=mixed)


def test_moderate_logits_match_original_objective_and_R_over_S(record_property):
    record_property('evidence',json.dumps(moderate_evidence()))


def test_extreme_wrong_confidence_retains_corrective_gradient(record_property):
    record_property('evidence',json.dumps(extreme_evidence()))


def test_exact_support_zero_loss_zero_gradient_and_empty():
    z=torch.tensor([[-1000.,1000.],[1000.,-1000.]],requires_grad=True)
    loss=supported_binary_ce(z,torch.tensor([0.,1.]),torch.tensor([0.,1.]))
    assert torch.equal(loss,torch.zeros_like(loss));loss.sum().backward()
    assert torch.equal(z.grad,torch.zeros_like(z))
    empty=torch.empty((0,2),requires_grad=True)
    supported_binary_ce(empty,torch.empty(0),torch.empty(0)).sum().backward()
    assert empty.grad.shape==(0,2)


@pytest.mark.parametrize('labels,prior,match',[
    ([1.],[0.],'support'),([0.],[1.],'support'),([.5],[.2],'binary'),
    ([float('nan')],[.2],'nonfinite'),([0.],[1.1],'prior')])
def test_invalid_labels_or_support_rejected(labels,prior,match):
    with pytest.raises(ValueError,match=match):supported_binary_ce(torch.zeros(1,2),torch.tensor(labels),torch.tensor(prior))


def test_common_shift_is_not_a_decision_response(record_property):
    record_property('evidence',json.dumps(translation_evidence()))


def test_decision_probability_uses_exact_supported_p0():
    z=torch.tensor([[0.,2.],[1.,0.],[0.,1.]])
    pi=torch.tensor([0.,1.,.2]);margin,p=decision_coordinates(z,pi)
    assert torch.equal(margin,torch.tensor([2.,-1.,1.]))
    assert p[0]==0 and p[1]==1 and torch.allclose(p[2],z[2].softmax(-1)[1])
    response=decision_response(z+torch.tensor([[0.,2.],[0.,2.],[0.,0.]]),z,pi)
    assert response['logit_gap_max_abs']==2 and response['p0_max_abs']==0


def test_actual_head_has_gap_probability_and_mixed_decision_responses(record_property):
    r=mechanism()
    for name in ['CN_feature_mask_decision_response','legal_global_input_only_decision_response','CN_global_decision_mixed_response']:
        assert r[name]['logit_gap_max_abs']>1e-8 and r[name]['p0_max_abs']>1e-8
    record_property('evidence',json.dumps(r))


def test_training_and_diagnostics_share_stable_CE_without_probability_BCE(monkeypatch,record_property):
    case,_,_=fixture();m,_=make(case,True)
    inp=c.train_input(case,1000,3,2026101407,1,np.random.default_rng(910),force_t=30)
    opt=torch.optim.AdamW(m.parameters(),lr=2e-4,weight_decay=1e-12,amsgrad=False)
    def forbidden(*args,**kwargs):raise AssertionError('production probability-domain BCE used')
    monkeypatch.setattr(F,'binary_cross_entropy',forbidden)
    captured={};row=c.update(m,opt,case,inp,1000,'cpu',do_step=False,diagnostics=captured)
    pi=m.prior[torch.tensor(inp['f'])];item=supported_binary_ce(captured['logits'],torch.tensor(inp['labels']),pi)
    assert row['loss']==pytest.approx(float(item.sum()*inp['ce_weight']),abs=1e-7)
    diag=diagnostics(m,case,inp,1000)
    for group in diag['relation_label_CE']:
        sel=(inp['f']==group['family'])&(inp['labels']==group['clean_label'])
        assert group['count']==int(sel.sum())
        if sel.any():assert group['CE']==pytest.approx(float(item[sel].mean()),abs=2e-6)
    # Non-diagnostic training dispatch must also request/use logits.
    row2=c.update(m,opt,case,inp,1000,'cpu',do_step=False,diagnostics=None)
    assert row2['loss']==pytest.approx(row['loss'],abs=1e-7)
    assert not opt.state  # No optimizer update or formal training occurred.
    record_property('evidence',json.dumps(dict(loss=row['loss'],ce_weight=inp['ce_weight'],
        independent_sum_R_over_S=float(item.sum()*inp['ce_weight']),non_diagnostic_loss=row2['loss'],
        diagnostic_groups=diag['relation_label_CE'],optimizer_updates=0)))
