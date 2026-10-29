import sys
from pathlib import Path
import numpy as np
import pytest
import torch
sys.path.insert(0,str(Path(__file__).parent/'package'))
from count_aux import count_loss,smooth_count_penalty,LAMBDA
S=18336
torch.set_num_threads(1)

@pytest.mark.parametrize('m,delta',[(3,883.77),(137,-40),(137,-10.55),(267,-14.35),(3,-.525),(0,4),(137,0)])
def test_count_gradient(m,delta):
    d=torch.tensor(delta,dtype=torch.float64,requires_grad=True);target=torch.tensor(float(m),dtype=torch.float64)
    rho,s,u,slope=smooth_count_penalty(d,target)
    g=torch.autograd.grad(rho,d)[0]
    assert torch.allclose(g,slope,atol=1e-13,rtol=1e-12)
    assert abs(float(g))<=1/float(s)
    eps=1e-5
    a=smooth_count_penalty(d.detach()+eps,target)[0];b=smooth_count_penalty(d.detach()-eps,target)[0]
    assert abs(float((a-b)/(2*eps)-g))<1e-7

@pytest.mark.parametrize('prob,labels',[
    ([.1,.2,.1,.1],[1,1,0,0]),([.8,.7,.6,.5],[1,0,0,0]),
    ([.05,.05,.95,.95],[1,1,0,0]),
    ([1e-9,1e-9,.1,.1],[1,1,0,0]),
    ([.1,1-1e-9,1-1e-9,.1],[1,0,0,0]),([.1,.2,.3,.4],[0,0,0,0])])
def test_present_logit_and_half_normal2(prob,labels):
    p=torch.tensor(prob,dtype=torch.float64);y=torch.tensor(labels)
    z=torch.stack([torch.zeros_like(p),torch.log(p)-torch.log1p(-p)],-1).requires_grad_()
    q=z.softmax(-1)[:,1];aux,rec=count_loss(q,[0]*len(p),[sum(labels)],[len(p)])
    ce=torch.nn.functional.cross_entropy(z,y,reduction='sum')/S
    ga=torch.autograd.grad(LAMBDA*aux,z,retain_graph=True)[0]
    gc=torch.autograd.grad(ce,z,retain_graph=True)[0]
    expected=LAMBDA*rec['penalty_count_derivative'][0]*p*(1-p)
    assert torch.allclose(ga[:,1],expected,atol=1e-14,rtol=1e-10)
    assert torch.allclose(ga[:,0],-expected,atol=1e-14,rtol=1e-10)
    half=torch.autograd.grad((ce+LAMBDA*aux)/2,z)[0]
    assert torch.allclose(half,(ga+gc)/2,atol=1e-14,rtol=1e-10)
    # Algebraic slot-ratio bounds, not a parameter-gradient bound.
    c=LAMBDA*S*abs(float(rec['penalty_count_derivative'][0]))
    assert (ga[:,1].abs()<=c*gc[:,1].abs()+1e-13).all()

def test_fixed_support_and_relation_mean():
    p=torch.tensor([.2,.6,.1,.3],dtype=torch.float64,requires_grad=True)
    loss,r=count_loss(p,[0,0,1,1],[2,0,2],[2,2,0],[1,0,2])
    expected=(torch.hypot(torch.tensor(1.),torch.tensor(-.2/2**.5)).double()-1+torch.hypot(torch.tensor(1.,dtype=torch.float64),torch.tensor(.4,dtype=torch.float64))-1)/2
    assert abs(float(loss-expected))<1e-7
    assert r['relation_denominator']==2
    assert r['included'].tolist()==[True,True,False]
    g=torch.autograd.grad(loss,p)[0]
    expected_g=torch.cat([r['penalty_count_derivative'][0].repeat(2),r['penalty_count_derivative'][1].repeat(2)])/2
    assert torch.allclose(g,expected_g,atol=1e-14,rtol=1e-12)
    with pytest.raises(ValueError):count_loss(p,[0,0,1,1],[2,0,1],[2,2,0],[1,0,2])

def test_all_fixed_differentiable_zero():
    p=torch.empty(0,dtype=torch.float64,requires_grad=True)
    a,r=count_loss(p,[],[0,2],[0,0],[0,2]);a.backward()
    assert float(a)==0 and r['relation_denominator']==0 and p.grad.numel()==0

def test_quantity_correct_structure_wrong():
    # 4-node star and path: same edge count, distinct degree spectrum.
    for p in [[1.,1.,1.,0.,0.,0.],[1.,0.,0.,1.,0.,1.]]:
        a,_=count_loss(torch.tensor(p),[0]*6,[3],[6]);assert float(a)==0

def test_zero_prediction_and_extreme_slope_finite():
    for p in [torch.zeros(18336,dtype=torch.float64),torch.ones(18336,dtype=torch.float64)]:
        p.requires_grad_();a,r=count_loss(p,[0]*len(p),[3],[len(p)])
        a.backward();assert torch.isfinite(a) and torch.isfinite(p.grad).all()
        assert float(p.grad.abs().max())<=1/(3**.5)

def test_small_residual_no_cancellation():
    d=torch.tensor(1e-12,dtype=torch.float64,requires_grad=True)
    r,_,_,_=smooth_count_penalty(d,torch.tensor(1.,dtype=torch.float64))
    assert float(r)>0 and abs(float(r)/5e-25-1)<1e-10

def test_ce_zero_lambda_original_arithmetic():
    z=torch.tensor([[1.,2.],[3.,-1.]],requires_grad=True);y=torch.tensor([0,1])
    ce=torch.nn.functional.cross_entropy(z,y,reduction='sum')/S
    a,_=count_loss(z.softmax(-1)[:,1],[0,0],[1],[2])
    weight=0.
    loss=ce/2 if weight==0 else ce/2+(weight/2)*a
    assert torch.equal(torch.autograd.grad(loss,z,retain_graph=True)[0],torch.autograd.grad(ce/2,z)[0])

def test_invalid_values_rejected():
    for p in [float('nan'),float('inf'),-.1,1.1]:
        with pytest.raises((FloatingPointError,ValueError)):count_loss(torch.tensor([p]),[0],[0],[1])

def test_undirected_mapping_and_id_gate():
    # Import only the historical slot mapper; no model/context import.
    original=Path(__file__).parent/'reference_source'
    sys.path.insert(0,str(original))
    from slots import TypedSlots
    space=TypedSlots(np.array([0,0,1,1]),((0,0),(0,1),(1,1)))
    fam,local=space.encode(np.array([[0,2,0,1],[2,0,1,0]]))
    ids=space.offsets[fam]+local
    assert ids[0]==ids[1] and ids[2]==ids[3]
    # This is exactly the original full-query gate: duplicates cannot replace missing IDs.
    expected=np.arange(space.total)
    bad=expected.copy();bad[-1]=bad[0]
    assert not np.array_equal(np.sort(bad),expected)

def test_no_K_damping_for_same_count():
    a,_=count_loss(torch.tensor([.5,.5,.5,.5],dtype=torch.float64),[0]*4,[1],[4])
    b,_=count_loss(torch.tensor([.25]*8,dtype=torch.float64),[0]*8,[1],[8])
    assert torch.equal(a,b)

def test_two_independent_inputs_accumulate_half_gradients():
    # A scalar toy parameter, not a network or optimizer update.
    w=torch.tensor(.3,dtype=torch.float64,requires_grad=True)
    def components(x,target):
        z=torch.stack([w*x,-w*x],-1)
        ce=torch.nn.functional.cross_entropy(z,target,reduction='sum')/S
        aux,_=count_loss(z.softmax(-1)[:,1],[0]*len(x),[int(target.sum())],[len(x)])
        return ce+LAMBDA*aux
    x1=torch.tensor([1.,2.],dtype=torch.float64);y1=torch.tensor([0,1])
    x2=torch.tensor([3.,-.4],dtype=torch.float64);y2=torch.tensor([1,1])
    expected=torch.autograd.grad((components(x1,y1)+components(x2,y2))/2,w)[0]
    (components(x1,y1)/2).backward();(components(x2,y2)/2).backward()
    assert torch.allclose(w.grad,expected,atol=1e-15,rtol=1e-12)
