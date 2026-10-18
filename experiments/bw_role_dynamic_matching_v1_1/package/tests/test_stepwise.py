"""Deterministic engineering fixtures only; no scientific outputs or updates."""
import copy
import itertools
import json
import numpy as np
import pytest
import torch
from hsgd.bw_preflight import common as c
from hsgd.fusion_checks import fixture,make,rng_equal
from hsgd.kernel import MarginalKernel
from role_degree.role_statistics import joint_profile,independent_degree_cf,target_cf
from stepwise_role.core import Layout,freeze,StepwiseGuide,transition_kl
from stepwise_role.sampling import reverse_step,binding,save_checkpoint,load_checkpoint,validate_arm


def setup_small():
    case=c.make_case(np.array([0,0,1,2]),np.array([[0,0],[2,3]]),((0,1),(0,2)))
    l=Layout.canonical(case['space'].types,case['space'].pairs,case['prior'])
    profile=joint_profile(l.types,case['clean'],case['space'].encode(case['clean'])[0],l.pairs)
    k=MarginalKernel(case['prior'],[1.,.8,.4,.1,0.])
    guide=StepwiseGuide(l,freeze(profile,l,k.alpha))
    state=np.array([1,0,0,1],dtype=np.uint8)
    return case,l,profile,k,guide,state


def direct_cf(l,p,w):
    """Independent direct edge-product implementation, NOT log/bincount engine."""
    result=[]
    for node in np.flatnonzero(l.types==0):
        products=np.ones(len(w),dtype=complex)
        for e,(u,v) in enumerate(l.slots.T):
            if node in (u,v): products*=1-p[e]+p[e]*np.exp(1j*w[:,l.families[e]])
        result.append(products)
    return np.mean(result,axis=0)


@pytest.mark.parametrize('s',[0,1,2,3,4])
def test_noisy_target_against_exhaustive_forward(s):
    _,l,_,k,g,_=setup_small()
    clean=np.array([1,0,1,0]) # actual reference: same type-0 node has both roles
    p=k.alpha[s]*clean+(1-k.alpha[s])*l.prior[l.families]
    maximum=0.
    for c0,w in g.omega.items():
        enumerated=np.zeros(32,dtype=complex)
        for bits in itertools.product((0,1),repeat=4):
            bits=np.array(bits);mass=np.prod(np.where(bits,p,1-p))
            deg=np.zeros((len(l.types),len(l.pairs)),int)
            for e,(u,v) in enumerate(l.slots.T):
                deg[u,l.families[e]]+=bits[e];deg[v,l.families[e]]+=bits[e]
            enumerated+=mass*np.exp(1j*deg[l.types==int(c0)]@w.T).mean(0)
        maximum=max(maximum,float(np.max(abs(enumerated-g.targets[c0][s]))))
    assert maximum<3e-14
    if s==0:
        exact=target_cf(g.frozen['payload']['frequencies']['payload']['profile'],g.omega['0'])
        assert np.max(abs(exact['0']-g.targets['0'][0]))<1e-14
    if s==4: # endpoint independent of clean role dependence
        assert np.max(abs(direct_cf(l,p,g.omega['0'])-g.targets['0'][s]))<1e-14


@pytest.mark.parametrize('t',[1,2,3,4])
def test_bridge_gradient_finite_difference_and_clean_endpoint(t):
    _,l,_,k,g,state=setup_small();p=np.array([.18,.64,.39,.72])
    q,loss,grad,_,error=g.posterior_gradient(p,state,k,t)
    z=np.log(p)-np.log1p(-p);numeric=[];eps=1e-5
    for e in range(4):
        zz=z.copy();zz[e]+=eps;pp=1/(1+np.exp(-zz))
        a=g.statistic(k.reverse(pp,state,l.families,t-1,t),t-1)[0]
        zz[e]-=2*eps;pp=1/(1+np.exp(-zz))
        b=g.statistic(k.reverse(pp,state,l.families,t-1,t),t-1)[0]
        numeric.append((a-b)/(2*eps))
    assert np.allclose(grad,numeric,atol=2e-8,rtol=2e-5)
    assert error<5e-14
    assert np.max(abs(q-p))<1e-14 if t==1 else True
    # Expected empirical CF agrees with an independent multiplication engine.
    details=g.statistic(q,t-1)[2]
    for c0,w in g.omega.items():
        direct=independent_degree_cf(l.types,l.slots,l.families,q,l.pairs,w)[c0]
        assert np.max(abs(direct-details['mean_cf'][c0]))<2e-14


def test_joint_dependence_target_response_and_zero_gradient_counterexample():
    _,l,pf,k,g,state=setup_small()
    other=copy.deepcopy(pf);other['groups']['0']['rows']=[[0,1],[1,0]];other['groups']['0']['counts']=[1,1]
    gg=StepwiseGuide(l,freeze(other,l,k.alpha))
    # Same per-relation degree marginals AND same frequency scales, different joint law.
    assert np.array_equal(g.omega['0'],gg.omega['0'])
    p=np.array([.17,.63,.46,.81])
    a,qa,ra=g.apply(p,state,k,1);b,qb,rb=gg.apply(p,state,k,1)
    assert ra['status']==rb['status']=='accepted'
    assert np.max(abs(qa-qb))>1e-6
    _,_,zero=g.apply(np.full(4,.5),state,k,1)
    assert zero['loss_before']>0 and zero['raw_logit_gradient_max']<1e-14
    assert zero['status']=='skipped' # positive discrepancy, no invented response


def test_reindexing_symmetry_and_incomplete_mask_rejection():
    _,l,profile,k,g,state=setup_small();p=np.array([.17,.63,.46,.81]);perm=np.array([1,0,2,3])
    edges=np.sort(perm[l.slots],axis=0)
    ll=Layout(l.types,edges,l.families,l.pairs,l.prior)
    gg=StepwiseGuide(ll,freeze(profile,ll,k.alpha))
    for t in (1,3):
        a,qa,ra=g.apply(p,state,k,t);b,qb,rb=gg.apply(p,state,k,t)
        assert abs(ra['loss_before']-rb['loss_before'])<2e-10
        assert np.allclose(qa,qb,atol=2e-10,rtol=1e-9)
    # Reversing storage orientation then canonicalizing cannot change variables.
    assert np.array_equal(np.sort(l.slots[::-1],axis=0),l.slots)
    with pytest.raises(ValueError,match='complete legal'):
        Layout(l.types,l.slots[:,:-1],l.families[:-1],l.pairs,l.prior)


def test_exact_support_saturation_and_kl_limits():
    l=Layout.canonical(np.array([0,0,1,1]),[(0,0),(0,1),(1,1)],[0.,.5,1.])
    edge=l.slots[:,[1,5]];fam=l.families[[1,5]]
    profile=joint_profile(l.types,edge,fam,l.pairs)
    k=MarginalKernel(l.prior,[1.,.7,0.]);g=StepwiseGuide(l,freeze(profile,l,k.alpha))
    p=np.array([0.,0.,.3,.8,1.,1.]);state=np.array([0,1,0,1,0,1])
    pp,q,r=g.apply(p,state,k,1)
    assert np.array_equal(pp[[0,1,4,5]],p[[0,1,4,5]])
    assert r['numerical_saturated_slots']==2 and q[0]==0 and q[-1]==1
    assert transition_kl(p,p)['sum']==0
    with pytest.raises(FloatingPointError): transition_kl(np.array([.1]),np.array([0.]))
    with pytest.raises(ValueError):g.apply(np.full(6,.5),state,k,1)
    with pytest.raises(ValueError):g.apply(np.array([0.,np.nan,.3,.8,1.,1.]),state,k,1)


def test_backtracking_failure_nonfinite_and_unresolvable_gradient(monkeypatch):
    _,_,_,k,g,state=setup_small();p=np.array([.17,.63,.46,.81])
    original=g.statistic
    def fail_candidates(q,s,gradient=False):
        loss,grad,detail=original(q,s,gradient)
        return (10. if not gradient else loss),grad,detail
    monkeypatch.setattr(g,'statistic',fail_candidates)
    pp,q,r=g.apply(p,state,k,1)
    assert r['reason']=='no_verified_descent' and r['trials']==8
    assert np.array_equal(pp,p)
    def nonfinite(*args,**kwargs):raise FloatingPointError('injected numerical guard')
    monkeypatch.setattr(g,'statistic',nonfinite)
    pp,q,r=g.apply(p,state,k,1)
    assert r['status']=='numerical_failure' and np.array_equal(pp,p)
    monkeypatch.setattr(g,'statistic',original)
    tiny=MarginalKernel(k.prior,[1.,1e-10,1e-20,0.]);tinyg=StepwiseGuide(g.layout,freeze(g.frozen['payload']['frequencies']['payload']['profile'],g.layout,tiny.alpha))
    pp,q,r=tinyg.apply(p,state,tiny,3)
    assert r['status']=='skipped' and np.array_equal(pp,p)


def case_guide(cn=False):
    case=fixture()[0];model,_=make(case,cn);model.eval()
    l=Layout.canonical(case['space'].types,case['space'].pairs,case['prior'])
    f=case['space'].encode(case['clean'])[0]
    profile=joint_profile(l.types,case['clean'],f,l.pairs)
    k=MarginalKernel(case['prior'],c.schedule(4));g=StepwiseGuide(l,freeze(profile,l,k.alpha))
    state=np.zeros(case['space'].total,dtype=np.uint8);state[case['clean_ids']]=1
    return case,model,k,g,state


@pytest.mark.parametrize('cn',[False,True])
def test_off_original_rng_full_small_chain_and_interruption(cn):
    from hsgd.runtime import reverse_step as original
    case,model,k,g,state=case_guide(cn)
    arm='D' if cn else 'C';off='B' if cn else 'A';seed=1789
    rng=np.random.default_rng(seed);before=c.rng_save(rng,'cpu')
    a,ra=original(model,case,state,k,4,20,seed,rng,'cpu',trace=True)
    arng=c.rng_save(rng,'cpu');c.rng_restore(before,rng,'cpu')
    b,rb=reverse_step(model,case,state,k,4,20,seed,rng,'cpu',arm=off,trace=True)
    assert np.array_equal(a,b);rng_equal(arng,c.rng_save(rng,'cpu'))
    c.rng_restore(before,rng,'cpu')
    saved=state.copy()
    with pytest.raises(InterruptedError):reverse_step(model,case,state,k,4,20,seed,rng,'cpu',arm=arm,guide=g,interrupt_block=2)
    assert np.array_equal(state,saved);rng_equal(before,c.rng_save(rng,'cpu'))
    for t in range(4,0,-1):
        rbefore=c.rng_save(rng,'cpu')
        value,record=reverse_step(model,case,state,k,t,20,seed,rng,'cpu',arm=arm,guide=g,trace=True)
        after=c.rng_save(rng,'cpu')
        # Guide uses no random input beyond original one full-slot uniform.
        c.rng_restore(rbefore,rng,'cpu');uniform=rng.random(case['space'].total)
        rng_equal(after,c.rng_save(rng,'cpu'))
        assert record['guidance']['time']==t and record['guidance']['next_time']==t-1
        assert np.array_equal(record['trace']['uniform'],uniform)
        assert record['exact_coverage']
        assert np.array_equal(value,uniform<record['trace']['q'])
        assert np.array_equal(state,saved) if t==4 else True
        state=value
    assert record['remaining_time']==0


def test_checkpoint_roundtrip_binding_tamper_and_pending_step(tmp_path):
    case,model,k,g,state=case_guide(False);rng=np.random.default_rng(1789)
    bound=binding(model,case,k,g,'C',20,1789)
    path=tmp_path/'committed.pt';start=c.rng_save(rng,'cpu')
    states=[]
    for t in range(4,0,-1):
        state,_=reverse_step(model,case,state,k,t,20,1789,rng,'cpu',arm='C',guide=g)
        states.append(state.copy())
        if t==3:save_checkpoint(path,state,2,rng,'cpu',bound)
    ending=c.rng_save(rng,'cpu')
    state,s=load_checkpoint(path,rng,'cpu',bound,len(state),4,2)
    with pytest.raises(InterruptedError):
        reverse_step(model,case,state,k,s,20,1789,rng,'cpu',arm='C',guide=g,interrupt_block=2)
    state,s=load_checkpoint(path,rng,'cpu',bound,len(state),4,2)
    for t in range(s,0,-1):
        state,_=reverse_step(model,case,state,k,t,20,1789,rng,'cpu',arm='C',guide=g)
        assert np.array_equal(state,states[4-t])
    rng_equal(ending,c.rng_save(rng,'cpu'))
    for field in ('python','torch_runtime_version','guide','alpha_sha256','seed','model_sha256','receipt_sha256'):
        wrong=copy.deepcopy(bound);wrong[field]='tampered'
        with pytest.raises(ValueError,match='binding'):load_checkpoint(path,rng,'cpu',wrong,len(state),4)
    with pytest.raises(ValueError,match='count'):load_checkpoint(path,rng,'cpu',bound,len(state),4,3)
    packet=torch.load(path,weights_only=False)
    packet['raw']=packet['raw']+b'corrupted RNG/payload'
    torch.save(packet,tmp_path/'corrupted.pt')
    with pytest.raises(ValueError,match='digest'):load_checkpoint(tmp_path/'corrupted.pt',rng,'cpu',bound,len(state),4)
    g.settings['delta_max']=.2
    with pytest.raises(ValueError,match='changed'):g.binding()


def test_model_arm_and_receipt_reject_wrong_native_weights():
    case,model,k,g,state=case_guide(False)
    with pytest.raises(ValueError):validate_arm(model,'D')
    with pytest.raises(ValueError):validate_arm(model,'C',{'completed_updates':10000},True)
    # Receipt shape is checked; actual BW import/verification is not claimed.
    receipt=dict(formal_training=True,completed_updates=6000,final=True,experiment_arm='baseline',
                 **{x:'a'*64 for x in ('checkpoint_sha256','source_sha256','config_sha256','data_sha256')})
    assert validate_arm(model,'C',receipt,True)


def test_proxy_is_not_expected_discrete_squared_error():
    _,l,_,k,g,state=setup_small();p=np.array([.17,.63,.46,.81])
    proxy=g.statistic(p,0)[0];expected=0.
    for bits in itertools.product((0,1),repeat=4):
        bits=np.array(bits);mass=np.prod(np.where(bits,p,1-p))
        expected+=mass*g.statistic(bits,0)[0]
    assert expected>proxy+1e-6


def test_fresh_process_recovery(tmp_path):
    import subprocess
    import sys
    from pathlib import Path
    case,model,k,g,state=case_guide(False);rng=np.random.default_rng(1789)
    bound=binding(model,case,k,g,'C',20,1789);path=tmp_path/'two_steps.pt'
    for t in range(4,0,-1):
        state,_=reverse_step(model,case,state,k,t,20,1789,rng,'cpu',arm='C',guide=g)
        if t==3:save_checkpoint(path,state,2,rng,'cpu',bound)
    expected=rng.random(10)
    output=tmp_path/'continued.npz'
    subprocess.run([sys.executable,str(Path(__file__).with_name('stepwise_resume_worker.py')),str(path),str(output)],check=True,timeout=60)
    with np.load(output) as z:
        assert np.array_equal(z['state'],state) and np.array_equal(z['uniform_after'],expected)


def test_underflow_and_factor_guard_never_normalize_untrusted_signal():
    # Probability-only engineering fixture: 30000 slots, not a sampled graph.
    types=np.r_[0,np.ones(30000,dtype=int)]
    l=Layout.canonical(types,[(0,1)],[.5])
    profile=joint_profile(types,np.empty((2,0),dtype=int),np.empty(0,dtype=int),l.pairs)
    k=MarginalKernel([.5],[1.,.5,0.]);g=StepwiseGuide(l,freeze(profile,l,k.alpha))
    p=np.full(30000,.5);state=np.zeros(30000,dtype=np.uint8)
    pp,q,r=g.apply(p,state,k,1)
    assert r['underflow_node_frequencies']>0 and r['reason']=='underflow_untrusted_gradient'
    assert np.array_equal(pp,p)
    g.omega['0']=np.full((32,1),np.pi) # injected contract violation, not a setting change
    with pytest.raises(FloatingPointError,match='factor lower bound'):
        g.statistic(p,0,True)
