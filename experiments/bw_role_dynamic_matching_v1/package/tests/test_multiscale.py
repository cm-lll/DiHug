import copy
import itertools
import numpy as np
import pytest
from hsgd.kernel import MarginalKernel
from role_degree.role_statistics import joint_profile
from stepwise_role.core import Layout
from multiscale_role.core import freeze,MultiScaleGuide
from multiscale_role.tail import empirical_tail,binomial_tail_states,combine


def example():
    # A triangle + cross-type edges: homogeneous edge has two endpoint terms.
    types=np.array([0,0,0,1,1]);pairs=[(0,0),(0,1),(1,1)]
    l=Layout.canonical(types,pairs,[.5,.25,0.])
    selected=np.array([0,1,3,6]);profile=joint_profile(types,l.slots[:,selected],l.families[selected],pairs)
    k=MarginalKernel(l.prior,[1.,.7,.2,0.]);g=MultiScaleGuide(l,freeze(profile,l,k.alpha))
    p=np.linspace(.13,.77,len(l.families));p[l.fixed0]=0
    state=np.zeros(len(p),dtype=np.uint8);state[[0,3]]=1
    return l,profile,k,g,p,state


def enumerate_cf_tail(l,g,prob,s):
    # Independent actual-graph enumeration; no DP, scan or production statistic.
    cf={c:np.zeros(96,complex) for c in g.omega}
    tail={c:np.zeros(len(g.channels[c])) for c in g.omega}
    free=np.flatnonzero((prob>0)&(prob<1));fixed=(prob==1).astype(int)
    for bits in itertools.product((0,1),repeat=len(free)):
        x=fixed.copy();x[free]=bits;mass=np.prod(np.where(x[free],prob[free],1-prob[free]))
        d=np.zeros((len(l.types),len(l.pairs)),int)
        for e,(u,v) in enumerate(l.slots.T):
            d[u,l.families[e]]+=x[e];d[v,l.families[e]]+=x[e]
        for c,w in g.omega.items():
            rows=d[l.types==int(c)];cf[c]+=mass*np.exp(1j*rows@w.T).mean(0)
            for j,ch in enumerate(g.channels[c]):
                degrees=rows[:,ch['families']].sum(1)
                # Independent scalar binomial sum, explicitly summing successes>=3.
                import math
                vals=[sum(math.comb(int(n),v)*ch['p']**v*(1-ch['p'])**(int(n)-v) for v in range(3,int(n)+1)) for n in degrees]
                tail[c][j]+=mass*np.mean(vals)
    return cf,tail


@pytest.mark.parametrize('s',[0,1,2,3])
def test_forward_noisy_target_and_conditional_statistics_enumeration(s):
    l,profile,k,g,p,state=example()
    clean=np.zeros(len(p));clean[[0,1,3,6]]=1
    noisy=k.alpha[s]*clean+(1-k.alpha[s])*l.prior[l.families]
    cf,tail=enumerate_cf_tail(l,g,noisy,s)
    target_cf,target_tail=g.target(s)
    for c in cf:
        assert np.max(abs(cf[c]-target_cf[c]))<3e-13
        assert np.max(abs(tail[c]-target_tail[c]),initial=0)<3e-13
    cf2,tail2=enumerate_cf_tail(l,g,p,s)
    loss,gp,detail=g.statistic(p,s,True)
    for c in cf2:
        assert np.max(abs(cf2[c]-detail['mean_cf'][c]))<3e-13
        assert np.max(abs(tail2[c]-detail['mean_tail'][c]),initial=0)<3e-13


@pytest.mark.parametrize('t',[1,2,3])
def test_original_posterior_and_complete_objective_gradient(t):
    l,pf,k,g,p,state=example()
    q,loss,grad,detail,error=g.posterior_gradient(p,state,k,t)
    eps=1e-5;num=np.zeros(len(p))
    for i in np.flatnonzero(~l.fixed0):
        z=np.log(p[i])-np.log1p(-p[i]);a=p.copy();b=p.copy()
        a[i]=1/(1+np.exp(-z-eps));b[i]=1/(1+np.exp(-z+eps))
        num[i]=(g.statistic(k.reverse(a,state,l.families,t-1,t),t-1)[0]-g.statistic(k.reverse(b,state,l.families,t-1,t),t-1)[0])/(2*eps)
    assert np.allclose(num,grad,atol=2e-8,rtol=2e-5)
    assert error<5e-14
    if t==1:assert np.max(abs(q-p))<1e-14


def test_tail_derivative_two_endpoints_and_probability_not_mean_degree():
    l,pf,k,g,p,state=example();q=p.copy()
    # Check each type/channel derivative directly in probability coordinates.
    for c,entries in g.tail_layout.items():
        for j,(ids,nodes) in enumerate(entries):
            ch=g.channels[c][j];n=l.sizes[int(c)]
            value,inc,detail=empirical_tail(q,ids,nodes,n,ch['p'],True)
            actual=np.bincount(ids,weights=inc,minlength=len(q))/n
            for e in np.unique(ids):
                a=q.copy();b=q.copy();a[e]+=1e-6;b[e]-=1e-6
                num=(empirical_tail(a,ids,nodes,n,ch['p'])[0]-empirical_tail(b,ids,nodes,n,ch['p'])[0])/(2e-6)
                assert abs(num-actual[e])<2e-9
    # Equal mean degree, different random degree law => different tail signal.
    ids=np.arange(4);nodes=np.zeros(4,dtype=int)
    a=empirical_tail(np.array([1.,1.,0.,0.]),ids,nodes,1,.5)[0]
    b=empirical_tail(np.full(4,.5),ids,nodes,1,.5)[0]
    assert a==0 and b>0


def test_full_joint_information_frequencies_not_evaluation_bank_and_no_duplicates():
    # Same per-relation marginals, different joint combinations.
    l=Layout.canonical(np.array([0,0,1,2]),[(0,1),(0,2)],[.5,.5])
    pf=joint_profile(l.types,l.slots[:,[0,2]],l.families[[0,2]],l.pairs)
    other=copy.deepcopy(pf);other['groups']['0']['rows']=[[0,1],[1,0]];other['groups']['0']['counts']=[1,1]
    a=MultiScaleGuide(l,freeze(pf,l,[1.,.5,0.]));b=MultiScaleGuide(l,freeze(other,l,[1.,.5,0.]))
    assert np.array_equal(a.omega['0'],b.omega['0'])
    assert (np.count_nonzero(a.omega['0'],axis=1)>=2).any()
    assert np.max(abs(a.target(0)[0]['0']-b.target(0)[0]['0']))>1e-5
    for c,channels in a.channels.items():
        masks=[tuple(x['families']) for x in channels];assert len(masks)==len(set(masks))
    p=np.array([.17,.63,.46,.81]);state=np.array([1,0,0,1]);k=MarginalKernel(l.prior,[1.,.5,0.])
    _,qa,ra=a.apply(p,state,k,1);_,qb,rb=b.apply(p,state,k,1)
    assert ra['status']==rb['status']=='accepted' and np.max(abs(qa-qb))>1e-6
    zero=a.apply(np.full(4,.5),state,k,1)[2]
    assert zero['loss_before']>0 and zero['reason']=='zero_or_roundoff_gradient'


def test_renumber_endpoint_symmetry_support_and_no_global_rng():
    l,pf,k,g,p,state=example();perm=np.array([2,0,1,4,3]);edges=np.sort(perm[l.slots],axis=0)
    ll=Layout(l.types,edges,l.families,l.pairs,l.prior);gg=MultiScaleGuide(ll,freeze(pf,ll,k.alpha))
    before=np.random.get_state()
    a,qa,ra=g.apply(p,state,k,1);b,qb,rb=gg.apply(p,state,k,1)
    assert np.allclose(qa,qb,atol=2e-10,rtol=1e-9)
    assert abs(ra['loss_before']-rb['loss_before'])<2e-10
    assert np.array_equal(np.sort(l.slots[::-1],axis=0),l.slots)
    assert np.array_equal(a[l.fixed0],p[l.fixed0])
    after=np.random.get_state();assert before[0]==after[0] and np.array_equal(before[1],after[1]) and before[2:]==after[2:]
    with pytest.raises(ValueError):g.apply(np.full(len(p),.5),state,k,1)
    bad=p.copy();bad[0]=np.nan
    with pytest.raises(ValueError):g.apply(bad,state,k,1)


def test_binomial_endpoints_underflow_and_untrusted_gradient_skip():
    assert np.array_equal(binomial_tail_states(np.array([0,1,2,3,100]),0),np.tile([1.,0,0,0],(5,1)))
    assert np.array_equal(binomial_tail_states(np.array([0,1,2,3,100]),1),np.array([[1.,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[0,0,0,1]]))
    # Probability-only bounded numerical stress; no graph is generated.
    types=np.r_[0,np.ones(4000,dtype=int)];l=Layout.canonical(types,[(0,1)],[.5])
    pf=joint_profile(types,np.empty((2,0),int),np.empty(0,int),l.pairs)
    k=MarginalKernel([.5],[1.,.5,0.]);g=MultiScaleGuide(l,freeze(pf,l,k.alpha))
    p=np.full(4000,.5);state=np.zeros(4000,dtype=np.uint8)
    pp,q,r=g.apply(p,state,k,1)
    assert r['underflow_node_frequencies']>0 and r['status']=='skipped' and np.array_equal(p,pp)


def test_controller_retained_and_failure_rollback(monkeypatch):
    l,pf,k,g,p,state=example()
    assert g.settings['delta_max']==.1 and g.settings['max_trials']==8
    old=g.statistic
    def no_descent(q,s,gradient=False):
        a,b,c=old(q,s,gradient)
        return (10. if not gradient else a),b,c
    monkeypatch.setattr(g,'statistic',no_descent)
    pp,q,r=g.apply(p,state,k,1)
    assert np.array_equal(pp,p) and r['trials']==8 and r['reason']=='no_verified_descent'
    assert r['components_before']==r['components_after']


@pytest.mark.parametrize('cn',[False,True])
def test_new_objective_existing_sampler_transaction_and_binding(tmp_path,cn):
    from test_stepwise import case_guide
    from hsgd.bw_preflight import common as c
    from hsgd.fusion_checks import rng_equal
    from hsgd.runtime import reverse_step as original
    from multiscale_role.sampling import reverse_step,binding,save_checkpoint,load_checkpoint
    case,model,k,old,state=case_guide(cn)
    profile=old.frozen['payload']['frequencies']['payload']['profile']
    g=MultiScaleGuide(old.layout,freeze(profile,old.layout,k.alpha))
    rng=np.random.default_rng(1789);before=c.rng_save(rng,'cpu');off='B' if cn else 'A';on='D' if cn else 'C'
    a,_=original(model,case,state,k,4,20,1789,rng,'cpu');after=c.rng_save(rng,'cpu')
    c.rng_restore(before,rng,'cpu')
    b,_=reverse_step(model,case,state,k,4,20,1789,rng,'cpu',arm=off)
    assert np.array_equal(a,b);rng_equal(after,c.rng_save(rng,'cpu'))
    c.rng_restore(before,rng,'cpu')
    with pytest.raises(InterruptedError):reverse_step(model,case,state,k,4,20,1789,rng,'cpu',arm=on,guide=g,interrupt_block=2)
    rng_equal(before,c.rng_save(rng,'cpu'))
    checks=[0]
    def interrupt_during_statistic():
        checks[0]+=1
        if checks[0]>=100:raise InterruptedError('injected mid-statistic safe stop')
    with pytest.raises(InterruptedError):
        reverse_step(model,case,state,k,4,20,1789,rng,'cpu',arm=on,guide=g,check_time=interrupt_during_statistic)
    rng_equal(before,c.rng_save(rng,'cpu'))
    x,record=reverse_step(model,case,state,k,4,20,1789,rng,'cpu',arm=on,guide=g,trace=True)
    assert record['guidance']['objective'].startswith('0.5*') and record['exact_coverage']
    assert np.array_equal(x,record['trace']['uniform']<record['trace']['q'])
    bound=binding(model,case,k,g,on,20,1789);assert 'multiscale_code_sha256' in bound
    save_checkpoint(tmp_path/'state.pt',x,3,rng,'cpu',bound)
    y,s=load_checkpoint(tmp_path/'state.pt',rng,'cpu',bound,len(x),4,3)
    assert np.array_equal(x,y) and s==3
    bad=dict(bound,multiscale_code_sha256='wrong')
    with pytest.raises(ValueError):load_checkpoint(tmp_path/'state.pt',rng,'cpu',bad,len(x),4,3)


def test_exact_one_support_and_internal_nonfinite_failure(monkeypatch):
    l=Layout.canonical(np.array([0,0,1,1]),[(0,0),(0,1),(1,1)],[0.,.5,1.])
    pf=joint_profile(l.types,l.slots[:,[1,5]],l.families[[1,5]],l.pairs)
    k=MarginalKernel(l.prior,[1.,.7,0.]);g=MultiScaleGuide(l,freeze(pf,l,k.alpha))
    p=np.array([0.,0.,.3,.8,1.,1.]);state=np.array([0,1,0,1,0,1])
    pp,q,r=g.apply(p,state,k,1)
    assert np.array_equal(pp[[0,1,4,5]],p[[0,1,4,5]]) and q[0]==0 and q[-1]==1
    def broken(*args,**kwargs):raise FloatingPointError('injected nonfinite statistic')
    monkeypatch.setattr(g,'statistic',broken)
    pp,q,r=g.apply(p,state,k,1)
    assert r['status']=='numerical_failure' and np.array_equal(pp,p)
