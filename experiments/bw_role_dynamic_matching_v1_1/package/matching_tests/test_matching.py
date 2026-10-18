"""Small ENGINEERING fixtures; no scientific model/sample or private BW inputs."""
import copy
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
import torch
from dynamic_role.core import exact_match,MatchingGuide,Layout,freeze,SETTINGS
from helpers import setup,chain_setup,REFERENCE


@pytest.mark.parametrize('seed',range(5))
def test_exhaustive_joint_assignment_and_bounds(seed):
    rng=np.random.default_rng(seed);mu=rng.random((5,3))*3;roles=rng.integers(0,4,(5,3)).astype(float);scale=np.array([2.,3.,4.]);dims=np.arange(3)
    a,r=exact_match(mu,roles,scale,dims)
    costs=[float((((mu-roles[list(p)])/scale)**2).sum()/3) for p in itertools.permutations(range(5))]
    assert abs(r['cost_sum']-min(costs))<1e-12
    assert r['assignment_objective_lower_bound']<=r['cost_sum']+1e-12
    assert r['primal_dual_gap_bound']<1e-10


@pytest.mark.parametrize('seed',range(5))
def test_sorting_equals_generic_exact_cost(seed):
    rng=np.random.default_rng(seed);mu=rng.random((9,2));roles=rng.integers(0,5,(9,2)).astype(float)
    a,x=exact_match(mu,roles,np.array([3.,5.]),np.array([1]))
    b,y=exact_match(mu,roles,np.array([3.,5.]),np.array([1]),method='generic')
    assert abs(x['cost_sum']-y['cost_sum'])<1e-12


def test_target_integer_multiplicity_joint_information_and_no_ids():
    _,l,pf,k,g,p,state=setup();bad=copy.deepcopy(pf);bad['groups']['0']['counts']=[1.5, .5]
    with pytest.raises(ValueError):freeze(bad,l,k.alpha,REFERENCE)
    bad=copy.deepcopy(pf);bad['groups']['0']['counts']=[1,2]
    with pytest.raises(ValueError):freeze(bad,l,k.alpha,REFERENCE)
    other=copy.deepcopy(pf);other['groups']['0']['rows']=[[0,1],[1,0]]
    gg=MatchingGuide(l,freeze(other,l,k.alpha,REFERENCE))
    # Equal one-dimensional histograms, different JOINT target and decisions.
    assert g.frozen['payload']['groups']['0']['scale']==gg.frozen['payload']['groups']['0']['scale']
    a,qa,ra=g.apply(p,state,k,1);b,qb,rb=gg.apply(p,state,k,1)
    assert ra['status']==rb['status']=='accepted'
    assert np.max(abs(qa-qb))>1e-6
    assert set(g.frozen['payload']['groups']['0'])=={'nodes','rows','counts','scale','effective_dimensions'}


def test_clean_gradient_finite_difference_including_both_endpoints():
    _,l,_,k,g,p,_=setup();assignment,_=g.match(p);loss,gp,_=g.loss_gradient(p,assignment)
    z=np.log(p)-np.log1p(-p);glogit=gp*p*(1-p)
    numeric=[]
    for j in range(len(p)):
        zz=z.copy();zz[j]+=1e-5;a=g.loss_gradient(1/(1+np.exp(-zz)),assignment,False)[0]
        zz[j]-=2e-5;b=g.loss_gradient(1/(1+np.exp(-zz)),assignment,False)[0]
        numeric.append((a-b)/2e-5)
    assert np.allclose(numeric,glogit,atol=SETTINGS['finite_difference_atol'],rtol=SETTINGS['finite_difference_rtol'])
    # Independent endpoint loop differentiates all type-normalized residuals.
    mu=g.expected_degrees(p);ref=np.zeros_like(p)
    for e,(u,v) in enumerate(l.slots.T):
        r=l.families[e]
        for node in (u,v):
            c=str(l.types[node]);spec=g._groups[c];i=np.flatnonzero(spec['nodes']==node)[0]
            ref[e]+=2*(mu[node,r]-spec['roles'][assignment[c][i],r])/spec['scale'][r]**2/(len(g._groups)*len(spec['nodes'])*len(spec['dims']))
    assert np.allclose(gp,ref,atol=1e-15)


def test_same_type_two_endpoint_gradient():
    from role_degree.role_statistics import joint_profile
    l=Layout.canonical([0,0,0],[(0,0)],[.5]);p=np.array([.2,.4,.8])
    profile=joint_profile(l.types,l.slots[:,[0,1]],np.array([0,0]),l.pairs)
    g=MatchingGuide(l,freeze(profile,l,[1.,.5,0.],REFERENCE));a,_=g.match(p)
    loss,gradient,_=g.loss_gradient(p,a)
    for e in range(3):
        pp=p.copy();pp[e]+=1e-5;aa=g.loss_gradient(pp,a,False)[0];pp[e]-=2e-5;bb=g.loss_gradient(pp,a,False)[0]
        assert abs((aa-bb)/2e-5-gradient[e])<2e-8


def test_unique_matching_renumbering_and_endpoint_symmetry():
    _,l,pf,k,g,p,state=setup();perm=np.array([1,0,2,3])
    ll=Layout(l.types,np.sort(perm[l.slots],axis=0),l.families,l.pairs,l.prior)
    gg=MatchingGuide(ll,freeze(pf,ll,k.alpha,REFERENCE))
    aa,ar=g.match(p);bb,br=gg.match(p)
    assert ar['0']['status']=='separated_at_declared_resolution'
    assert np.array_equal(aa['0'][::-1],bb['0'])
    a,qa,ra=g.apply(p,state,k,3);b,qb,rb=gg.apply(p,state,k,3)
    assert np.allclose(a,b,atol=2e-10,rtol=1e-9)
    assert np.allclose(qa,qb,atol=2e-10,rtol=1e-9)
    assert abs(ra['loss_before']-rb['loss_before'])<2e-10
    assert np.array_equal(np.sort(l.slots[::-1],axis=0),l.slots)


def test_different_role_tie_counterexample_and_duplicate_roles():
    mu=np.array([[.5],[.5]]);roles=np.array([[0.],[1.]])
    a,r=exact_match(mu,roles,np.ones(1),np.array([0]));b,rr=exact_match(mu[::-1],roles,np.ones(1),np.array([0]))
    assert r['status']=='nonidentical_role_tie_witness' and r['witnesses'][0]['exact_float_zero']
    assert np.array_equal(a,b) and not np.array_equal(a[::-1],b)
    assert not np.array_equal(mu-roles[a],(mu[::-1]-roles[b])[::-1])
    _,dup=exact_match(mu,np.ones((2,1)),np.ones(1),np.array([0]))
    assert dup['status']=='duplicate_roles_only_at_declared_resolution'


def test_symmetric_graph_tied_gradients_are_not_claimed_equivariant():
    _,l,pf,k,g,p,state=setup();p=np.full(4,.5)
    a,audit=g.match(p);loss,gp,_=g.loss_gradient(p,a)
    assert audit['0']['status']=='nonidentical_role_tie_witness'
    assert loss>0 and np.max(abs(gp))>0 # hard selection can break symmetry
    assert not np.array_equal(gp[:2],gp[:2][::-1])


@pytest.mark.parametrize('t',[1,2,3,4])
def test_controller_fixed_vs_reoptimized_and_original_posterior(t):
    _,l,_,k,g,p,state=setup();pp,q,r=g.apply(p,state,k,t)
    assert r['status']=='accepted'
    assert r['loss_after_fixed']<r['loss_before']-r['descent_threshold']
    assert r['loss_after_reoptimized']<=r['loss_after_fixed']+r['reoptimization_allowance']
    assert np.allclose(q,k.reverse(pp,state,l.families,t-1,t),atol=5e-14,rtol=0)
    if t==1:assert np.allclose(q,pp,atol=5e-14,rtol=0)
    # Same p -> same CLEAN target loss/gradient independent of t/state.
    assert r['loss_before']==g.apply(p,1-state,k,1)[2]['loss_before']


def test_exact_support_saturation_zero_gradient_and_incomplete_slots():
    from role_degree.role_statistics import joint_profile
    from hsgd.kernel import MarginalKernel
    l=Layout.canonical(np.array([0,0,1,1]),[(0,0),(0,1),(1,1)],[0.,.5,1.])
    profile=joint_profile(l.types,l.slots[:,[1,5]],l.families[[1,5]],l.pairs)
    k=MarginalKernel(l.prior,[1.,.7,0.]);g=MatchingGuide(l,freeze(profile,l,k.alpha,REFERENCE))
    p=np.array([0.,0.,.3,.8,1.,1.]);state=np.array([0,1,0,1,0,1])
    pp,q,r=g.apply(p,state,k,1)
    assert np.array_equal(pp[[0,1,4,5]],p[[0,1,4,5]]) and r['numerical_saturated_slots']==2
    binary=np.array([0.,0.,0.,1.,1.,1.]);out,q,rr=g.apply(binary,state,k,1)
    assert rr['status']=='skipped' and rr['reason']=='no_movable_probabilities'
    assert np.array_equal(out,binary)
    with pytest.raises(ValueError):Layout(l.types,l.slots[:,:-1],l.families[:-1],l.pairs,l.prior)
    with pytest.raises(ValueError):g.apply(np.full(len(p),.5),state,k,1)
    with pytest.raises(ValueError):g.apply(np.full(len(p),np.nan),state,k,1)


def test_backtracking_failure_nonfinite_and_zero_gradient(monkeypatch):
    _,_,_,k,g,p,state=setup();original=g.loss_gradient
    def stuck(pp,assignments,gradient=True):
        loss,gp,detail=original(pp,assignments,gradient)
        return (loss if gradient else 1e6),gp,detail
    monkeypatch.setattr(g,'loss_gradient',stuck)
    # Forced engineering failure affects rematch too -> numerical failure retained.
    out,q,r=g.apply(p,state,k,1)
    assert r['status']=='numerical_failure' and np.array_equal(out,p)
    def zero(pp,assignments,gradient=True):
        loss,gp,detail=original(pp,assignments,gradient)
        return loss,np.zeros_like(pp) if gradient else None,detail
    monkeypatch.setattr(g,'loss_gradient',zero)
    out,q,r=g.apply(p,state,k,1)
    assert r['status']=='skipped' and r['reason']=='zero_or_roundoff_gradient'
    def nonfinite(*args,**kwargs):raise FloatingPointError('engineering injected nonfinite')
    monkeypatch.setattr(g,'loss_gradient',nonfinite)
    out,q,r=g.apply(p,state,k,1)
    assert r['status']=='numerical_failure' and np.array_equal(out,p)


def test_target_external_mutation_cannot_change_cached_roles():
    _,_,_,_,g,p,_=setup();a,ar=g.match(p);before=g.binding();copy_target=g.frozen
    copy_target['payload']['groups']['0']['rows'][0][0]=99
    assert before==g.binding()
    g._groups['0']['roles'][0,0]=99
    with pytest.raises(ValueError,match='mutation'):g.binding()


@pytest.mark.parametrize('cn',[False,True])
def test_original_disabled_path_rng_coverage_and_partial_step_rollback(cn):
    from dynamic_role.sampling import reverse_step
    from hsgd.runtime import reverse_step as original
    from hsgd.bw_preflight.common import rng_save
    from hsgd.fusion_checks import rng_equal
    case,m,k,g,state=chain_setup(cn);a=np.random.default_rng(91);b=np.random.default_rng(91)
    x,r=original(m,case,state,k,4,20,1789,a,'cpu',backend='upstream',trace=True)
    y,s=reverse_step(m,case,state,k,4,20,1789,b,'cpu',arm='B' if cn else 'A',guide=None,trace=True)
    assert np.array_equal(x,y) and a.bit_generator.state==b.bit_generator.state
    rng=np.random.default_rng(62);before=rng_save(rng,'cpu')
    with pytest.raises(InterruptedError):reverse_step(m,case,state,k,4,20,1789,rng,'cpu',arm='D' if cn else 'C',guide=g,interrupt_block=1)
    rng_equal(before,rng_save(rng,'cpu'))
    z,r=reverse_step(m,case,state,k,4,20,1789,rng,'cpu',arm='D' if cn else 'C',guide=g,trace=True)
    assert r['exact_coverage'] and r['covered_slots']==case['space'].total
    assert np.array_equal(r['trace']['coverage'],np.ones(case['space'].total))


def test_small_full_chain_and_fresh_process_resume(tmp_path):
    from dynamic_role.sampling import reverse_step,binding,save_checkpoint,load_checkpoint
    case,m,k,g,state=chain_setup();rng=np.random.default_rng(91)
    bound=binding(m,case,k,g,'C',20,1789)
    for t in (4,3):state,_=reverse_step(m,case,state,k,t,20,1789,rng,'cpu',arm='C',guide=g)
    ck=tmp_path/'commit.pt';save_checkpoint(ck,state,2,rng,'cpu',bound)
    for t in (2,1):state,_=reverse_step(m,case,state,k,t,20,1789,rng,'cpu',arm='C',guide=g)
    nxt=rng.random(10);dest=tmp_path/'resumed.npz'
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    subprocess.run([sys.executable,str(Path(__file__).with_name('resume_worker.py')),str(ck),str(dest)],check=True,env=env,timeout=60)
    with np.load(dest) as z:assert np.array_equal(z['state'],state) and np.array_equal(z['next_uniform'],nxt)
    bad=copy.deepcopy(bound);bad['guide']['target_sha256']='x'*64
    with pytest.raises(ValueError):load_checkpoint(ck,np.random.default_rng(3),'cpu',bad,case['space'].total,4,2)
    bad=copy.deepcopy(bound);bad['guide']['scipy_runtime_version']='different'
    with pytest.raises(ValueError):load_checkpoint(ck,np.random.default_rng(3),'cpu',bad,case['space'].total,4,2)


def test_output_overlap_rejected_before_any_write(tmp_path):
    source=tmp_path/'source';source.mkdir();(source/'evidence').write_text('preserve')
    before=list(source.rglob('*'))
    cmd=[sys.executable,str(Path(__file__).resolve().parents[1]/'run_dynamic.py'),'pack',
         '--export',str(source),'--short-run',str(tmp_path/'short'),'--output',str(source/'bad')]
    r=subprocess.run(cmd,capture_output=True,text=True,timeout=10)
    assert r.returncode!=0 and list(source.rglob('*'))==before


def test_probability_entropy_exact_boundaries():
    from dynamic_role.runner import distributions
    _,l,_,_,_,_,_=setup();r=distributions(np.array([0.,1.,.5,.5]),l)
    assert abs(r['mean_entropy']-np.log(2)/2)<1e-15 and r['exact_zero']==r['exact_one']==1


def test_near_boundary_probabilities_no_epsilon_and_no_product_underflow():
    _,l,_,k,g,p,state=setup();p=np.array([np.nextafter(0.,1.),np.nextafter(1.,0.),0.,1.])
    pp,q,r=g.apply(p,state,k,1)
    assert pp[2]==0 and pp[3]==1 and np.isfinite(pp).all() and np.isfinite(q).all()
    assert r['numerical_saturated_slots']==2
    a,_=g.match(p);loss,gradient,_=g.loss_gradient(p,a)
    assert np.isfinite(loss) and np.isfinite(gradient).all()


def test_controller_has_no_rng_consumption_and_recomputes_matching(monkeypatch):
    import random
    from hsgd.bw_preflight.common import rng_save
    from hsgd.fusion_checks import rng_equal
    _,_,_,k,g,p,state=setup();rng=np.random.default_rng(125);before=rng_save(rng,'cpu')
    original=g.match;calls=[]
    def watch(prob,check=lambda:None):calls.append(prob.copy());return original(prob,check)
    monkeypatch.setattr(g,'match',watch)
    pp,q,r=g.apply(p,state,k,1);g.apply(p,state,k,2)
    assert len(calls)==4 and np.array_equal(calls[0],p) and np.array_equal(calls[2],p)
    rng_equal(before,rng_save(rng,'cpu'))


def test_direct_hidden_worker_cannot_write_existing_evidence(tmp_path):
    out=tmp_path/'historical';out.mkdir();(out/'evidence.json').write_text('{"preserve":true}')
    before={p.name:p.read_bytes() for p in out.iterdir()}
    command=[sys.executable,str(Path(__file__).resolve().parents[1]/'run_dynamic.py'),'check',
        '--portable',str(tmp_path/'missing.json'),'--portable-sha256','0'*64,'--output',str(out),
        '--worker','--deadline','12345','--cleanup-deadline','12351']
    result=subprocess.run(command,capture_output=True,text=True,timeout=10)
    assert result.returncode!=0 and {p.name:p.read_bytes() for p in out.iterdir()}==before
