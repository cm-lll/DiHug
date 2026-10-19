"""Independent small CPU truth and constrained-decoder tests; no real model."""
import copy
import itertools
import math
from functools import lru_cache
import numpy as np
import pytest
from joint_decode.contracts import Schema,EdgeLaw,canonical_assignment,validate_assignment,CONFIG
from joint_decode.exact import Universe,distributions,draw,metrics,uniform_assignment_counterexample
from joint_decode.decoder import decode
from joint_decode.experiments import homo_case,hetero_case,independent_six_census


@lru_cache(None)
def homogeneous():
    s,t,c,tr=homo_case();return s,t,c,tr,Universe(s)


@lru_cache(None)
def heterogeneous():
    s,t,r,w=hetero_case();return s,t,r,w,Universe(s)


def test_independent_regular_census_all32768_and_two_topologies():
    s,t,c,tr,u=homogeneous();m=u.role_mask(t);oracle=independent_six_census()
    assert len(u.codes)==32768 and m.sum()==70
    assert oracle['feasible']==70 and oracle['six_cycles']==60 and oracle['two_triangles']==10
    assert oracle['independent_combinatorial_counts']==dict(six_cycles=60,two_triangles=10)
    assert int((m&u.connected).sum())==60 and int((m&~u.connected).sum())==10
    assert np.array_equal(np.sort(np.unique(u.topology[m],return_counts=True)[1]),[10,60])
    assert (u.triangles[m&u.connected]==0).all() and (u.triangles[m&~u.connected]==2).all()
    assert len(np.unique(u.topology))==156


def test_joint_roles_are_not_separate_marginals():
    s,t,r,w,u=heterogeneous();joint=u.role_mask(t);marg=u.role_mask(t,True)
    code=int(np.sum(w.astype(np.uint32)<<np.arange(s.s)))
    assert joint.sum()==54 and marg.sum()==162 and marg[code] and not joint[code]
    with pytest.raises(ValueError,match='COMPLETE'):validate_assignment(s,t,s.degree(w))


@pytest.mark.parametrize('case',[0,1,2])
def test_five_distributions_full_normalization_and_exact_hard_roles(case):
    s,t,c,tr,u=homogeneous();p=np.full(s.s,.5) if case==0 else np.where(c if case==1 else tr,.8,.2)
    ds,info,logq,mask=distributions(u,t,EdgeLaw.from_probabilities(p))
    assert info['C_size']==info['admissible_C_size']==70
    rng=np.random.default_rng(CONFIG['sampling_seed']);uniforms=rng.random(4096)
    for name,w in ds.items():
        assert abs(w.sum()-1)<1e-12 and (w>=0).all()
        samples=draw(w,uniforms)
        if name!='independent':assert mask[samples].all() and (w[~mask]==0).all()
    if case==0:
        assert abs(info['q_C']-70/32768)<1e-14
        assert np.allclose(ds['uniform_feasible'],ds['conditional_T1'],atol=1e-16)
        assert np.allclose(ds['conditional_T1'],ds['conditional_T2'],atol=1e-16)
        assert info['MAP']['exact_FP64_tied_maximizers']==70


@pytest.mark.parametrize('preferred',[1,2])
def test_temperature_changes_distribution_entropy_score_tradeoff(preferred):
    s,t,c,tr,u=homogeneous();p=np.where(c if preferred==1 else tr,.8,.2)
    ds,_,score,_=distributions(u,t,EdgeLaw.from_probabilities(p))
    h=[];mean=[]
    for name in ('conditional_T1','conditional_T2'):
        w=ds[name];take=w>0;h.append(-np.sum(w[take]*np.log(w[take])));mean.append(w@score)
    assert h[1]>h[0] and mean[1]<mean[0]
    assert np.max(abs(ds['conditional_T1']-ds['conditional_T2']))>.01


def test_fixed_seed_and_shared_uniform_reproducibility():
    _,t,_,_,u=homogeneous();ds,_,_,_=distributions(u,t,EdgeLaw(np.zeros(15)))
    a=np.random.default_rng(199).random(4096);b=np.random.default_rng(199).random(4096)
    assert np.array_equal(draw(ds['conditional_T1'],a),draw(ds['conditional_T1'],b))


def test_uniform_role_assignment_mixture_is_not_correct_anonymous_conditioning():
    s,t,r,w,u=heterogeneous();law=EdgeLaw.from_probabilities(CONFIG['hetero_probabilities_canonical_family_slots'])
    ds,_,score,mask=distributions(u,t,law);d=uniform_assignment_counterexample(u,ds['conditional_T1'],score,mask)
    assert d['distinct_complete_role_assignments']==18
    assert d['max_log_assignment_partition']>d['min_log_assignment_partition']
    assert d['TV_uniform_assignment_mixture_vs_true_conditional']>.01
    matrices=u.degrees[np.flatnonzero(ds['conditional_T1']>0)].reshape(54,-1)
    assert len(np.unique(matrices,axis=0))==18 # roles not permanently locked


def test_expected_degree_exactly_matching_does_not_imply_sampled_matching():
    s=Schema(np.zeros(4,dtype=int),[(0,0)]);x=np.zeros(s.s,np.uint8);x[s.encode_edges(np.array([[0,2],[1,3]]))]=1
    t=s.profile(x);u=Universe(s);ds,info,_,mask=distributions(u,t,EdgeLaw.from_probabilities(np.full(s.s,1/3)))
    assert np.allclose(ds['independent']@u.degrees[:,:,0],np.ones(4),atol=1e-14)
    assert abs(info['q_C']-16/243)<1e-14 and info['q_C']<.07


@pytest.mark.parametrize('seed',range(4))
def test_fixed_joint_decoder_vs_exact_enumeration(seed):
    s,t,r,w,u=heterogeneous();law=EdgeLaw(np.random.default_rng(seed).normal(size=s.s));d=canonical_assignment(s,t)
    bits,report=decode(s,t,d,law,seconds=10)
    feasible=np.all(u.degrees==d[None],axis=(1,2));best=float(law.log_prob(u.bits)[feasible].max())
    assert bits is not None and report['optimal'] and report['status']=='optimal_fixed_matching'
    assert np.array_equal(s.degree(bits),d) and abs(report['log_q']-best)<2e-7
    assert report['log_q_gap'] is not None and abs(report['log_q_gap'])<2e-7
    assert report['no_anonymous_global_optimality_claim'] and report['no_sampling_claim']


@pytest.mark.parametrize('preferred',[0,1,2])
def test_same_type_binary_solver_and_double_endpoint_degrees(preferred):
    s,t,c,tr,u=homogeneous();law=EdgeLaw.from_probabilities(np.full(15,.5) if preferred==0 else np.where(c if preferred==1 else tr,.8,.2))
    d=canonical_assignment(s,t);bits,report=decode(s,t,d,law)
    assert report['optimal'] and np.array_equal(s.degree(bits),np.full((6,1),2)) and bits.sum()==6
    score=law.log_prob(u.bits);assert abs(report['log_q']-score[u.role_mask(t)].max())<2e-7
    assert any('MILP' in b['algorithm'] for b in report['blocks'])


def test_true_zero_one_support_and_finite_logit_saturation_are_different():
    s=Schema(np.array([0,0]),[(0,0)]);target0=s.profile(np.array([0]));d0=canonical_assignment(s,target0)
    law=EdgeLaw(np.array([1000.]))
    assert law.support[0]==-1 and law.log_prob([0])==-1000 and np.exp(law.logp[0])==1.
    bits,rep=decode(s,target0,d0,law);assert rep['optimal'] and bits[0]==0 and rep['log_q']==-1000
    bits,rep=decode(s,target0,d0,EdgeLaw.from_probabilities([1.]))
    assert bits is None and rep['status']=='infeasible'
    target1=s.profile(np.array([1]));d1=canonical_assignment(s,target1)
    bits,rep=decode(s,target1,d1,EdgeLaw.from_probabilities([0.]))
    assert bits is None and rep['status']=='infeasible'
    with pytest.raises(ValueError,match='empty'):distributions(Universe(s),target1,EdgeLaw.from_probabilities([0.]))
    bits,rep=decode(s,target1,d1,EdgeLaw.from_probabilities([1.]))
    assert rep['optimal'] and bits[0]==1


def test_same_type_LP_relaxation_can_be_fractional_and_support_infeasible():
    s,t,c,tr,u=homogeneous();one=np.zeros(s.s,np.uint8);one[s.encode_edges(np.array([[0,2,4],[1,3,5]]))]=1
    target=s.profile(one);d=canonical_assignment(s,target)
    support=np.where(tr,-1,0)
    # Two triangles, degree1 per vertex: fractional 1/2 edges satisfies LP,
    # but no integral matching exists in either odd component.
    fractional=tr*.5
    inc=np.zeros((6,15))
    for e,(a,b) in enumerate(s.slots.T):inc[a,e]+=1;inc[b,e]+=1
    assert np.array_equal(inc@fractional,np.ones(6))
    bits,rep=decode(s,target,d,EdgeLaw(np.zeros(15),support))
    assert bits is None and rep['status']=='infeasible'


def test_fixed_match_support_failure_does_not_imply_anonymous_failure():
    s=Schema(np.array([0,0,1]),[(0,1)]);reference=np.array([1,0]);t=s.profile(reference)
    law=EdgeLaw(np.zeros(2),[-1,0]);d=canonical_assignment(s,t)
    bits,rep=decode(s,t,d,law)
    assert bits is None and rep['status']=='infeasible'
    ds,info,score,mask=distributions(Universe(s),t,law)
    assert info['admissible_C_size']==1 and ds['conditional_T1'][1]==1.


def test_zero_degrees_and_duplicate_role_counts():
    s=Schema(np.array([0,0,1,1]),[(0,0),(0,1),(1,1)]);x=np.zeros(s.s,np.uint8);t=s.profile(x)
    assert all(g['counts']==[2] for g in t['groups'].values())
    bits,rep=decode(s,t,canonical_assignment(s,t),EdgeLaw(np.full(s.s,20.)))
    assert rep['optimal'] and bits.sum()==0 and np.array_equal(s.degree(bits),np.zeros((4,3),int))
    bad=copy.deepcopy(t);bad['groups']['0']['counts']=[1]
    with pytest.raises(ValueError):canonical_assignment(s,bad)


def test_distribution_relabel_equivariance_and_tied_MAP_counterexample():
    s,t,r,w,u=heterogeneous();p=np.array(CONFIG['hetero_probabilities_canonical_family_slots']);law=EdgeLaw.from_probabilities(p)
    perm=np.array([2,0,1,4,5,3]);slot_map,code_map=u.relabel_codes(perm)
    moved=np.empty_like(p);moved[slot_map]=p
    a,_,_,_=distributions(u,t,law);b,_,_,_=distributions(u,t,EdgeLaw.from_probabilities(moved))
    for name in ('independent','uniform_feasible','conditional_T1','conditional_T2'):
        assert np.allclose(a[name],b[name][code_map],atol=1e-12,rtol=1e-12)
    uniform,info,_,_=distributions(u,t,EdgeLaw(np.zeros(s.s)))
    assert info['MAP']['exact_FP64_tied_maximizers']==54
    chosen=info['MAP']['representative_code'];witness=None
    for pp in u.permutations:
        _,codes=u.relabel_codes(pp)
        if codes[chosen]!=chosen:witness=pp;break
    assert witness is not None # numbered deterministic tie rule NOT equivariant


def test_time_limited_feasible_solution_never_marked_optimal(monkeypatch):
    import joint_decode.decoder as module
    s=Schema(np.array([0,0,1,1]),[(0,1)]);x=np.array([1,0,0,1]);t=s.profile(x);d=canonical_assignment(s,t)
    native=module.linprog
    def limited(*args,**kwargs):
        sol=native(*args,**kwargs);sol.status=1;sol.message='ENGINEERING simulated solver limit after valid incumbent';return sol
    monkeypatch.setattr(module,'linprog',limited)
    bits,rep=decode(s,t,d,EdgeLaw(np.array([2.,0.,1.,3.])))
    assert bits is not None and rep['status']=='feasible_unproven' and rep['optimal'] is False


def test_engineering_score_has_no_reference_edge_input():
    from joint_decode.imdb import engineering_logits
    s,t,r,w,u=heterogeneous()
    assert not np.array_equal(r,w) and np.array_equal(engineering_logits(s),engineering_logits(s))
    import inspect
    assert list(inspect.signature(engineering_logits).parameters)==['schema']


def test_uniform_feasible_metrics_against_analytic_values():
    s,t,c,tr,u=homogeneous();ds,_,logq,mask=distributions(u,t,EdgeLaw(np.zeros(15)))
    samples=draw(ds['uniform_feasible'],np.random.default_rng(77).random(4096))
    m=metrics(u,ds['uniform_feasible'],logq,mask,samples,70)
    assert abs(m['entropy_nats']-math.log(70))<1e-12
    assert abs(m['effective_graph_count_exp_entropy']-70)<1e-10
    assert abs(m['maximum_single_graph_probability']-1/70)<1e-14
    assert abs(m['expected_two_graph_edge_symmetric_difference']-7.2)<1e-12
    assert abs(m['connected_rate']-6/7)<1e-14 and abs(m['mean_triangles']-2/7)<1e-14
    assert m['exact_type_preserving_topology_classes']==2


def test_independent_metrics_against_analytic_values():
    s,t,c,tr,u=homogeneous();ds,_,logq,mask=distributions(u,t,EdgeLaw(np.zeros(15)))
    samples=draw(ds['independent'],np.random.default_rng(77).random(4096))
    m=metrics(u,ds['independent'],logq,mask,samples,32768)
    connected={1:1}
    for n in range(2,7):connected[n]=2**math.comb(n,2)-sum(math.comb(n-1,k-1)*connected[k]*2**math.comb(n-k,2) for k in range(1,n))
    assert connected[6]==26704 and abs(m['connected_rate']-26704/32768)<1e-14
    assert abs(m['entropy_nats']-15*math.log(2))<1e-12
    assert abs(m['mean_triangles']-2.5)<1e-12 and abs(m['expected_two_graph_edge_symmetric_difference']-7.5)<1e-12
    assert abs(m['mean_original_log_q_nats']+15*math.log(2))<1e-12


def test_all_type_preserving_joint_assignments_are_realizable_in_complete_schema():
    s,t,r,w,u=heterogeneous();mask=u.role_mask(t)
    matrices,counts=np.unique(u.degrees[mask].reshape(-1,18),axis=0,return_counts=True)
    expected=1
    for g in t['groups'].values():expected*=math.factorial(g['nodes'])//math.prod(math.factorial(v) for v in g['counts'])
    assert len(matrices)==expected==18 and np.array_equal(counts,np.full(18,3))
