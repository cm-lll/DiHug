"""Deterministic tiny engineering fixtures, NOT notebook B1 observations."""
import copy
import itertools
import json
from pathlib import Path
import sys
import subprocess
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tests'))
from test_multiscale import example
from hsgd.kernel import MarginalKernel
from stepwise_role.core import Layout
from role_degree.role_statistics import joint_profile
from multiscale_role.core import MultiScaleGuide,freeze
from multiscale_role.evaluation import genre64,degree_w1
from b1_attribution.math import (TOL,components,posterior_chain,sum_certificate,replay,alignment,
                                 independent_cf_gradient,first_order,genre_laws)
from b1_attribution.runner import old_case_binding,compare_metrics
from b1_attribution.entry import admit,validate_done
from test_legacy_repairs import fixture as four_type_fixture


def scalar_fd(f,p,free):
    result=np.zeros_like(p);h=1e-5
    for j in np.flatnonzero(free):
        a=p.copy();b=p.copy();z=np.log(p[j])-np.log1p(-p[j])
        a[j]=1/(1+np.exp(-(z+h)));b[j]=1/(1+np.exp(-(z-h)))
        result[j]=(f(a)-f(b))/(2*h)
    return result


@pytest.mark.parametrize('t',[1,2])
def test_components_preserve_weights_support_posterior_and_gradient_sum(t):
    l,pf,k,g,p,state=example();q,factor,_=posterior_chain(p,state,k,g,t)
    loss,raw=g.statistic(q,t-1,True)[:2]
    detail,parts=components(q,g,t-1,True,factor);certificate,total=sum_certificate(parts,raw*factor)
    assert abs(loss-detail['total_loss'])<1e-15 and certificate['passed']
    max_fd=0.
    for key,value in parts.items():
        def f(x):return components(k.reverse(x,state,l.families,t-1,t),g,t-1)[0]['losses'][key]
        fd=scalar_fd(f,p,(p>0)&(p<1)&~l.fixed0&~l.fixed1)
        max_fd=max(max_fd,float(abs(fd-value).max()))
        assert np.allclose(fd,value,atol=TOL['finite_difference_atol'],rtol=TOL['finite_difference_rtol'])
        assert np.array_equal(value[l.fixed0|l.fixed1],np.zeros((l.fixed0|l.fixed1).sum()))
    assert max_fd<1e-10


def test_replay_only_chosen_step_identical_without_calling_apply_again(monkeypatch):
    l,pf,k,g,p,state=example();old_p,old_q,record=g.apply(p,state,k,1)
    total=g.posterior_gradient(p,state,k,1)[2]
    monkeypatch.setattr(g,'apply',lambda *a,**kw:(_ for _ in ()).throw(AssertionError('new candidate search forbidden')))
    pp,qq,dz,detail=replay(p,state,k,g,1,record,total)
    assert np.array_equal(pp,old_p) and np.array_equal(qq,old_q)
    assert detail['new_candidate_searches']==0 and abs(dz).max()<=record['accepted_delta']
    bad=copy.deepcopy(record);bad['accepted_delta']*=.9
    with pytest.raises(ValueError):replay(p,state,k,g,1,bad,total)


def test_four_types_eight_components_match_original_total():
    case,l,b,col,cp=four_type_fixture();pf=joint_profile(l.types,case['clean'],case['space'].encode(case['clean'])[0],l.pairs)
    k=MarginalKernel(l.prior,b['alpha']);g=MultiScaleGuide(l,freeze(pf,l,k.alpha))
    p=b['p0'];state=b['state'];q,factor,_=posterior_chain(p,state,k,g,1)
    detail,parts=components(q,g,0,True,factor);native=g.posterior_gradient(p,state,k,1)
    assert len(parts)==8 and abs(detail['total_loss']-native[1])<1e-15
    assert sum_certificate(parts,native[2])[0]['passed']


def test_recorded_skipped_zero_gradient_stays_zero():
    l,pf,k,g,p,state=example();p=(p>.4).astype(float);p[l.fixed0]=0
    oldp,oldq,record=g.apply(p,state,k,1);total=g.posterior_gradient(p,state,k,1)[2]
    pp,qq,dz,r=replay(p,state,k,g,1,record,total)
    assert record['status']=='skipped' and r['candidate_updates_replayed']==0
    assert np.array_equal(pp,p) and np.array_equal(qq,oldq) and not total.any() and not dz.any()


def test_independent_Genre64_gradient_and_first_order_small_step():
    l,pf,k,g,p,state=example();q,factor,_=posterior_chain(p,state,k,g,1)
    w=np.zeros((64,len(l.pairs)));w[:,1]=np.linspace(.001,.1,64)
    spec=dict(Genre_type=1,omega64=w.tolist(),CF_normalizer=1)
    loss,gradient,_=independent_cf_gradient(q,g,0,spec);gz=gradient*factor
    fd=scalar_fd(lambda x:genre64(k.reverse(x,state,l.families,0,1),g,0,spec)['error'],p,(p>0)&(p<1)&~l.fixed0)
    assert abs(gz-fd).max()<1e-11 and loss==pytest.approx(genre64(q,g,0,spec)['error'],abs=1e-15)
    info,parts=components(q,g,0,True,factor);total=sum(parts.values(),np.zeros_like(p));denom=abs(total).max();eta=1e-5
    dz=-eta*total/denom;free=(p>0)&(p<1)
    pp=p.copy();z=np.log(p[free])-np.log1p(-p[free]);pp[free]=1/(1+np.exp(-(z+dz[free])))
    actual=genre64(pp,g,0,spec)['error']-loss
    attributed=first_order(parts,gz,eta,denom,actual,dz)
    assert abs(attributed['additivity_error'])<1e-20 and abs(attributed['actual_minus_first_order'])<1e-11


@pytest.mark.parametrize('left,right',[(np.zeros(4),np.zeros(4)),(np.ones(4),np.zeros(4)),(np.ones(4),-np.ones(4))])
def test_zero_norm_is_undefined_and_opposition_is_negative(left,right):
    result=alignment(left,right)
    if not np.any(left) or not np.any(right):assert result['cosine'] is None and result['cosine_status']=='undefined_zero_norm'
    else:assert result['cosine']==pytest.approx(-1.) and result['inner_product']<0


def genre_example():
    types=np.array([0,0,0,1,1]);pairs=[(0,0),(0,1),(1,1)]
    l=Layout.canonical(types,pairs,[0.,.3,0.]);chosen=np.flatnonzero(l.families==1)[[0,1,3]]
    pf=joint_profile(types,l.slots[:,chosen],l.families[chosen],pairs);k=MarginalKernel(l.prior,[1.,.5,0.]);g=MultiScaleGuide(l,freeze(pf,l,k.alpha))
    q=np.repeat(l.prior, [int((l.families==r).sum()) for r in range(3)]);q[l.families==1]=np.linspace(.1,.8,6)
    return l,g,q


def test_complete_PMF_CDF_tail_and_W1_intervals_against_enumeration():
    l,g,q=genre_example();after=q.copy();after[l.families==1]*=.9
    report,a=genre_laws(q,after,g,0,genre='1');expected=np.zeros(a['before_pmf'].shape)
    free=np.flatnonzero((q>0)&(q<1))
    for bits in itertools.product((0,1),repeat=len(free)):
        x=np.zeros(len(q),int);x[free]=bits;mass=np.prod(np.where(bits,q[free],1-q[free]))
        d=np.zeros(len(l.types),int)
        for u,v in l.slots[:,x==1].T:d[u]+=1;d[v]+=1
        expected+=mass*np.bincount(d[l.types==1],minlength=len(expected))/(l.types==1).sum()
    assert abs(expected-a['before_pmf']).max()<1e-15
    old=degree_w1(q,g,0,only_types={'1'})['1'][0]['W1']
    assert report['states']['before']['W1']==pytest.approx(old,abs=1e-14)
    assert sum(report['states']['before']['W1_intervals'].values())==pytest.approx(old,abs=1e-14)
    assert np.array_equal(a['before_pmf'].cumsum(),a['before_cdf'])
    p=g.channels['1'][0]['p'];D90=report['D90']
    assert report['states']['before']['P_degree_gt_D90']==pytest.approx(expected[D90+1:].sum(),abs=1e-15)
    original=components(q,g,0)[0]['details']['1']['probes'][0]['value']
    assert report['states']['before']['original_tail_probe']==pytest.approx(original,abs=1e-15)
    assert a['target_anonymous_role_pmf'].shape[0]==len(g.frozen['payload']['profile']['groups']['1']['rows'])


def test_type_renumbering_and_same_type_two_endpoint_contributions():
    l,pf,k,g,p,state=example();before,parts=components(p,g,0,True)
    # Swap within coarse domains; canonical slot mapping is recomputed.
    perm=np.array([2,0,1,4,3]);newslots=perm[l.slots];newslots.sort(axis=0)
    lookup={tuple(e):i for i,e in enumerate(l.slots.T)};mapping=np.array([lookup[tuple(e)] for e in newslots.T])
    qp=np.empty_like(p);qp[mapping]=p
    after,other=components(qp,g,0,True)
    assert abs(before['total_loss']-after['total_loss'])<1e-14
    for key in parts:assert abs(parts[key]-other[key][mapping]).max()<1e-13
    # The per-component finite differences above include the same-type family
    # (0,0), so both endpoints must contribute for the derivative to match.


@pytest.mark.parametrize('bad',['arm','time','index','summary_case','unfinished'])
def test_only_original_B1_completed_receipt_accepted(bad):
    old=dict(arm='B',t=1,index=5);s=dict(status='complete',completed_records=6,input_hash_verification_complete=True,input_hashes_unchanged=True,records=[{}]*5+[copy.deepcopy(old)])
    old_case_binding(s,old)
    if bad=='arm':old['arm']='A'
    elif bad=='time':old['t']=20
    elif bad=='index':old['index']=4
    elif bad=='summary_case':s['records'][5]['extra']='tampered'
    else:s['status']='incomplete'
    with pytest.raises(ValueError):old_case_binding(s,old)


def test_metric_drift_refused_not_new_case_or_step():
    current=dict(Genre64=dict(error=.1),expected_degree_PMF_W1={'3':[dict(W1=.2)]})
    compare_metrics(current,copy.deepcopy(current))
    old=copy.deepcopy(current);old['Genre64']['error']+=.01
    with pytest.raises(ValueError):compare_metrics(current,old)


def test_missing_real_inputs_never_substituted_or_written(tmp_path):
    short=tmp_path/'absent_short';export=tmp_path/'absent_export';out=tmp_path/'out'
    with pytest.raises(ValueError,match='missing real inputs'):admit(out,ROOT,short,export,'0'*64,'0'*64)
    assert not out.exists() and not short.exists() and not export.exists()
    r=subprocess.run([sys.executable,str(ROOT/'run_b1.py'),'inspect','--short-run',str(short),'--export',str(export)],capture_output=True,text=True,timeout=10)
    assert r.returncode==2 and json.loads(r.stdout)['status']=='missing_real_inputs'


def test_B1_output_overlap_rejected_before_any_write(tmp_path):
    short=tmp_path/'original';short.mkdir();before=list(short.iterdir());out=short/'must_not_exist'
    r=subprocess.run([sys.executable,str(ROOT/'run_b1.py'),'diagnose','--short-run',str(short),'--export',str(tmp_path/'export'),
        '--summary-sha256','0'*64,'--case-sha256','0'*64,'--output',str(out)],capture_output=True,text=True,timeout=10)
    assert r.returncode!=0 and 'overlaps protected' in r.stderr and list(short.iterdir())==before


def test_empty_B1_artifacts_never_pass_completed_label(tmp_path):
    record=dict(status='complete',original_step_reproduced=True,gradient_sum_verified=True,
                original_metrics_reproduced=True,input_hash_verification_complete=True,new_candidate_searches=0,output_hashes={})
    (tmp_path/'summary.json').write_text(json.dumps(record))
    with pytest.raises(ValueError,match='missing B1'):validate_done(tmp_path)
