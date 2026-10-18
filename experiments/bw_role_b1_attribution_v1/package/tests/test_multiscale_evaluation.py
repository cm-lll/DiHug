import itertools
import json
from pathlib import Path
import numpy as np
import pytest
from multiscale_role.evaluation import full_pmf_rows,degree_w1,validate_spec
from multiscale_role.short import POLICY,classify,run
from multiscale_role.core import MultiScaleGuide,freeze
from stepwise_role.core import Layout
from role_degree.role_statistics import joint_profile


def test_exact_pmf_against_enumeration_including_fixed_boundaries():
    p=np.array([[0.,1.,.2,.7],[.1,.4,.5,.9]])
    actual,error=full_pmf_rows(p)
    expected=np.zeros((2,5))
    for bits in itertools.product((0,1),repeat=4):
        expected[:,sum(bits)]+=np.prod(np.where(bits,p,1-p),axis=1)
    assert np.max(abs(actual-expected))<2e-15 and error<2e-15


def test_W1_of_expected_distribution_not_expected_graph_W1():
    l=Layout.canonical(np.zeros(4,dtype=int),[(0,0)],[1/6])
    profile=joint_profile(l.types,l.slots[:,:1],l.families[:1],l.pairs)
    g=MultiScaleGuide(l,freeze(profile,l,[1.,.5,0.]))
    q=np.zeros(6);q[[0,5]]=.5
    value=degree_w1(q,g,0)['0'][0]['W1']
    expected_graph_W1=0.
    for a,b in itertools.product((0,1),repeat=2):
        x=np.zeros(6);x[[0,5]]=[a,b]
        expected_graph_W1+=.25*degree_w1(x,g,0)['0'][0]['W1']
    assert value<1e-15 and abs(expected_graph_W1-.25)<1e-15


def test_no_original64_substitute_and_predeclared_mixed_decision():
    with pytest.raises(ValueError):validate_spec({},None,'bad')
    before=dict(Genre64=dict(error=.1),expected_degree_PMF_W1={'3':[dict(families=[8],W1=.2)]})
    after=dict(Genre64=dict(error=.09),expected_degree_PMF_W1={'3':[dict(families=[8],W1=.3)]})
    result=classify(before,after,dict(Genre_type=3))
    assert result['any_deterioration'] and not result['dual_improvement']
    after['expected_degree_PMF_W1']['3'][0]['W1']=.19
    assert classify(before,after,dict(Genre_type=3))['dual_improvement']
    assert len(POLICY['fixed_cases'])==6


def test_missing_real_files_nonzero_no_fake_pass(tmp_path):
    import time
    out=tmp_path/'missing'
    code=run(tmp_path/'absent.json','a'*64,tmp_path/'eval.json','b'*64,out,time.monotonic()+10)
    result=json.loads((out/'summary.json').read_text())
    assert code!=0 and result['status']=='protocol_failed' and result['completed_records']==0
    assert result['model_calls']==0 and result['input_hashes_unchanged'] is None


def test_safe_stop_without_inputs_does_not_start_work(tmp_path):
    import time
    # A deterministic deadline test, not running six scientific inputs.
    out=tmp_path/'deadline'
    code=run(tmp_path/'unused.json','a'*64,tmp_path/'unused_eval.json','b'*64,out,time.monotonic()-1)
    result=json.loads((out/'summary.json').read_text())
    assert code!=0 and result['status']=='incomplete' and result['completed_records']==0
