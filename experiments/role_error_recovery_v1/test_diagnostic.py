import copy
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
import pytest
import torch
from diagnostic import slots,degrees,edit_variants,metrics,sweep,c,compare_validation,query_digest
from hsgd.kernel import MarginalKernel
from role_condition.panel import graph_case,skeleton,mg_initial
from role_condition.model import extract_target
from capacity_readout import CapacityDenoiser
from test_late_fusion import SMALL

def fixture(condition):
    case=graph_case(skeleton()|mg_initial(condition))
    clean=np.zeros(case["space"].total,dtype=np.uint8);clean[case["clean_ids"]]=1
    return case,clean

@pytest.mark.parametrize("condition",["balanced","extreme"])
def test_pressure_identity_counts_legality_and_location(condition):
    case,x=fixture(condition);before=x.copy()
    selected=30
    out=edit_variants(case,condition,x,x,selected,np.random.default_rng(8))
    ee,ff=slots(case);mg=case["space"].pairs.index((2,3))
    for i in range(0,6,2):
        a,b=out[i:i+2]
        assert a[0]["status"]==b[0]["status"]=="ready"
        assert a[2].sum()==b[2].sum()
        assert np.all(ff[a[2]]==mg) and not np.any(a[2]&b[2])
        assert (ee[:,a[2]]==selected).any(0).all()
        assert not (ee[:,b[2]]==selected).any()
        assert (a[1][a[2]]==int(condition=="extreme")).all()
        if condition=="extreme" and i in (2,4):
            degree=degrees(a[1],ee,ff,34,10)
            assert degree[selected,mg]==(5 if i==2 else 6)
    assert np.array_equal(x,before)
    again=edit_variants(case,condition,x,x,selected,np.random.default_rng(8))
    for a,b in zip(out,again):assert np.array_equal(a[1],b[1])

def test_infeasible_does_not_replace_or_reduce():
    case,x=fixture("balanced");ordinary=x.copy();ee,ff=slots(case)
    ordinary[(ee==30).any(0)&(ff==8)]=0
    out=edit_variants(case,"balanced",x,ordinary,30,np.random.default_rng(8))
    assert all(info["status"]=="infeasible" and state is None and mask is None for info,state,mask in out)

def test_fixed_support_never_edited():
    case,x=fixture("extreme");case["prior"][8]=0
    out=edit_variants(case,"extreme",x,x,30,np.random.default_rng(8))
    assert all(xx is None for _,xx,_ in out)

def test_flow_identity_and_na():
    case,x=fixture("balanced");p=np.repeat(case["prior"],case["space"].counts)
    result=metrics(case,x,x,np.zeros(len(x),bool),30,p,p)
    assert result["categories"]["added_false"]["p0"] is None
    assert result["categories"]["remaining_true"]["n"]==int(x.sum())
    assert result["flows"]["global"]["net"]==pytest.approx(p.sum()-x.sum())

def test_both_endpoints_and_query_exact_coverage():
    case,x=fixture("balanced");ee,ff=slots(case);degree=degrees(x,ee,ff,34,10)
    assert degree.sum()==2*x.sum()
    plan=c.plan_for(case,[11,2,3],4);coverage=np.zeros(len(x),int)
    for b in range(4):
        ids,fam,query=plan.block(b);coverage[ids]+=1
        assert np.array_equal(ee[:,ids],query) and np.array_equal(ff[ids],fam)
    assert np.array_equal(coverage,np.repeat(case["active"].astype(int),case["space"].counts))
    assert query_digest(plan)==query_digest(c.plan_for(case,[11,2,3],4))

def test_actual_probability_entry_no_sampling_or_target_identity_input():
    torch.set_num_threads(1);case,x=fixture("extreme")
    model=CapacityDenoiser(4,10,family_pairs=case["space"].pairs,prior=case["prior"],T=64,**SMALL)
    model.set_target(extract_target(case));model.eval()
    plan=c.plan_for(case,[4,1,8],4);kernel=MarginalKernel(case["prior"],c.schedule(64))
    before=torch.get_rng_state().clone()
    p,q,_=sweep(model,case,x,kernel,1,plan,lambda:None)
    np.testing.assert_allclose(p,q,atol=1e-12,rtol=0)
    changed=dict(case);changed["clean_ids"]=np.array([],dtype=int);changed["clean"]=np.zeros((2,0),dtype=int)
    pp,qq,_=sweep(model,changed,x,kernel,1,plan,lambda:None)
    np.testing.assert_array_equal(p,pp);np.testing.assert_array_equal(q,qq)
    assert torch.equal(before,torch.get_rng_state())
    fixed=np.repeat(case["prior"]==0,case["space"].counts)
    assert (p[fixed]==0).all()

def test_validation_replay_fails_on_mismatch():
    row=dict(id="test",condition="balanced",t=1,counts_by_family_label=[[2,3]],ce=.2,
             ce_sum_by_family_label=[[.1,.2]],expected_edges=[.4],expected_edge_bias=[-.3])
    old=dict(rows=[copy.deepcopy(row) for _ in range(12)],mean_ce=.2)
    assert compare_validation(old,old)["max_absolute_error"]==0
    new=copy.deepcopy(old);new["rows"][0]["expected_edges"][0]=.8
    with pytest.raises(AssertionError):compare_validation(old,new)

def test_metrics_same_category_on_paired_input():
    case,x=fixture("balanced")
    info,xx,mask=edit_variants(case,"balanced",x,x,30,np.random.default_rng(8))[0]
    p=np.repeat(case["prior"],case["space"].counts)
    a=metrics(case,x,x,mask,30,p,p);b=metrics(case,x,xx,mask,30,p,p)
    assert a["categories"]==b["categories"]
    assert a["categories"]["deleted_true"]["n"]==1
