"""Small pure-algorithm checks; no model calls or diffusion chains."""
from collections import Counter
import itertools
import math
from pathlib import Path
import sys
import numpy as np
import pytest
from scipy.special import logsumexp

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE.parent/"role_robust_training_v1"))
from algorithms import dp_cost,gale_ryser,flow_witness,degrees,role_rows,checked
from core import RoleTarget
from capacity_readout import allocation_marginals,MAX_DP_CELLS


@pytest.mark.parametrize("target",[[4]*6,[18,2,1,1,1,1],[8,6,4,3,2,1],[12,4,3,2,2,1],[10,10,1,1,1,1]])
def test_full_movie_side_compatibility(target):
    rows=[2]*6+[1]*12
    edges,proof=flow_witness(rows,target)
    assert proof["max_flow"]==24 and len(set(map(tuple,edges)))==24
    assert np.array_equal(np.bincount(edges[:,0],minlength=18),rows)
    assert np.array_equal(np.bincount(edges[:,1],minlength=6),target)


def test_sum_and_slot_capacity_are_not_sufficient():
    assert sum([2,2,0])==sum([3,1]) and max([3,1])<=3
    assert not gale_ryser([2,2,0],[3,1])[0]
    with pytest.raises(ValueError,match="infeasible"):flow_witness([2,2,0],[3,1])


def test_exact_DP_cost_and_original_guard_without_huge_allocation(monkeypatch):
    assert MAX_DP_CELLS==200000
    x=dp_cost([289,60,22,9,7,4,3,3,2,2]+[1]*9)
    assert x["cells"]=="17005599129600" and x["forward_backward_bytes_lower_bound"]=="272089586073600"
    assert not x["guard_passed"]
    assert dp_cost([6])["special_case"] is not None and dp_cost([6])["forward_backward_bytes_lower_bound"]=="0"
    def forbidden(*a,**kw):raise AssertionError("large allocation reached")
    ll=np.zeros((410,19));monkeypatch.setattr(np,"full",forbidden)
    with pytest.raises(RuntimeError,match="resource limit"):allocation_marginals(ll,np.array([289,60,22,9,7,4,3,3,2,2]+[1]*9))


def test_small_marginals_match_independent_twelve_assignment_truth():
    counts=np.array([2,1,1]);ll=np.array([[.1,.2,-.4],[-.8,.9,0],[.4,-.2,.6],[.2,.7,-.1]])
    marginal,info=allocation_marginals(ll,counts)
    states=[x for x in itertools.product(range(3),repeat=4) if np.array_equal(np.bincount(x,minlength=3),counts)]
    assert len(states)==12
    scores=np.array([sum(ll[i,r] for i,r in enumerate(x)) for x in states]);weights=np.exp(scores-logsumexp(scores))
    exact=np.zeros_like(ll)
    for x,w in zip(states,weights):
        for i,r in enumerate(x):exact[i,r]+=w
    np.testing.assert_allclose(marginal,exact,atol=2e-12,rtol=0)
    np.testing.assert_allclose(marginal.sum(0),counts,atol=2e-12,rtol=0)
    assert info["cells"]==int(dp_cost(counts)["cells"])


def test_joint_roles_not_independent_marginals_and_no_IDs():
    schema=((0,1),(0,2));sizes=(2,2,2)
    a=RoleTarget(((((1,0),1),((0,1),1)),(((1,0),1),((0,0),1)),(((0,1),1),((0,0),1))),sizes,schema)
    b=RoleTarget(((((1,1),1),((0,0),1)),(((1,0),1),((0,0),1)),(((0,1),1),((0,0),1))),sizes,schema)
    assert a.fingerprint!=b.fingerprint
    for r in range(2):assert Counter(v[r] for v,n in a.roles[0] for _ in range(n))==Counter(v[r] for v,n in b.roles[0] for _ in range(n))
    assert set(a.as_dict())=={"roles","type_counts","schema"}


def test_role_duplicate_merging_and_order():
    a=RoleTarget(((((1,),2),),(((1,),2),)),(2,2),((0,1),))
    b=RoleTarget(((((1,),1),((1,),1)),(((1,),2),)),(2,2),((0,1),))
    assert a==b


def test_witness_relabel_preserves_complete_joint_multisets():
    types=np.array([0,0,1,1]);edges=np.array([[0,2],[0,3],[1,2]]);f=np.array([0,0,0]);pairs=((0,1),)
    before=role_rows(degrees(types,edges,f,pairs),types)
    perm=np.array([1,0,3,2]);after=np.sort(perm[edges],axis=1)
    assert before==role_rows(degrees(types,after,f,pairs),types)


def test_hash_must_be_pinned_not_current_self_attestation(tmp_path):
    p=tmp_path/"input";p.write_text("payload")
    with pytest.raises(ValueError,match="pinned input hash"):checked(p,"0"*64)
    with pytest.raises(ValueError):checked(p,None)


def test_preparation_has_no_model_execution_or_sampling_entry():
    src=(HERE/"prepare.py").read_text()
    for forbidden in ("CapacityDenoiser(","reverse_step(","optimizer.step(","make_model(",".backward("):
        assert forbidden not in src
    assert 'scientific_generation_authorized' in (HERE/"config.json").read_text()


def test_public_preparation_rejects_protected_output_before_writes(tmp_path):
    import subprocess
    run=tmp_path/"old_run";run.mkdir();(run/"keep").write_bytes(b"unchanged")
    panel=tmp_path/"old_panel";panel.mkdir();before=list(run.rglob("*"))
    r=subprocess.run([sys.executable,str(HERE/"prepare.py"),"--run",str(run),"--panel",str(panel),
                      "--output",str(run/"invalid_output")],capture_output=True,timeout=30)
    assert r.returncode!=0 and b"new separate output required" in r.stderr
    assert list(run.rglob("*"))==before and (run/"keep").read_bytes()==b"unchanged"
