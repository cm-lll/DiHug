import itertools
import numpy as np
import pytest
from scaling import scale,required,ReadoutFailure

def ok(L,c):
    p,r=required(L,np.array(c));assert r['row_residual']<=1e-8 and r['column_residual']<=1e-8;return p

@pytest.mark.parametrize('L,c',[(np.zeros((3,1)),[3]),(np.zeros((4,2)),[3,1]),(np.array([[40.,0],[0,40.]]),[1,1]),(np.array([[0.,-np.inf],[-np.inf,0.]]),[1,1]),(np.array([[0.,0,-np.inf],[0.,0,0.],[0.,-np.inf,0.]]),[1,1,1])])
def test_basic(L,c):ok(L,c)

def test_uniform_and_zero_frequency():
    p=ok(np.zeros((4,3)),[3,0,1]);np.testing.assert_allclose(p,np.tile([.75,0,.25],(4,1)),atol=1e-12)

def test_invariance():
    L=np.array([[2.,-.4,1.],[3,2.,.3],[-2.,1.,2.],[.2,.5,.1]])
    p=ok(L,[2,1,1]);ix=[2,0,3,1];jx=[2,0,1]
    np.testing.assert_allclose(ok(L[ix][:,jx],[1,2,1]),p[ix][:,jx],atol=2e-8,rtol=0)
    np.testing.assert_allclose(ok(L+np.array([2,-6,4,1])[:,None]+np.array([8,-4,3]),[2,1,1]),p,atol=2e-8,rtol=0)

def test_support_and_boundary_distinct():
    p,r=scale([[0,-np.inf],[0,-np.inf]],np.array([1,1]));assert p is None and r['status']=='support_infeasible'
    p,r=scale([[0,0],[-np.inf,0]],np.array([1,1]));assert r['support_feasible'] and r['status']=='not_converged' and r['iterations']==1000
    with pytest.raises(ReadoutFailure):required([[0,0],[-np.inf,0]],np.array([1,1]))

def test_underflow_not_support():
    p,r=scale([[0,-1e6],[-1e6,0]],np.array([1,1]));assert r['status']=='numerical_underflow' and r['underflow_count']==2 and r['support_density']==1

def test_large_constants_and_numeric_range():
    np.testing.assert_allclose(ok(np.full((2,2),1e200),[1,1]),.5)
    p,r=scale([[1e308,-1e308],[-1e308,1e308]],np.array([1,1]))
    assert p is None and r['status']=='numerical_range_exceeded'

def test_resource_stop_and_no_rng():
    import scaling
    state=np.random.get_state()
    ok(np.ones((3,2)),[1,2])
    after=np.random.get_state();assert state[0]==after[0] and np.array_equal(state[1],after[1]) and state[2:]==after[2:]
    with pytest.raises(InterruptedError):scale(np.ones((3,2)),np.array([1,2]),lambda:(_ for _ in ()).throw(InterruptedError()))
    old=scaling.MAX_ELEMENTS
    try:
        scaling.MAX_ELEMENTS=1
        with pytest.raises(MemoryError):scale(np.ones((3,2)),np.array([1,2]))
    finally:scaling.MAX_ELEMENTS=old

@pytest.mark.parametrize('L,c',[(np.array([[np.nan]]),[1]),(np.array([[np.inf]]),[1]),(np.zeros((2,2)),[1,0]),(np.zeros((1,1)),[1.])])
def test_invalid(L,c):
    with pytest.raises(ValueError):scale(L,np.array(c))

def test_exact_enumeration_counterexample():
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'role_robust_training_v1'))
    import core
    from capacity_readout import allocation_marginals
    L=np.array([[np.log(4),0],[0,np.log(4)]])
    exact,_=allocation_marginals(L,np.array([1,1]));weights=np.exp([L[0,0]+L[1,1],L[0,1]+L[1,0]])
    np.testing.assert_allclose(exact,[[weights[0]/weights.sum(),weights[1]/weights.sum()],[weights[1]/weights.sum(),weights[0]/weights.sum()]])
    approximate=ok(L,[1,1]);np.testing.assert_allclose(approximate,[[.8,.2],[.2,.8]],atol=1e-12)
    assert abs(exact[0,0]-approximate[0,0])>.14
