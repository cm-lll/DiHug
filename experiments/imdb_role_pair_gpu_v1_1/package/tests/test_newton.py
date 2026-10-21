import itertools
import numpy as np
import pytest
from newton_solver import solve,dual,reduce_support,required

@pytest.mark.parametrize('L,c',[(np.zeros((3,1)),[3]),(np.zeros((4,2)),[3,1]),(np.array([[80.,0],[0,80.]]),[1,1]),(np.array([[0.,0],[-np.inf,0.]]),[1,1]),(np.zeros((4,3)),[2,0,2])])
def test_basic(L,c):
    p,lp,r=solve(L,np.array(c));assert r.get('new_contract_pass'),r
    np.testing.assert_allclose(p.sum(1),1,atol=1e-8);np.testing.assert_allclose(p.sum(0),c,atol=1e-8)

def test_forced_zero_proof_exhaustive():
    # All64 supports for3 nodes,2 columns; exact capacity assignments provide independent union.
    for mask in range(64):
        s=np.array([(mask>>i)&1 for i in range(6)],bool).reshape(3,2);c=np.array([2,1]);allx=[]
        for a in itertools.product(range(2),repeat=3):
            if np.array_equal(np.bincount(a,minlength=2),c) and all(s[i,j] for i,j in enumerate(a)):allx.append(a)
        keep,components,r=reduce_support(s,c)
        assert r['feasible']==bool(allx)
        if allx:
            union=np.zeros_like(s)
            for a in allx:union[np.arange(3),a]=True
            np.testing.assert_array_equal(keep,union)

def test_gradient_hessian():
    L=np.array([[2.,1.,-.5],[.2,1.,2.],[1.,.2,-.8],[2.,3.,1.]])
    c=np.array([2,1,1]);b=np.array([.4,-.3,.1]);f,g,H,*_=dual(L,c,b)
    eps=1e-5
    for j in range(3):
        d=np.eye(3)[j]*eps
        fp,gp,*_=dual(L,c,b+d);fm,gm,*_=dual(L,c,b-d)
        np.testing.assert_allclose((fp-fm)/(2*eps),g[j],atol=1e-9)
        np.testing.assert_allclose((gp-gm)/(2*eps),H[:,j],atol=1e-9)
    np.testing.assert_allclose(H@np.ones(3),0,atol=1e-14)

def test_permutations_and_constants():
    L=np.array([[2,1,0],[3,2,.5],[-2,1,2],[.2,.5,.1]])
    p,_,r=solve(L,np.array([2,1,1]));assert r['new_contract_pass']
    nodes=[2,0,3,1];cols=[2,0,1]
    q,_,s=solve(L[nodes][:,cols],np.array([1,2,1]));assert s['new_contract_pass']
    np.testing.assert_allclose(q,p[nodes][:,cols],atol=2e-8,rtol=0)
    q,_,s=solve(L+np.array([3,-4,1,5])[:,None]+np.array([2,-7,3]),np.array([2,1,1]))
    np.testing.assert_allclose(q,p,atol=2e-8,rtol=0)

def test_components_infeasible_and_tail_contract():
    p,lp,r=solve(np.array([[0.,-np.inf],[-np.inf,0.]]),np.ones(2,dtype=int));assert r['new_contract_pass'] and len(r['components'])==2
    p,lp,r=solve(np.array([[0.,-np.inf],[0,-np.inf]]),np.ones(2,dtype=int));assert r['status']=='support_infeasible'
    p,lp,r=solve(np.array([[0.,-1000],[-1000,0.]]),np.ones(2,dtype=int))
    assert r['status']=='converged_bounded_tail' and not r['original_contract_pass']
    assert np.isfinite(lp).all() and r['rounded_positive_count']==2 and r['total_omitted_mass_upper']<1e-280
    np.testing.assert_array_equal(p,np.eye(2))

def test_same_entropy_solution_not_DP():
    L=np.array([[np.log(4),0],[0,np.log(4)]])
    p,_,r=solve(L,np.ones(2,dtype=int));np.testing.assert_allclose(p,[[.8,.2],[.2,.8]],atol=1e-12)
    assert abs(p[0,0]-16/17)>.14

def test_high_precision_fixed_cases():
    import mpmath as mp
    mp.mp.dps=80
    cases=[np.array([[np.log(4.),0],[0,np.log(4.)]]),np.array([[2.,1,0],[0,1,3.],[1.,2,.5]]),np.array([[80.,0],[0,80.]])]
    for L in cases:
        n,k=L.shape;c=np.ones(k,dtype=int)
        def probs(b):
            bb=list(b)+[mp.mpf('0')];out=[]
            for row in L:
                w=[mp.exp(mp.mpf(float(x))+v) for x,v in zip(row,bb)];out.append([x/sum(w) for x in w])
            return out
        def grad(*b):
            p=probs(b);return tuple(sum(p[i][j] for i in range(n))-int(c[j]) for j in range(k-1))
        b=mp.findroot(grad,tuple(mp.mpf(0) for _ in range(k-1)),tol=mp.mpf('1e-65'),maxsteps=100)
        if k==2:b=[b[0] if hasattr(b,'rows') else b]
        ref=np.array([[float(x) for x in row] for row in probs(b)])
        p,lp,r=solve(L,c);assert r['new_contract_pass'],r;np.testing.assert_allclose(p,ref,atol=2e-8,rtol=0)

def test_stop_no_rng():
    before=np.random.get_state();p,lp,r=solve(np.zeros((3,2)),np.array([1,2]),lambda:(_ for _ in ()).throw(InterruptedError('STOP')))
    assert p is None and r['status']=='solver_failed'
    after=np.random.get_state();assert np.array_equal(before[1],after[1]) and before[2:]==after[2:]
