import numpy as np
import pytest
from scaling import scale,required
from adapter import reader_context

def test_hall_infeasible_without_empty_rows_or_columns():
    x=np.array([[0,-np.inf,-np.inf],[0,-np.inf,-np.inf],[-np.inf,0,0.]])
    p,r=scale(x,np.ones(3,dtype=int));assert p is None and r['status']=='support_infeasible'

def test_kkt_and_duplicate_column_frequency():
    L=np.array([[2,1,1],[1,2,2],[0,3,3],[2,1,1.]])
    p,r=required(L,np.array([2,1,1]));q,s=required(L[:,:2],np.array([2,2]))
    np.testing.assert_allclose(p[:,0],q[:,0],atol=1e-8)
    np.testing.assert_allclose(p[:,1]+p[:,2],q[:,1],atol=1e-8)
    kkt=np.log(p)-L;np.testing.assert_allclose(kkt-kkt[:,:1]-kkt[:1,:]+kkt[0,0],0,atol=1e-12)

def test_dispatch_restores_even_on_failure():
    class Module:pass
    m=Module();original=lambda L,c:('old',{});m.allocation_marginals=original
    with pytest.raises(RuntimeError):
        with reader_context(m):
            assert m.allocation_marginals is not original
            raise RuntimeError('engineering interruption')
    assert m.allocation_marginals is original
    with reader_context(m,'exact'):assert m.allocation_marginals is original
