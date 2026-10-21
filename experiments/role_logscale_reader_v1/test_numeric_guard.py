import json
import numpy as np
import pytest
from scaling import scale,required,ReadoutFailure

def test_extreme_receipts_remain_finite_or_explicitly_rejected():
    cases=[np.array([[1e308,0.],[1e308,-1e308]]),
           np.array([[0.,-8e307],[-8e307,0.],[0.,-8e307]])]
    for L in cases:
        c=np.array([1,L.shape[0]-1]);p,r=scale(L,c)
        json.dumps(r,allow_nan=False)
        assert p is None or np.isfinite(p).all()
        if r['status']!='converged':
            with pytest.raises(ReadoutFailure):required(L,c)

def test_successful_log_probabilities_remain_regular():
    p,r=required(np.array([[600.,0.],[0.,600.]]),np.array([1,1]))
    assert (p>0).all() and np.isfinite(p).all() and r['underflow_count']==0
