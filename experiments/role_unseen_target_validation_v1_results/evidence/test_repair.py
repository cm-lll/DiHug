from repair_audit import valid_TV,numeric_stat
import pytest

def test_legitimate_sum_rounding():
    for value in (1.,1.-2**-53,1.+2**-52):
        valid_TV(value,{'a':6},{'b':6},6)

def test_material_error_rejected():
    with pytest.raises(AssertionError):valid_TV(.99,{'a':6},{'b':6},6)

def test_zero_exact_and_mismatch():
    valid_TV(0.,{'a':6},{'a':6},6)
    with pytest.raises(AssertionError):valid_TV(1e-16,{'a':6},{'a':6},6)

def test_boolean_indicator_quantiles():
    result=numeric_stat([False,True,None])
    assert result['mean']==.5 and result['P90']==.9 and result['NA']==1
    assert result['values']==[False,True,None]
