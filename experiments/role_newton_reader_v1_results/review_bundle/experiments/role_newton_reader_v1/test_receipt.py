import json
import numpy as np
from prepare import save

def test_numpy_scalar_receipts_and_nonfinite_rejection(tmp_path):
    p=tmp_path/'receipt.json';save(p,dict(ok=np.bool_(True),n=np.int64(3),x=np.float64(.25)))
    assert json.loads(p.read_text())==dict(ok=True,n=3,x=.25)
    import pytest
    with pytest.raises(ValueError):save(tmp_path/'bad.json',dict(x=float('nan')))
