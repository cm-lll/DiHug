import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import generation_common as g
import pytest
def test_verified_read_rejects_invalid_capacity():
    p,lp,r=g.newton.solve(g.np.zeros((3,2)),g.np.array([1,2]))
    assert g.verify(p,lp,g.np.zeros((3,2)),g.np.array([1,2]),r)['independent_column']<=1e-8
    bad=p.copy();bad[0,0]+=.01
    with pytest.raises(AssertionError):g.verify(bad,lp,g.np.zeros((3,2)),g.np.array([1,2]),r)
def test_failed_read_saves_and_does_not_fallback(tmp_path,monkeypatch):
    monkeypatch.setattr(g,'check',lambda:None)
    reader=g.Reader('Newton',tmp_path);reader.begin(1)
    monkeypatch.setattr(g.newton,'solve',lambda *args:(None,None,dict(status='iteration_limit',new_contract_pass=False)))
    with pytest.raises(g.newton.ReadoutFailure):reader(g.np.zeros((2,2)),g.np.array([1,1]))
    assert (tmp_path/'failed_read.npz').exists() and (tmp_path/'failed_read.json').exists()
def test_reader_rng_unchanged(tmp_path,monkeypatch):
    monkeypatch.setattr(g,'check',lambda:None);g.np.random.seed(32);before=g.np.random.get_state()
    reader=g.Reader('Newton',tmp_path);reader.begin(1);reader(g.np.array([[0.,1.],[1.,0.]]),g.np.array([1,1]))
    after=g.np.random.get_state();assert g.np.array_equal(before[1],after[1]) and before[2:]==after[2:]
def test_numpy_receipt_serialization(tmp_path):
    p=tmp_path/'receipt.json';g.save(p,dict(passed=g.np.bool_(True),steps=g.np.int64(1)))
    assert g.read(p)==dict(passed=True,steps=1)
