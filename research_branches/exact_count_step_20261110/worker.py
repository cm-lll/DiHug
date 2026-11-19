"""Isolated hook into frozen four-boundary diagnostic: count gradient only."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
import sys,json,importlib.util
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results'/('exact_count_step_20261110'+os.environ.get('EXACT_STEP_SUFFIX',''))
OLD=ROOT/'results/hard_clean_zero_safe_20261110'
sys.path.insert(0,str(ROOT/'research_branches/hard_clean_zero_safe_20261110'))
spec=importlib.util.spec_from_file_location('frozen_four_step',ROOT/'research_branches/hard_clean_zero_safe_20261110/check_real.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
from exact_count import exact,tests

core=None
class BoundHardScore(base.HardScore):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        global core
        core=self

original_loo=base.loo_gradient
calls=[]
def count_only(probability,bits,costs):
    dg=original_loo(probability,bits,costs)
    original=dg.copy();values,gradient=exact(probability,core.f,core.ref['counts'])
    dg[0]=gradient
    assert np.array_equal(dg[1:],original[1:]),'non-count estimator modified'
    n=len(calls);protocol=json.loads((OUT/'protocol.json').read_text());task=protocol['tasks'][n]
    replay=np.load(OLD/task['task']/'failed_batch_replay.npz')
    assert bits.shape==replay['bits'].shape
    calls.append(dict(task=task['task'],only_gradient_row_zero_changed=True,exact_risk=float(values.sum()),
        sampled_count_risk=float(costs[:,0].mean()),exact_margin_norm=float(np.linalg.norm(gradient)),
        LOO_margin_norm=float(np.linalg.norm(original[0])),
        max_probability_difference_from_old_recomputation=float(abs(probability-replay['probability']).max()),
        hard_draw_bit_differences_from_old=int(np.count_nonzero(bits!=replay['bits'])),
        note='ordinary GPU numerical recomputation is not certified bitwise identical; input/RNG contracts are separately enforced'))
    base.save(OUT/'gradient_injection.json',calls)
    return dg

if __name__=='__main__':
    assert tests()['passed']
    base.OUT=OUT
    base.HardScore=BoundHardScore
    base.loo_gradient=count_only
    raise SystemExit(base.main())
