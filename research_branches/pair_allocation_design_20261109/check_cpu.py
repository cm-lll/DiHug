"""CPU algebra/archived-edit audit only. No model import, fit, or graph sampling."""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import hashlib
import json
import math
import resource
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/pair_allocation_design_20261109'
SRC = ROOT / 'results/exchange_closure_accounting_20261109/verified/all_exchanges.json'

def sig(x):
    return 1 / (1 + math.exp(-x))

def main():
    start = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=False)
    digest = hashlib.sha256(SRC.read_bytes()).hexdigest()
    rows = json.loads(SRC.read_text())
    assert len(rows) == 172
    tests = []
    # Fixed graph background and pair selection; F_e=1,F_f=1+delta.
    # These are numerical examples, not model logits or generated graphs.
    max_fd = 0.0
    for ze in (-3., 0., 2.):
        for zf in (-2., 0., 3.):
            for delta in (-0.3, 0., 0.7):
                q = sig(zf-ze)
                pf, pe = sig(zf), sig(ze)
                conditional = pf*(1-pe)/(pf*(1-pe)+pe*(1-pf))
                assert abs(q-conditional) < 1e-13
                g = delta*q*(1-q)
                eps = 1e-5
                fd = delta*(sig(zf+eps-ze)-sig(zf-eps-ze))/(2*eps)
                max_fd = max(max_fd, abs(g-fd))
                assert abs(g-fd) < 1e-10
                for shift in (-10., 10.):
                    assert abs(sig(zf+shift-ze-shift)-q) < 1e-13
                tests.append(dict(ze=ze,zf=zf,delta=delta,q=q,g_f=g,g_e=-g))
    # Fixed linear toy Jacobians: common component cancels; identical ones give zero.
    c = (4., 7., -2.)
    he, hf = (1., 2., 3.), (2., 0., 4.)
    difference = [b-a for a,b in zip(he,hf)]
    shifted = [(b+d)-(a+d) for a,b,d in zip(he,hf,c)]
    assert difference == shifted
    assert all(a-a == 0 for a in he)
    # Original independent Bernoulli risk != conditional one-edge risk.
    # F00=0,F10=1,F01=2,F11=4 at z_e=z_f=0.
    # dR/dz_f=.625; conditional derivative=.25.
    unconditional = .5*.25*(2-0)+.5*.25*(4-1)
    assert abs(unconditional-.625) < 1e-15
    conditional_grad = .25
    pairs = []
    for row in rows:
        assert row['relation_inventory_unchanged']
        delta = row['geometric_total_change']
        pairs.append(dict(added_slot=row['added_slot'],removed_slot=row['removed_slot'],
            delta_score=delta,preferred='new_position' if delta<0 else ('old_position' if delta>0 else 'tie'),
            triangle_error_change=row['triangle_error_change'],
            isolate_error_change=row['isolates_error_change'],
            actual_model_probability=None,actual_parameter_response=None))
    result = dict(scope='CPU design; no production change', archived_pairs=len(pairs),
        preferred_new=sum(r['delta_score']<0 for r in pairs),
        preferred_new_triangle_cost=sum(r['delta_score']<0 and r['triangle_error_change']>0 for r in pairs),
        algebra_cases=len(tests),max_finite_difference_error=max_fd,
        common_logit_offset_cancels=True,identical_jacobians_yield_zero=True,
        original_risk_gradient_example=unconditional,conditional_risk_gradient_example=conditional_grad,
        original_risk_unbiased=False,asset_unchanged=hashlib.sha256(SRC.read_bytes()).hexdigest()==digest,
        GPU_hours=0,model_calls=0,updates=0,new_graphs=0,
        seconds=time.monotonic()-start,RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    assert result['asset_unchanged']
    for name,obj in [('summary',result),('algebra_cases',tests),('archived_pairs',pairs),
                     ('binding',{str(SRC):digest,str(Path(__file__)):hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})]:
        (OUT/(name+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__ == '__main__':
    main()
