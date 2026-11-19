"""Pre-model engineering amendment: same objective, unbiased budgeted estimator."""
import itertools,json,time
from pathlib import Path
import numpy as np
OUT=Path(__file__).resolve().parents[2]/'results/rooted_connection_20261109T015142'
p=OUT/'protocol.json';old=json.loads(p.read_text())
if (OUT/'protocol_revision0.json').exists():raise RuntimeError('already revised; do not overwrite frozen rule')
(OUT/'protocol_revision0.json').write_text(json.dumps(old,indent=2,ensure_ascii=False))
new=dict(old,revision=1,gradient='one hard graph; uniform min(1024,M) mutable edges without replacement from independent auxiliary RNG; exact per-edge conditional differences times M/m; unbiased for same hard score expectation',
    reason='CPU full slots cost1.17/7.43s and CUDA1.73/2.11s before network; bounded1024 estimator fixed before any model/quality observation to retain complete factorial with margin',
    estimator_limitation='extra edge-subsampling variance; not identical to full-slot gradient realization; no claim variance solved')
p.write_text(json.dumps(new,indent=2,ensure_ascii=False))
# Independent finite HT identity on a nonuniform vector, ALL 20 subsets.
d=np.array([-2.,.1,.7,3.,-5.,11.]);est=[]
for ids in itertools.combinations(range(6),3):
 v=np.zeros(6);v[list(ids)]=2*d[list(ids)];est.append(v)
assert np.max(abs(np.mean(est,axis=0)-d))<1e-14
(OUT/'HT_exact_check.json').write_text(json.dumps(dict(passed=True,subsets=20,max_error=float(np.max(abs(np.mean(est,axis=0)-d))),fixed_edge_sample=1024,models_called=0)))
(OUT/'leases').mkdir(exist_ok=True)
ended=(OUT/'gpu_cost_receipt.json').stat().st_mtime
(OUT/'leases/engineering_cost.json').write_text(json.dumps(dict(id='engineering_cost',pid=-1,start_ticks='completed',gpu=0,acquired_epoch=ended-180,released_epoch=ended,returncode=0,accounting='conservative180s timeout envelope; numerical GPU cost worker measured15.477s; no live process')))
