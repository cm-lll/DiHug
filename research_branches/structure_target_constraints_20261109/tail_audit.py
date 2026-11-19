"""Read saved exact target arrays; report joint and per-metric violation mass."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
p=ROOT/'results/structure_target_constraints_20261109/completed'
z=np.load(p/'targets.npz');f=z['features'];eps=z['bounds'];ids=z['canonical']
valid=np.all(f<=eps+1e-12,1)
o={'all_bounds_satisfying_graph_count':int(valid.sum()),
   'all_bounds_satisfying_isomorphism_classes':len(set(ids[valid])),
   'per_metric_exceedance':{n:(z[n]@(f>eps+1e-12)).tolist() for n in
       ['sparse_base','unit_energy','bounded_expected_error','zero_error_boundary']}}
(p/'tail_audit.json').write_text(json.dumps(o,indent=2)+'\n')
print(json.dumps(o,indent=2))
