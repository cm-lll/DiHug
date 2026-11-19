"""Read-only summary of the predeclared full-slot cost panels."""
from pathlib import Path
import json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/rooted_quantile_cpu_20261109'
rows=[]
for ds in ['IMDB','PubMed']:
    tag=f'{ds}_CE_CE_t500_draw0'
    with np.load(ROOT/f'results/hard_structure_edge_direction_cpu_20261108/{tag}_flips.npz') as z:
        sign=z['flip_sign'];metrics={k:z[k+'_flip'] for k in ['total_degree_W1','joint_degree_TV','triangle_abs_error']}
    for method,path in [('RC',ROOT/f'results/rooted_feedback_audit_20261109/{tag}_full.npz'),('QUANTILE',OUT/f'{ds}_full_reference.npz')]:
        with np.load(path) as z:d=z['conditional_difference']*sign
        good=d<-1e-12
        row=dict(dataset=ds,method=method,slots=len(d),rewarded=int(good.sum()),three_worse=int((good&np.logical_and.reduce([v>1e-12 for v in metrics.values()])).sum()))
        for k,v in metrics.items():row[k+'_worse']=int((good&(v>1e-12)).sum())
        rows.append(row)
# The original read-only summary is checked, not overwritten.
assert rows==json.loads((OUT/'full_cost_panel_directions.json').read_text())
binding=json.loads((OUT/'binding.json').read_text())
assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in binding.items())
decision=dict(status='completed_stop',candidate='fixed_projection_quantile_v1',formal_training_started=False,
    adoption='not_recommended_yet',reason='IMDB score-direction improvement, PubMed mixed and additional middle-noise costs',
    no_new_GPU=True,sources_unchanged=True,known_example_clarification='aggregate no benefit does not mean no benefit within each type',
    source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')})
assert not (OUT/'decision.json').exists()
(OUT/'decision.json').write_text(json.dumps(decision,indent=2));print(json.dumps(decision,indent=2))
