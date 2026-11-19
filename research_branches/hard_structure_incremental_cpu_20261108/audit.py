"""Read-only verification and summary of completed CPU artifacts."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='1'
import json,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/hard_structure_incremental_cpu_20261108'
OLD=ROOT/'results/hard_structure_rebar_cpu_20261108'
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()
bind=json.loads((OUT/'asset_binding.json').read_text())
source=json.loads((OUT/'source_binding.json').read_text())
assert all(sha(p)==h for p,h in bind.items())
assert all(sha(p)==h for p,h in source.items())
rows=json.loads((OUT/'results.json').read_text())
checks=json.loads((OUT/'synthetic_tests.json').read_text())
max_archive=0.
for row in rows:
    with np.load(OUT/(row['panel']+'_full_maps.npz')) as a, np.load(OLD/(row['panel']+'_samples.npz')) as old:
        g=a['gradients'];p=a['probability'];ix=old['selected']
        assert g.shape==(3,row['slots']) and np.isfinite(g).all()
        assert np.all((p>0)&(p<1))
        assert abs(g[:,ix]-old['edge_conditional'][:3]).max()<1e-11
        max_archive=max(max_archive,float(abs(g[:,ix]-old['edge_conditional'][:3]).max()))
summary={}
for ds in ['IMDB','PubMed']:
    data=[r for r in rows if r['dataset']==ds]
    summary[ds]=dict(panels=len(data),slots=data[0]['slots'],
        full_seconds_range_including_cache=[min(r['full_seconds_per_draw_including_cache'] for r in data),max(r['full_seconds_per_draw_including_cache'] for r in data)],
        matched_subset_speedup_range=[min(r['reference_same_slots']['same_slots_speedup'] for r in data),max(r['reference_same_slots']['same_slots_speedup'] for r in data)])
result=dict(status='passed',datasets=summary,assets_unchanged=len(bind),execution_sources_unchanged=len(source),
    full_maps=36,archived_full_map_subset_max_abs_error=max_archive,
    archived_all64_subset_max_abs_error=max(r['max_archived_gradient_error'] for r in rows),
    real_score_difference_max_abs_error=max(r['reference_same_slots']['max_abs_D_difference_error'] for r in rows),
    small_graph_score_difference_max_abs_error=max(r['max_abs_difference_error'] for r in checks),
    exact_expectation_gradient_error=checks[-1]['max_abs_difference_error'],
    GPU_hours=0,model_calls=0,quality_improvement_tested=False,
    note='Same score/estimator preserved; measured cost and equivalence only')
(OUT/'audit.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
