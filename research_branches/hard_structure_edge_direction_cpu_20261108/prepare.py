import datetime,json,hashlib
from pathlib import Path
R=Path(__file__).resolve().parents[2];H=Path(__file__).parent;O=R/'results/hard_structure_edge_direction_cpu_20261108';O.mkdir(exist_ok=False);OLD=R/'results/hard_structure_response_20261108T182154'
start=datetime.datetime.fromisoformat('2026-11-08T18:42:43+08:00').timestamp()
p=dict(start_epoch=start,science_stop_epoch=start+720,delivery_stop_epoch=start+900,max_cpu_threads=1,max_rss_bytes=2*1024**3,max_GPU=0,model_calls=0,updates=0,new_random_draws=0,samples=[0,1,2,3],times=[1,500,1000],parent_conditions=['IMDB_CE','IMDB_PS','PubMed_CE','PubMed_PS'],arms=['CE','CE_STRUCT'],score_tolerance=1e-12,metric_tolerance=1e-12,gradient_weight='p(1-p)*abs(D1-D0); descriptors not network influence',numeric_reference_indices=8,synthetic_cases=16,scope='96 reused hard auxiliary graphs, all legal edge flips; neither mutate originals nor reverse generate; no coefficient/score changes')
(O/'protocol.json').write_text(json.dumps(p,indent=2))
source=list(H.glob('*.py'))+[R/'research_branches/hard_structure_incremental_cpu_20261108/incremental.py',R/'research_branches/hard_structure_rebar_cpu_20261108/objective.py']
sha=lambda x:hashlib.sha256(x.read_bytes()).hexdigest()
(O/'source_binding.json').write_text(json.dumps({str(x):sha(x) for x in source},indent=2))
files=[OLD/'protocol.json',OLD/'probes.json',OLD/'assets.json',R/'results/output_boundary_replay_20261108T081654/tasks.json']+list(OLD.glob('*fixed_inputs.pt'))
for c in p['parent_conditions']:
 for arm in p['arms']:
  for t in p['times']:files.append(OLD/f'{c}_{arm}_t{t}.npz')
(O/'asset_binding.json').write_text(json.dumps({str(x):sha(x) for x in files},indent=2));print(O)
