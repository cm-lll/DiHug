import json,hashlib,datetime
from pathlib import Path
R=Path(__file__).resolve().parents[2];O=R/'results/hard_structure_gradient_gpu_20261108T180629';O.mkdir(exist_ok=False)
start=datetime.datetime.fromisoformat('2026-11-08T18:06:29+08:00').timestamp()
p=dict(start_epoch=start,science_stop_epoch=start+480,delivery_stop_epoch=start+600,gpu_seconds_limit=360,max_forward=4,max_backward=8,max_updates=0,aux_draws_per_condition=1,aux_seeds=dict(IMDB_CE=2026110801,IMDB_PS=2026110801,PubMed_CE=2026110802,PubMed_PS=2026110802),purpose='gradient interface and cost only; no quality inference, lambda selection or optimizer update')
(O/'protocol.json').write_text(json.dumps(p,indent=2))
paths=list(Path(__file__).parent.glob('*.py'))+[R/'research_branches/hard_structure_rebar_cpu_20261108/objective.py',R/'research_branches/hard_structure_incremental_cpu_20261108/incremental.py']
(O/'source_binding.json').write_text(json.dumps({str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in paths},indent=2))
print(O)
