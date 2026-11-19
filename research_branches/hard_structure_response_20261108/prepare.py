"""Freeze before new model results. One coefficient; no selection by outcome."""
import datetime,json,hashlib
from pathlib import Path
R=Path(__file__).resolve().parents[2];H=Path(__file__).parent;O=R/'results/hard_structure_response_20261108T182154';O.mkdir(exist_ok=False)
start=datetime.datetime.fromisoformat('2026-11-08T18:21:54+08:00').timestamp()
p=dict(start_epoch=start,science_stop_epoch=start+1020,delivery_stop_epoch=start+1200,gpu_seconds_limit=720,steps=8,coefficient=.1,auxiliary_draws=1,evaluation_draws=32,times=[1,500,1000],max_updates=64,max_forwards=100,aux_seeds={'IMDB':2026110811,'PubMed':2026110812},evaluation_seeds={'IMDB':2026110813,'PubMed':2026110814},evaluation_mc_seeds={'IMDB':2026110815,'PubMed':2026110816},purpose='temporary matched parameter response, no formal training or T1000',coefficient_rationale='single fixed small auxiliary coefficient; prior four gradient ratios imply .016-.136 relative magnitude on that limited panel only, no guarantee on subsequent inputs',decision='No adoption from eight steps. Report paired hard-score changes and Monte Carlo SE, all relation/positive/negative/degree/triangle costs; no thresholds tuned on outcome. Any state, finite or source mismatch stops branch.')
(O/'protocol.json').write_text(json.dumps(p,indent=2))
sources=list(H.glob('*.py'))+[R/'research_branches/hard_structure_rebar_cpu_20261108/objective.py',R/'research_branches/hard_structure_incremental_cpu_20261108/incremental.py']
(O/'source_binding.json').write_text(json.dumps({str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in sources},indent=2));print(O)
