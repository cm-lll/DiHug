import datetime,json,hashlib
from pathlib import Path
R=Path(__file__).resolve().parents[2];H=Path(__file__).parent;O=R/'results/joint_transport_cpu_20261108';O.mkdir(exist_ok=False);OLD=R/'results/hard_structure_response_20261108T182154';EDGE=R/'results/hard_structure_edge_direction_cpu_20261108'
start=datetime.datetime.fromisoformat('2026-11-08T19:19:30+08:00').timestamp()
p=dict(start_epoch=start,science_stop_epoch=start+1020,delivery_stop_epoch=start+1200,max_CPU_threads=1,max_RSS_bytes=2*1024**3,GPU=0,model_calls=0,updates=0,real_aux_samples=[0,1,2,3],times=[1,500,1000],slots_per_relation_presence_stratum=16,slot_selection_seed=2026110830,scale='max(1,mean reference raw feature), per type/coordinate',coordinate_weight='1/(nonempty_block_count * active_coordinate_count_in_block * scale)',blocks=['relation degrees','typed triangle participation'],type_weight='equal',features='same counts as old MMD; remove log1p and change scale, therefore whole distance adaptation not solver-only ablation',score_tolerance=1e-12,synthetic_graphs=16,full_benchmarks=['IMDB_CE/CE/t500/draw0','PubMed_CE/CE/t500/draw0'],full_benchmark_forecast_limit_seconds=60,new_real_auxiliary_draws=0,scope='96 registered auxiliary graphs; fixed stratified edge panel plus two conditional full-slot cost benchmarks; no formal objective changes')
(O/'protocol.json').write_text(json.dumps(p,indent=2));sha=lambda x:hashlib.sha256(x.read_bytes()).hexdigest()
files=[OLD/'protocol.json',OLD/'probes.json',R/'results/output_boundary_replay_20261108T081654/tasks.json',EDGE/'examples.csv']+list(OLD.glob('*fixed_inputs.pt'))
for x in list(OLD.glob('*.npz'))+list(EDGE.glob('*flips.npz')):files.append(x)
(O/'asset_binding.json').write_text(json.dumps({str(x):sha(x) for x in files},indent=2))
sources=list(H.glob('*.py'))+[R/'research_branches/hard_structure_rebar_cpu_20261108/objective.py',R/'research_branches/hard_structure_edge_direction_cpu_20261108/metrics.py']
(O/'source_binding.json').write_text(json.dumps({str(x):sha(x) for x in sources},indent=2));print(O)
