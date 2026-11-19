"""CPU-only freeze before any new trajectories or scores."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import ast,hashlib,json
from pathlib import Path
from datetime import datetime
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/terminal_structure_real_20261109T233600'
SRC=ROOT/'research_branches/cn_multiseed_20261104'
SCI=SRC/'research_branches/typed_structure_early_cn_expansion_review/experiments/typed_structure_early_cn_expansion_v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
OUT.mkdir(exist_ok=False)
seeds=[2026110941,2026110942,2026110943,2026110944]
times={str(s):sorted(np.random.default_rng([s,314159]).choice(np.arange(1,1001),8,replace=False).tolist()) for s in seeds}
models={};bindings={}
for ds,ts in [('IMDB',2026110411),('PubMed',2026110421)]:
    folder=ROOT/f'results/cn_multiseed_20261104T042546/run/{ds}_{ts}_A_late'
    p=folder/'readout_ema_0.99.pt';h=sha(p);assert json.loads(p.with_suffix('.sha256.json').read_text())['sha256']==h
    d=SCI/'package/assets'/f'{ds}.npz'
    models[ds]=dict(train_seed=ts,readout=str(p),data=str(d),architecture='A late CN',updates=3000,readout_identity='EMA0.99')
    for f in [p,d,folder/'final.sha256.json',folder/'environment.json']:bindings[str(f)]=sha(f)
for f in list((SCI/'package').rglob('*.py'))+[SCI/'boundary_transactions.py',ROOT/'research_branches/terminal_structure_audit_20261109/transition_objective.py',Path(__file__).with_name('worker.py')]:bindings[str(f)]=sha(f)
# Extract only unchanged pure score functions; never execute the old runner.
source=ROOT/'research_branches/component_target_cpu_20261109/run_cpu.py'
text=source.read_text();tree=ast.parse(text);keep={'load','bfs','descriptor','dist','features','terms','PAIRS','TRIPLES','PI','TI'}
parts=['import itertools\nimport numpy as np\nfrom scipy.optimize import linear_sum_assignment\nfrom scipy.spatial.distance import cdist\n']
for n in tree.body:
    name=n.name if isinstance(n,(ast.FunctionDef,ast.ClassDef)) else n.targets[0].id if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) else None
    if name in keep:parts.append(ast.get_source_segment(text,n))
score=OUT/'frozen_score.py';score.write_text('\n\n'.join(parts)+'\n');bindings[str(source)]=sha(source);bindings[str(score)]=sha(score)
protocol=dict(start_epoch=datetime.fromisoformat('2026-11-09T23:36:00+08:00').timestamp(),gpu_stop_epoch=datetime.fromisoformat('2026-11-10T00:41:00+08:00').timestamp(),delivery_epoch=datetime.fromisoformat('2026-11-10T00:56:00+08:00').timestamp(),wall_minutes=80,gpu_hours=2.5,max_gpus=2,models=models,bindings=bindings,sample_seeds=seeds,times=times,
    objective='Seven frozen terminal errors evaluated separately; no chosen composite, parameter updates, or optimizer. Old_F descriptive only, not adopted loss.',score_terms=['joint','relation_W1','low','hub','node_triangle_W1','typed_triangle_L1','component_distance'],
    estimator='Per term independent-path leave-one-out baseline; sum sampled transition score gradients multiplied by 1000/8, path mean. Four paths is only feasibility/noise screening.',
    policy='Frozen EMA0.99 A; production R1 T1000. At eight uniformly preselected times retain autograd graph of the SAME forward used by the production draw; no probability replay substitution.',
    numerics='Production deterministic settings unchanged. No prior repeated-update tolerance waived. Torch/NumPy posterior max abs <=1e-12 and transition logp abs <=1e-8. Nonfinite/zero probability/state mutation stop affected branch.',
    scope='8 chains, at most64 gradient-bearing network forwards/backwards, zero updates. Parameter state/RNG checked. CPU full original/LCC metrics after GPU release.',
    time_sampling='Same uniform independent time-subset rule per path across datasets, independent from chain RNG; no resampling missing low times.',
    resource='Host GPU0 IMDB and GPU1 PubMed after idle recheck. Actual allocation-to-exit charged. Independent watchdog and per-step deadline.')
(OUT/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
print(json.dumps(dict(out=str(OUT),times=times,readouts={k:bindings[v['readout']] for k,v in models.items()}),indent=2))
