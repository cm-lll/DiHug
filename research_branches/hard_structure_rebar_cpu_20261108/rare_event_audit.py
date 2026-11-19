"""Replay ONLY the recorded uniform draws. No new samples, graph scores or models."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='1'
import csv,json
from pathlib import Path
import numpy as np
import torch
torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/hard_structure_rebar_cpu_20261108'
panels=json.loads((OUT/'panels.json').read_text())
coordinates=json.loads((OUT/'selected_edges.json').read_text())
cfg=json.loads((OUT/'protocol.json').read_text())
rows=[]
for panel_id,panel in enumerate(panels):
    name=panel['panel'];cc=[c for c in coordinates if c['panel']==name]
    indices=torch.tensor([c['active_index'] for c in cc])
    p=torch.tensor([c['probability'] for c in cc])
    with np.load(OUT/f'{name}_samples.npz') as z:
        diff=z['REBAR']-z['edge_conditional'];means=diff.mean(0);ses=diff.std(0,ddof=1)/8
    counts={}
    for phase,seed,n in [('development',cfg['development_seed'],32),('evaluation',cfg['evaluation_seed'],64)]:
        gen=torch.Generator().manual_seed(seed+panel_id)
        bits=[]
        for _ in range(n):
            u=torch.rand(panel['mutable_slots'],generator=gen)
            torch.rand(panel['mutable_slots'],generator=gen)
            bits.append((torch.logit(p)+u[indices].log()-torch.log1p(-u[indices])>=0).numpy())
        b=np.array(bits);counts[phase]=np.minimum(b.sum(0),n-b.sum(0))
    for j,c in enumerate(cc):
        pmin=min(c['probability'],1-c['probability'])
        rows.append(dict(**c,minor_probability=pmin,expected_minor_events_64=64*pmin,
            probability_no_minor_event_64=(1-pmin)**64,
            observed_minor_events_dev32=int(counts['development'][j]),
            observed_minor_events_eval64=int(counts['evaluation'][j]),
            REBAR_minus_conditional_mean=float(means[j]),paired_empirical_SE=float(ses[j]),
            empirical_SE_units=float(abs(means[j])/max(ses[j],1e-15)),
            interpretation='Descriptive empirical SE can miss rare tails; not a calibrated normal test'))
with (OUT/'rare_events.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(OUT/'rare_events.json').write_text(json.dumps(rows,indent=2))
report=dict(selected_coordinates=len(rows),zero_minor_events_eval64=sum(x['observed_minor_events_eval64']==0 for x in rows),
    above_4_empirical_SE=[x for x in rows if x['empirical_SE_units']>4],
    new_graph_scores=0,new_random_draw_sequences=0,model_calls=0,GPU_hours=0,
    conclusion='Do not use 64-draw variance reduction alone as proof of gradient accuracy, especially near deterministic probabilities.')
(OUT/'rare_event_audit.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
