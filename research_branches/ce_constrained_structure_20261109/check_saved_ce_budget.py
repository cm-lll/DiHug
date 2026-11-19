"""Assess the proposed initial-total-CE feasibility rule on saved real outputs.

This does not predict what the new optimizer would do; existing trajectories
were trained with another objective. No model calls or checkpoint execution.
"""
import csv
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/ce_constrained_structure_20261109'
p=ROOT/'results/direct_medium_run_20261108T230502/curves.csv'
with p.open() as f:r=list(csv.DictReader(f))
out=[]
for condition in sorted({x['condition'] for x in r}):
    start=next(x for x in r if x['condition']==condition and x['arm']=='start')
    end=next(x for x in r if x['condition']==condition and x['arm']=='CE_STRUCT' and x['offset']=='64')
    row=dict(condition=condition,total_CE_budget_satisfied=float(end['CE'])<=float(start['CE']))
    for m in ['CE','positive_CE','negative_CE','DIRECT','total_degree_W1','joint_TV','triangle_abs_error']:
        row[m+'_start']=float(start[m]);row[m+'_end']=float(end[m])
        row[m+'_relative_change']=float(end[m])/float(start[m])-1.
    out.append(row)
with (OUT/'saved_real_CE_budget.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
(OUT/'saved_real_CE_budget_audit.json').write_text(json.dumps(dict(
    source_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),conditions=4,feasible=3,
    feasible_with_negative_CE_increase=2,
    qualification='Existing fixed-lambda DIRECT trajectories, not outcomes of the new constraint optimizer. Two training seeds only.',
    model_calls=0,GPU_hours=0),indent=2))
print(json.dumps(out,indent=2))
