"""Aggregate completed paired CPU reports; no model construction or GPU access."""
import csv,json,hashlib,time
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[2]
O=R/'results/direct_structure_response_20261108T205816'
read=lambda p:json.loads(p.read_text())
def csvread(name):return list(csv.DictReader((O/name).open()))
graphs=csvread('conditional_graph_paired.csv');probes=csvread('paired_metrics.csv')
metrics=['DIRECT','type_equal_total_degree_W1','type_equal_relation_degree_W1','type_equal_joint_degree_TV','triangles','triangle_abs_error']
summary=[]
for c in sorted({r['condition'] for r in graphs}):
 rows=[r for r in graphs if r['condition']==c];s={'condition':c,'times':[int(r['t']) for r in rows]}
 for k in metrics:
  for arm in ['CE','STRUCT','delta']:s[k+'_'+arm]=float(np.mean([float(r[k+'_'+arm]) for r in rows]))
  s[k+('_decreased_times' if k=='triangles' else '_improved_times')]=[int(r['t']) for r in rows if float(r[k+'_delta'])<0]
 summary.append(s)
out={'scope':'raw conditional independent-edge p0 auxiliary graphs, NOT full reverse chains; three viewed development states per condition; 32 paired auxiliary draws each; CE/PS parents share a training seed within each dataset',
     'summary':summary,'graph_pairs':graphs,
     'probe_pairs':probes,'receipt':read(O/'receipt.json'),'lease':read(O/'lease.json'),
     'postprocessing_source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')},
     'created_epoch':time.time()}
(O/'decision_evidence.json').write_text(json.dumps(out,indent=2))
print(json.dumps(summary,indent=2))
print('probe absolute deltas')
for r in probes:print(r['condition'],r['t'],*[f'{k}={float(r[k+"_delta"]):+.8g}' for k in ['CE','positive_CE','negative_CE','expected_degree_MAE','expected_triangles']])
