import csv,json,time,hashlib
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[2];O=R/'results/hard_structure_edge_direction_cpu_20261108';rows=list(csv.DictReader((O/'graph_summary.csv').open()));relations=list(csv.DictReader((O/'relations.csv').open()));examples=list(csv.DictReader((O/'examples.csv').open()))
def pooled(rs):
 good=sum(int(x['score_improving_flips']) for x in rs);mass=sum(float(x['score_improving_gradient_mass']) for x in rs);out=dict(graphs=len(rs),score_improving_flips=good,score_improving_gradient_mass=mass)
 for m in ['triangle_abs_error','total_degree_W1','joint_degree_TV']:
  bad=sum(int(x[m+'_worse_count']) for x in rs);out[m+'_worse_count']=bad;out[m+'_worse_fraction']=bad/good if good else None;out[m+'_worse_gradient_mass_fraction']=sum(float(x[m+'_worse_gradient_mass_fraction'] or 0)*float(x['score_improving_gradient_mass']) for x in rs)/mass if mass else None
 for m in ['triangle_and_degree_worse_count','all_three_worse_count']:out[m]=sum(int(x[m]) for x in rs)
 out['triangle_min']=min(int(x['triangles']) for x in rs);out['triangle_max']=max(int(x['triangles']) for x in rs);return out
by_panel=[]
for c in sorted(set(x['condition'] for x in rows)):
 for arm in ['CE','CE_STRUCT']:
  for t in [1,500,1000]:
   rs=[x for x in rows if x['condition']==c and x['arm']==arm and int(x['t'])==t]
   if rs:by_panel.append(dict(condition=c,arm=arm,t=t,**pooled(rs)))
state=[]
for ds in ['IMDB','PubMed']:
 for s in ['under','equal','over']:
  rs=[x for x in rows if x['condition'].startswith(ds) and x['triangle_state']==s]
  if rs:state.append(dict(dataset=ds,state=s,**pooled(rs)))
def savecsv(name,rs):
 with (O/name).open('w') as f:
  w=csv.DictWriter(f,list(rs[0]));w.writeheader();w.writerows(rs)
savecsv('panel_summary.csv',by_panel);savecsv('state_summary.csv',state)
# Deterministic first example in predefined traversal; no strongest-effect selection.
representatives=[]
for ds in ['IMDB','PubMed']:
 for t in [1,500,1000]:
  for kind in ['score_down_triangle_up','score_down_all_three_up']:
   xs=[x for x in examples if x['condition'].startswith(ds) and int(x['t'])==t and x['kind']==kind]
   if xs:representatives.append(xs[0])
(O/'representative_examples.json').write_text(json.dumps(representatives,indent=2))
checks={}
for file in ['source_binding.json','asset_binding.json']:
 checks[file]={p:hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in json.loads((O/file).read_text()).items()}
old=R/'results/joint_output_formal_projected_20261108T043000/source_binding.json'
checks['production']={p:hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in json.loads(old.read_text()).items()}
audit=dict(all_unchanged=all(all(v.values()) for v in checks.values()),checks=checks,outputs_sha256={str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in O.glob('*flips.npz')},at_epoch=time.time())
(O/'audit.json').write_text(json.dumps(audit,indent=2));assert audit['all_unchanged'];print(json.dumps(dict(states=state,representatives=representatives),indent=2))
