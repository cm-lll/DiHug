"""Post-hoc descriptive CPU audit on EXACT SAME 32 registered auxiliary draws.
No new random streams, model calls, weights or candidate selection.
These conditional p0 graphs are NOT reverse-chain generated graphs.
"""
import sys,json,csv,time
from pathlib import Path
import numpy as np
import torch
torch.set_num_threads(1)
R=Path(__file__).resolve().parents[2];O=R/'results/hard_structure_response_20261108T182154';sys.path.insert(0,str(R/'research_branches/hard_structure_rebar_cpu_20261108'))
from objective import JointLocalScore
read=lambda p:json.loads(Path(p).read_text());P=read(O/'protocol.json');probes=read(O/'probes.json');rows=[];tick=time.time()
def tv(a,b):
 ha={tuple(k):v/len(a) for k,v in zip(*np.unique(a,axis=0,return_counts=True))};hb={tuple(k):v/len(b) for k,v in zip(*np.unique(b,axis=0,return_counts=True))}
 return .5*sum(abs(ha.get(k,0)-hb.get(k,0)) for k in ha.keys()|hb.keys())
for ds in ['IMDB','PubMed']:
 fixed=torch.load(O/f'{ds}_fixed_inputs.pt',map_location='cpu')['inputs']
 # Recover immutable type inventory from frozen inputs' case data, validated via historical task hash.
 tasks=read(R/'results/output_boundary_replay_20261108T081654/tasks.json');task=next(x for x in tasks if x['dataset']==ds);name=task['row']['config']['datasets'][ds]['file'];wanted=task['row']['config']['datasets'][ds]['sha256']
 # Exact package path used by the production science entry; verify full content hash.
 import hashlib
 data=R/'research_branches/typed_structure_early_cn_expansion_review/experiments/typed_structure_early_cn_expansion_v1/package'/name
 assert hashlib.sha256(data.read_bytes()).hexdigest()==wanted
 with np.load(data) as z:types=z['types'].copy()
 for t,seed,inp in fixed:
  rr=next(x for x in probes if x['condition']==ds+'_CE' and x['t']==t);r=len(rr['relations']);core=JointLocalScore(types,inp['query'],inp['f'],inp['labels'],r);target=core.features(core.reference).numpy();target_tri=target[:,r:].sum()/3
  for parent in ['CE','PS']:
   c=ds+'_'+parent
   for arm in ['CE','CE_STRUCT']:
    prob=np.load(O/f'{c}_{arm}_t{t}.npz')['probability'];rng=np.random.default_rng(P['evaluation_mc_seeds'][ds]+t);scores=[]
    for sample in range(P['evaluation_draws']):
     bits=(rng.random(len(prob))<prob).astype(np.float64)
     with torch.no_grad():feature=core.features(torch.from_numpy(bits)).numpy();score=float(core(torch.from_numpy(bits)))
     scores.append(score);degree_w1=[];relation_w1=[];joint_tv=[]
     for ty,ix in core.nodes.items():
      ix=ix.numpy();mask=core.masks[ty].numpy()[:r];a=feature[ix,:r];b=target[ix,:r]
      degree_w1.append(float(np.mean(abs(np.sort(a.sum(1))-np.sort(b.sum(1))))))
      relation_w1.append(float(np.mean(abs(np.sort(a[:,mask],axis=0)-np.sort(b[:,mask],axis=0)))))
      joint_tv.append(tv(a[:,mask],b[:,mask]))
     rows.append(dict(condition=c,arm=arm,t=t,sample=sample,score=score,type_equal_total_degree_W1=float(np.mean(degree_w1)),type_equal_relation_degree_W1=float(np.mean(relation_w1)),type_equal_joint_degree_TV=float(np.mean(joint_tv)),triangles=float(feature[:,r:].sum()/3),triangle_abs_error=float(abs(feature[:,r:].sum()/3-target_tri))))
    saved=next(x for x in probes if x['condition']==c and x['arm']==arm and x['t']==t)
    assert np.allclose(scores,saved['hard_score_samples'],rtol=0,atol=1e-12),'not same auxiliary sample panel'
fields=list(rows[0]);
with (O/'conditional_graph_samples.csv').open('w') as f:
 w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)
pairs=[]
for c in sorted(set(x['condition'] for x in rows)):
 for t in P['times']:
  a=[x for x in rows if x['condition']==c and x['t']==t and x['arm']=='CE'];b=[x for x in rows if x['condition']==c and x['t']==t and x['arm']=='CE_STRUCT'];row=dict(condition=c,t=t)
  for k in fields[4:]:
   d=np.array([y[k]-x[k] for x,y in zip(a,b)]);row[k+'_CE']=float(np.mean([x[k] for x in a]));row[k+'_STRUCT']=float(np.mean([x[k] for x in b]));row[k+'_delta']=float(d.mean());row[k+'_paired_MC_SE']=float(d.std(ddof=1)/np.sqrt(len(d)))
  pairs.append(row)
with (O/'conditional_graph_paired.csv').open('w') as f:
 w=csv.DictWriter(f,list(pairs[0]));w.writeheader();w.writerows(pairs)
(O/'conditional_graph_audit.json').write_text(json.dumps(dict(posthoc_descriptive=True,new_random_draws=0,reconstructed_existing_draws=len(rows),existing_scores_equal=True,seconds=time.time()-tick,metric_definition='Type equal total/relation degree W1 and joint relation-degree TV on registered conditional p0 Bernoulli draws; triangle absolute error per draw then averaged, not abs(mean). No full reverse generation.'),indent=2))
print(json.dumps(pairs,indent=2))
