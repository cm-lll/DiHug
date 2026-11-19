"""Saved final graphs/logs only: score geometry audit, no model call or fitting."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import json,time,csv,hashlib,resource
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from score import RootedScore,RootedIncremental,kernel
torch.set_num_threads(1)
ROOT=Path(__file__).resolve().parents[2];RUN=ROOT/'results/rooted_connection_20261109T015142';OUT=ROOT/'results/rooted_connection_postmortem_20261109'
def read(p):return json.loads(p.read_text())
def write(name,rows):
 (OUT/(name+'.json')).write_text(json.dumps(rows,indent=2,ensure_ascii=False,allow_nan=False))
 if isinstance(rows,list) and rows:
  with (OUT/(name+'.csv')).open('w',newline='') as f:
   w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
 began=time.time();OUT.mkdir(exist_ok=False);write('protocol',dict(scope='64 saved final graphs;32 present/32 absent slots per graph with fixed seed, selected before score observation',seed=2026110911,CPU_seconds_cap=180,GPU=0,new_model_calls=0,interpretation='balanced selected-slot audit, not population error rate or network gradient; final graph differs from training conditional graph'))
 cores={};refs={};graph_rows=[];edges=[];rng=np.random.default_rng(2026110911)
 for ds in ['IMDB','PubMed']:
  z=np.load(RUN/f'{ds}_cpu_cost_panel.npz');core=RootedScore(z['types'],z['query'],z['families'],z['reference'],10);cores[ds]=core
  refs[ds]=core.features(core.reference).numpy()
 jobs=read(RUN/'evaluation_jobs.json')
 for job in jobs:
  assert time.time()-began<180
  ds=job['dataset'];core=cores[ds];ref=refs[ds];p=Path(job['folder'])/'graph.npz';binding=read(p.parent/'binding.json')
  assert hashlib.sha256(p.read_bytes()).hexdigest()==binding['graph_sha256']
  z=np.load(p);assert np.array_equal(z['types'],core.types)
  edge=z['edges'];a=np.zeros((core.n,core.n),dtype=np.uint8);a[edge[0],edge[1]]=a[edge[1],edge[0]]=1
  q=core.query.numpy();bits=a[q[0],q[1]].astype(float);assert int(bits.sum())==edge.shape[1]
  inc=RootedIncremental(core,bits);ident={k:job[k] for k in ['dataset','train_seed','arm','sample_seed']}
  xx=xy=yy=0.
  for t,c in inc.cache.items():
   if c['mask'].any():xx+=c['xx'].mean();xy+=kernel(c['x'],c['y']).mean();yy+=float(core.target_self[t])
  nt=len(inc.cache);tri=float(inc.raw[:,10:core.base_width].sum()/3);rt=float(ref[:,10:core.base_width].sum()/3)
  graph_rows.append(dict(**ident,score=inc.base_score,self_term=xx/nt,cross_term=-2*xy/nt,reference_term=yy/nt,edges=int(bits.sum()),triangles=tri,reference_triangles=rt))
  selected=[]
  for value in [0,1]:
   pool=np.flatnonzero(bits==value);selected.extend(rng.choice(pool,min(32,len(pool)),replace=False).tolist())
  indices=np.array(sorted(selected),dtype=int);diff=inc.differences(indices);flips,_=inc.raw_flips(indices)
  for k,e in enumerate(indices):
   ntg=float(flips[k,:,10:core.base_width].sum()/3)
   edges.append(dict(**ident,slot=int(e),present=int(bits[e]),family=int(core.families[e]),score_change=float((1-2*bits[e])*diff[k]),triangle_error_change=abs(ntg-rt)-abs(tri-rt),triangle_change=ntg-tri))
 write('graphs',graph_rows);write('edge_directions',edges)
 validation=[];curves=[]
 for task in read(RUN/'tasks.json'):
  folder=RUN/'run'/task['task_id'];baseline=ROOT/'results/cn_multiseed_20261104T042546/run'/task['baseline_task']
  for arm,where in [(task['arm'],folder),(task['architecture'],baseline)]:
   path=where/'validation.json'
   if not path.exists():continue
   for point in read(path):
    for mode in ['raw','ema_0_99']:
     for r in point[mode]['rows']:
      pos=sum(x['positive_count'] for x in r['family']);neg=sum(x['negative_count'] for x in r['family'])
      validation.append(dict(dataset=task['dataset'],train_seed=task['train_seed'],arm=arm,updates=point['updates'],readout=mode,t=r['t'],input_sha256=r['input_sha256'],CE=r['ce_sum_over_S'],positive_CE=sum(x['positive_count']*(x['positive_ce'] or 0) for x in r['family'])/pos if pos else None,negative_CE=sum(x['negative_count']*(x['negative_ce'] or 0) for x in r['family'])/neg if neg else None,expected_edges=sum(x['expected_edges'] for x in r['family'])))
  log=[json.loads(s) for s in (folder/'training.jsonl').read_text().splitlines()]
  for start in range(0,3000,500):
   for label,lo,hi in [('low',1,100),('middle',101,700),('high',701,1000)]:
    rows=[r for r in log[start:start+500] if lo<=r['t']<=hi]
    curves.append(dict(dataset=task['dataset'],train_seed=task['train_seed'],arm=task['arm'],end=start+500,noise=label,n=len(rows),CE=float(np.mean([r['unweighted_CE'] for r in rows])),score=float(np.mean([r['auxiliary_loss'] for r in rows])),gradient_norm_mean=float(np.mean([r['gradient_norm'] for r in rows])),gradient_norm_max=float(max(r['gradient_norm'] for r in rows))))
 write('validation',validation);write('training_bins',curves)
 write('receipt',dict(seconds=time.time()-began,peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,graphs=len(graph_rows),selected_edge_checks=len(edges),model_calls=0,GPU=0))
 print(json.dumps(read(OUT/'receipt.json')))
if __name__=='__main__':main()
