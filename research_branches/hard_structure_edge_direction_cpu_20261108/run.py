import os,sys,json,time,csv,hashlib,traceback,resource
from pathlib import Path
import numpy as np
import torch
torch.set_num_threads(1)
R=Path(__file__).resolve().parents[2];H=Path(__file__).parent;O=R/'results/hard_structure_edge_direction_cpu_20261108';OLD=R/'results/hard_structure_response_20261108T182154'
sys.path.insert(0,str(R/'research_branches/hard_structure_incremental_cpu_20261108'));sys.path.insert(0,str(R/'research_branches/hard_structure_rebar_cpu_20261108'))
from incremental import IncrementalScore
from objective import JointLocalScore
from metrics import Metrics,brute
read=lambda p:json.loads(Path(p).read_text());P=read(O/'protocol.json');oldp=read(OLD/'protocol.json');oldrows=read(OLD/'probes.json');start=time.time();rows=[];relations=[];examples=[];status=dict(status='running',completed_graphs=0,edge_flips=0,reference_checks=0,new_random_draws=0,model_calls=0,GPU=0,updates=0)
def save(n,x):(O/n).write_text(json.dumps(x,indent=2))
def check():
 if time.time()>P['science_stop_epoch'] or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>P['max_rss_bytes']:raise TimeoutError('CPU budget')
def csvwrite(name,data):
 if not data:return
 with (O/name).open('w') as f:
  w=csv.DictWriter(f,list(data[0]));w.writeheader();w.writerows(data)
def statistics(ix,score_delta,changes,mass):
 good=ix&(score_delta<-P['score_tolerance']);mg=float(mass[good].sum());out=dict(slots=int(ix.sum()),score_improving_flips=int(good.sum()),score_improving_gradient_mass=mg)
 for name in ['triangle_abs_error','total_degree_W1','joint_degree_TV']:
  bad=good&(changes[name]>P['metric_tolerance']);better=good&(changes[name]<-P['metric_tolerance']);out[name+'_worse_count']=int(bad.sum());out[name+'_better_count']=int(better.sum());out[name+'_worse_fraction']=float(bad.sum()/good.sum()) if good.any() else None;out[name+'_worse_gradient_mass_fraction']=float(mass[bad].sum()/mg) if mg else None
 both=good&(changes['triangle_abs_error']>P['metric_tolerance'])&(changes['total_degree_W1']>P['metric_tolerance']);allbad=both&(changes['joint_degree_TV']>P['metric_tolerance']);out['triangle_and_degree_worse_count']=int(both.sum());out['all_three_worse_count']=int(allbad.sum());return out
def main():
 sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
 for filename in ['source_binding.json','asset_binding.json']:
  for path,value in read(O/filename).items():assert sha(path)==value,path
 # Independent finite tiny-graph regression, including same-type endpoint histogram collisions.
 rng=np.random.default_rng(2026110820);maxerr=0.;checks=0
 for trial in range(P['synthetic_cases']):
  n=6;types=np.array([0,0,0,1,1,1]);q=np.array([(i,j) for i in range(n) for j in range(i+1,n)]).T;f=np.array([types[i]+types[j] for i,j in q.T]);y=rng.integers(0,2,len(f));bits=rng.integers(0,2,len(f));met=Metrics(types,q,f,y,bits,3);change=met.changes();base=brute(types,q,f,y,bits,3)
  for e in range(len(f)):
   flip=bits.copy();flip[e]=1-flip[e];new=brute(types,q,f,y,flip,3)
   for key in base:maxerr=max(maxerr,abs(new[key]-base[key]-change[key][e]));assert abs(new[key]-base[key]-change[key][e])<1e-12
   checks+=1
 save('synthetic_tests.json',dict(graphs=P['synthetic_cases'],edge_checks=checks,max_abs_error=maxerr))
 tasks=read(R/'results/output_boundary_replay_20261108T081654/tasks.json')
 for ds in ['IMDB','PubMed']:
  task=next(x for x in tasks if x['dataset']==ds);cfg=task['row']['config'];data=R/'research_branches/typed_structure_early_cn_expansion_review/experiments/typed_structure_early_cn_expansion_v1/package'/cfg['datasets'][ds]['file'];assert sha(data)==cfg['datasets'][ds]['sha256']
  with np.load(data) as z:types=z['types'].copy()
  fixed=torch.load(OLD/f'{ds}_fixed_inputs.pt',map_location='cpu')['inputs']
  for t,seed,inp in fixed:
   q,f,y=inp['query'],inp['f'],inp['labels'];r=len(next(x for x in oldrows if x['condition']==ds+'_CE' and x['t']==t)['relations']);core=JointLocalScore(types,q,f,y,r)
   for parent in ['CE','PS']:
    condition=ds+'_'+parent
    for arm in P['arms']:
     prob=np.load(OLD/f'{condition}_{arm}_t{t}.npz')['probability'];saved=next(x for x in oldrows if x['condition']==condition and x['arm']==arm and x['t']==t);aux=np.random.default_rng(oldp['evaluation_mc_seeds'][ds]+t)
     for draw in P['samples']:
      check();tick=time.perf_counter();bits=(aux.random(len(prob))<prob).astype(float);inc=IncrementalScore(core,bits);assert abs(inc.base_score-saved['hard_score_samples'][draw])<1e-12
      diff=inc.differences(batch_size=128);met=Metrics(types,q,f,y,bits,r);change=met.changes();flip=change['flip_sign']*diff;mass=prob*(1-prob)*abs(diff)
      # Independent old full objective check for 8 pre-fixed slots, no outcome-dependent index selection.
      selected=np.unique(np.linspace(0,len(prob)-1,P['numeric_reference_indices'],dtype=int));referr=0.
      for e in selected:
       nb=bits.copy();nb[e]=1-nb[e]
       with torch.no_grad():value=float(core(torch.from_numpy(nb)))-inc.base_score
       referr=max(referr,abs(value-flip[e]));assert abs(value-flip[e])<1e-11;status['reference_checks']+=1
      state='under' if met.triangles<met.ref_triangles else ('over' if met.triangles>met.ref_triangles else 'equal')
      base=dict(condition=condition,arm=arm,t=t,draw=draw,triangle_state=state,triangles=met.triangles,reference_triangles=met.ref_triangles,score=inc.base_score,degree_W1=met.base_w1,joint_TV=met.base_tv)
      row=dict(**base,**statistics(np.ones(len(prob),dtype=bool),flip,change,mass),score_reference_max_error=referr,seconds=time.perf_counter()-tick)
      rows.append(row)
      for rel in range(r):
       for direction,mask in [('add',bits==0),('delete',bits==1)]:relations.append(dict(**base,relation=rel,direction=direction,**statistics((f==rel)&mask,flip,change,mass)))
      # Registered edge-order examples: first conflicting + first jointly improving; not representative frequency claims.
      for kind,mask in [('score_down_triangle_up',(flip<-1e-12)&(change['triangle_abs_error']>1e-12)),('score_down_all_three_up',(flip<-1e-12)&(change['triangle_abs_error']>1e-12)&(change['total_degree_W1']>1e-12)&(change['joint_degree_TV']>1e-12)),('score_down_all_three_down',(flip<-1e-12)&(change['triangle_abs_error']<-1e-12)&(change['total_degree_W1']<-1e-12)&(change['joint_degree_TV']<-1e-12))]:
       ix=np.flatnonzero(mask)
       if len(ix):
        e=int(ix[0]);examples.append(dict(**base,kind=kind,slot=e,i=int(q[0,e]),j=int(q[1,e]),relation=int(f[e]),present=bool(bits[e]),reference_label=int(y[e]),probability=float(prob[e]),common_neighbors=int(change['common_neighbors'][e]),score_flip=float(flip[e]),triangle_error_flip=float(change['triangle_abs_error'][e]),degree_W1_flip=float(change['total_degree_W1'][e]),joint_TV_flip=float(change['joint_degree_TV'][e])))
      np.savez_compressed(O/f'{condition}_{arm}_t{t}_draw{draw}_flips.npz',score_flip=flip,triangle_abs_error_flip=change['triangle_abs_error'],total_degree_W1_flip=change['total_degree_W1'],joint_degree_TV_flip=change['joint_degree_TV'],gradient_mass=mass,flip_sign=change['flip_sign'])
      status['completed_graphs']+=1;status['edge_flips']+=len(prob);save('progress.json',status);csvwrite('graph_summary.csv',rows);csvwrite('relations.csv',relations);csvwrite('examples.csv',examples)
     print(json.dumps(dict(condition=condition,arm=arm,t=t,completed=status['completed_graphs'],elapsed=time.time()-start)),flush=True)
 assert status['completed_graphs']==96 and status['reference_checks']==768
 for filename in ['source_binding.json','asset_binding.json']:
  for path,value in read(O/filename).items():assert sha(path)==value,path
 assert not torch.cuda.is_initialized();status['status']='complete';save('assets_unchanged.json',dict(unchanged=True,cuda_initialized=False))
try:main()
except BaseException as exc:status.update(status='failed_or_stopped',error=repr(exc),traceback=traceback.format_exc())
finally:status.update(seconds=time.time()-start,end_epoch=time.time(),peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024);save('receipt.json',status);print(json.dumps(status),flush=True)
raise SystemExit(0 if status['status']=='complete' else 2)
