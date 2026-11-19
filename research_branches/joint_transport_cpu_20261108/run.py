import sys,time,json,csv,itertools,resource,hashlib,traceback
from pathlib import Path
import numpy as np
import torch
torch.set_num_threads(1)
R=Path(__file__).resolve().parents[2];O=R/'results/joint_transport_cpu_20261108';OLD=R/'results/hard_structure_response_20261108T182154';EDGE=R/'results/hard_structure_edge_direction_cpu_20261108'
sys.path.insert(0,str(R/'research_branches/hard_structure_rebar_cpu_20261108'));sys.path.insert(0,str(R/'research_branches/hard_structure_edge_direction_cpu_20261108'))
from objective import JointLocalScore
from metrics import Metrics,brute
from transport import JointTransport,assignment_value
read=lambda p:json.loads(Path(p).read_text());P=read(O/'protocol.json');oldp=read(OLD/'protocol.json');start=time.time();status=dict(status='running',graphs=0,selected_flips=0,full_benchmark_flips=0,bound_checks=0,model_calls=0,GPU=0,updates=0);rows=[];edges=[];bench=[];cases=[];scales=[]
def save(n,x):(O/n).write_text(json.dumps(x,indent=2))
def csvsave(n,x):
 if not x:return
 with (O/n).open('w') as f:
  w=csv.DictWriter(f,list(x[0]));w.writeheader();w.writerows(x)
def check():
 if time.time()>=P['science_stop_epoch'] or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>P['max_RSS_bytes']:raise TimeoutError('CPU budget')
def stats(score,metrics,prob,iw):
 good=score<-P['score_tolerance'];mass=abs(score)*prob*(1-prob);den=float(iw[good].sum());gm=float((iw[good]*mass[good]).sum());out=dict(improving_sample_slots=int(good.sum()),improving_HT_slots=den,improving_HT_gradient_mass=gm)
 for key in ['triangle_abs_error','total_degree_W1','joint_degree_TV']:
  bad=good&(metrics[key]>1e-12);out[key+'_bad_sample_slots']=int(bad.sum());out[key+'_bad_HT_slots']=float(iw[bad].sum());out[key+'_bad_fraction_HT']=float(iw[bad].sum()/den) if den else None;out[key+'_bad_gradient_fraction_HT']=float((iw[bad]*mass[bad]).sum()/gm) if gm else None
 bad=good
 for key in ['triangle_abs_error','total_degree_W1','joint_degree_TV']:bad=bad&(metrics[key]>1e-12)
 out['all_three_bad_sample_slots']=int(bad.sum());out['all_three_bad_HT_slots']=float(iw[bad].sum());return out
def main():
 sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
 for fn in ['source_binding.json','asset_binding.json']:
  for path,h in read(O/fn).items():assert sha(path)==h,path
 rng=np.random.default_rng(2026110831);error=0.;cnt=0;bound_error=0.
 for trial in range(P['synthetic_graphs']):
  types=np.array([0,0,0,1,1,1]);q=np.array([(i,j) for i in range(6) for j in range(i+1,6)]).T;f=np.array([types[i]+types[j] for i,j in q.T]);y=rng.integers(0,2,len(f));bits=rng.integers(0,2,len(f));core=JointLocalScore(types,q,f,y,3);ot=JointTransport(core,bits)
  for c in ot.cache.values():
   cost=c['cost'];exact=min(sum(cost[i,j] for i,j in enumerate(perm))/len(cost) for perm in itertools.permutations(range(len(cost))));assert abs(exact-assignment_value(cost))<1e-12
  assert abs(ot.value(ot.refraw)[0])<1e-12
  perm=np.r_[rng.permutation(3),3+rng.permutation(3)];assert abs(ot.value(ot.raw[perm])[0]-ot.base)<1e-12
  flips,bs=ot.differences(np.arange(len(f)),True)
  for e in range(len(f)):
   b=bits.copy();b[e]=1-b[e]
   with torch.no_grad():raw=core.features(torch.as_tensor(b,dtype=torch.float64)).numpy()
   val,_=ot.value(raw);error=max(error,abs(val-ot.base-flips[e]));assert abs(val-ot.base-flips[e])<1e-12;metric=brute(types,q,f,y,b,3)
   for key in metric:bound_error=max(bound_error,metric[key]-bs[e][key]);assert metric[key]<=bs[e][key]+1e-10
   cnt+=1
 save('synthetic_tests.json',dict(graphs=P['synthetic_graphs'],flip_checks=cnt,max_score_difference=error,max_bound_violation=bound_error,exact_small_permutations=True,reference_zero=True,type_permutation_invariant=True))
 tasks=read(R/'results/output_boundary_replay_20261108T081654/tasks.json');examples=list(csv.DictReader((EDGE/'examples.csv').open()))
 for dsid,ds in enumerate(['IMDB','PubMed']):
  task=next(x for x in tasks if x['dataset']==ds);cfg=task['row']['config'];data=R/'research_branches/typed_structure_early_cn_expansion_review/experiments/typed_structure_early_cn_expansion_v1/package'/cfg['datasets'][ds]['file'];assert sha(data)==cfg['datasets'][ds]['sha256']
  with np.load(data) as z:types=z['types'].copy()
  fixed=torch.load(OLD/f'{ds}_fixed_inputs.pt',map_location='cpu')['inputs'];oldrows=read(OLD/'probes.json')
  for t,seed,inp in fixed:
   q,f,y=inp['query'],inp['f'],inp['labels'];r=len(next(x for x in oldrows if x['condition']==ds+'_CE' and x['t']==t)['relations']);core=JointLocalScore(types,q,f,y,r)
   for parent in ['CE','PS']:
    c=ds+'_'+parent
    for arm in ['CE','CE_STRUCT']:
     prob=np.load(OLD/f'{c}_{arm}_t{t}.npz')['probability'];aux=np.random.default_rng(oldp['evaluation_mc_seeds'][ds]+t)
     for draw in P['real_aux_samples']:
      check();tick=time.perf_counter();bits=(aux.random(len(prob))<prob).astype(float);ot=JointTransport(core,bits);met=Metrics(types,q,f,y,bits,r);archive=np.load(EDGE/f'{c}_{arm}_t{t}_draw{draw}_flips.npz');assert np.array_equal(archive['flip_sign'],1-2*bits)
      # Frozen random priority shared across arms. Select by relation/presence only, never by error or score.
      ranks=np.random.default_rng(P['slot_selection_seed']+dsid*100000+t*10+draw).random(len(prob));selected=[];inverse=[]
      for rel in range(r):
       for present in [0,1]:
        ix=np.flatnonzero((f==rel)&(bits==present));k=min(len(ix),P['slots_per_relation_presence_stratum']);chosen=ix[np.argsort(ranks[ix])[:k]];selected.extend(chosen.tolist());inverse.extend([len(ix)/k]*k if k else [])
      selected=np.asarray(selected,dtype=int);inverse=np.asarray(inverse);costtick=time.perf_counter();delta,bounds=ot.differences(selected,True);seconds=time.perf_counter()-costtick
      metrics={key:archive[key+'_flip'][selected] for key in ['triangle_abs_error','total_degree_W1','joint_degree_TV']};bases=dict(triangle_abs_error=abs(met.triangles-met.ref_triangles),total_degree_W1=met.base_w1,joint_degree_TV=met.base_tv)
      for j,bd in enumerate(bounds):
       for key in metrics:assert bases[key]+metrics[key][j]<=bd[key]+1e-9
       status['bound_checks']+=1
      base=dict(condition=c,arm=arm,t=t,draw=draw,triangles=met.triangles,reference_triangles=met.ref_triangles,triangle_state='under' if met.triangles<met.ref_triangles else ('over' if met.triangles>met.ref_triangles else 'equal'),selected_slots=len(selected),all_slots=len(prob))
      for method,score in [('MMD',archive['score_flip'][selected]),('JOINT_W1',delta)]:rows.append(dict(**base,method=method,base_score=float(ot.base) if method=='JOINT_W1' else None,seconds=seconds if method=='JOINT_W1' else None,**stats(score,metrics,prob[selected],inverse)))
      for j,e in enumerate(selected):edges.append(dict(**base,slot=int(e),relation=int(f[e]),present=int(bits[e]),probability=float(prob[e]),inverse_inclusion_probability=float(inverse[j]),MMD_flip=float(archive['score_flip'][e]),JOINT_W1_flip=float(delta[j]),**{k+'_flip':float(v[j]) for k,v in metrics.items()}))
      # Retest every old pre-registered representative separately, never pool it into stratified statistics.
      ex=[x for x in examples if x['condition']==c and x['arm']==arm and int(x['t'])==t and int(x['draw'])==draw]
      if ex:
       dscores=ot.differences([int(x['slot']) for x in ex])
       for x,dv in zip(ex,dscores):cases.append(dict(**x,JOINT_W1_flip=float(dv),JOINT_W1_still_improves=bool(dv<-1e-12)))
      if parent=='CE' and arm=='CE' and t==500 and draw==0:
       forecast=seconds/len(selected)*len(prob);br=dict(dataset=ds,forecast_seconds=forecast,performed=False)
       if forecast<P['full_benchmark_forecast_limit_seconds'] and time.time()+forecast+60<P['science_stop_epoch']:
        check();bt=time.perf_counter();full=ot.differences(np.arange(len(prob)));br.update(performed=True,seconds=time.perf_counter()-bt,slots=len(prob));np.savez_compressed(O/f'{ds}_full_cost_map.npz',JOINT_W1_flip=full);status['full_benchmark_flips']+=len(prob)
        fm={k:archive[k+'_flip'] for k in metrics};br.update({method:stats(sc,fm,prob,np.ones(len(prob))) for method,sc in [('MMD',archive['score_flip']),('JOINT_W1',full)]})
       bench.append(br);save('full_cost_benchmarks.json',bench)
      if t==1 and parent=='CE' and arm=='CE' and draw==0:
       for ty,cache in ot.cache.items():scales.append(dict(dataset=ds,type=int(ty),nodes=len(cache['nodes']),scale=cache['scale'].tolist(),weights=cache['weights'].tolist(),active_degree=cache['active_degree'],active_triangle=cache['active_triangle'],block_count=cache['block_count']))
       save('frozen_scales.json',scales)
      status['graphs']+=1;status['selected_flips']+=len(selected);save('progress.json',status);csvsave('graph_summary.csv',rows);csvsave('selected_edges.csv',edges);csvsave('old_case_retest.csv',cases)
     print(json.dumps(dict(condition=c,arm=arm,t=t,graphs=status['graphs'],elapsed=time.time()-start)),flush=True)
 assert status['graphs']==96
 for fn in ['source_binding.json','asset_binding.json']:
  for path,h in read(O/fn).items():assert sha(path)==h,path
 assert not torch.cuda.is_initialized();save('assets_unchanged.json',dict(unchanged=True,cuda_initialized=False));status['status']='complete'
try:main()
except BaseException as e:status.update(status='failed_or_stopped',error=repr(e),traceback=traceback.format_exc())
finally:status.update(seconds=time.time()-start,end_epoch=time.time(),peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024);save('receipt.json',status);print(json.dumps(status),flush=True)
raise SystemExit(0 if status['status']=='complete' else 2)
