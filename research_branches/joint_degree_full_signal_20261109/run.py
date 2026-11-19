"""Same-coordinate and full-slot CPU feedback on frozen archived hard draws."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,json,csv,time,hashlib,resource,signal,traceback,datetime
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
torch.set_num_threads(1)
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'research_branches/joint_degree_distance_20261109'))
from run import JointDegree
sys.path.insert(0,str(ROOT/'research_branches/rooted_connection_20261109'))
from score import RootedScore,RootedIncremental
sys.path.insert(0,str(ROOT/'research_branches/explicit_structure_cpu_20261109'))
from explicit_score import ExplicitScore
OUT=ROOT/'results/joint_degree_full_signal_20261109'
PREV=ROOT/'results/joint_degree_signal_20261109/verified'
OLD=ROOT/'results/hard_structure_response_20261108T182154'
EDGE=ROOT/'results/hard_structure_edge_direction_cpu_20261108'
START=time.monotonic();BIND={}
TICKET_START=datetime.datetime.fromisoformat('2026-11-09T14:32:08+08:00').timestamp()
SCIENCE_STOP=TICKET_START+780;HARD_STOP=TICKET_START+900
def bind(p):
 p=Path(p);BIND[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest();return p
def read(p):return json.loads(bind(p).read_text())
def save(n,x):
 (OUT/(n+'.json')).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
 if isinstance(x,list) and x and isinstance(x[0],dict):
  with (OUT/(n+'.csv')).open('w',newline='') as f:
   w=csv.DictWriter(f,list(x[0]));w.writeheader();w.writerows(x)
def limits():
 if time.time()>=SCIENCE_STOP:raise TimeoutError('scientific stop; preserve partial results')
 if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024>=2048:raise MemoryError('2GiB RSS cap')
def cosine(a,b):
 den=np.linalg.norm(a)*np.linalg.norm(b);return float(a@b/den) if den else None
def compare(a,b,slots,scope,tag,method,families):
 x=a[slots];y=b[slots];both=(abs(x)>1e-15)&(abs(y)>1e-15);opp=(x*y<0)&both
 mass=(abs(x)+abs(y))/2;den=float(mass.sum());avg=(np.linalg.norm(x)+np.linalg.norm(y))/2
 return dict(panel=tag,scope=scope,method=method,slots=len(slots),cosine=cosine(x,y),
  both_nonzero=int(both.sum()),opposed=int(opp.sum()),opposed_abs_mass_share=float(mass[opp].sum()/den) if den else None,
  mean_norm_over_average_norm=float(np.linalg.norm((x+y)/2)/avg) if avg else None,
  relative_difference=float(np.linalg.norm(x-y)/avg) if avg else None)
def main():
 OUT.mkdir(exist_ok=False);signal.alarm(max(1,int(HARD_STOP-time.time())))
 save('protocol',dict(start_BJT='2026-11-09T14:32:08+08:00',scientific_stop_BJT='2026-11-09T14:45:08+08:00',hard_stop_BJT='2026-11-09T14:47:08+08:00',
  scope='12 archived probability panels, same union of old selected slots on both hard draws; PubMed t1000 CE/PS parents full slots. No new draws/model/updates.',
  limits=dict(threads=1,RSS_MiB=2048,GPU=0,wall_seconds=900),max_coordinate_evaluations=78874,
  frozen='old explicit TV vs replace TV/8 by geometric joint-degree/8; all other components unchanged; no coefficient selection',
  statistics='full-slot and selected-union kept separate; directions and magnitudes per relation/label/time/history; two draws descriptive, not converged variance estimate',
  subset_noise='Analytic uniform without-replacement HT M/m variance conditional on frozen hard draw; not sampling new subsets or claiming parameter noise',
  decision='No success gate or automatic training; retain costs and decide next parameter-level feasibility from complete evidence.'))
 for p in [Path(__file__),ROOT/'research_branches/joint_degree_distance_20261109/run.py',ROOT/'research_branches/rooted_connection_20261109/score.py',ROOT/'research_branches/explicit_structure_cpu_20261109/explicit_score.py']:bind(p)
 oldbindings=read(PREV/'binding.json')
 def checked(p):
  p=bind(p);assert oldbindings.get(str(p))==BIND[str(p)],str(p);return p
 rows=read(PREV/'slot_feedback.json');identity=read(PREV/'conditional_identity.json')
 unions=defaultdict(set);original={}
 for r in rows:
  tag=f"{r['dataset']}_{r['parent']}_CE_t{r['t']}";unions[tag].add(r['slot']);original[(tag,r['draw'],r['slot'])]=r
 save('frozen_slots',{k:sorted(v) for k,v in unions.items()})
 fixtures={};inputs={}
 for ds in ['PubMed','IMDB']:
  with np.load(checked(ROOT/f'results/rooted_connection_20261109T015142/{ds}_cpu_cost_panel.npz')) as z:fixtures[ds]={k:z[k].copy() for k in ['types','query','families','reference']}
  inputs[ds]={int(t):inp for t,seed,inp in torch.load(checked(OLD/f'{ds}_fixed_inputs.pt'),map_location='cpu',weights_only=False)['inputs']}
 pairrows=[];relpairs=[];summaries=[];relrows=[];costrows=[];tests=[];count=0;calibrated=False
 tasks=[(ds,parent,t) for ds in ['PubMed','IMDB'] for parent in ['CE','PS'] for t in [1000,1,500]]
 for ds,parent,t in tasks:
  limits();tag=f'{ds}_{parent}_CE_t{t}';inp=inputs[ds][t];q,f,y=map(np.asarray,(inp['query'],inp['f'],inp['labels']))
  z=fixtures[ds];n=len(z['types']);i=np.argsort(q[0]*n+q[1]);j=np.argsort(z['query'][0]*n+z['query'][1])
  assert np.array_equal(q[:,i],z['query'][:,j]) and np.array_equal(f[i],z['families'][j]) and np.array_equal(y[i],z['reference'][j])
  jd=JointDegree(dict(types=z['types'],query=q,families=f,reference=y));core=RootedScore(z['types'],q,f,y,10);sc=ExplicitScore(core)
  with np.load(checked(OLD/f'{tag}.npz')) as z:p=z['probability'].astype(float)
  assert p.shape==y.shape and np.isfinite(p).all() and np.all((p>=0)&(p<=1))
  M=len(p);allslots=ds=='PubMed' and t==1000;union=np.array(sorted(unions[tag]),dtype=int);slots=np.arange(M) if allslots else union
  vectors=[];started=time.monotonic();ident=next(r for r in identity if r['panel']==tag)
  for draw in [0,1]:
   with np.load(checked(EDGE/f'{tag}_draw{draw}_flips.npz')) as z:bits=(1-z['flip_sign'])/2
   inc=RootedIncremental(core,bits);base=sc.terms(inc.raw);joint0,_=jd.terms(inc.raw[:,:10]);refiso=int((sc.ref[:,:10].sum(1)==0).sum());iso0=int((inc.raw[:,:10].sum(1)==0).sum())
   base_tri=float(sc.triangle_counts(inc.raw).sum());target_tri=float(sc.reftri.sum())
   arrays={k:np.full(M,np.nan) for k in ['old_delta','new_delta','old_g','new_g','triangle_change','triangle_error_change','hub_error_change','isolates_error_change','joint_distance_change','closure_change','low_change']}
   for start in range(0,len(slots),32):
    limits();tick=time.monotonic();ix=slots[start:start+32];raw,_=inc.raw_flips(ix);val=sc.terms(raw)
    for k,e in enumerate(ix):
     joint,_=jd.terms(raw[k,:,:10]);dt=joint['TV']-joint0['TV'];dj=joint['joint']-joint0['joint']
     assert abs(dt-(val['joint_TV'][k]-base['joint_TV'][0]))<1e-10
     a=float(val['total'][k]-base['total'][0]);b=a+(dj-dt)/8;factor=float(p[e]*(1-p[e])*(1-2*bits[e]))
     tri=float(sc.triangle_counts(raw[k]).sum());isoc=int((raw[k,:,:10].sum(1)==0).sum())
     vals=[a,b,factor*a,factor*b,tri-base_tri,abs(tri-target_tri)-abs(base_tri-target_tri),float(val['hub'][k]-base['hub'][0]),abs(isoc-refiso)-abs(iso0-refiso),dj,float(val['closure'][k]-base['closure'][0]),float(val['low'][k]-base['low'][0])]
     for key,value in zip(arrays,vals):arrays[key][e]=value
     if (tag,draw,int(e)) in original:
      r=original[(tag,draw,int(e))]
      assert abs(a-r['old_flip_delta'])<1e-10 and abs(b-r['new_flip_delta'])<1e-10
      assert abs(arrays['new_g'][e]-r['new_logit_derivative'])<1e-11
    count+=len(ix)
    if not calibrated and count>=256:
     elapsed=time.monotonic()-started;estimate=elapsed/count*78874
     save('calibration',dict(first_coordinates=count,seconds=elapsed,linear_upper_panel_seconds=estimate,remaining_scientific_seconds=SCIENCE_STOP-time.time(),safety_factor=1.5))
     if estimate*1.5>SCIENCE_STOP-time.time():raise TimeoutError('full panel estimate exceeds remaining budget; no scope reduction')
     calibrated=True
    if start%2048==0:save('progress',dict(panel=tag,draw=draw,coordinates_completed=count,elapsed_seconds=time.monotonic()-START))
   # Independent complete feature recalculation on three fixed coordinates.
   for e in slots[np.linspace(0,len(slots)-1,3,dtype=int)]:
    bb=bits.copy();bb[e]=1-bb[e]
    with torch.no_grad():rr=core.features(torch.from_numpy(bb)).numpy()
    vv=sc.terms(rr);jj,_=jd.terms(rr[:,:10]);actual=float(vv['total'][0]-base['total'][0]+((jj['joint']-joint0['joint'])-(jj['TV']-joint0['TV']))/8)
    err=abs(actual-arrays['new_delta'][e]);assert err<1e-10;tests.append(dict(panel=tag,draw=draw,slot=int(e),full_feature_delta_error=err))
   assert all(np.isfinite(a[slots]).all() for a in arrays.values())
   np.savez_compressed(OUT/f'{tag}_draw{draw}.npz',slots=slots,union_slots=union,probability=p,reference=y,bits=bits,relation=f,**arrays)
   vectors.append(arrays)
   scopes=[('union',union)]+([('full',np.arange(M))] if allslots else [])
   ce=(p-y)/(n*(n-1)/2)
   for scope,ss in scopes:
    for method in ['old','new']:
     delta=arrays[method+'_delta'][ss];g=arrays[method+'_g'][ss];good=delta<-1e-12;bad=good&(arrays['triangle_error_change'][ss]>1e-12);mass=float(abs(g[good]).sum())
     summaries.append(dict(panel=tag,dataset=ds,parent=parent,train_seed=ident['train_seed'],updates=ident['updates'],readout=ident['readout'],t=t,draw=draw,scope=scope,method=method,slots=len(ss),
      rewarded=int(good.sum()),triangle_cost=int(bad.sum()),hub_cost=int((good&(arrays['hub_error_change'][ss]>1e-12)).sum()),isolates_cost=int((good&(arrays['isolates_error_change'][ss]>0)).sum()),
      rewarded_abs_mass=mass,triangle_cost_abs_mass=float(abs(g[bad]).sum()),triangle_cost_mass_share=float(abs(g[bad]).sum()/mass) if mass else None,
      gradient_norm=float(np.linalg.norm(g)),cosine_CE=cosine(g,ce[ss]),CE_opposed=int((g*ce[ss]<0).sum()),
      prediction_saturated=int(((p[ss]==0)|(p[ss]==1)).sum())))
     for rel in np.unique(f[ss]):
      for label in [0,1]:
       sub=ss[(f[ss]==rel)&(y[ss]==label)];gg=arrays[method+'_g'][sub];dd=arrays[method+'_delta'][sub];ok=dd<-1e-12;cost=ok&(arrays['triangle_error_change'][sub]>1e-12)
       relrows.append(dict(panel=tag,draw=draw,scope=scope,method=method,relation=int(rel),label=label,support=len(sub),
        norm=float(np.linalg.norm(gg)),cosine_CE=cosine(gg,ce[sub]),CE_opposed=int((gg*ce[sub]<0).sum()),rewarded=int(ok.sum()),triangle_cost=int(cost.sum()),
        triangle_cost_abs_mass=float(abs(gg[cost]).sum()),rewarded_abs_mass=float(abs(gg[ok]).sum())))
     if scope=='full':
      m=min(1024,M);costrows.append(dict(panel=tag,draw=draw,method=method,M=M,m=m,relative_MSE=M/m-1,relative_RMS=float(np.sqrt(M/m-1)),
       absolute_expected_error_norm_squared=float((M/m-1)*(g@g)),scope='analytic logit HT sampling error conditioned on this hard draw, not parameter error'))
   save('panel_summary',summaries);save('relation_label_summary',relrows)
   print(tag,draw,'slots',len(slots),'seconds',round(time.monotonic()-started,2),flush=True)
  for scope,ss in [('union',union)]+([('full',np.arange(M))] if allslots else []):
   for method in ['old','new']:
    x=vectors[0][method+'_g'];yy=vectors[1][method+'_g'];pairrows.append(compare(x,yy,ss,scope,tag,method,f))
    for rel in np.unique(f[ss]):
     for label in [0,1]:
      sub=ss[(f[ss]==rel)&(y[ss]==label)]
      relpairs.append(dict(**compare(x,yy,sub,scope,tag,method,f),relation=int(rel),label=label))
  save('paired_draws',pairrows);save('paired_draws_by_relation_label',relpairs);save('analytic_HT_noise',costrows);save('tests',tests)
 assert count<=78874 and len(pairrows)==28 and len(summaries)==56 and len(tests)==72
 assert not torch.cuda.is_initialized();limits()
 assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in BIND.items())
 save('binding',BIND);save('receipt',dict(status='complete',worker_seconds=time.monotonic()-START,ticket_elapsed_seconds=time.time()-TICKET_START,
  peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,coordinates=count,hard_states=24,full_slot_states=4,models=0,updates=0,new_graphs=0,GPU_hours=0,assets_unchanged=True))
 print((OUT/'receipt.json').read_text(),flush=True)
if __name__=='__main__':
 try:main()
 except BaseException as exc:
  if OUT.exists():save('failure',dict(error=repr(exc),traceback=traceback.format_exc(),worker_seconds=time.monotonic()-START))
  raise
