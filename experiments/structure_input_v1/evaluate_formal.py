"""Frozen evaluator, all scalar fields and full structured statistics retained."""
from .formal_runtime import *
import pathlib
import json,csv,itertools,statistics as st,traceback,math
import numpy as np
from lcc_eval import metrics as M
import evaluation_fields as E
def write(name,rows):
 save(OUT/'evaluation'/(name+'.json'),rows)
 if not rows:return
 with (OUT/'evaluation'/(name+'.csv')).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
def avg(v):return st.mean(v) if v and all(x is not None for x in v) else None
def sd(v):return st.stdev(v) if len(v)>1 and all(x is not None for x in v) else None
def main():
 began=time.time();dest=OUT/'evaluation';dest.mkdir(exist_ok=True)
 def check():
  if time.time()>read(OUT/'ledger.json')['deadline_epoch']-60:raise TimeoutError('CPU-only report repair limit; original scientific deadline unchanged')
 F=M.frozen(SCI/'lcc_eval/vendor/frozen_metrics.py',SCI/'lcc_eval/vendor/aligned_evaluate.py',SCI/'lcc_eval/vendor/hgen_utils.py');refs={};fields={};graphs={}
 for ds in ['IMDB','PubMed']:
  z=np.load(ASSETS/SCI.relative_to(ROOT)/'package/assets'/f'{ds}.npz');g=M.make_graph(z['types'],z['edges'].T,z['schema']);names=next(x['config']['datasets'][ds]['type_names'] for x in read(OUT/'tasks.json') if x['dataset']==ds);ref=read(BASE/'evaluation'/f'reference_{ds}.json');refs[ds]=(g,ref['statistics'],names,ref['raw']);fields[ds]=set();save(dest/f'reference_{ds}.json',ref)
 jobs=[]
 for tid in TASKS:
  ds,seed,_=tid.split('_',2)
  for arm in ['IID','STRUCTURE']:
   for ss in SAMPLES:
    root=BASE if arm=='IID' else OUT
    jobs.append(dict(dataset=ds,train_seed=int(seed),arm=arm,task_id=tid+'_'+arm,updates=3000,sample_seed=ss,folder=str(root/'run'/tid/'ema_0.99'/f'sample_{ss}'),reused=arm=='IID',base_task=tid))
 contracts=read(OUT/'prepared/sampling_contracts.json');inventory=[];rows=[]
 for j in jobs:
  ident={k:j[k] for k in ['dataset','train_seed','arm','task_id','updates','sample_seed']};ds=j['dataset'];ref,rs,names,reference=refs[ds];folder=pathlib.Path(j['folder']);gp=folder/'graph.npz';inv=dict(ident,file=str(gp),reused=j['reused']);dd=dest/j['task_id']/f"u{j['updates']}"/str(j['sample_seed'])
  try:
   check();b=read(folder/'binding.json');gh=sha(gp)
   if gh!=b['graph_sha256'] or any(b[k]!=j[k] for k in ['dataset','train_seed','updates','sample_seed']) or b['configuration']!=('A_late' if j['reused'] else 'STRUCTURE_late'):raise ValueError('graph identity/hash')
   rp=(BASE if j['reused'] else OUT)/'run'/j['base_task']/'readout_ema_0.99.pt'
   if sha(rp)!=b['model_sha256']:raise ValueError('EMA readout graph binding')
   steps=[json.loads(x) for x in (folder/'steps.jsonl').read_text().splitlines()];expected=contracts[f"{ds}_{j['sample_seed']}"]['steps']
   if len(steps)!=1000 or any(r['time']!=1000-i or not r['exact_coverage'] or r['uniform_sha256']!=expected[i]['uniform_sha256'] or r['query_plan_sha256']!=expected[i]['query_plan_sha256'] for i,r in enumerate(steps)):raise ValueError('all1000 paired sampling contract')
   z=np.load(gp);g=M.make_graph(z['types'],z['edges'],z['schema']);lg,mapping=M.lcc(g);dd.mkdir(parents=True,exist_ok=True)
   np.savez_compressed(dd/'lcc.npz',types=lg['types'],edges=lg['edges'],schema=lg['schema'],original_ids=lg['original_ids'])
   save(dd/'mapping.json',dict(source=str(gp),graph_sha256=gh,mapping=mapping,node_retention=len(lg['types'])/len(g['types']),edge_retention=lg['edges'].shape[1]/g['edges'].shape[1] if g['edges'].shape[1] else None,type_original=np.bincount(g['types'],minlength=4).tolist(),type_retained=np.bincount(lg['types'],minlength=4).tolist()))
   for version,gg in [('original',g),('lcc',lg)]:
    cached=dd/(version+'.json');oldpath=BASE/'evaluation'/j['base_task']/str(j['sample_seed'])/(version+'.json')
    if cached.exists() or j['reused']:
     old=read(cached if cached.exists() else oldpath)
     if old['graph_binding']['graph_sha256']!=gh:raise ValueError('cached metric hash')
     stats,comp,raw,dist,ae=[old[k] for k in ['statistics','comparison','raw','distance','absolute_error']]
    else:
     stats=M.stats(gg,F,names,check);comp=M.compare(gg,stats,ref,rs,F,names);raw,dist=E.scalar(stats,rs,comp,names);ae={k:abs(v-reference[k]) if v is not None and reference.get(k) is not None else None for k,v in raw.items()}
    if not math.isclose(dist['all10_compat_W1'],.4*dist['main_incident4_W1'],abs_tol=1e-10):raise ValueError('W1 denominator')
    cached=dd/(version+'.json')
    save(cached,dict(ident,version=version,statistics=stats,comparison=comp,raw=raw,distance=dist,absolute_error=ae,reference=reference,graph_binding=b,reused_metrics=j['reused']))
    graphs[(ds,j['train_seed'],j['arm'],j['updates'],j['sample_seed'],version)]=(gg,stats)
    for kind,vals in [('raw',raw),('absolute_error',ae),('distance',dist)]:
     for metric,value in vals.items():fields[ds].add((kind,metric));rows.append(dict(ident,version=version,kind=kind,metric=metric,value=value,reason=None if value is not None else 'undefined/unavailable; see complete statistic JSON'))
   inv['status']='complete'
  except BaseException as exc:inv.update(status='missing' if isinstance(exc,FileNotFoundError) else 'failed',error=repr(exc),traceback=traceback.format_exc())
  inventory.append(inv);save(dest/'inventory.json',inventory)
 lookup={(r['dataset'],r['train_seed'],r['arm'],r['updates'],r['sample_seed'],r['version'],r['kind'],r['metric']):r['value'] for r in rows}
 for inv in inventory:
  for version in ['original','lcc']:
   for kind,metric in fields[inv['dataset']]:
    key=tuple(inv[k] for k in ['dataset','train_seed','arm','updates','sample_seed'])+(version,kind,metric)
    if key not in lookup:lookup[key]=None;rows.append(dict(**{k:inv[k] for k in ['dataset','train_seed','arm','task_id','updates','sample_seed']},version=version,kind=kind,metric=metric,value=None,reason=inv.get('error','unavailable')))
 means=[];paired=[];across=[]
 for ds,seeds in [('IMDB',[2026110411,2026110412]),('PubMed',[2026110421,2026110422])]:
  for ver in ['original','lcc']:
   for kind,metric in sorted(fields[ds]):
    sm={}
    for seed in seeds:
     for arm in ['IID','STRUCTURE']:
      vals=[lookup[(ds,seed,arm,3000,ss,ver,kind,metric)] for ss in SAMPLES];mu=avg(vals);sm[(seed,arm)]=mu
      means.append(dict(dataset=ds,train_seed=seed,arm=arm,updates=3000,version=ver,kind=kind,metric=metric,mean=mu,sample_std=sd(vals),samples=4,valid_samples=sum(v is not None for v in vals)))
     for ss in SAMPLES+['mean']:
      a=sm[(seed,'STRUCTURE')] if ss=='mean' else lookup[(ds,seed,'STRUCTURE',3000,ss,ver,kind,metric)]
      b=sm[(seed,'IID')] if ss=='mean' else lookup[(ds,seed,'IID',3000,ss,ver,kind,metric)]
      paired.append(dict(dataset=ds,train_seed=seed,sample_seed=ss,version=ver,kind=kind,metric=metric,IID=b,STRUCTURE=a,difference=a-b if a is not None and b is not None else None))
    for arm in ['IID','STRUCTURE']:
     vals=[sm[(seed,arm)] for seed in seeds]
     across.append(dict(dataset=ds,arm=arm,version=ver,kind=kind,metric=metric,mean=avg(vals),training_seed_std=sd(vals),training_seed_means=vals,training_seeds=2,samples_per_model=4))
 diversity=[]
 for ds,seed,arm,u in sorted(set(k[:4] for k in graphs)):
  for ver in ['original','lcc']:
   for ss1,ss2 in itertools.combinations(SAMPLES,2):
    keys=[(ds,seed,arm,u,s,ver) for s in [ss1,ss2]]
    if not all(k in graphs for k in keys):continue
    (g1,s1),(g2,s2)=[graphs[k] for k in keys];c=M.compare(g1,s1,g2,s2,F,refs[ds][2]);a=set(map(tuple,g1['original_ids'][g1['edges']].T));b=set(map(tuple,g2['original_ids'][g2['edges']].T))
    diversity.append(dict(dataset=ds,train_seed=seed,arm=arm,updates=u,version=ver,sample_seed1=ss1,sample_seed2=ss2,labeled_Jaccard=len(a&b)/len(a|b) if a|b else None,edge_difference=len(a^b),exact_duplicate=a==b and set(g1['original_ids'])==set(g2['original_ids']),comparison=c))
 for name,data in [('per_graph_all_metrics',rows),('per_model_four_sample_mean',means),('paired_STRUCTURE_minus_IID',paired),('across_training_seeds',across)]:write(name,data)
 save(dest/'sample_diversity.json',diversity);save(dest/'receipt.json',dict(at=now(),expected_graphs=32,complete=sum(x['status']=='complete' for x in inventory),new_expected=16,reused_expected=16,errors=[x for x in inventory if x['status']!='complete'],seconds=time.time()-began))
 print('evaluation',sum(x['status']=='complete' for x in inventory),'/32',flush=True)
if __name__=='__main__':main()


