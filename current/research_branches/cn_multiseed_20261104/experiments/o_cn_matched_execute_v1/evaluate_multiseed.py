"""Complete frozen metrics, separate prospective/historical/posthoc panels."""
import csv,itertools,json,math,pathlib,statistics,time,traceback
import numpy as np
from common_run import ROOT,SCI,OUT,read,save,sha,now
from lcc_eval import metrics as M
import evaluation_fields as E

def avg(xs):return statistics.mean(xs)if xs and all(x is not None for x in xs)else None
def sd(xs):return statistics.stdev(xs)if len(xs)>1 and all(x is not None for x in xs)else None
def write(name,rows):
 save(OUT/'evaluation'/(name+'.json'),rows)
 with(OUT/'evaluation'/(name+'.csv')).open('w',newline='')as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)

def main():
 start=time.time();dest=OUT/'evaluation';dest.mkdir(exist_ok=True);ledger=read(OUT/'ledger.json');approval=read(OUT/'approval.json')
 for p,h in approval['source_files'].items():
  if sha(p)!=h:raise ValueError('source binding drift '+p)
 def check():
  if time.time()>=ledger['deadline_epoch']-90:raise TimeoutError('delivery reserve')
 F=M.frozen(SCI/'lcc_eval/vendor/frozen_metrics.py',SCI/'lcc_eval/vendor/aligned_evaluate.py',SCI/'lcc_eval/vendor/hgen_utils.py')
 refs={};fields={};rows=[];inventory=[];errors=[];graphs={};contracts=read(OUT/'prepared/sampling_contracts.json')
 for ds in ['IMDB','PubMed']:
  z=np.load(SCI/'package/assets'/f'{ds}.npz');g=M.make_graph(z['types'],z['edges'].T,z['schema']);names=read(SCI/'package/config.json')['datasets'][ds]['type_names']
  stats=M.stats(g,F,names,check);comp=M.compare(g,stats,g,stats,F,names);raw,dist=E.scalar(stats,stats,comp,names)
  refs[ds]=(g,stats,names,raw);fields[ds]={(kind,m)for kind,d in [('raw',raw),('absolute_error',raw),('distance',dist)]for m in d}
  save(dest/f'reference_{ds}.json',dict(statistics=stats,comparison=comp,raw=raw,distance=dist))
 jobs=read(OUT/'tasks.json')+read(OUT/'reused_jobs.json')
 for job in jobs:
  ds,seed,arm=job['dataset'],job['train_seed'],job['arm'];ref,rs,names,reference=refs[ds]
  for ss in job['sample_seeds']:
   ident=dict(dataset=ds,train_seed=seed,arm=arm,sample_seed=ss,updates=3000,readout='EMA0.99',task_id=job['task_id'],selection='posthoc_case'if seed==2026110321 else 'unselected_seeds')
   folder=pathlib.Path(job['folder'])/'ema_0.99'/f'sample_{ss}';gp=folder/'graph.npz';inv=dict(ident,graph_file=str(gp))
   try:
    check();b=read(folder/'binding.json')
    if sha(gp)!=b['graph_sha256']or any(b[k]!=v for k,v in [('dataset',ds),('train_seed',seed),('sample_seed',ss),('updates',3000),('configuration',arm+'_late')]):raise ValueError('graph identity/hash')
    if sha(pathlib.Path(job['folder'])/'readout_ema_0.99.pt')!=b['model_sha256']:raise ValueError('wrong readout')
    steps=[json.loads(x)for x in(folder/'steps.jsonl').read_text().splitlines()];expected=contracts[f'{ds}_{ss}']['steps']
    if len(steps)!=1000 or any(r['time']!=1000-i or not r['exact_coverage']or r['uniform_sha256']!=expected[i]['uniform_sha256']or r['query_plan_sha256']!=expected[i]['query_plan_sha256']for i,r in enumerate(steps)):raise ValueError('paired sampling contract')
    z=np.load(gp);g=M.make_graph(z['types'],z['edges'],z['schema']);lg,mapping=M.lcc(g);dd=dest/job['task_id']/str(ss);dd.mkdir(parents=True,exist_ok=True)
    if not set(map(tuple,lg['original_ids'][lg['edges']].T))<=set(map(tuple,g['edges'].T)):raise ValueError('LCC new edge')
    np.savez_compressed(dd/'lcc.npz',types=lg['types'],edges=lg['edges'],schema=lg['schema'],original_ids=lg['original_ids'])
    save(dd/'mapping.json',dict(source=str(gp),source_sha256=sha(gp),mapping=mapping,node_retention=len(lg['types'])/len(g['types']),edge_retention=lg['edges'].shape[1]/g['edges'].shape[1]if g['edges'].shape[1]else None,type_original=np.bincount(g['types'],minlength=4).tolist(),type_retained=np.bincount(lg['types'],minlength=4).tolist()))
    for version,gg in [('original',g),('lcc',lg)]:
     if job.get('evaluation_folder'):
      prior=pathlib.Path(job['evaluation_folder'])/str(ss)/(version+'.json');old=read(prior)
      if old['graph_binding']['graph_sha256']!=sha(gp):raise ValueError('old metric graph binding')
      st,comp,raw,dist,ae=[old[k]for k in ['statistics','comparison','raw','distance','absolute_error']]
     else:
      st=M.stats(gg,F,names,check);comp=M.compare(gg,st,ref,rs,F,names);raw,dist=E.scalar(st,rs,comp,names);ae={k:abs(v-reference[k])if v is not None and reference.get(k)is not None else None for k,v in raw.items()}
     if dist['main_incident4_W1']is not None and not math.isclose(dist['all10_compat_W1'],.4*dist['main_incident4_W1'],abs_tol=1e-10):raise ValueError('W1 convention')
     save(dd/(version+'.json'),dict(ident,version=version,statistics=st,comparison=comp,raw=raw,distance=dist,absolute_error=ae,reference=reference,graph_binding=b,reused_metrics=bool(job.get('evaluation_folder'))))
     graphs[(ds,seed,arm,ss,version)]=(gg,st)
     for kind,vals in [('raw',raw),('absolute_error',ae),('distance',dist)]:
      for metric,value in vals.items():fields[ds].add((kind,metric));rows.append(dict(ident,version=version,kind=kind,metric=metric,value=value,reason=None if value is not None else 'undefined/unavailable; see full JSON'))
    inv.update(status='complete',graph_sha256=sha(gp))
   except BaseException as exc:
    inv.update(status='missing'if isinstance(exc,FileNotFoundError)else'failed',reason=repr(exc));errors.append(dict(ident,error=repr(exc),traceback=traceback.format_exc()))
   inventory.append(inv);save(dest/'inventory.json',inventory);save(dest/'errors.json',errors)
 lookup={(r['dataset'],r['train_seed'],r['arm'],r['sample_seed'],r['version'],r['kind'],r['metric']):r['value']for r in rows}
 for inv in inventory:
  for version in ['original','lcc']:
   for kind,metric in fields[inv['dataset']]:
    key=(inv['dataset'],inv['train_seed'],inv['arm'],inv['sample_seed'],version,kind,metric)
    if key not in lookup:lookup[key]=None;rows.append(dict(**{k:inv[k]for k in ['dataset','train_seed','arm','sample_seed','updates','readout','task_id','selection']},version=version,kind=kind,metric=metric,value=None,reason=inv.get('reason','unavailable')))
 pre=read(OUT/'preregistration.json');S=pre['sample_seeds'];panels=[]
 for ds,old in [('IMDB',[2026110211,2026110212]),('PubMed',[2026110221,2026110222])]:
  new=pre['training_seeds'][ds]
  panels.extend([dict(name='prospective_four_samples',dataset=ds,seeds=new,samples=S),dict(name='historical_two_samples',dataset=ds,seeds=old,samples=S[:2]),dict(name='combined_common_two_samples',dataset=ds,seeds=old+new,samples=S[:2])])
 panels.append(dict(name='posthoc_known_bad_A_case',dataset='PubMed',seeds=[2026110321],samples=S))
 means=[];pairs=[];across=[]
 for panel in panels:
  ds=panel['dataset'];seeds=panel['seeds'];samples=panel['samples']
  for version in ['original','lcc']:
   for kind,metric in sorted(fields[ds]):
    sm={}
    for seed in seeds:
     for arm in ['O','A']:
      vals=[lookup.get((ds,seed,arm,ss,version,kind,metric))for ss in samples];mu=avg(vals);sm[(seed,arm)]=mu
      means.append(dict(panel=panel['name'],dataset=ds,train_seed=seed,arm=arm,version=version,kind=kind,metric=metric,value=mu,sample_std=sd(vals),valid_samples=sum(v is not None for v in vals),expected_samples=len(samples)))
     for ss in samples+['sample_mean']:
      a=sm[(seed,'A')]if ss=='sample_mean'else lookup.get((ds,seed,'A',ss,version,kind,metric));o=sm[(seed,'O')]if ss=='sample_mean'else lookup.get((ds,seed,'O',ss,version,kind,metric))
      pairs.append(dict(panel=panel['name'],dataset=ds,train_seed=seed,sample_seed=ss,version=version,kind=kind,metric=metric,A=a,O=o,difference=a-o if a is not None and o is not None else None))
    for arm in ['O','A']:
     vals=[sm[(seed,arm)]for seed in seeds];across.append(dict(panel=panel['name'],dataset=ds,arm=arm,version=version,kind=kind,metric=metric,training_seed_means=vals,mean=avg(vals),training_seed_std=sd(vals),expected_training_seeds=len(seeds),expected_samples_per_model=len(samples)))
    deltas=[sm[(seed,'A')]-sm[(seed,'O')]if sm[(seed,'A')]is not None and sm[(seed,'O')]is not None else None for seed in seeds]
    across.append(dict(panel=panel['name'],dataset=ds,arm='A-O',version=version,kind=kind,metric=metric,training_seed_means=deltas,mean=avg(deltas),training_seed_std=sd(deltas),expected_training_seeds=len(seeds),improved_seeds=sum(x<0 for x in deltas if x is not None),worsened_seeds=sum(x>0 for x in deltas if x is not None),tied_seeds=sum(x==0 for x in deltas if x is not None),direction_counts_apply_to='errors/distances only; raw direction requires reference'))
 diversity=[]
 for job in jobs:
  ds,seed,arm=job['dataset'],job['train_seed'],job['arm']
  for version in ['original','lcc']:
   for left,right in itertools.combinations(job['sample_seeds'],2):
    ks=[(ds,seed,arm,s,version)for s in [left,right]];rec=dict(dataset=ds,train_seed=seed,arm=arm,version=version,sample_seeds=[left,right],metrics=None)
    if all(k in graphs for k in ks):
     (g1,s1),(g2,s2)=[graphs[k]for k in ks];c=M.compare(g1,s1,g2,s2,F,refs[ds][2]);a=set(map(tuple,g1['original_ids'][g1['edges']].T));b=set(map(tuple,g2['original_ids'][g2['edges']].T));rec['metrics']=dict(labeled_edge_Jaccard=len(a&b)/len(a|b)if a|b else None,labeled_edge_difference=len(a^b),exact_labeled_duplicate=a==b and set(g1['original_ids'])==set(g2['original_ids']),degree_W1=c['total_degree_W1'],triangle_composition_TV=c['triangle_composition_TV'],MP2_composition_TV=c['MP_composition_TV'][1],MP3_composition_TV=c['MP_composition_TV'][2],limitation='Shared graphs within model; not independent training repeats; labeled overlap not isomorphism invariant')
    diversity.append(rec)
 for n,v in [('per_graph_all_metrics',rows),('per_training_seed',means),('paired_A_minus_O',pairs),('across_training_seeds',across)]:write(n,v)
 save(dest/'sample_diversity.json',diversity);save(dest/'panels.json',panels)
 save(dest/'receipt.json',dict(at=now(),expected_original_graphs=56,complete_original_graphs=sum(x['status']=='complete'for x in inventory),new_models=9,new_graphs=36,reused_graphs=20,all_metrics_exported=True,errors=errors,seconds=time.time()-start))
 print('Evaluation finished',len(inventory),'graph identities',len(errors),'errors')

if __name__=='__main__':main()
