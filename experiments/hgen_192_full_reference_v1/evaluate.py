"""Apply the existing frozen all-metric evaluator, including separate induced LCC."""
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import sys,time,csv,collections,shutil,traceback,itertools
import numpy as np
from common import *
METRICS=PROJECT/'research_branches/count_aux_length_run/experiments/typed_structure_count_aux_length_v1'
sys.path.insert(0,str(METRICS))
import summarize as S
from lcc_eval import metrics as M
NAMES={'IMDB':['Director','Actor','Movie','Genre'],'PubMed':['Gene','Disease','Chemical','Species']}
def dump(p,x):S.write_json(p,x)
def write_csv(p,rows):
 p=pathlib.Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 if not rows:p.write_text('');return
 keys=list(dict.fromkeys(k for row in rows for k in row))
 with p.open('w',newline='') as f:w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rows)
def mean(v):return sum(v)/len(v) if v and all(x is not None for x in v) else None
def leaves(x,p=''):
 if isinstance(x,dict):
  for k,v in x.items():yield from leaves(v,p+'.'+str(k) if p else str(k))
 elif isinstance(x,list):
  for i,v in enumerate(x):yield from leaves(v,p+'['+str(i)+']')
 else:yield p,x
def main():
 begin=time.time();records=[];inventory=[];graphs={};refs={};F=S.factory();errors=[];derived=OUT/'derived';derived.mkdir(exist_ok=True)
 def check():
  if time.time()>read(OUT/'ledger.json')['delivery_deadline_epoch']-120:raise TimeoutError('delivery reserve')
 for d in PROTOCOL['datasets']:
  z=np.load(str(OUT/'data'/d/'reference.npz'));g=M.make_graph(z['types'],z['edges'].T,z['schema']);st=M.stats(g,F,NAMES[d],check);refs[d]=(g,st);dump(derived/(d+'_reference.json'),S.evaluate_graph(g,g,st,F,NAMES[d],check))
  for s,sample in itertools.product(PROTOCOL['train_seeds'][d],PROTOCOL['sample_seeds']):
   folder=OUT/'run'/cell(d,s)/('sample_'+str(sample));path=folder/'binding.json';base=dict(dataset=d,train_seed=s,sample_seed=sample,arm='HGEN',checkpoint='D3750_G750',readout='native_final',model='author_HGEN',fusion='not_applicable')
   item=dict(**base,status='missing',reason='worker or assembly not completed');inventory.append(item)
   if not path.exists():continue
   b=read(path)
   if (b['dataset'],b['train_seed'],b['sample_seed'])!=(d,s,sample) or sha(folder/'graph.npz')!=b['graph_sha256']:raise ValueError('identity/hash binding')
   item.update(status='complete' if b['complete'] and b['simple'] else 'failed_or_incomplete',reason=b['reason'],self_loops=b['self_loops'],binding=b)
   if not b['simple']:continue
   z=np.load(str(folder/'graph.npz'));g=M.make_graph(z['types'],z['edges'],z['schema']);kept,mapping=M.lcc(g);target=derived/(cell(d,s)+'_'+str(sample));target.mkdir(exist_ok=True);mapping.update(source_sha256=sha(folder/'graph.npz'),source=str(folder/'graph.npz'))
   np.savez_compressed(str(target/'lcc.npz'),types=kept['types'],edges=kept['edges'],families=kept['families'],schema=kept['schema'],new_to_original=kept['original_ids'],old_to_new=np.array(mapping['old_to_new']));mapping['derived_sha256']=sha(target/'lcc.npz');save(target/'mapping.json',mapping)
   item.update(node_retention=len(kept['types'])/192,edge_retention=kept['edges'].shape[1]/g['edges'].shape[1] if g['edges'].shape[1] else None,type_counts_original=np.bincount(g['types'],minlength=4).tolist(),type_counts_lcc=np.bincount(kept['types'],minlength=4).tolist())
   for version,gg in [('original',g),('lcc',kept)]:
    check();val=S.evaluate_graph(gg,*refs[d],F,NAMES[d],check)
    # The frozen evaluator's unavailable-native-powerlaw label is historical;
    # actual package availability here differs. No frozen scalar is replaced.
    val['statistics']['HGEN_native_powerlaw_reason']='not executed by frozen common evaluator; original HGEN native extra is reported separately'
    val.update(**base,version=version,status=item['status'],graph_source_sha256=b['graph_sha256'],reference_source_sha256=sha(OUT/'data'/d/'reference.npz'),evaluation_source_sha256='c0f5d92db2f891ebde70befb4e0e29adc50bcb632bdeff5866173d79aa56ed16')
    records.append(val);dump(target/(version+'.json'),val);graphs[d,s,sample,version]=(gg,val['statistics'])
 means=[];diversity=[];scalars=[];details=[]
 for r in records:
  ident={k:r[k] for k in ('dataset','train_seed','sample_seed','arm','checkpoint','version','status')}
  for group in ('raw','absolute_error','distribution_error'):
   for k,v in r[group].items():scalars.append(dict(**ident,group=group,metric=k,value=v,reference=r['reference'].get(k) if group!='distribution_error' else 0,reason=r['undefined'].get(group,{}).get(k) if v is None else None))
  for group in ('statistics','comparison'):
   for k,v in leaves(r[group]):details.append(dict(**ident,group=group,field=k,value=v,reason='undefined_or_unavailable' if v is None else None))
 for d in PROTOCOL['datasets']:
  for s,v in itertools.product(PROTOCOL['train_seeds'][d],('original','lcc')):
   rr=[r for r in records if (r['dataset'],r['train_seed'],r['version'])==(d,s,v)];good=len(rr)==2 and all(x['status']=='complete' for x in rr)
   for group in ('raw','absolute_error','distribution_error'):
    keys=set(refs[d][1]) if False else set()
    for r in rr:keys.update(r[group])
    # No intersection: preserve all fields even if one generated graph is absent.
    if not keys:keys.update(read(derived/(d+'_reference.json'))[group])
    for k in sorted(keys):
     value=mean([r[group].get(k) for r in rr]) if good else None
     means.append(dict(dataset=d,train_seed=s,method='HGEN',checkpoint='D3750_G750',version=v,group=group,metric=k,value=value,reason=None if value is not None else 'both_complete_samples_required_or_undefined'))
   ab=[graphs.get((d,s,z,v)) for z in PROTOCOL['sample_seeds']]
   diversity.append(dict(dataset=d,train_seed=s,version=v,metrics=S.diversity(ab[0][0],ab[1][0],ab[0][1],ab[1][1],F) if all(x is not None for x in ab) else None,reason=None if all(x is not None for x in ab) else 'both_graphs_required'))
 result=dict(records=records,inventory=inventory,cell_means=means,diversity=diversity,errors=errors,status='complete' if len(records)==32 and all(x['status']=='complete' for x in inventory) else 'incomplete',seconds=time.time()-begin)
 dump(OUT/'evaluation.json',result);write_csv(OUT/'tables/per_graph_all_scalar_metrics.csv',scalars);write_csv(OUT/'tables/per_graph_all_detail_fields.csv',details);write_csv(OUT/'tables/training_seed_means.csv',means);dump(OUT/'tables/sample_diversity.json',diversity);dump(OUT/'tables/inventory.json',inventory)
 # Diffusion controls are existing results only, independently trained methods;
 # equal integer seeds would not make GAN and diffusion RNG trajectories paired.
 prior=read(PROJECT/'results/count_aux_results_closeout_20261030_final/evaluation.json');cross=[]
 for row in means:cross.append(row)
 for row in prior['cell_means']:
  if row['checkpoint'] in (3000,6000):cross.append(dict(row,method='E-earlyCN_'+row['arm']))
 write_csv(OUT/'tables/cross_method_training_seed_means.csv',cross)
 agg=[]
 for d,v in itertools.product(PROTOCOL['datasets'],('original','lcc')):
  keys={(x['group'],x['metric']) for x in means if (x['dataset'],x['version'])==(d,v)}
  for group,k in sorted(keys):
   vals=[x['value'] for x in means if (x['dataset'],x['version'],x['group'],x['metric'])==(d,v,group,k)];valid=all(x is not None for x in vals) and len(vals)==4
   agg.append(dict(dataset=d,version=v,group=group,metric=k,training_seed_count=4,value=mean(vals) if valid else None,minimum=min(vals) if valid else None,maximum=max(vals) if valid else None,reason=None if valid else 'all_four_seeds_required_or_undefined'))
 write_csv(OUT/'tables/HGEN_four_seed_summary.csv',agg)
 # Copy authoritative definitions, including all gaps, without redefining metrics.
 refsrc=PROJECT/'results/count_aux_results_closeout_20261030_final'
 shutil.copy2(str(refsrc/'METRIC_DIRECTORY.md'),str(OUT/'METRIC_DIRECTORY.md'));shutil.copy2(str(refsrc/'tables/metric_catalog.csv'),str(OUT/'tables/metric_catalog.csv'))
 print(result['status'],len(records),'all metric versions',time.time()-begin)
if __name__=='__main__':
 try:main()
 except BaseException:save(OUT/'evaluation_failure.json',dict(traceback=traceback.format_exc(),epoch=time.time()));raise
