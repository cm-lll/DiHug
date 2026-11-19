"""CPU complete metrics, per-model sampling and paired-seed aggregation."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import sys,time,json,csv,itertools,traceback
from pathlib import Path
import numpy as np
import metrics as M
from support import Score,case_for,s,base,ROOT
OLD=ROOT/'research_branches/seed_coverage_repair_release/experiments/seed_coverage_repair_v1/lcc_eval'
F=M.frozen(OLD/'vendor/frozen_metrics.py',OLD/'vendor/aligned_evaluate.py',OLD/'vendor/hgen_utils.py')
def read(p):return json.loads(Path(p).read_text())
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def table(p,rows):
 save(p.with_suffix('.json'),rows)
 if rows:
  with p.with_suffix('.csv').open('w') as f:w=csv.DictWriter(f,list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
def flat(x,p=''):
 if isinstance(x,dict):return {k:v for a,b in x.items() for k,v in flat(b,p+'.'+str(a) if p else str(a)).items()}
 if isinstance(x,list):return {k:v for i,b in enumerate(x) for k,v in flat(b,p+'['+str(i)+']').items()}
 return {p:x}
def main():
 out=Path(sys.argv[1]);p=read(out/'protocol.json');start=time.time();rows=[];allrows=[];missing=[];pairs=[];rngpairs=[];references={}
 def check():
  if time.time()>p['delivery_epoch']-40:raise TimeoutError('CPU delivery reserve')
 for job in p['jobs']:
  ds=job['dataset'];names=job['type_names'];z=np.load(job['data']);ref=M.make_graph(z['types'],z['edges'].T,z['schema']);rs=read(out/'cpu_check'/f'{ds}_reference.json');references[ds]=rs;case=case_for(ds);ids,fam,q=s.w.common.plan_for(case,2026111402,1).block(0);labels=np.isin(ids,case['clean_ids']).astype(np.uint8);score=Score(ds,case,q,fam,labels)
  for arm in ('baseline','T_eager'):
   graphs=[]
   for ss in p['sample_seeds']:
    check();folder=out/job['id']/(ds+'_'+arm)/f'final_sample_{ss}';gp=folder/'graph.npz';ident=dict(dataset=ds,train_seed=job['train_seed'],arm=arm,sample_seed=ss)
    if not gp.exists() or not (folder/'summary.json').exists():missing.append(dict(**ident,reason='no completed whole chain'));continue
    try:
     zz=np.load(gp);assert np.array_equal(zz['types'],z['types']) and np.array_equal(zz['schema'],z['schema']);g=M.make_graph(zz['types'],zz['edges'],zz['schema']);assert np.all(case['active'][g['families']]);st=M.stats(g,F,names,check);cmp=M.compare(g,st,ref,rs,F,names);lc,mapping=M.lcc(g);ls=M.stats(lc,F,names,check);lcmp=M.compare(lc,ls,ref,rs,F,names)
     ff,ll=case['space'].encode(g['edges']);bits=np.zeros(case['space'].total,dtype=np.uint8);bits[case['space'].offsets[ff]+ll]=1;cost=score(bits[ids])
     full=dict(raw=dict(statistics=st,comparison=cmp),LCC=dict(statistics=ls,comparison=lcmp,mapping=mapping),reference=rs,terminal_score=dict(joint_heat_degree=cost[0],count=cost[1],by_type=cost[2]),score_scaling=score.s,LCC_note='Postprocessed only; comparison still against full reference; cannot call connectivity a model gain',triangle_note='Zero triangles are forced by DBLP schema; composition of two zero measures stored as convention 0, not a quality success' if ds=='DBLP_HGEN' else None)
     save(folder/'full_metrics.json',full);graphs.append((ss,g))
     for view,stat,comparison in [('raw',st,cmp),('LCC',ls,lcmp)]:
      rr=dict(**ident,view=view,nodes=stat['nodes'],edges=stat['edges'],reference_edges=rs['edges'],edge_abs_error=abs(stat['edges']-rs['edges']),degree_W1=comparison['main_incident_W1'],joint_TV=comparison['main_joint_degree_TV'],triangles=stat['triangles'],reference_triangles=rs['triangles'],triangle_abs_error=abs(stat['triangles']-rs['triangles']),triangle_composition_TV=comparison['triangle_composition_TV'],four_cycles=stat['four_cycles'],four_cycle_abs_error=abs(stat['four_cycles']-rs['four_cycles']),four_cycle_composition_TV=comparison['four_cycle_composition_TV'],MP2=stat['mp'][1]['total'],MP2_abs_error=abs(stat['mp'][1]['total']-rs['mp'][1]['total']),MP2_TV=comparison['MP_composition_TV'][1],MP3=stat['mp'][2]['total'],MP3_abs_error=abs(stat['mp'][2]['total']-rs['mp'][2]['total']),MP3_TV=comparison['MP_composition_TV'][2],components=stat['components'],LCC=stat['lcc'],isolates=stat['isolates'],maximum_degree=stat['maximum_degree'],mean_clustering=stat['average_clustering'],reference_overlap=comparison['EO_jaccard'],joint_score=cost[0] if view=='raw' else None,count_score=cost[1] if view=='raw' else None)
      rows.append(rr);allrows.extend(dict(**ident,view=view,metric=k,value=v) for k,v in flat(full[view]).items())
    except BaseException:missing.append(dict(**ident,reason='evaluation failed',error=traceback.format_exc()))
    table(out/'per_graph',rows);save(out/'missing.json',missing)
   for (sa,a),(sb,b) in itertools.combinations(graphs,2):
    aa=set(map(tuple,a['edges'].T));bb=set(map(tuple,b['edges'].T));pairs.append(dict(dataset=ds,train_seed=job['train_seed'],arm=arm,sample_seed1=sa,sample_seed2=sb,common=len(aa&bb),union=len(aa|bb),jaccard=len(aa&bb)/len(aa|bb) if aa|bb else None,identical=aa==bb))
  for ss in p['sample_seeds']:
   folders=[out/job['id']/(ds+'_'+a)/f'final_sample_{ss}' for a in ('baseline','T_eager')]
   if not all((x/'summary.json').exists() for x in folders):continue
   a,b=[read(x/'identity.json') for x in folders];records=[[json.loads(l) for l in (x/'steps.jsonl').read_text().splitlines()] for x in folders]
   match=len(records[0])==len(records[1])==1000 and all(x['uniform_hash']==y['uniform_hash'] and x['query_hash']==y['query_hash'] and x['time']==y['time'] for x,y in zip(*records))
   rngpairs.append(dict(dataset=ds,train_seed=job['train_seed'],sample_seed=ss,initial_state_equal=a['initial_hash']==b['initial_hash'],all_1000_uniform_and_query_hashes_equal=match))
 summaries=[];meta=['dataset','train_seed','arm','sample_seed','view']
 for key in sorted(set((r['dataset'],r['train_seed'],r['arm'],r['view']) for r in rows)):
  ss=[r for r in rows if tuple(r[k] for k in ['dataset','train_seed','arm','view'])==key];row=dict(zip(['dataset','train_seed','arm','view'],key));row['samples']=len(ss)
  for k in ss[0]:
   if k in meta:continue
   vals=[r[k] for r in ss if isinstance(r[k],(int,float))];row[k+'_mean']=float(np.mean(vals)) if vals else None;row[k+'_sample_sd']=float(np.std(vals,ddof=1)) if len(vals)>1 else None
  summaries.append(row)
 differences=[]
 for j in p['jobs']:
  for view in ('raw','LCC'):
   a=next((x for x in summaries if x['dataset']==j['dataset'] and x['train_seed']==j['train_seed'] and x['arm']=='baseline' and x['view']==view),None);b=next((x for x in summaries if x['dataset']==j['dataset'] and x['train_seed']==j['train_seed'] and x['arm']=='T_eager' and x['view']==view),None)
   if a and b and a['samples']==b['samples']==4:differences.append(dict(dataset=j['dataset'],train_seed=j['train_seed'],view=view,samples_per_model=4,**{k:b[k]-a[k] if b[k] is not None and a[k] is not None else None for k in a if k.endswith('_mean')}))
 cross=[]
 for key in sorted(set((r['dataset'],r['arm'],r['view']) for r in summaries)):
  ss=[r for r in summaries if (r['dataset'],r['arm'],r['view'])==key and r['samples']==4];r=dict(zip(['dataset','arm','view'],key));r['training_models']=len(ss)
  if not ss:continue
  for k in ss[0]:
   if not k.endswith('_mean'):continue
   v=[x[k] for x in ss if x[k] is not None];r[k]=float(np.mean(v)) if v else None;r[k+'_between_training_seed_sd']=float(np.std(v,ddof=1)) if len(v)>1 else None
  cross.append(r)
 for name,values in [('per_graph',rows),('all_metrics',allrows),('model_summary',summaries),('paired_seed_difference',differences),('across_training_seeds',cross),('sample_overlap_pairs',pairs),('sampling_contract',rngpairs)]:table(out/name,values)
 save(out/'reference_metrics.json',references);save(out/'evaluation_receipt.json',dict(status='complete' if len([x for x in rows if x['view']=='raw'])==32 and not missing and len(rngpairs)==16 and all(x['initial_state_equal'] and x['all_1000_uniform_and_query_hashes_equal'] for x in rngpairs) else 'partial',graphs=len([x for x in rows if x['view']=='raw']),missing=missing,seconds=time.time()-start,GPU=0))
 lines=['# 新数据集完整资产与192节点试点','完整源图和可追溯诱导子图已准备；未进行完整大图训练。','下面为每模型先汇总四次采样后的原图结果；两训练seed仍只是筛查。','|数据|seed|方案|图数|度W1|联合TV|边数|三角数|MP3|LCC|','|---|---|---|---|---|---|---|---|---|---|']
 for r in summaries:
  if r['view']=='raw':lines.append('|'+ '|'.join(str(r.get(k)) for k in ['dataset','train_seed','arm','samples','degree_W1_mean','joint_TV_mean','edges_mean','triangles_mean','MP3_mean','LCC_mean'])+'|')
 lines+=['DBLP三角结构为schema强制零，不能据此宣布闭合收益。ACM完整原始版本保留7类型，部分小图关系支持很少。','完整明细、采样标准差、逐seed配对差、原图/LCC及缺项见本目录CSV/JSON。正式主线未替换，未根据结果调参。']
 (out/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
