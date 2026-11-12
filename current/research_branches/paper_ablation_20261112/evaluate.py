"""CPU whole-panel evaluation. Never writes into any reused asset directory."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import sys,json,time,csv,itertools,signal,traceback
from pathlib import Path
from scoring import *
sys.path.insert(0,str(ROOT/'research_branches/new_datasets_20261112'))
import metrics as M
import networkx as nx
OLD=ROOT/'research_branches/seed_coverage_repair_release/experiments/seed_coverage_repair_v1/lcc_eval'
F=M.frozen(OLD/'vendor/frozen_metrics.py',OLD/'vendor/aligned_evaluate.py',OLD/'vendor/hgen_utils.py')
def read(p):return json.loads(Path(p).read_text())
def table(path,rows):
 base.write(path.with_suffix('.json'),rows)
 if rows:
  with path.with_suffix('.csv').open('w') as f:
   w_=csv.DictWriter(f,list(dict.fromkeys(k for r in rows for k in r)));w_.writeheader();w_.writerows(rows)
def flat(x,p=''):
 if isinstance(x,dict):return {k:v for a,b in x.items() for k,v in flat(b,p+'.'+str(a) if p else str(a)).items()}
 if isinstance(x,list):return {k:v for i,b in enumerate(x) for k,v in flat(b,p+'['+str(i)+']').items()}
 return {p:x}
def typed_iso(a,b):
 if len(a['types'])!=len(b['types']) or a['edges'].shape[1]!=b['edges'].shape[1]:return 'false: inventory'
 def graph(g):
  x=nx.Graph();x.add_nodes_from((i,{'type':int(t)}) for i,t in enumerate(g['types']))
  x.add_edges_from((int(u),int(v),{'family':int(f)}) for (u,v),f in zip(g['edges'].T,g['families']));return x
 x,y=graph(a),graph(b)
 if sorted((x.nodes[i]['type'],d) for i,d in x.degree())!=sorted((y.nodes[i]['type'],d) for i,d in y.degree()):return 'false: typed degree necessary condition'
 old=signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError()))
 try:
  signal.setitimer(signal.ITIMER_REAL,2)
  return bool(nx.is_isomorphic(x,y,node_match=nx.algorithms.isomorphism.categorical_node_match('type',-1),edge_match=nx.algorithms.isomorphism.categorical_edge_match('family',-1)))
 except TimeoutError:return 'unknown: 2 second timeout'
 finally:signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old)
def main():
 out=Path(sys.argv[1]);p=read(out/'protocol.json');start=time.time();(out/'evaluation').mkdir(exist_ok=True)
 rows=[];allrows=[];missing=[];pairs=[];contracts=[];cache={};refs={};scorecache={}
 def check():
  if time.time()>p['delivery_epoch']-30:raise TimeoutError('delivery reserve')
 inventory=list(p['reused_graphs'])
 for j in p['jobs']:
  for ss in p['sample_seeds']:
   inventory.append(dict(dataset=j['dataset'],train_seed=j['train_seed'],arm=j['architecture']+'_'+j['reward'],sample_seed=ss,
       parent=j['parent'],path=str(out/j['id']/(j['dataset']+'_T_eager')/f'final_sample_{ss}/graph.npz')))
 for item in inventory:
  check();ds=item['dataset'];parent=p['parents'][item['parent']];names=parent['type_names'];gp=Path(item['path']);ident={k:item[k] for k in ('dataset','train_seed','arm','sample_seed')}
  if not gp.exists() or not (gp.parent/'summary.json').exists():missing.append(dict(**ident,reason='no complete chain'));continue
  try:
   if ds not in refs:
    z=np.load(parent['data']);re= z['edges'];re=re.T if re.shape[0]!=2 else re
    ref=M.make_graph(z['types'],re,z['schema']);rs=M.stats(ref,F,names,check);case=case_from(parent['data']);ids,fam,q=w.common.plan_for(case,2026111402,1).block(0);labels=np.isin(ids,case['clean_ids']).astype(np.uint8)
    refs[ds]=(ref,rs,case,ids);scorecache[ds]=(TypedScore(ds,case,q,fam,labels),GenericScore(ds,case,q,fam,labels))
   ref,rs,case,ids=refs[ds];typed,generic=scorecache[ds]
   assert base.sha(gp)==read(gp.parent/'summary.json')['graph_sha256']
   zz=np.load(gp);assert np.array_equal(zz['types'],ref['types']) and np.array_equal(zz['schema'],ref['schema'])
   g=M.make_graph(zz['types'],zz['edges'],zz['schema']);assert np.all(case['active'][g['families']])
   st=M.stats(g,F,names,check);cmp=M.compare(g,st,ref,rs,F,names);lc,mapping=M.lcc(g);ls=M.stats(lc,F,names,check);lcmp=M.compare(lc,ls,ref,rs,F,names)
   ff,ll=case['space'].encode(g['edges']);b=np.zeros(case['space'].total,np.uint8);b[case['space'].offsets[ff]+ll]=1
   tc,gc=typed(b[ids]),generic(b[ids]);iso=typed_iso(g,ref)
   full=dict(raw=dict(statistics=st,comparison=cmp),LCC=dict(statistics=ls,comparison=lcmp,mapping=mapping),reference=rs,
       typed_score=tc,generic_score=gc,typed_reference_isomorphic=iso,graph_path=str(gp),graph_sha256=base.sha(gp))
   tag='_'.join(str(ident[k]) for k in ident);base.write(out/'evaluation'/(tag+'.json'),full)
   cache[(ds,item['train_seed'],item['arm'],item['sample_seed'])]=(g,st,gp)
   for view,stat,comparison in [('raw',st,cmp),('LCC',ls,lcmp)]:
    rows.append(dict(**ident,view=view,nodes=stat['nodes'],edges=stat['edges'],reference_edges=rs['edges'],edge_abs_error=abs(stat['edges']-rs['edges']),degree_W1=comparison['main_incident_W1'],joint_TV=comparison['main_joint_degree_TV'],triangles=stat['triangles'],reference_triangles=rs['triangles'],triangle_abs_error=abs(stat['triangles']-rs['triangles']),triangle_composition_TV=comparison['triangle_composition_TV'],four_cycles=stat['four_cycles'],four_cycle_abs_error=abs(stat['four_cycles']-rs['four_cycles']),MP2=stat['mp'][1]['total'],MP2_abs_error=abs(stat['mp'][1]['total']-rs['mp'][1]['total']),MP2_TV=comparison['MP_composition_TV'][1],MP3=stat['mp'][2]['total'],MP3_abs_error=abs(stat['mp'][2]['total']-rs['mp'][2]['total']),MP3_TV=comparison['MP_composition_TV'][2],components=stat['components'],LCC=stat['lcc'],isolates=stat['isolates'],maximum_degree=stat['maximum_degree'],reference_overlap=comparison['EO_jaccard'],typed_joint_score=tc[0] if view=='raw' else None,typed_count_score=tc[1] if view=='raw' else None,generic_joint_score=gc[0] if view=='raw' else None,generic_count_score=gc[1] if view=='raw' else None))
    allrows.extend(dict(**ident,view=view,metric=k,value=v) for k,v in flat(full[view]).items())
  except Exception:missing.append(dict(**ident,reason='evaluation failure',error=traceback.format_exc()))
  table(out/'per_graph',rows);base.write(out/'missing.json',missing)
 for model in sorted(set(k[:3] for k in cache)):
  gs=[(k[-1],v) for k,v in cache.items() if k[:3]==model]
  for (sa,(a,ast,_)),(sb,(b,bst,_)) in itertools.combinations(gs,2):
   check();ea=set(map(tuple,a['edges'].T));eb=set(map(tuple,b['edges'].T));common=len(ea&eb);sim=M.compare(a,ast,b,bst,F,p['parents'][next(x['parent'] for x in inventory if (x['dataset'],x['train_seed'],x['arm'])==model)]['type_names'])
   pairs.append(dict(dataset=model[0],train_seed=model[1],arm=model[2],sample_seed1=sa,sample_seed2=sb,
      common=common,jaccard=common/len(ea|eb) if ea|eb else None,common_over_first=common/len(ea) if ea else None,
      common_over_second=common/len(eb) if eb else None,identical_edges=ea==eb,typed_isomorphic=typed_iso(a,b),
      pair_degree_W1=sim['main_incident_W1'],pair_joint_TV=sim['main_joint_degree_TV'],pair_MP3_TV=sim['MP_composition_TV'][2]))
 for j in p['jobs']:
  for ss in p['sample_seeds']:
   a=cache.get((j['dataset'],j['train_seed'],j['architecture'],ss));b=cache.get((j['dataset'],j['train_seed'],j['architecture']+'_'+j['reward'],ss))
   if not a or not b:continue
   ia,ib=[read(v[2].parent/'identity.json') for v in (a,b)];ra,rb=[[json.loads(l) for l in (v[2].parent/'steps.jsonl').read_text().splitlines()] for v in (a,b)]
   def norm(x):return x['time'],x.get('uniform_hash',x.get('uniform_sha256')),x.get('query_hash',x.get('query_plan_sha256'))
   contracts.append(dict(job=j['id'],sample_seed=ss,initial_equal=ia.get('initial_hash',ia.get('initial_state_sha256'))==ib['initial_hash'],
       all_1000_random_queries_equal=len(ra)==len(rb)==1000 and all(norm(x)==norm(y) for x,y in zip(ra,rb))))
 summaries=[];keys=['dataset','train_seed','arm','view'];meta=keys+['sample_seed']
 for key in sorted(set(tuple(r[k] for k in keys) for r in rows)):
  rr=[r for r in rows if tuple(r[k] for k in keys)==key];v=dict(zip(keys,key));v['samples']=len(rr)
  for k in rr[0]:
   if k in meta:continue
   vals=[r[k] for r in rr if isinstance(r[k],(int,float))];v[k+'_mean']=float(np.mean(vals)) if vals else None;v[k+'_sample_sd']=float(np.std(vals,ddof=1)) if len(vals)>1 else None
  summaries.append(v)
 cross=[]
 for key in sorted(set((r['dataset'],r['arm'],r['view']) for r in summaries)):
  ss=[r for r in summaries if (r['dataset'],r['arm'],r['view'])==key and r['samples']==4];v=dict(zip(['dataset','arm','view'],key));v['training_models']=len(ss)
  if not ss:continue
  for k in ss[0]:
   if not k.endswith('_mean'):continue
   vs=[r[k] for r in ss if r[k] is not None];v[k]=float(np.mean(vs)) if vs else None;v[k+'_between_seed_sd']=float(np.std(vs,ddof=1)) if len(vs)>1 else None
  cross.append(v)
 differences=[]
 for ds,seed,view in sorted(set((r['dataset'],r['train_seed'],r['view']) for r in summaries)):
  by={r['arm']:r for r in summaries if (r['dataset'],r['train_seed'],r['view'])==(ds,seed,view) and r['samples']==4}
  for after,before in [('A','O'),('O_typed','O'),('A_typed','A'),('A_typed','O_typed'),('A_typed','A_generic'),('A_generic','A')]:
   if after in by and before in by:
    differences.append(dict(dataset=ds,train_seed=seed,view=view,contrast=after+' minus '+before,**{k:by[after][k]-by[before][k] for k in by[after] if k.endswith('_mean') and by[after][k] is not None and by[before][k] is not None}))
  if all(x in by for x in ['O','A','O_typed','A_typed']):
   differences.append(dict(dataset=ds,train_seed=seed,view=view,contrast='interaction=(A_typed-A)-(O_typed-O)',**{k:(by['A_typed'][k]-by['A'][k])-(by['O_typed'][k]-by['O'][k]) for k in by['A'] if k.endswith('_mean') and all(by[x][k] is not None for x in ['O','A','O_typed','A_typed'])}))
 for name,data in [('per_graph',rows),('all_metrics',allrows),('model_summary',summaries),('across_training_seeds',cross),('paired_differences',differences),('sample_diversity_pairs',pairs),('sampling_contract',contracts)]:table(out/name,data)
 base.write(out/'reference_metrics.json',{d:x[1] for d,x in refs.items()})
 complete=len([r for r in rows if r['view']=='raw'])==80 and not missing and len(contracts)==48 and all(r['initial_equal'] and r['all_1000_random_queries_equal'] for r in contracts)
 base.write(out/'evaluation_receipt.json',dict(status='complete' if complete else 'partial',graphs=len(rows)//2,missing=missing,seconds=time.time()-start))
 lines=['# 匹配CN与异质终态评分补证','每数据集两个既有开发训练seed、每模型四图；不宣称独立确认或全面优势。','generic是类型盲评分消融，不是官方GDPO复现。额外反馈计算、评分尺度/分组变化、原EMA/新raw差异均须披露。',
 '|数据|训练seed|方案|样本|度W1|联合TV|三角绝对误差|MP3绝对误差|LCC|','|---|---|---|---|---|---|---|---|---|']
 for r in summaries:
  if r['view']=='raw':lines.append('|'+ '|'.join(str(r.get(k)) for k in ['dataset','train_seed','arm','samples','degree_W1_mean','joint_TV_mean','triangle_abs_error_mean','MP3_abs_error_mean','LCC_mean'])+'|')
 lines+=['质量与多样性分别报告，不因低重合认定高质量，不筛去坏图；同构超时为未知，不排除近似复制。','完整关系/类型/hub/低度、原图/LCC、均值与采样标准差、逐seed差及交互见CSV。部分结果不得冒充完整消融。',f'缺项：{len(missing)}；完成评价图：{len(rows)//2}/80。']
 (out/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
