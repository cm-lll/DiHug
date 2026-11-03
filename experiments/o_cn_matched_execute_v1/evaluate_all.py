"""Frozen full evaluator, original/LCC, paired seeds and missing-aware nesting."""
import argparse,csv,itertools,math,statistics,time,traceback
from common_run import ROOT,SCI,OUT,read,save,sha,now
import numpy as np
from lcc_eval import metrics as M
import evaluation_fields as E
def csvwrite(p,rows):
 fields=list(dict.fromkeys(k for r in rows for k in r))
 with p.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def avg(x):return statistics.mean(x) if x and all(v is not None for v in x) else None
def sd(x):return statistics.stdev(x) if len(x)>1 and all(v is not None for v in x) else None
def main():
 started=time.time();dest=OUT/'evaluation';dest.mkdir(exist_ok=True);ledger=read(OUT/'ledger.json');approval=read(OUT/'approval.json')
 for p,h in approval['source_files'].items():
  if sha(p)!=h:raise ValueError('bound source modified before evaluation '+p)
 def check():
  if time.time()>=ledger['deadline_epoch']-90:raise TimeoutError('final delivery reserve')
 F=M.frozen(SCI/'lcc_eval/vendor/frozen_metrics.py',SCI/'lcc_eval/vendor/aligned_evaluate.py',SCI/'lcc_eval/vendor/hgen_utils.py')
 rows=[];inventory=[];graphs={};refs={};fields={};full=[];errors=[];contracts=read(OUT/'prepared/sampling_contracts.json')
 for ds in ['IMDB','PubMed']:
  z=np.load(SCI/'package/assets'/f'{ds}.npz');ref=M.make_graph(z['types'],z['edges'].T,z['schema']);names=read(SCI/'package/config.json')['datasets'][ds]['type_names'];rs=M.stats(ref,F,names,check);rc=M.compare(ref,rs,ref,rs,F,names);raw,dist=E.scalar(rs,rs,rc,names)
  refs[ds]=(ref,rs,names,raw);fields[ds]={(k,m) for k,d in [('raw',raw),('absolute_error',raw),('distance',dist)] for m in d}
  save(dest/f'reference_{ds}.json',dict(statistics=rs,comparison=rc,raw=raw,distance=dist))
 for job in read(OUT/'tasks.json')+read(OUT/'reused_A_jobs.json'):
  ds=job['dataset'];seed=job['train_seed'];arm=job['arm'];ref,rs,names,reference=refs[ds]
  for ss in [2026102821,2026102822]:
   identity=dict(dataset=ds,train_seed=seed,arm=arm,sample_seed=ss,updates=3000,readout='EMA0.99',task_id=job['task_id']);folder=__import__('pathlib').Path(job['folder'])/'ema_0.99'/f'sample_{ss}';inv=dict(identity,graph_file=str(folder/'graph.npz'))
   try:
    check();b=read(folder/'binding.json');gp=folder/'graph.npz'
    if sha(gp)!=b['graph_sha256'] or any(b[k]!=v for k,v in [('dataset',ds),('train_seed',seed),('sample_seed',ss),('updates',3000),('configuration',arm+'_late')]):raise ValueError('graph identity/hash')
    w=__import__('pathlib').Path(job['folder'])/'readout_ema_0.99.pt'
    if sha(w)!=b['model_sha256']:raise ValueError('weight hash')
    step=[__import__('json').loads(x) for x in (folder/'steps.jsonl').read_text().splitlines()];expected=contracts[f'{ds}_{ss}']['steps']
    if len(step)!=1000 or any(r['time']!=1000-i or not r['exact_coverage'] or r['uniform_sha256']!=expected[i]['uniform_sha256'] or r['query_plan_sha256']!=expected[i]['query_plan_sha256'] for i,r in enumerate(step)):raise ValueError('complete paired T1000 step record')
    z=np.load(gp);g=M.make_graph(z['types'],z['edges'],z['schema']);lg,mapping=M.lcc(g);dd=dest/job['task_id']/str(ss);dd.mkdir(parents=True,exist_ok=True)
    if not set(map(tuple,lg['original_ids'][lg['edges']].T))<=set(map(tuple,g['edges'].T)):raise ValueError('LCC introduced edge')
    np.savez_compressed(dd/'lcc.npz',types=lg['types'],edges=lg['edges'],schema=lg['schema'],original_ids=lg['original_ids'])
    save(dd/'mapping.json',dict(source=str(gp),source_sha256=sha(gp),mapping=mapping,node_retention=len(lg['types'])/len(g['types']),edge_retention=lg['edges'].shape[1]/g['edges'].shape[1] if g['edges'].shape[1] else None,
         type_original=np.bincount(g['types'],minlength=4).tolist(),type_retained=np.bincount(lg['types'],minlength=4).tolist()))
    for version,gg in [('original',g),('lcc',lg)]:
     check()
     if arm=='A':
      old=read(__import__('pathlib').Path(job['evaluation_folder'])/str(ss)/(version+'.json'));s=old['statistics'];comp=old['comparison'];raw=old['raw'];dist=old['distance'];ae=old['absolute_error']
      if old['graph_binding']['graph_sha256']!=sha(gp):raise ValueError('reused A metric/graph identity')
     else:
      s=M.stats(gg,F,names,check);comp=M.compare(gg,s,ref,rs,F,names);raw,dist=E.scalar(s,rs,comp,names);ae={k:abs(v-reference[k]) if v is not None and reference.get(k) is not None else None for k,v in raw.items()}
     if dist['main_incident4_W1'] is not None and not math.isclose(dist['all10_compat_W1'],.4*dist['main_incident4_W1'],abs_tol=1e-10,rel_tol=1e-10):raise ValueError('W1 convention')
     save(dd/f'{version}.json',dict(identity,version=version,statistics=s,comparison=comp,raw=raw,distance=dist,absolute_error=ae,reference=reference,graph_binding=b))
     graphs[(ds,seed,arm,ss,version)]=(gg,s)
     for kind,vals in [('raw',raw),('absolute_error',ae),('distance',dist)]:
      for metric,v in vals.items():fields[ds].add((kind,metric));rows.append(dict(identity,version=version,kind=kind,metric=metric,value=v,reason=None if v is not None else 'undefined/unavailable; full JSON retains source reason'))
    inv.update(status='complete',graph_sha256=sha(gp))
   except BaseException as e:
    inv.update(status='missing' if isinstance(e,FileNotFoundError) else 'failed',reason=repr(e));errors.append(dict(identity,error=repr(e),traceback=traceback.format_exc()))
   inventory.append(inv);save(dest/'inventory.json',inventory);save(dest/'errors.json',errors)
 lookup={(r['dataset'],r['train_seed'],r['arm'],r['sample_seed'],r['version'],r['kind'],r['metric']):r['value'] for r in rows}
 for inv in inventory:
  for ver in ['original','lcc']:
   for kind,metric in fields[inv['dataset']]:
    k=(inv['dataset'],inv['train_seed'],inv['arm'],inv['sample_seed'],ver,kind,metric)
    if k not in lookup:
     rows.append(dict(dataset=k[0],train_seed=k[1],arm=k[2],sample_seed=k[3],version=ver,kind=kind,metric=metric,value=None,reason=inv.get('reason','metric unavailable')));lookup[k]=None
 means=[];pairs=[];across=[];diversity=[]
 for ds in ['IMDB','PubMed']:
  seeds=sorted({j['train_seed'] for j in read(OUT/'tasks.json') if j['dataset']==ds})
  for ver in ['original','lcc']:
   for kind,metric in sorted(fields[ds]):
    seedmeans={}
    for seed in seeds:
     mu={}
     for arm in ['O','A']:
      x=[lookup.get((ds,seed,arm,ss,ver,kind,metric)) for ss in [2026102821,2026102822]];mu[arm]=avg(x);seedmeans[(seed,arm)]=mu[arm]
      means.append(dict(dataset=ds,train_seed=seed,arm=arm,version=ver,kind=kind,metric=metric,value=mu[arm],sample_std=sd(x),valid_samples=sum(v is not None for v in x),expected_samples=2))
     for ss in [2026102821,2026102822,'sample_mean']:
      left=mu['A'] if ss=='sample_mean' else lookup.get((ds,seed,'A',ss,ver,kind,metric));right=mu['O'] if ss=='sample_mean' else lookup.get((ds,seed,'O',ss,ver,kind,metric))
      pairs.append(dict(dataset=ds,train_seed=seed,sample_seed=ss,comparison='A-O',version=ver,kind=kind,metric=metric,left=left,right=right,difference=left-right if left is not None and right is not None else None))
    for arm in ['O','A']:
     x=[seedmeans[(seed,arm)] for seed in seeds];across.append(dict(dataset=ds,arm=arm,version=ver,kind=kind,metric=metric,training_seed_means=x,mean=avg(x),training_seed_std=sd(x),expected_training_seeds=2))
   for seed in seeds:
    for arm in ['O','A']:
     keys=[(ds,seed,arm,ss,ver) for ss in [2026102821,2026102822]];r=dict(dataset=ds,train_seed=seed,arm=arm,version=ver,samples=[2026102821,2026102822],metrics=None)
     if all(k in graphs for k in keys):
      (g1,t1),(g2,t2)=[graphs[k] for k in keys];c=M.compare(g1,t1,g2,t2,F,refs[ds][2]);a=set(map(tuple,g1['original_ids'][g1['edges']].T));b=set(map(tuple,g2['original_ids'][g2['edges']].T));r['metrics']=dict(labeled_edge_Jaccard=len(a&b)/len(a|b) if a|b else None,labeled_edge_difference=len(a^b),exact_labeled_duplicate=a==b and set(g1['original_ids'])==set(g2['original_ids']),degree_W1=c['total_degree_W1'],triangle_composition_TV=c['triangle_composition_TV'],MP2_composition_TV=c['MP_composition_TV'][1],MP3_composition_TV=c['MP_composition_TV'][2],limitation='two samples; labeled edge differences not isomorphism invariant')
     diversity.append(r)
 for n,v in [('per_graph_all_metrics',rows),('per_training_seed',means),('paired_A_minus_O',pairs),('across_training_seeds',across)]:save(dest/f'{n}.json',v);csvwrite(dest/f'{n}.csv',v)
 save(dest/'sample_diversity.json',diversity)
 core={('absolute_error','edges'),('absolute_error','triangles'),('distance','main_incident4_W1'),('distance','main_joint_degree_TV'),('distance','triangle_composition_TV'),('raw','lcc'),('raw','isolates')}
 decision=[r for r in means if r['version']=='original' and (r['kind'],r['metric']) in core];csvwrite(dest/'decision_table.csv',decision);save(dest/'decision_table.json',decision)
 lines=['# O→A普通单输入晚CN匹配面板','',f'完成原图{sum(r["status"]=="complete" for r in inventory)}/16；只使用最终3000步EMA。','原图主结果、LCC另表；普通单输入CE。训练模型内先汇总两个S，再跨两个训练seed；不将16图当独立训练重复。','','|数据集|训练seed|组别|指标|两图均值|采样SD|','|---|---|---|---|---|---|']
 for r in decision:lines.append('|'+ '|'.join(str(r[k]) for k in ['dataset','train_seed','arm','metric','value','sample_std'])+'|')
 lines+=['','## 配对方向与结构代价','']
 for ds in ['IMDB','PubMed']:
  for kind,metric in sorted(core):
   if kind=='raw':continue
   p=[r for r in pairs if r['dataset']==ds and r['version']=='original' and r['sample_seed']=='sample_mean' and r['kind']==kind and r['metric']==metric]
   values=[r['difference'] for r in p];label='缺项' if any(x is None for x in values) else '两seed同向改善' if all(x<0 for x in values) else '两seed同向增加误差' if all(x>0 for x in values) else '混合/持平'
   lines.append(f'- {ds} {metric}: A−O={values}，{label}。')
 lines+=['','完整关系、尾部、闭合、元路径、类型覆盖、连通和所有不利值见全表；不据平均边数挑赢家。未满足全面优越、协同或稳定修复的证明条件。历史normal2/raw/外部基线另表不混入。',f'缺失/错误：{errors}']
 (OUT/'ONE_PAGE_RESULTS_ZH.md').write_text('\n'.join(lines)+'\n')
 save(dest/'receipt.json',dict(at=now(),expected_original_graphs=16,complete_original_graphs=sum(r['status']=='complete' for r in inventory),all_metrics_exported=True,errors=errors,seconds=time.time()-started))
if __name__=='__main__':main()
