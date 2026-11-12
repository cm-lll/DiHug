"""CPU complete raw/LCC panel, existing R1 controls, no checkpoint selection."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import sys,json,time,importlib.util
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('existing_eval',ROOT/'research_branches/terminal_structure_real_20261109/evaluate.py')
ev=importlib.util.module_from_spec(spec);spec.loader.exec_module(ev)
def read(p):return json.loads(Path(p).read_text())
def main():
 out=Path(sys.argv[1]);p=read(out/'protocol.json');start=time.time();ev.torch.set_num_threads(1)
 cache=out/'evaluations';cache.mkdir(exist_ok=True);rows=[];allmetrics=[];missing=[];checks=[];refs={}
 def check():
  if time.time()>p['delivery_epoch']-30:raise TimeoutError('delivery deadline')
 numeric=['edges','edge_abs_error','degree_W1','joint_TV','triangles','triangle_abs_error','triangle_composition_TV','MP2','MP2_abs_error','MP2_TV','MP3','MP3_abs_error','MP3_TV','LCC','components','isolates','four_cycles','type_coverage']
 for ds,parent in p['parents'].items():
  data=np.load(parent['data']);edges=data['edges'];edges=edges.T if edges.shape[0]!=2 else edges
  names=['Director','Actor','Movie','Genre'] if ds=='IMDB' else ['Gene','Disease','Chemical','Species']
  ref=ev.M.make_graph(data['types'],edges,data['schema']);rs=ev.M.stats(ref,ev.F,names,check);refs[ds]=rs
  for arm in ('R1','R10','R20'):
   for ss in p['sample_seeds']:
    check();baseline=next(x for x in p['reused'] if x['dataset']==ds and x['sample_seed']==ss)
    directory=Path(baseline['directory']) if arm=='R1' else out/(ds+'_'+arm)/'stage_4'/f'final_sample_{ss}'
    if not (directory/'summary.json').exists():missing.append(dict(dataset=ds,arm=arm,sample_seed=ss,reason='chain incomplete'));continue
    summary=read(directory/'summary.json');ident=read(directory/'identity.json');steps=[json.loads(x) for x in (directory/'steps.jsonl').read_text().splitlines()]
    assert summary['status']=='complete' and summary['steps']==1000 and summary['graph_sha256']==ev.sha(directory/'graph.npz')
    assert ident['R']==int(arm[1:]) and ident['updates']==4 and ident['sample_seed']==ss and ident['train_seed']==parent['train_seed']
    bs=[json.loads(x) for x in (Path(baseline['directory'])/'steps.jsonl').read_text().splitlines()]
    assert [x['time'] for x in steps]==list(range(1000,0,-1))
    assert [x['uniform_hash'] for x in steps]==[x['uniform_hash'] for x in bs]
    assert ident['initial_hash']==read(Path(baseline['directory'])/'identity.json')['initial_hash']
    assert all(x['exact_coverage'] and x['blocks']==int(arm[1:]) for x in steps)
    if arm!='R1':
     state=ev.torch.load(out/(ds+'_'+arm)/'stage_4/final_complete.pt',map_location='cpu',weights_only=False)
     assert state['updates']==state['ema']['updates']==4 and state['recovery']['exact_resume_allowed']
     assert {float(v['step']) for v in state['optimizer']['state'].values()}=={4.}
     del state
    z=np.load(directory/'graph.npz');g=ev.M.make_graph(z['types'],z['edges'],z['schema'])
    assert np.array_equal(g['types'],ref['types']) and np.array_equal(g['schema'],ref['schema'])
    st=ev.M.stats(g,ev.F,names,check);cmp=ev.M.compare(g,st,ref,rs,ev.F,names);lg,mapping=ev.M.lcc(g);ls=ev.M.stats(lg,ev.F,names,check);lc=ev.M.compare(lg,ls,ref,rs,ev.F,names)
    full=dict(graph_sha256=summary['graph_sha256'],raw=dict(statistics=st,comparison=cmp),LCC=dict(statistics=ls,comparison=lc,mapping=mapping))
    ev.save(cache/f'{ds}_{arm}_{ss}.json',full)
    for view in ('raw','LCC'):
     st=full[view]['statistics'];cmp=full[view]['comparison']
     rows.append(dict(dataset=ds,arm=arm,train_seed=parent['train_seed'],sample_seed=ss,view=view,reference_edges=rs['edges'],reference_triangles=rs['triangles'],
      edges=st['edges'],edge_abs_error=abs(st['edges']-rs['edges']),degree_W1=cmp['main_incident4_W1'],joint_TV=cmp['main_joint_degree_TV'],triangles=st['triangles'],triangle_abs_error=abs(st['triangles']-rs['triangles']),triangle_composition_TV=cmp['triangle_composition_TV'],
      MP2=st['mp'][1]['total'],MP2_abs_error=abs(st['mp'][1]['total']-rs['mp'][1]['total']),MP2_TV=cmp['MP_composition_TV'][1],MP3=st['mp'][2]['total'],MP3_abs_error=abs(st['mp'][2]['total']-rs['mp'][2]['total']),MP3_TV=cmp['MP_composition_TV'][2],LCC=st['lcc'],components=st['components'],isolates=st['isolates'],four_cycles=st['four_cycles'],type_coverage=len(names)-len(cmp['missing_types'])))
     allmetrics.extend(dict(dataset=ds,arm=arm,sample_seed=ss,view=view,metric=k,value=v) for k,v in ev.flat(full[view]).items())
    checks.append(dict(dataset=ds,arm=arm,sample_seed=ss,full_1000=True,paired_initial_full_slot_uniforms=True,full_coverage=True,query_partition_intentionally_differs_across_R=True))
 summaries=[];deltas=[]
 for ds in p['parents']:
  for arm in ('R1','R10','R20'):
   for view in ('raw','LCC'):
    rr=[x for x in rows if (x['dataset'],x['arm'],x['view'])==(ds,arm,view)]
    if not rr:continue
    record=dict(dataset=ds,arm=arm,view=view,training_models=1,samples=len(rr))
    for k in numeric:
     vals=[x[k] for x in rr if x[k] is not None];record[k+'_mean']=float(np.mean(vals)) if vals else None;record[k+'_sample_sd']=float(np.std(vals,ddof=1)) if len(vals)>1 else None
    summaries.append(record)
    if arm!='R1':
     for x in rr:
      b=next(y for y in rows if (y['dataset'],y['arm'],y['view'],y['sample_seed'])==(ds,'R1',view,x['sample_seed']))
      deltas.append(dict(dataset=ds,arm=arm,view=view,sample_seed=x['sample_seed'],**{k:x[k]-b[k] if x[k] is not None and b[k] is not None else None for k in numeric}))
 for name,table in [('per_graph',rows),('all_metrics',allmetrics),('model_summary',summaries),('paired_differences',deltas)]:ev.save(out/(name+'.json'),table);ev.csvsave(out/(name+'.csv'),table)
 ev.save(out/'reference_metrics.json',refs)
 for path,h in p['bindings'].items():assert ev.sha(path)==h,path
 ev.save(out/'verification.json',dict(checks=checks,missing=missing,graphs=len(rows)//2,expected=12,old_assets_unchanged=True))
 ev.save(out/'evaluation_receipt.json',dict(seconds=time.time()-start,GPU=0,missing=missing))
 report=['# 分块训练/采样匹配试点结果','',f'完成 {len(rows)//2}/12 张评价图（4张R1复用、8张新图）。每数据集一个开发训练seed；采样SD不是训练seed SD。','', '| 数据集 | 查询方式 | 视图 | 度W1均值 | 三角绝对误差均值 | 联合度TV均值 | MP3绝对误差均值 | LCC均值 |','|---|---|---|---:|---:|---:|---:|---:|']
 for r in summaries:
  report.append('| '+ ' | '.join(str(r[k]) for k in ['dataset','arm','view','degree_W1_mean','triangle_abs_error_mean','joint_TV_mean','MP3_abs_error_mean','LCC_mean'])+' |')
 report+=['','只比较从共同R1更新2起点到更新4的短适配，不能称从头分块训练或跨seed稳定性。终点反馈仍是clean-head有偏代理，不是精确转移策略梯度。全部指标及采样SD见CSV；缺项：'+str(missing)]
 (out/'REPORT_ZH.md').write_text('\n'.join(report)+'\n')
if __name__=='__main__':main()
