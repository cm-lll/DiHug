"""Pairing audit and evidence report; never selects/restarts scientific work."""
import csv,json,pathlib,statistics,time,zipfile
from common_run import ROOT,CHECKOUT,SCI,HERE,OUT,read,save,sha,now

def main():
 ledger=read(OUT/'ledger.json');jobs=read(OUT/'tasks.json');reused=read(OUT/'reused_jobs.json');evaldir=OUT/'evaluation';failures=[];checks=[]
 for p in read(OUT/'protected_parent_assets.json'):
  if sha(p['path'])!=p['sha256']:raise ValueError('protected historical asset changed '+p['path'])
 groups={}
 for j in jobs+reused:groups.setdefault((j['dataset'],j['train_seed']),{})[j['arm']]=j
 for(ds,seed),pair in groups.items():
  logs={}
  for arm in ['O','A']:
   p=pathlib.Path(pair[arm]['folder'])/'training.jsonl';logs[arm]=[json.loads(x)for x in p.read_text().splitlines()]if p.exists()else[]
  n=min(map(len,logs.values()));same=all(logs['O'][i]['update']==logs['A'][i]['update']and logs['O'][i]['input_sha256']==logs['A'][i]['input_sha256']for i in range(n))
  check=dict(dataset=ds,train_seed=seed,posthoc_case=seed==2026110321,compared_updates=n,main_input_exact=same,complete=n==3000,dropout_trajectory='architecture-dependent; no claim of complete Torch path equality')
  checks.append(check)
  if not same or n!=3000:failures.append(check)
 save(OUT/'final_pairing_audit.json',dict(checks=checks,failures=failures))
 means=read(evaldir/'per_training_seed.json');pairs=read(evaldir/'paired_A_minus_O.json');across=read(evaldir/'across_training_seeds.json')
 core=[('absolute_error','edges'),('distance','main_incident4_W1'),('absolute_error','triangles'),('distance','triangle_composition_TV'),('distance','main_joint_degree_TV'),('absolute_error','MP2_total'),('distance','MP2_composition_TV'),('absolute_error','MP3_total'),('distance','MP3_composition_TV'),('raw','lcc')]
 selected=[r for r in means if r['version']=='original'and(r['kind'],r['metric'])in core]
 save(OUT/'decision_table.json',selected)
 with(OUT/'decision_table.csv').open('w',newline='')as f:
  w=csv.DictWriter(f,fieldnames=list(selected[0])if selected else ['missing']);w.writeheader();w.writerows(selected)
 lines=['# 类型化CN多训练seed匹配补证','',
 '普通单输入CE、3000更新、EMA.99、R1/T1000；原图为主，LCC独立。主要指标为incident4度W1、逐图三角数量绝对误差及联合度TV。',
 '新seed在生成前冻结；已知较差A0321是事后选择案例，不混入普通seed平均。旧2样本、新4样本分表；合并证据只用共同的两个采样seed，先模型内平均，再对训练seed等权汇总/样本标准差。',
 '公共初态、主Xt/时间/查询及采样计划/全槽随机数配对；CN增加上下文Dropout调用，因此不声称完整Torch随机调用轨迹一致。O还移除CN参数及头末37列，参数量差14816。','',
 '|面板|数据集|组别|指标|训练seed均值|训练seed样本SD|','|---|---|---|---|---|---|']
 for r in across:
  if r['version']=='original'and(r['kind'],r['metric'])in core and r['arm']in ['O','A']:
   lines.append('|'+ '|'.join(str(r.get(k))for k in ['panel','dataset','arm','metric','mean','training_seed_std'])+'|')
 lines+=['','## 逐seed配对方向','']
 for panel in ['prospective_four_samples','combined_common_two_samples','posthoc_known_bad_A_case']:
  for ds in ['IMDB','PubMed']:
   for kind,metric in core:
    if kind=='raw':continue
    rr=[r for r in pairs if r['panel']==panel and r['dataset']==ds and r['version']=='original'and r['kind']==kind and r['metric']==metric and r['sample_seed']=='sample_mean']
    if not rr:continue
    values=[r['difference']for r in rr];label='缺项'if any(v is None for v in values)else'均改善'if all(v<0 for v in values)else'均恶化'if all(v>0 for v in values)else'混合或持平'
    lines.append(f'- {panel} / {ds} / {metric}：A−O={[(r["train_seed"],r["difference"])for r in rr]}；{label}。')
 lines+=['','## 最差训练seed与采样波动','']
 for panel in ['prospective_four_samples','combined_common_two_samples']:
  for ds in ['IMDB','PubMed']:
   for arm in ['O','A']:
    rr=[r for r in selected if r['panel']==panel and r['dataset']==ds and r['arm']==arm and r['metric']=='main_incident4_W1']
    if rr and all(r['value']is not None for r in rr):
     worst=max(rr,key=lambda r:r['value']);lines.append(f'- {panel} / {ds} / {arm}：W1最大seed={worst["train_seed"]}，模型采样均值{worst["value"]}，该模型采样SD={worst["sample_std"]}。')
 receipt=read(evaldir/'receipt.json');lines+=['','不合成总分、不将采样图当独立训练重复、不使用“统计显著”。组成/连通/逐关系代价保留在全表，不以总边数接近宣布结构恢复。',
 f'全图身份完成{receipt["complete_original_graphs"]}/56；评价错误：{receipt["errors"]}；配对异常：{failures}。',
 '缺项、未定义和未实现字段保留null及原因。HGEN未安装原生依赖等原有缺项沿用，不新增评价定义。','',
 '科学源码175373c；仅本轮执行配置/队列/采样数量与报告接线变化，原模型/CE/AdamW/EMA/噪声核/后验未改。旧结果与旧账本保持原样。']
 (OUT/'ONE_PAGE_RESULTS_ZH.md').write_text('\n'.join(lines)+'\n')
 leases=[read(p)for p in(OUT/'leases').glob('*.json')];hours=sum((r.get('released_epoch',time.time())-r['acquired_epoch'])/3600 for r in leases)
 result=dict(at=now(),wall_minutes=(time.time()-ledger['start_epoch'])/60,GPU_hours=hours,within_wall=time.time()<=ledger['deadline_epoch'],within_GPU_hours=hours<=ledger['GPU_hour_limit'],
    completed_new_models=sum((pathlib.Path(j['folder'])/'summary.json').exists()and read(pathlib.Path(j['folder'])/'summary.json')['status']=='complete'for j in jobs),expected_new_models=9,expected_new_chains=36,evaluation=receipt,pairing=checks,pairing_failures=failures,protected_assets_unchanged=True,all_own_leases_released=all(r.get('released_epoch')for r in leases))
 save(OUT/'FINAL_RECEIPT.json',result)
 archive=OUT.parent/(OUT.name+'_review.zip')
 with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=4)as z:
  for p in OUT.rglob('*'):
   if p.is_file()and p.suffix not in ['.pt']and 'temporary'not in p.relative_to(OUT).parts:z.write(p,str(p.relative_to(ROOT)))
  for p in HERE.glob('*.py'):z.write(p,str(p.relative_to(CHECKOUT)))
 with zipfile.ZipFile(archive)as z:
  if z.testzip()is not None:raise ValueError('archive corrupt')
 save(OUT/'archive_receipt.json',dict(path=str(archive),sha256=sha(archive),bytes=archive.stat().st_size,weights_remain_local=True))
 print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':main()
