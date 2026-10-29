"""Package only existing evidence; no model, new graph, or mutable source import."""
import argparse,pathlib,json,csv,shutil,datetime,time
import numpy as np
from audit import read,sha
def csvfile(p,rows):
 if not rows:return
 keys=list(dict.fromkeys(k for r in rows for k in r))
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def main():
 p=argparse.ArgumentParser();p.add_argument('--sealed',required=True);p.add_argument('--source-zip',required=True);a=p.parse_args()
 out=pathlib.Path(__file__).resolve().parent;sealed=pathlib.Path(a.sealed);run=sealed/'run';ev=read(run/'unified_evaluation.json');audit=read(sealed/'independent_audit.json');summary=read(run/'launcher_summary.json');activation=read(run/'activation.json')
 assert audit['status']=='passed' and not (out/'run').exists()
 for f in run.rglob('*'):
  if not f.is_file() or f.suffix=='.pt':continue
  assert not f.is_symlink()
  dest=out/'run'/f.relative_to(run);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,dest)
 for name in ['approval.json','startup_receipt.json','source_binding.json','prelaunch.json','launcher_process.json','launcher.log','independent_audit.json','independent_audit.log']:shutil.copy2(sealed/name,out/name)
 shutil.copy2(a.source_zip,out/'runtime_source.zip')
 (out/'LOCAL_BINARY_ASSETS.json').write_text(json.dumps(dict(assets=audit['weights'],policy='All 48 initial/checkpoint/final-readout files retained at original local paths; public archive contains hashes and sizes, not large weights.'),indent=2))
 curves=[];validation=[];families=[];allvalues=[];overview=[];costs=[]
 for cell in audit['training']:
  folder=run/cell['directory'];identity={k:cell[k] for k in ['dataset','arm','train_seed']}
  for line in (folder/'training.jsonl').read_text().splitlines():
   x=json.loads(line);curves.append(dict(**identity,update=x['update'],loss=x['loss'],gradient_norm=x['gradient_norm'],normal2_seconds=x['normal2_seconds'],total_update_seconds=x['total_update_seconds'],input_sha256=x['input_sha256'],t0=x['t'][0],t1=x['t'][1]))
  for val in read(folder/'validation.json'):
   for kind in ['raw','ema_0.99']:
    for x in val[kind]['rows']:
     validation.append(dict(**identity,updates=val['updates'],readout=kind,t=x['t'],input_sha256=x['input_sha256'],ce_sum_over_S=x['ce_sum_over_S'],p0_sum=sum(z['expected_edges'] for z in x['family'])))
     for family in x['family']:families.append(dict(**identity,updates=val['updates'],readout=kind,t=x['t'],**family))
  costs.append(dict(**identity,train_worker_seconds=cell['seconds'],normal2_seconds=cell['normal2_seconds'],total_update_seconds=cell['total_update_seconds'],chain_seconds=sum(x['seconds'] for x in audit['graphs'] if x['dataset']==cell['dataset'] and x['arm']==cell['arm']),peak_allocated_mib=cell['peak_allocated_mib'],peak_reserved_mib=cell['peak_reserved_mib'],peak_rss_mib=cell['peak_rss_mib']))
 for row in ev['records']:
  identity={k:row[k] for k in ['dataset','arm','train_seed','sample_seed','version']}
  for group in ['raw','reference','absolute_error','distribution_error']:
   allvalues.extend(dict(**identity,group=group,metric=k,value=v,reason=row['undefined'].get(group,{}).get(k)) for k,v in row[group].items())
  overview.append(dict(**identity,nodes=row['raw']['nodes'],edges=row['raw']['edges'],edge_absolute_error=row['absolute_error']['edges'],W1_incident4=row['distribution_error']['main_incident4_W1'],joint_TV=row['distribution_error']['main_joint_degree_TV'],triangles=row['raw']['triangles'],triangle_absolute_error=row['absolute_error']['triangles'],triangle_TV=row['distribution_error']['triangle_composition_TV'],LCC=row['raw']['lcc'],MP2_TV=row['distribution_error']['MP2_composition_TV'],MP3_TV=row['distribution_error']['MP3_composition_TV']))
 for name,rows in [('training_curves',curves),('validation',validation),('validation_by_relation',families),('all_metric_values',allvalues),('overview',overview),('costs',costs)]:csvfile(out/(name+'.csv'),rows)
 for name in ['cell_means','paired','paired_means','lcc_minus_original']:csvfile(out/(name+'.csv'),ev[name])
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axs=plt.subplots(2,2,figsize=(12,7))
 for i,d in enumerate(['IMDB','PubMed']):
  for arm in ['A','C','D','E']:
   rows=[r for r in curves if r['dataset']==d and r['arm']==arm];y=np.array([x['loss'] for x in rows]);axs[i,0].plot(np.arange(100,3001),np.convolve(y,np.ones(100)/100,'valid'),label=arm)
   rows=[r for r in validation if r['dataset']==d and r['arm']==arm and r['readout']=='ema_0.99' and r['t']==1];axs[i,1].plot([r['updates'] for r in rows],[r['ce_sum_over_S'] for r in rows],marker='o',label=arm)
  axs[i,0].set_title(d+' normal2 CE, trailing 100 updates');axs[i,1].set_title(d+' fixed t=1 EMA validation CE')
  for ax in axs[i]:ax.legend();ax.set_xlabel('effective updates');ax.grid(alpha=.2)
 fig.tight_layout();fig.savefig(out/'training_curves.png',dpi=150);plt.close(fig)
 def means(d,arm,v,key):
  rows=[r for r in overview if r['dataset']==d and r['arm']==arm and r['version']==v];return sum(r[key] for r in rows)/2
 lines=['# 最后小图A/C/D/E筛选结果','', '完成8×3000次normal2更新及16条T1000链，全部使用最终EMA0.99。无补抽、筛图、重启或新增模型。科学及自动评价于2026-10-29 06:44:59完成，199.94分钟，早于原科学停止07:45:03；独立审计和发布因系统盘恢复延后，实际日期另记，**不声称全部交付在旧280分钟内完成**。','', '## 主要结果（先逐图算误差，再平均两个采样seed）','', '|数据集|组|边数绝对误差↓|incident4 W1↓|联合度TV↓|三角绝对误差↓|三角组成TV↓|LCC节点|','|---|---|---:|---:|---:|---:|---:|---:|']
 for d in ['IMDB','PubMed']:
  for arm in ['A','C','D','E']:
   values=[means(d,arm,'original',k) for k in ['edge_absolute_error','W1_incident4','joint_TV','triangle_absolute_error','triangle_TV','LCC']];lines.append('|'+d+'|'+arm+'|'+'|'.join(f'{v:.5g}' for v in values)+'|')
 lines += ['', '## 筛选建议与代价','', '**IMDB优先D，E保留为有取舍的备选。** D相对A，W1从0.19010降至0.09505，三角绝对误差3.5→1.5，MP3组成TV0.06739→0.03097，LCC178→183.5。E的W1和联合度TV进一步降至0.08594、0.19271，但边数绝对误差16.5高于D的10，三角组成TV0.04755高于D的0.03565；没有所有指标都改善。按表现接近时优先简单方案的预定原则，D不需要正式大图RolX字典。', '', 'IMDB C只有1/0个三角，参考45；D为43/44、E为43/43。DAM两图均值A/C/D/E为41/0.5/41/40；DAA为2/0/0.5/1.5，AAM为1.5/0/2/1.5，不能只看三角总量宣布闭合完全恢复。Genre最大/次大度均值分别为A72/50.5、C70/30、D74/24、E72/25；完整参考及排序谱见原始统计，不以单一最大度取代尾部分布。', '', '**PubMed优先C，保留A为匹配的CN对照；不推荐直接把E定为主方案。** C边数绝对误差37、W1 0.47396、联合度TV0.69531及LCC190.5在本面板较好，但三角组成TV0.10145劣于A的0.06472，MP3组成TV0.08934也略差于A的0.08489。E三角总数绝对误差167最好，却欠边121.5、W1 0.58854，不能用三角数收益掩盖数量/度代价。D的MP2/MP3组成TV较小，但三角总数误差903，数量与组成的判断必须分开。', '', '整体优先收拢为D（IMDB）和C（PubMed）两个候选，A继续作为完整图匹配基线。这是探索性筛选建议，不是唯一赢家证明；不自动拟合大图字典或启动完整图实验。', '', '## 原图与LCC分开','', '|数据集|组|LCC节点均值|LCC边数绝对误差↓|LCC固定参考权重W1↓|LCC联合度TV↓|','|---|---|---:|---:|---:|---:|']
 for d in ['IMDB','PubMed']:
  for arm in ['A','C','D','E']:lines.append('|'+d+'|'+arm+'|'+'|'.join(f'{means(d,arm,"lcc",k):.5g}' for k in ['nodes','edge_absolute_error','W1_incident4','joint_TV'])+'|')
 lines += ['', 'LCC连通和无孤立是选择规则，不是额外学习成果。PubMed所有图仅移除无边节点，边数/三角/元路径数量不变；IMDB会移除部分带边小分量。按实际保留节点重算容量和分母，主W1保留参考类型权重；全体类型都在，但逐类型流失仍需看inventory和mapping。', '', '## 审计、成本及证据边界','', '独立CPU审计：16原图+16LCC；48份本地权重哈希，最终raw/EMA与checkpoint张量一致；5682项公共初态张量比较；训练输入及采样随机流配对；全部24000更新和16000步覆盖日志；123556项聚合算术与null处理。关键离散指标独立复算，包括关系边数、三角、四环、MP1/2/3总量、W1/联合度TV及LCC映射。MP/闭合的完整组成沿用冻结评价，并未在本次独立重枚举所有模式。', '', '训练/采样墙钟及显存见costs.csv，不能把八进程各自RSS峰值相加称为同一时刻峰值。raw权重和验证保留，但只生成EMA，不能认定3000步下EMA优于raw。每数据集仅一个新训练seed、每模型两图，不证明稳定显著性或大图有效性。旧1000步实验不是本轮训练长度的严格配对对照。', '', '全指标保留于all_metric_values.csv及run/unified_evaluation.json；逐图类型三角、闭合机会、MP组成、度谱、LCC映射及样本差异均在run/derived_lcc_evaluation。HGEN legacy K4不等于真正四环；原生powerlaw依赖、社区标签及未实现graphlet/orbit仍列缺项，未填造数字。', '', '## 复核','', '先校验SHA256SUMS，将runtime_source.zip解至新的数据盘临时目录；运行 `python audit.py --run run --source <解包后的typed_structure_final_screen_v1目录> --output <新的回执路径> --public-only`。CPU线程设为1，不加载权重或模型，不激活正式入口。完整本地权重审计省略--public-only，但需使用原run路径，不能把公开包不存在的权重视为已验证。', '', '发布时刻晚于科学完成；完成后停止，不增加训练seed、采样图或其他模块。']
 (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')
 (out/'README.md').write_text('# Final screen results\n\n见RESULTS.md。源基线472e533fd28cfc067702b1b75082ac5105140a2d。全部科学输出只读归档；权重保留本地，公开哈希见LOCAL_BINARY_ASSETS.json。公开包包含16原图、16LCC、完整日志、评价、固定源ZIP、审计脚本与回执。\n')
 (out/'delivery_timing.json').write_text(json.dumps(dict(original_start_epoch=activation['start_epoch'],original_science_stop_epoch=activation['deadline_epoch']-1200,original_final_deadline_epoch=activation['deadline_epoch'],scientific_and_auto_evaluation_finished_epoch=summary['finished_epoch'],independent_audit_started_epoch=audit['started_epoch'],independent_audit_finished_epoch=audit['finished_epoch'],report_built_epoch=time.time(),reason_late='root partition recovery followed by explicitly requested result audit/publication; original scientific clock unchanged'),indent=2))
 print('report complete',out)
if __name__=='__main__':main()
