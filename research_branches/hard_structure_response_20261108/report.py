"""Render all conditions after CPU audit; no new scientific calculation."""
import json,csv,time,datetime,hashlib,zipfile
from pathlib import Path
R=Path(__file__).resolve().parents[2];O=R/'results/hard_structure_response_20261108T182154';H=Path(__file__).parent
read=lambda p:json.loads(Path(p).read_text());p=read(O/'protocol.json');lease=read(O/'lease.json');receipt=read(O/'receipt.json');summary=read(O/'summary.json');audit=read(O/'audit.json')
rows=list(csv.DictReader((O/'paired_metrics.csv').open()));updates=read(O/'updates.json')
fmt=lambda x:f'{float(x):+.6g}'
lines=['# 八步局部参数响应：CE与硬图结构辅助','',
'本票只在已有局部诊断父状态上进行配对更新，不是正式模型训练或完整扩散生成。四条件来自IMDB 2026110411和PubMed 2026110422各两份局部CE/PS状态；这些父状态共同来源于PS历史轨迹，不能叫四个独立训练seed或纯CE从头对照。',
'',f"实际状态：{receipt['status']}；{receipt['updates']}次临时更新，{receipt['forwards']}前向，{receipt['backwards']}反向；完整T1000链0。GPU租用{(lease['end_epoch']-lease['start_epoch']):.3f}秒／{lease['gpu_hours']:.6f} GPU小时，worker{receipt['seconds']:.3f}秒，峰值保留显存{receipt['peak_reserved_MiB']:.1f}MiB。",
'','## 固定规则与对照',
'从每份父状态完整恢复后，CE和CE_STRUCT分别继续同样8份历史输入。结构辅助系数0.1、每步一份独立伯努利辅助图；不做权重搜索、梯度裁剪、保护或投影。原CE、AdamW、EMA和学习率不变。此系数是本轮工程选择：上票有限面板上未缩放结构/CE梯度比.163—1.360，取.1使其约.016—.136；这不是最优性推导，后续输入也不保证此比例。',
'结构目标是按类型匹配节点关系度与类型三角参与量的联合分布，Gaussian MMD V统计与尺度沿用CPU票。每条边使用p(1-p)[D(B_-e,1)-D(B_-e,0)]作为logit差的梯度。该梯度通过停梯度线性代理注入，代理标量值记录本次硬图评分；代理高阶导数/单次标量变化不应被解释成精确期望目标变化。',
'独立辅助NumPy随机流在诊断检查点外层保存，主训练随机流不受其消耗。完整恢复边界和采样均沿用上一票已通过的执行策略；新辅助分支没有额外重复下一步的GPU恢复试验，不把仅保存了辅助RNG声称为已通过该额外试验。异常停止，不将中断半步保存成可精确恢复状态。',
'','## 评价定义',
'新固定前向加噪输入t=1/500/1000，每时刻一份；未用于本轮参数更新或辅助抽样。仍来自同一参考图，且seed/父状态已用于开发，不是跨图或独立最终确认。raw读出，EMA保存但不评价。',
'每输入另固定32份共同均匀随机数，为两臂构造条件p0硬图评分；表中MC SE是成对评分差的经验标准误，不是训练seed置信区间。32抽样可能漏掉稀有事件，0差异不证明总体无差异。不是模型T1000生成图。',
'节点预期关系度由边概率加和，三角预期量由三条不同无向边概率乘积求和，后者仅在本轮条件p0独立伯努利分布下成立；不是完整逆过程图的期望三角。degree MAE是期望节点度与真值的误差，不是随机图度MAE的期望。三角参与通道全节点相加后除3，合法无自环槽不重复计边。',
'低度/hub按各类型参考度Q25/Q90预先固定（低度<=Q25，hub>Q25且>=Q90，中间其余），用于已知X0前向样本；空组保留null和支持数。自由生成匿名节点不使用此按编号解释。',
'','## 各父状态三时间等权均值差：CE_STRUCT减CE','',
'|父状态|硬评分差|节点预期度MAE差|三角数量绝对误差差|正边CE差|非边CE差|',
'|---|---:|---:|---:|---:|---:|']
for s in summary:lines.append('|'+s['condition']+'|'+'|'.join(fmt(s[k+'_mean_delta']) for k in ['hard_score_mean','expected_degree_MAE','triangle_abs_error','positive_CE','negative_CE'])+'|')
lines+=['','负值表示对应误差下降。三时间平均仅描述当前三个固定输入，不替代逐时间结果。','','## 全部12个配对输入','',
'|父状态|t|硬评分差 ± MC SE|度MAE差|三角误差差|正边CE差|非边CE差|',
'|---|---:|---:|---:|---:|---:|---:|']
for x in rows:lines.append('|'+x['condition']+'|'+x['t']+'|'+fmt(x['hard_score_paired_delta'])+' ± '+f"{float(x['hard_score_paired_MC_SE']):.3g}"+'|'+'|'.join(fmt(x[k+'_delta']) for k in ['expected_degree_MAE','triangle_abs_error','positive_CE','negative_CE'])+'|')
lines+=['','## 工程与成本','',
'四条件CE第一步与既存同父CE更新的模型、AdamW、EMA、RNG、计数逐位一致；逐步输入及完整RNG与历史记录核对，评价不改训练状态。最终两臂主RNG一致，计数与EMA更新数一致；结果保存为diagnostic_only，不覆盖父资产。audit.json列出绑定/环境/输入核对。',
f"源码/资产未改核验：{audit['all_bindings_unchanged']}。HEAD {audit['git_head']}，隔离代码按source_binding.json绑定，未创建或推送新commit。原183份生产源码不变。缺torch-scatter加速的既有警告未修，不安装依赖。",'']
for ds in ['IMDB','PubMed']:
 for arm in ['CE','CE_STRUCT']:
  u=[x for x in updates if x['condition'].startswith(ds+'_') and x['arm']==arm]
  if u:lines.append(f"- {ds} {arm}: {len(u)}更新，更新内部均值{sum(x['update_seconds'] for x in u)/len(u):.3f}秒，辅助计算均值{sum(x['auxiliary_seconds'] for x in u)/len(u):.3f}秒；不含全部状态加载/评价/保存。")
lines+=['','## 结论边界与交付',
'本面板判断局部响应和代价，不凭8步或一个评分决定正式采用；完整生成/跨训练seed泛化尚未验证。不能将本票结果外推为整个REBAR方法有效或无效：实际采用的是逐边条件求和估计，REBAR仅为此前CPU对照。没有自动扩大训练或调整系数。',
'逐关系全部结果：paired_relations.csv；低度/hub支持和代价：paired_nodegroups.csv；原始逐输入数据probes.json，训练记录updates.json，逐指标基线/候选/起点/差paired_metrics.csv。临时检查点及概率保留本地，审查包不包含这些大体积或原始标签资产。',
'详细研究判断另见DECISION_ZH.md；不得只抽取上述平均收益忽略逐时间和逐关系代价。']
(O/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines[lines.index('## 各父状态三时间等权均值差：CE_STRUCT减CE'):]))
