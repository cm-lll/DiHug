import csv,json,time,hashlib,zipfile,subprocess
from pathlib import Path
R=Path(__file__).resolve().parents[2];O=R/'results/hard_structure_edge_direction_cpu_20261108';H=Path(__file__).parent
read=lambda p:json.loads(Path(p).read_text());rows=list(csv.DictReader((O/'graph_summary.csv').open()));panels=list(csv.DictReader((O/'panel_summary.csv').open()));receipt=read(O/'receipt.json');p=read(O/'protocol.json');audit=read(O/'audit.json')
lines=['# CPU逐边评分方向核查','',
'结论：确认现有联合MMD评分与部分目标结构指标存在实际方向不一致，不能继续把评分下降等同于结构改善。未发现本票计数或增量实现错误。尚不能把这认定为历史坏seed或上一轮全部代价的唯一原因。',
'',f"24已存概率面板×原32辅助抽样的前4份=96图，全部合法槽共{receipt['edge_flips']:,}次独立翻边。每次均从同一个冻结硬图出发，不连续编辑。16合成小图240边的独立重算误差≤3.33e−16；实际图768固定槽与原完整评分重算一致。单CPU线程，worker{receipt['seconds']:.3f}秒，峰值RSS{receipt['peak_RSS_MiB']:.2f}MiB，0GPU/去噪网络/训练/完整反向链。正式源、字典、父资产与概率不改。",
'','## 范围与口径',
'四父状态为IMDB0411、PubMed0422各两条局部CE/PS分支；上一票各自CE/CE_STRUCT八步终点，t1/500/1000，共24概率面板。只有两个训练seed，不把相关分支和翻边当独立训练重复。辅助图是给定Xt的条件p0独立伯努利样本，不是T1000生成图。',
'本票没有新增真实面板随机样本：复用原seed和序列前4份，并逐图核对原评分。16新随机微图仅为CPU实现测试，单独计数，不能把receipt的new_random_draws=0解释成连合成测试随机数都没用。',
'对于硬图B和槽e，增量代码计算D1−D0。实际翻转的评分差为(1−2B_e)(D1−D0)，小于−1e−12记作评分改善。独立指标差大于1e−12记作恶化。三角指标是全图三角数绝对误差；度W1是按节点类型分别排序总度后求W1再类型等权；联合度TV是同节点关系度向量的经验分布TV再类型等权。均保留合法无向槽、无自环/重复边。',
'同时报告改善翻边数及|p(1−p)(D1−D0)|加权质量，后者只表示本次输出logit估计器的幅度，不等于网络更新影响、翻边发生概率或最终图的因果贡献。没有显著性检验。',
'','## 评分改善的翻边中，三角误差反而上升的比例','',
'|数据集|t|改善翻边数|三角恶化占比|对应梯度幅度占比|三项同时恶化数|',
'|---|---:|---:|---:|---:|---:|']
for ds in ['IMDB','PubMed']:
 for t in [1,500,1000]:
  rs=[x for x in rows if x['condition'].startswith(ds) and int(x['t'])==t];good=sum(int(x['score_improving_flips']) for x in rs);mass=sum(float(x['score_improving_gradient_mass']) for x in rs);bad=sum(int(x['triangle_abs_error_worse_count']) for x in rs);weighted=sum(float(x['triangle_abs_error_worse_gradient_mass_fraction'] or 0)*float(x['score_improving_gradient_mass']) for x in rs)/mass
  lines.append(f"|{ds}|{t}|{good:,}|{100*bad/good:.4f}%|{100*weighted:.4f}%|{sum(int(x['all_three_worse_count']) for x in rs):,}|")
lines+=['','以上各行覆盖两个相关父状态×两更新臂×四抽样，共16图。不能把操作数当成样本量去做统计显著性结论。全部逐父状态/更新臂表见panel_summary.csv，逐关系加删方向见relations.csv。',
'','## 具体可复核例子',
'按预先冻结遍历顺序取到的PubMed CE父、CE终点、t500、辅助图0：原图2246个三角，参考1937。添加槽6927（节点51—89，关系4，当前概率0.0087657，共同邻居1个）：',
'- 三角2246→2247，绝对误差309→310；',
'- 类型等权度W1增加0.0067567568；',
'- 类型等权联合度TV增加0.0033783784；',
'- 联合MMD评分却从0.009649156下降到0.009647411，差−1.744823e−6。',
'该边不是参考边，但这不是判断它不好的依据：本例依据是三项结构指标同时恶化。并非选最大反例；examples.csv保留每图按槽序首个反例及首个共同改善例。原槽和概率在冻结输入及概率文件中可追溯。',
'','## 能说明什么，不能说明什么',
'1. PubMed过闭合32图（1984—2595三角，参考1937）有40726个评分改善却增大三角误差的翻边，占改善操作27.67%；以本票logit梯度幅度加权为11.11%。其中19230个操作同时增大三角误差、度W1和联合度TV。低噪声和中噪声、两父状态、两终点都出现该现象，不是仅一张图的特例。仍只是本面板证据。',
'2. IMDB过闭合16图中，评分改善却继续增大三角误差的操作为0；但上一票网络更新后这些评价输入的三角期望仍上升。这说明不能用“评分在每处都鼓励加三角”解释全部结果。多目标组合、共享参数对其他时间的响应、有限更新及抽样误差仍可能参与，不能在本票CPU反事实中分离。',
'3. 度W1方向冲突在两数据集都存在；具体按图比例和加权量保留，不能只展示三角方向较好的IMDB就说评分已适配。',
'4. 计数验证通过、评分与旧实现相同，当前问题是目标选择与指标对齐，不是发现一个可立即修复的加减号/三角重计数错误。',
'','## 为什么包含三角仍可能奖励三角误差变大',
'当前评分先对各节点关系度、类型三角参与量做log1p及参考尺度变换，再用Gaussian核比较每类节点的联合经验分布。它在同一标量中汇总所有节点与坐标的相似度变化；没有要求每个结构指标都改善。其他坐标或节点的相似度改善，可以抵消某些总量/边缘分布代价。字段里包含三角，并不等于评分对总三角误差单调。',
'这是代码允许的补偿，并非宣称log1p、核带宽或类型等权中的某一个已被单独证明是原因。三项指标一起变差，也不意味着所有未测结构特征都变差；因此结论是与本项目重点指标方向不一致，不是证明MMD在所有任务上错误。',
'','## 唯一优先下一步',
'先重新定义可解释的类型内联合结构距离，保留本轮通过核验的逐边条件差分工具；不要继续仅调辅助系数或直接扩大训练。优先研究同节点关系度和类型三角向量的联合运输距离：先写清它与边缘度误差及三角总量误差的界，再用现有硬图做CPU方向和成本核验。它仍不保证所有指标每步同时改善，不是把更换距离包装成已验证修复。',
'本票没有实现替代目标、没有修改正式训练，也没有对新目标作效果结论。若方向核验和成本不成立就停止；不重开门控、CN入口或更新保护搜索。',
'','## 复现与资产',
'prepare.py冻结范围/哈希，supervise.py限制墙钟与RSS并屏蔽CUDA，run.py执行已保存图的全边反事实，metrics.py独立计数，summarize.py汇总。既有目录拒绝重启，复现需新目录和新截止。正式183份源码及输入哈希在audit.json核对；全部翻边数值NPZ保留本地，轻量审查包包含完整汇总、例子和源码，不包含权重/原始图标签。']
(O/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
assert receipt['status']=='complete' and audit['all_unchanged']
sha=lambda x:hashlib.sha256(x.read_bytes()).hexdigest()
delivery=dict(at_epoch=time.time(),wall_seconds=time.time()-p['start_epoch'],worker_seconds=receipt['seconds'],CPU_threads=1,GPU_hours=0,model_calls=0,training_updates=0,T1000_chains=0,reused_real_aux_graphs=96,new_synthetic_unit_cases=16,formal_training_authorized=False,git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip(),source_hashes={str(x):sha(x) for x in H.glob('*.py')},warnings=['No zero-error guarantee for all graph properties','4 correlated auxiliary draws/panel, not independent training repetitions','Finite edge edits are not network/Adam or reverse-chain interventions'],failure_records=[])
assert delivery['wall_seconds']<900
(O/'delivery.json').write_text(json.dumps(delivery,indent=2))
z=O.with_name(O.name+'_review.zip')
with zipfile.ZipFile(z,'w',zipfile.ZIP_DEFLATED) as f:
 for x in O.iterdir():
  if x.suffix in ['.json','.csv','.md','.log']:f.write(x,'results/'+x.name)
 for x in H.glob('*.py'):f.write(x,'code/'+x.name)
 for x in [R/'research_branches/hard_structure_rebar_cpu_20261108/objective.py',R/'research_branches/hard_structure_incremental_cpu_20261108/incremental.py']:f.write(x,'frozen_reference/'+x.name)
print(json.dumps(dict(package=str(z),sha256=sha(z),bytes=z.stat().st_size,wall_seconds=delivery['wall_seconds']),indent=2))
