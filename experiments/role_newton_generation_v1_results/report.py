"""Render all fixed targets, preserving tradeoffs and per-training-seed records."""
import json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
OUT=Path('/data2/lyh/gnn_project/DiHug/results/role_newton_generation_20261021_v1')
def read(p):return json.loads(p.read_text())
def main():
    audit=read(OUT/'audit.json');assert audit['status']=='passed'
    ag=read(OUT/'aggregate.json');per=read(OUT/'per_seed.json');f=read(OUT/'freeze.json')
    table=['|目标|DP Genre W1|Newton Genre W1|DP MG边数|Newton MG边数|DP最大/次大G度|Newton最大/次大G度|', '|---|---:|---:|---:|---:|---:|---:|']
    for r in ag:
        a,b=r['DP'],r['Newton'];table.append(f"|{r['condition']}|{a['G_W1']:.4f}|{b['G_W1']:.4f}|{a['family8_edges']:.3f}|{b['family8_edges']:.3f}|{a['G_max']:.3f}/{a['G_second_max']:.3f}|{b['G_max']:.3f}/{b['G_second_max']:.3f}|")
    lines=['# 固定权重DP/Newton完整生成对照','',
      '全部528条新T64/R4链已执行，0训练/GPU/BW/大图网络。三套广覆盖final6000权重严格核对来源/配置/依赖。',
      '原模型/角色似然/角色编码/噪声/Bayes/查询/RNG不改，仅比较精确DP读取与同熵目标Newton读取。',
      '容量约束在软读取层，不是生成图硬度数约束；没有配额、概率控制、修图、筛图或补抽。',
      '各表先在训练seed内平均8图，再等权平均三seed；不是24或528次独立训练重复。','',
      '## 全部目标的实际离散图',*table,'',
      'balanced/extreme为原控制，dev_single/dev_double为旧开发目标，train_06为广训练中间正控制，combo_*为新组合，value_*为未见度值。',
      'W1来自实际图，不是预测期望度。目标各MG边数均24，不要求逐图W1=0或逐节点匹配。','',
      '## 其他结构与多样性','|目标|DP/Newton总边|DP/Newton路径闭合|DP/Newton连通率|DP/Newton类型同构类/8图|DP/Newton平均边差|',
      '|---|---:|---:|---:|---:|---:|']
    for r in ag:
        a,b=r['DP'],r['Newton'];fmt=lambda x:'NA' if x is None else f'{x:.4f}'
        lines.append(f"|{r['condition']}|{a['edges']:.2f}/{b['edges']:.2f}|{fmt(a['closure_path'])}/{fmt(b['closure_path'])}|{a['connected']:.3f}/{b['connected']:.3f}|{a['mean_topology_classes']:.2f}/{b['mean_topology_classes']:.2f}|{a['mean_pair_edge_difference']:.2f}/{b['mean_pair_edge_difference']:.2f}|")
    lines +=['','每8图的类型/关系保持精确同构检查只描述本批重复，不证明总体多样性保持；边差使用固定标签，另提供去编号同构类。',
      '完整各关系、各类型W1/联合TV、排序度谱、hub≥8/12、低度/零度、MP1/2/3、闭合和连通原始值见generation.json/per_seed.json。',
      '新目标没有唯一拓扑真值，闭合/连通是变化描述，不将旧图结构强制作为答案。','',
      '## 新轨迹的每次读取验收',
      f"实际读取计数：{audit['read_counts']}；每个新Newton Xt都独立核验，没有复用旧DP轨迹代替。",
      f"新轨迹Newton最大指标：{audit['Newton_maxima']}。",
      '每次读取保留L/P/logP及完整64步图状态，支持数学零/约束零/舍入零分开处理。',
      'logP<-700尾部舍入只用于输出，原支持/目标不改；每行/总遗漏质量门1e-280/1e-276，实际角色上下文误差界门1e-250。',
      '行/归一列≤1e-8，独立full-support KKT≤2e-10，log归一≤2e-12；资源/单矩阵时限与上一票相同。',
      '原/新尾部合同状态分别计数。没有成功读取以外的输出进入网络；没有隐式回退DP。','',
      '## 成本和可靠性','|范围|链数|链平均秒|链P90秒|链最大秒|累计进程链秒|', '|---|---:|---:|---:|---:|---:|']
    for arm in ('DP','Newton'):
        s=audit['timings'][arm+'_chains'];lines.append(f"|{arm}|{s['defined']}|{s['mean']:.4f}|{s['P90']:.4f}|{max(s['values']):.4f}|{sum(s['values']):.3f}|")
    receipts=audit['receipts'];lines +=['',
      f"六个单线程CPU子进程CPU忙时合计{sum(x['cpu_seconds'] for x in receipts):.3f}秒；最大子进程RSS{max(x['peak_rss_mib'] for x in receipts):.2f}MiB；整体同时峰未测。",
      f"独立审计墙钟{audit['wall_seconds']:.3f}秒，CPU{audit['cpu_seconds']:.3f}秒。链计时含图/轨迹/读数保存与评价，比较两臂日志/保存策略见源码。",
      'Newton额外保存L/P/logP，DP不保存不存在的对偶量，因此链总耗时不是纯求解核函数比较；分别保留每次读取时长。',
      '原固定验证在六个作业加载后均复现，容差2e-8；实际误差见每作业validation.json。',
      '39CPU测试及零跳过集合核对；交付解包回执另存，不伪装成科学面板结果。',
      '首次common同名导入、预检NumPy标量序列化故障均保留；只做工程修复，正式链之前修好，不重置2700秒时钟。',
      '最终总墙钟见publication/completion receipt；准备/工程失败/生成/评价/审计/交付全部计入。','',
      '## 可审查边界',
      '两臂配对先验、查询、逐槽uniform和RNG摘要逐步一致；轨迹本身允许分歧，trajectory_divergence.json记录首个不同时间。',
      '不要求同一图、不使用生成后匹配。只能就这三套已训练权重、固定11目标和8seed报告结果。',
      '没有证明任意新目标/大图/新后端都可靠；没有测试大图完整前向。下一步建议由实际质量和成本审查决定，不自动执行。','']
    (ROOT/'RESULTS.md').write_text('\n'.join(lines));print('rendered RESULTS')
if __name__=='__main__':main()
