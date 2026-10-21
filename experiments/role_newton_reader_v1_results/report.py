"""Factual results report, no scientific execution."""
import json,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
R=Path('/data2/lyh/gnn_project/DiHug/results/role_newton_20261021_v1')
def read(p):return json.loads(p.read_text())
def main():
    assert read(R/'final_audit.json')['status']=='passed'
    a=read(R/'audit.json');small=read(R/'small_partial.json');large=read(R/'large_partial.json');allr=small+large
    ref=read(R/'reference_cost.json')['rows'];pred=read(R/'prediction_summary.json')
    lines=['# 同熵目标的约化对偶阻尼Newton：CPU固定面板结果','',
      '**固定面板数值失败得到解决：544/544满足新接口，538满足原普通概率接口，6只满足事前声明的有界尾部接口。**',
      '不能将这解释成等价于精确DP、完整图后验，或生成质量/多样性已保持。没有训练、新完整链、大图网络或GPU/BW。','',
      '## 目标与唯一求解器',
      '基线9b010a6b41b39755a9bb36f34fa210b32b8fd689；原DP、原缩放和所有旧结果未改。',
      'F(b)=Σ_i LSE_j(L_ij+b_j)−cᵀb；P_i=softmax(L_i+b)，g=Pᵀ1−c，H=diag(Pᵀ1)−PᵀP。',
      '熵系数1，完整联合角色及频数不变。阻尼仅用于Newton方向，不进入F或g；没有混选其他求解器。',
      '每个支持分量固定末列偏置为0；行/列常数规范化后偏置零初始化，所有首次/重复均冷启动，无跨时间暖启动。',
      '200迭代、4000函数评估、协作30秒单矩阵时限，最多32次线搜索减半、8次方向阻尼尝试；',
      '初始阻尼1e-6、通过后/3最低1e-12、失败×10最高1e12，Armijo1e-4，最大偏置增量1000。',
      'float64；行绝对/归一列残差≤1e-8、独立平稳残差≤2e-10。2m元素和256MiB workspace估算门。',
      '所有配置在完整面板前冻结，JSON修复后只更新序列化，不改变solver.py或任何科学设置。','',
      '## 输入来源与支持分类',
      '528小矩阵来自132个封存小图Xt，按9b010中643载荷清单逐项比对NPZ原SHA。',
      '旧16大图LL没有保存数组字节，无法核对历史LL哈希；本票明确采用确定性重建，非重新抽样：',
      '核对原完整数据SHA/匿名角色/噪声源码、重建后旧边状态SHA，重算旧缩放的状态、迭代、行列残差和下溢数均逐项相同。',
      '新544个LL分别保存并哈希；16个大图LL仅能称来源和状态核验的重建，不伪称原存数组完全认证。',
      '实际544矩阵均满支持，无支持不可行或约束强制零。对稀疏支持，整数maxflow＋残余有向图SCC只删除所有可行分配中不可能使用的位置。',
      '3节点2角色全部64支持掩码与穷举可行分配边并集一致；身份分量/边界/零频数/不可行均有检查。',
      '这些支持证书不作为网络硬角色答案；实际满支持输入甚至无需整数指派。','',
      '## 下溢表示合同必须区分',
      '内部保留所有合法有限logP。普通接口对logP<−700输出舍入0，但该位置不从目标/支持删除。',
      '每行遗漏质量≤舍入数×exp(−700)，总量为各行和，门分别1e-280/1e-276。没有epsilon或角色截断。',
      f"全544矩阵最大总遗漏质量上界{a['maximum_global_tail_mass_bound']:.9g}；6个大图案例有舍入，另538个无此舍入。",
      '上下文误差界为每行遗漏上界×max_j||value_j||∞；这是任意有限角色value的可审计条件界，不包含后续网络误差。',
      '小图528分配无尾部舍入，因此132个实际模型读取的尾部上下文误差界为0；另独立核对FP32转换与乘法界。',
      f"观察到的上下文FP32算术最大误差{pred['observed_context_float32_error_max']:.9g}，在所声明算术界内；不是尾部误差。",
      '不能把6个新合同通过写成原严格普通概率接口通过。内部忽略尾部的梯度误差每列≤总遗漏质量上界，Hessian每元素≤其3倍。','',
      '## 正确性、最优性和失败',
      '最终35项CPU测试零失败/跳过，包含原缩放21项，以及梯度/Hessian有限差分、节点/列置换、行列常数、',
      '支持穷举、分量、尾部、同目标非DP反例、80位高精度、STOP/RNG、JSON和导入回归。',
      '独立矩阵审计重建logP−L的行/列可加势，重新核验行列约束、log概率归一化、普通输出及遗漏上界，不依赖success字段。',
      f"最大独立平稳残差{a['maximum_independent_stationarity']:.9g}。所有原缩放收敛的{a['same_target_pairs']}矩阵同目标比较通过，",
      f"最大P差{a['max_same_target_P_difference']:.9g}，小于事前1e-6审查标记；目标值差在预定相对1e-8内。",
      '预定高精度3案例（4:1二阶、三阶中等、80强集中）80位mpmath核对见high_precision.json。',
      '|高精度案例|Newton最大P误差|原缩放最大P误差|','|---|---:|---:|']
    for p in read(R/'high_precision.json')['cases']:lines.append(f"|{p['case']}|{p['newton_max_abs']:.6g}|{p['scaling_max_abs']:.6g}|")
    lines+=['','|范围|原缩放通过|Newton原接口|Newton新尾部接口|仍失败|','|---|---:|---:|---:|---:|']
    for name,rows in [('小图',small),('大图',large)]:
        lines.append(f"|{name}|{sum(x['input']['old_scaling']['status']=='converged' for x in rows)}/{len(rows)}|{sum(x['result']['status']=='converged_original' for x in rows)}|{sum(x['result']['status']=='converged_bounded_tail' for x in rows)}|{sum(not x['result'].get('new_contract_pass',False) for x in rows)}|")
    lines+=['','原102个矩阵的容量残差失败全部得到解决；不只是删除下溢拒绝。新最大行/列残差如下：',
      f"行{max(x['result']['row_residual'] for x in allr):.9g}，归一列{max(x['result']['column_residual'] for x in allr):.9g}。",
      '初次面板因NumPy bool不能JSON序列化退出，attempt01包含日志/回执/初始输出/原freeze；只修序列化后同一账本重试。',
      '两辅助统计脚本又遇同名历史panel.py导入冲突，失败日志保留，改为直接导入原scaling并补回归。',
      '以上未改变求解器、参数、输入或时钟；没有数值失败后换方法、温度、初值或删难例。','',
      '## 完整模块成本：首次与重复，不比较迭代次数冒充加速',
      '|范围|原缩放首次合计/重复合计 秒|Newton首次合计/重复合计 秒|Newton首次中位/P90/最坏 ms|',
      '|---|---:|---:|---:|']
    for scope in ['small','large']:
        rr=[x for x in ref if x['scope']==scope];cost=a['costs'][scope]
        lines.append(f"|{scope}|{sum(x['first']['seconds'] for x in rr):.4f}/{sum(x['repeat']['seconds'] for x in rr):.4f}|{cost['first']['total']:.4f}/{cost['repeat']['total']:.4f}|{cost['first']['median']*1000:.3f}/{cost['first']['P90']*1000:.3f}/{cost['first']['max']*1000:.3f}|")
    lines+=['首次及重复均重新初始化、包括支持分析/方向/线搜索/输出/独立残差验收，不是只计Cholesky或Newton迭代。',
      '旧缩放成本来自本票同544矩阵重新首次/重复，旧失败也保留。不能将失败耗时解释为完成读取吞吐。',
      '全部Newton首次支持分析合计'+f"{sum(v['support_seconds'] for v in [x['result'] for x in allr]):.6f}秒；",
      '小图迭代中位/最大4/14、大图6.5/13；函数评估最大39/29，逐矩阵全部保存。',
      '矩阵加载/哈希、16个LL重建、原缩放再核验、IO、测试及交付额外计入唯一总账，不包含在solve单次计时内。',
      '当前输入已是固定LL；不能把约0.68秒的大图16矩阵求解总和冒充完整度似然构造或大图NN成本。',
      f"读取面板单进程峰RSS{a['worst_process_rss_mib']:.2f}MiB，包含库导入；预定大矩阵tracemalloc最大{max(x['traced_peak_bytes'] for x in a['memory'])/1024**2:.2f}MiB，输入预载，含输出，不等于进程RSS。",
      'NumPy/SciPy底层工作区可能不完全进入tracemalloc，整体同时峰未测；没有大DP或1000时刻预缓存。','',
      '## 原132个Xt上的固定权重预测（不采样）',
      '三个广覆盖final6000严格加载和原来源绑定通过；只读已验收P作同一LL的缓存输入，重新计算LL/频数逐位核对。',
      '原4块查询摘要核对，Torch/NumPy RNG前后不变；原模型/编码/预测头/Bayes后验完全不改。',
      '这132个状态全部完成，补上原缩放拒绝的55状态。缓存只省重复求解，不能将NN计时当完整端到端读取。',
      '|t|配对数|最大p0差（vsDP）|最大q差（vsDP）|MG预计边数差均值|','|---|---:|---:|---:|---:|']
    for t in (1,8,32,64):
        rr=[x for x in pred['rows'] if x['t']==t];rels=[x for r in rr for x in r['relations']];mg=[x for x in rels if x['family']==8]
        lines.append(f"|{t}|{len(rr)}|{max(x['p0_max_abs'] for x in rels):.9g}|{max(x['q_max_abs'] for x in rels):.9g}|{np.mean([x['expected_edge_delta'] for x in mg]):+.9g}|")
    lines+=['同熵目标的数值问题改善，没有消除近似目标与DP的固有差异；t32约.122的p0差仍存在。',
      '小图低t的q差较小不保证完整轨迹/拓扑统计保持；没有把概率期望当实际生成分布。','',
      '## 范围、依赖与下一步',
      'Python3.9.23、NumPy1.26.4、SciPy1.11.0、mpmath1.3.0、Torch2.1.0、PyG2.3.1、pytest7.4.4，单CPU线程/作业。',
      '原权重、旧实现和结果均保留，main不改。所有链0、训练0、GPU/BW0、大图网络0。',
      '唯一3600秒账本从1792552133.6867504首次准备开始，失败/修复/LL重建/复验/发布均计入；最终总墙钟见publication_receipt。',
      '结论仅限：本固定面板上同一熵目标能更稳定、以更低求解成本求解，6矩阵需显式新输出合同。不是DP等价或质量认证。',
      '建议下一票独立批准45分钟CPU固定权重完整生成对照：3套广权重×DP/Newton×11目标×8配对seed=528条新T64/R4链，',
      '参考原精确路径约18分钟生成及本票读取消耗估计留出预检/评价/异常保存余量；真实新状态成本仍可能变化。',
      '必须配对先验、查询和逐槽uniform，统计实际图角色/边数/尾部/闭合/连通/同构重复；分歧如实报告，不筛图补抽。',
      '此处仅建议和预算，不实现/启动下一票；也不自动大图NN、训练、温度搜索或另一求解器。']
    ROOT.mkdir(exist_ok=True);(ROOT/'RESULTS.md').write_text('\n'.join(lines)+'\n');print('report wrote',len(lines),'lines')
if __name__=='__main__':main()
