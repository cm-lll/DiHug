"""Build factual report from receipts, no model/sampling."""
import json,statistics,time
from collections import Counter
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
R=Path('/data2/lyh/gnn_project/DiHug/results/role_logscale_20261021_v1')
def read(p):return json.loads(p.read_text())
def main():
    x=read(R/'summary.json');a=read(R/'audit_guard.json');assert a['status']=='passed' and a['posterior_audited_all132']
    allr=read(R/'all_family_differences.json');allocation=read(R/'all_allocation_differences.json')
    names=['balanced','extreme','dev_single','dev_double','train_06','combo_1','combo_2','combo_3','value_0','value_7','value_11']
    lines=['# 容量约束对数行列缩放：CPU候选审查结果','',
      '**实现的支持/数值拒绝合同检查通过；当前候选不适合作为精确DP的直接替代。**',
      '132个真实旧链Xt中77个完整预测配对通过，55个低噪声状态未收敛并拒绝；没有回退或放宽门。',
      '所有132状态的完整来源和数组均已保存，包括失败时未接受的迭代矩阵；没有用其推进新链。','',
      '## 数学和固定合同',
      '对完整联合角色似然L及整数频数c，最大化 <P,L> − ΣP logP，约束P≥0、P1=1、Pᵀ1=c；熵系数固定1。',
      'float64、零列势初始化、行/列常数规范化；每次先行后列，最多1000完整迭代；行绝对误差和列误差/max(1,c)均≤1e-8。',
      '不可能位置−inf保持0；零频数列移除再恢复；支持不可行由独立整型流区分。最多2m元素和256MiB估算workspace，不创建任何大DP表。',
      '下溢及数值范围超限拒绝，不加epsilon、不改温度、不截断角色、不硬分配、不改预测或生成规则。',
      '这不是原容量分布精确边际或图后验。n=2、c=(1,1)，对角似然4、非对角1：DP对角16/17=.941176，缩放对角4/5=.8。',
      '两者行列和完全一致仍有.141176固有差异；软频数约束不等于生成图硬度配额。','',
      '## 权重、验证与测试',
      '基线7b45d502e34b283192e19212c343b79a5910298a，三广覆盖6000权重严格加载并与完整6000检查点逐张量核对，',
      '旧源码/配置/数据及实际Python、Torch、PyG、NumPy、SciPy等绑定均通过；没有重训或导入短测权重。',
      '|训练seed|权重SHA256|原固定验证最大差|','|---|---|---:|']
    for s in (2026102011,2026102012,2026102013):
        b=read(R/f'seed{s}/binding.json');v=read(R/f'seed{s}/validation.json')
        lines.append(f"|{s}|{b['weight']['sha256']}|{v['maximum_difference']}|")
    lines+=['','21项最终完整CPU回归零失败/跳过：单角色/整数频数/均匀/强集中/零支持/稀疏可行/Hall不可行/边界慢收敛、',
      '节点和角色置换、行列加常数、下溢/数值范围、零频数、RNG不变、KKT、重复列频数合并、适配器异常恢复、DP穷举及非等价反例。',
      '可行三角支持例[[0,0],[-inf,0]]在1000次仍不满足行残差，是可行但未收敛，不伪称支持不可行。',
      '初版14项、数值范围/资源补充16项、原完整19项及交付21项原始记录全部保留，不能将早期通过代替最终集合。',
      '交付前防御审查增加非有限势/logP及目标值的立即拒绝，防止失败结果污染日志；不改数学或数值门。',
      '初始scaling.py原字节及冻结SHA保留；全部528实际似然矩阵的新旧输出逐位相同、状态/迭代一致（无NN调用）。',
      '大图相同16矩阵重新成本测量另存在guard_cost；本报告大图数字采用交付版本，原记录不覆盖。','',
      '## 同真实状态比较',
      '11旧目标×3模型×原第一个采样seed2026102031×t1/8/32/64。旧产物没有全Xt，先重放33条原路径，',
      '核对全部64步state/query/uniform/RNG及旧末图哈希；为失败载荷补存再次重放相关旧路径，也完整核对原哈希。',
      '这是只读重放，不增加独立科学样本。exact/scaled使用同Xt、时间、条件和完整4块计划；无抽样或更新。',
      '|时刻|通过/请求|最大逐行TV|最大p0差（全关系）|最大q差（全关系）|MG期望边数差均值（近似−DP）|',
      '|---|---:|---:|---:|---:|---:|']
    for t in [1,8,32,64]:
        z=[r for r in allr if r['t']==t];b=x['by_time'][str(t)]
        lines.append(f"|{t}|{b['completed']}/33|{b['max_allocation_TV']:.6g}|{max(r['p0_max_abs'] for r in z):.6g}|{max(r['q_max_abs'] for r in z):.6g}|{b['MG_expected_shift']['mean']:+.6g}|")
    lines+=['','最大角色上下文差.095421、最大软目标关系度差1.496607；所有值都是同状态预测差异，不是实际采样图分布误差。',
      't32全部33状态通过，但MG最大p0差.121504，超过事前描述性.01敏感性标记；原模型确实会响应替换后的读取。',
      '对应q最大差约.006517，不能以单步较小q差保证全链相同或质量保持；t64是角色似然无信息端点，分配和预测一致。',
      '完成子集MG期望漂移很小且多数略向下；55拒绝状态无近似网络预测，因此不能宣称没有重新引入MG过连。',
      '每关系/时间/目标/seed的p0/q、出生/消失/净变化见all_family_differences.json；全部allocation、熵、残差和输入LL保留。','',
      '### 拒绝状态清单（列内数字为t；空表示四时刻均通过）',
      '|目标|2026102011|2026102012|2026102013|','|---|---|---|---|']
    for n in names:
        cells=[','.join(str(t) for t in [1,8,32,64] if any(y['condition']==n and y['seed']==s and y['t']==t for y in x['failed_states'])) or '—' for s in (2026102011,2026102012,2026102013)]
        lines.append('|'+n+'|'+'|'.join(cells)+'|')
    lines+=['','全部55状态支持可行，仅未达迭代门；t1有31、t8有24。失败payload中的所有类型都重算并保存，但未接受矩阵不进入网络。',
      '拒绝不是近似与DP不同造成的判错，也没有通过行归一化掩盖列频数误差。','',
      '## 小图成本（输入/精度不变）',
      '|时刻/已通过子集|DP首次分配中位ms|缩放首次分配中位ms|缩放最大迭代|','|---|---:|---:|---:|']
    for t in [1,8,32,64]:
        z=[v for v in allocation if v['t']==t]
        lines.append(f"|{t}|{statistics.median(v['exact']['seconds'] for v in z)*1000:.3f}|{statistics.median(v['scaled']['seconds'] for v in z)*1000:.3f}|{max(v['scaled']['iterations'] for v in z)}|")
    cost=read(R/'small_cost_full.json');L=cost['likelihood'];mem=cost['memory']
    lines += [f"完整seed11共{len(L)}个状态的四类型度似然：冷构造中位{statistics.median(v['cold_likelihood_seconds'] for v in L)*1000:.3f}ms，单时间缓存复用中位{statistics.median(v['warm_likelihood_seconds'] for v in L)*1000:.3f}ms。",
      '重复求解各3次保存在逐状态分配回执；初次诊断分配的LL已在恢复旧链时缓存，不冒称完整冷启动读取。',
      '网络实际前向和构图/传输等runtime计时均保存在逐状态exact_timing/scaled_timing，network_seconds包含一次读取，不是纯Transformer核时间。']
    for method in ['exact','scaled']:
        z=[v['traced_peak_bytes'] for v in mem if v['method']==method]
        lines.append(f"{method}：三预定目标t8、四类型共12次traced函数分配峰值范围{min(z)}…{max(z)}字节（包含输出，不含输入解压，不是进程RSS）；trace下耗时另列，不与低干扰计时混用。")
    lines+=['初版small_cost.json只有27个已通过状态且解压进入trace；已保留，并用small_cost_full.json补齐44状态及预载输入测量，未改算法。',
      '上方首次/重复/trace及网络分解属于原配对诊断版本；交付防御检查版对全部528个LL的独立算法耗时另见numerical_guard_equivalence，不把两版计时混称。',
      '小图高噪声缩放较快，低噪声已通过子集反而更慢，失败状态还承担1000次迭代；不能给统一加速结论。','',
      '## 完整大图库存读取模块：真实前向噪声，不是模型概率或生成链',
      '数据SHA/schema/完整角色向量频数逐项核验。IMDB1653节点4079边，PubMed1565节点12284边。',
      'T1000、固定t20/500和noise seed2026102141；只构造一时刻度PMF缓存。没有实例化大图网络或建立大DP。',
      '|数据/时刻|类型 n×k|缩放状态|迭代|行残差|列残差|下溢数|冷LL秒|缩放秒|',
      '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for name in ['IMDB','PubMed']:
        data=read(R/'guard_cost'/f'{name}_reader.json')
        for r in data['rows']:
            z=r['scaling'];lines.append(f"|{name}/{r['t']}|{r['type_name']} {r['n']}×{r['k']}|{z['status']}|{z['iterations']}|{z['row_residual']:.3g}|{z['column_residual']:.3g}|{z['underflow_count']}|{r['loglikelihood_seconds']:.3f}|{z['seconds']:.3f}|")
        lines.append(f"{name}子进程墙钟{data['wall_seconds']:.3f}s，含两时刻冷目标/LL构造、支持、缩放和各2次重复缩放；峰RSS{data['peak_rss_mib']:.2f}MiB。")
    lines += ['','16个类型/时刻中仅5个满足全门：IMDB t500四类、PubMed t500 Gene。其余11个拒绝，其中6个还有数值下溢。',
      '最大实际矩阵643×277=178111元素，float64单矩阵约1.36MiB、128字节/元素workspace估算约21.74MiB；内存门通过。',
      '当前先阻塞的是低噪声缩放收敛与数值动态范围，而非容量DP式组合内存；完整度似然卷积和多次缩放仍有明显CPU成本。',
      '这些数据包含真正完整度似然，并非人工矩阵压力测试；但没有大图NN前向，不能据此承诺大图生成速度。','',
      '## 资源、审计与决策',
      '唯一3600秒账本从1792547195.9825604（北京时间2026-10-21 09:46:35.983）起计，旧3h预算及截止无关；交付核验后另存最终累计回执。',
      '初始三诊断子进程各约165…167秒、单进程峰RSS672.23MiB；失败载荷补录约147秒并行，交付版大图读取耗时和RSS见上方实际回执。',
      '子任务run计时不含顶层库导入，父进程时限及本票总墙钟包含全部启动。原jobs.seconds是整批观察窗口上界，包含较早退出子进程的等待时间，不冒充其实际计算时间。',
      '大图RSS含Python/Torch库导入，不能当作纯矩阵工作区；整体并发峰及CPU忙时未独立采集。',
      '三CPU单线程工作进程，GPU0、训练更新0、新独立生成样本0、近似完整链0；全部子进程已清理回收。',
      '全部132载荷/行列残差/支持/实际posterior/时刻1的q=p0/旧路径哈希及16大图身份审计通过；审计通过不等于55失败状态通过。',
      'Python3.9.23/Torch2.1.0/PyG2.3.1/NumPy1.26.4/SciPy1.11.0/NetworkX2.8.7/pytest7.4.4，实际环境保存在每seed绑定。',
      '实现正确性：规定的支持、对称性、KKT及拒绝合同检查通过；适合作为替代：当前不通过，不能无损切换。',
      '差异分为：解析反例证明的固有目标差别；55小图及11大图矩阵的数值失败；已收敛情况下学习网络对上下文变化的响应。',
      '没有完整新链，不能把预测差直接判定成最终质量退化或多样性损失。',
      '不建议下一步直接完整生成。当前低t会中断，无法可靠外推整条近似链耗时；若后续审查批准继续，建议另给60分钟纯工程数值票，',
      '仍用原132状态/16读取矩阵固定面板，先解决同一目标的数值可用性再议完整生成；这只是预算建议，本轮不实现或执行。',
      '本票完成后停止，不调温度、不另选算法、不微调或重训、不访问BW/notebook，也不自动运行完整对照。']
    ROOT.mkdir(exist_ok=True);(ROOT/'RESULTS.md').write_text('\n'.join(lines)+'\n');print('report',len(lines),'lines')
if __name__=='__main__':main()
