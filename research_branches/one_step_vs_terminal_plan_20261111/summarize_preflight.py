"""Summarize measured engineering costs without claiming training efficacy."""
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/one_step_terminal_gpu_preflight_20261111'
def main():
    rows=json.loads((OUT/'rows.json').read_text()); release=json.loads((OUT/'release.json').read_text())
    verify=json.loads((OUT/'verification.json').read_text());done=json.loads((OUT/'complete.json').read_text())
    costs={}
    old=json.loads((ROOT/'results/one_step_vs_terminal_plan_20261111/budget.json').read_text())
    for ds in ['IMDB','PubMed']:
        rr=[r for r in rows if r['dataset']==ds]
        f=float(np.median([r['forward_seconds'] for r in rr])); sc=float(np.median([r['score16_seconds'] for r in rr]));b=np.median([r['VJP_seconds'] for r in rr],axis=0)
        costs[ds]=dict(conditions=len(rr),median_forward_seconds=f,median_score16_seconds=sc,median_VJP_seconds=b.tolist(),
            max_cached_p_difference=max(r['cached_p_max_abs'] for r in rr),
            S1_core_seconds_estimate=f+sc/16+float(b[0]),S16_core_seconds_estimate=f+sc+float(b[1]),
            eager_4chains_8times_seconds_historical_estimate=4*old['historical_chain_seconds'][ds]+32*(f+float(b[2]))+sc/4,
            caveat='No optimizer/IO/validation costs included; S1 score extrapolated from16. Old chain timing contains transition probes: conservative mixed-instrumentation estimate, NOT fresh fullchain throughput.')
    result=dict(status='engineering_complete',forwards=done['forwards'],VJPs=done['VJPs'],
        updates=0,new_reverse_chains=0,conditional_clean_draws=done['conditional_clean_draws'],
        GPU_hours=release['GPU_hours'],wall_seconds=release['wall_seconds'],
        peak_allocated_MiB=done['peak_allocated_bytes']/2**20,peak_reserved_MiB=done['peak_reserved_bytes']/2**20,
        costs=costs,verification=verify['status'],decision='Proceed to freeze matched pilot and resource budget; no effect winner or final training-size claim.',
        unresolved=['Matched complete optimizer/RNG start for all arms','Final reward scale, clipping and trainable parameters',
                    'Online recovery and optimizer-update preflight','Equal-resource pilot size and new complete-chain timing',
                    'Saved trajectory states are not forward-noise inputs: cannot compare learning quality here'])
    (OUT/'summary.json').write_text(json.dumps(result,indent=2))
    lines=['# 三方案接线与成本检查','',
        '三臂固定名称：S1一步单硬图；S16多样本相对奖励加权反馈；T-eager完整链终点反馈适配。S16各图分别评分，使用留一基线，不挑赢家；本轮无标准差归一化/奖励截断，不冒称原GDPO完整复现。',
        '',f"已完成64次前向、192次VJP、1024份条件干净硬图抽样；0参数更新、0新完整链。GPU{release['gpu']}占用{release['GPU_seconds']:.2f}秒/{release['GPU_hours']:.5f}小时，含加载、CPU评分等待及保存；墙钟{release['wall_seconds']/60:.2f}分钟，峰值分配{result['peak_allocated_MiB']:.1f}MiB。进程退出与释放见release.json/gpu_after.txt。",
        '', '全部模型参数/缓冲区与全局Python、NumPy、Torch CPU/CUDA RNG未改变；局部抽样使用独立固定SeedSequence。schema活动支持IMDB13120、PubMed18336，无向槽各一次；保留逐图成本、逐类型联合误差。S1与LOO有限枚举/有限差分独立检查通过。',
        '', '共享异质评分为类型内同节点联合关系度×热传播核MMD，加逐关系数量平方误差；本轮两项单位系数仅作工程约定。T-eager用最终图为各保存Xt的clean-head目标；这不是旧transition score，也不是无偏终点策略梯度。三者共享一次前向只为计时，非正式训练曝光安排。',
        '', '## 实测及外推', '', '| 数据集 | 前向秒 | 16图评分秒 | 单次VJP秒（S1/S16/eager） | S1核心秒估计 | S16核心秒估计 | 完整链K4×8时点分钟估计 |', '|---|---:|---:|---|---:|---:|---:|']
    for ds,c in costs.items():lines.append(f"| {ds} | {c['median_forward_seconds']:.3f} | {c['median_score16_seconds']:.3f} | {'/'.join(f'{v:.3f}' for v in c['median_VJP_seconds'])} | {c['S1_core_seconds_estimate']:.3f} | {c['S16_core_seconds_estimate']:.3f} | {c['eager_4chains_8times_seconds_historical_estimate']/60:.2f} |")
    lines+=['','S1/S16未计Adam、恢复、验证、保存；完整链沿用旧严格模式带诊断的历史成本，不是本轮新整链实测。不能据此承诺正式吞吐，更不能把三臂相同更新数称为公平计算预算。',
        '', '## 结论边界与下一步', '',
        '工程反馈路径可计算且有限。既有概率与重算概率微差逐状态保留，没有修改原重算容差或宣称逐位复现。该面板使用旧自由生成Xt，无新前向加噪训练输入，也未测试Adam更新/恢复，不能把工程通过写成已完成训练接线。',
        '', '下一步唯一方向：冻结共同父状态、优化器/奖励处理及等GPU预算的有界三臂学习试点，纳入必要更新/恢复检查；完整链收集成本按实际重新校准。无需增加“短时必须改善”的效果门槛，但不在未绑定正式资源与训练协议时启动长训。没有三方案效果赢家。']
    (OUT/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n');print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
