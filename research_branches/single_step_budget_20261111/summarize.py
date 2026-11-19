"""Summarize input/output noise; reuse previously verified exact-three estimator."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import json,hashlib,csv,time,datetime
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/single_step_budget_20261111'
def dump(name,x):(OUT/name).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False))
def table(name,x):
    with (OUT/name).open('w') as f:w=csv.DictWriter(f,fieldnames=list(x[0]));w.writeheader();w.writerows(x)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert json.loads((OUT/'audit.json').read_text())['passed']
# This is an explicitly separate, post-main analysis. Freeze before computing
# any hybrid outcomes; exact components are fixed by existing verified code.
pp=OUT/'hybrid_reuse_protocol.json'
if pp.exists():raise RuntimeError('refuse to overwrite post-analysis protocol')
dump(pp.name,dict(frozen_epoch=time.time(),scope='Secondary reuse analysis, main all-MC outcomes already viewed. No new model, input, hard graph, or update.',choice='Existing exact_hard_moments scheme, hard risk rows 0(count),2(present triangle),3(absent triangle) exact; rows1(joint)4(component) same MC. No outcome-based selection of rows.',purpose='Assess input allocation using the already available hybrid estimator, not reintroduce all-MC as production default.',bindings={str(p):sha(p) for p in [OUT/'estimates.json',OUT/'truth.json',ROOT/'research_branches/exact_hard_moments_20261110/moments.py',Path(__file__)]}))
original=json.loads((OUT/'estimates.json').read_text());truth=json.loads((OUT/'truth.json').read_text());tm={(r['seed'],r['update']):r for r in truth};allrows=[]
for method in ['all_MC','existing_exact3_MC2']:
    for a in original:
        t=tm[a['seed'],a['update']];G=np.array(t['gradient']);norm=np.linalg.norm(G);g=np.array(a['gradient']);oracle=np.array(a['conditional_gradient'])
        if method=='existing_exact3_MC2':g[[0,2,3]]=oracle[[0,2,3]]
        ie=oracle-G;oe=g-oracle;d=-g.mean(0);dn=np.linalg.norm(d);d=d/dn if dn else d;slopes=G@d
        allrows.append(dict(method=method,seed=a['seed'],update=a['update'],repeat=a['repeat'],B=a['B'],K=a['K'],hard_budget=a['hard_budget'],relative_MSE=float(np.sum((g-G)**2)/norm**2),input_relative_MSE=float(np.sum(ie**2)/norm**2),output_relative_MSE=float(np.sum(oe**2)/norm**2),relative_cross=float(2*np.sum(ie*oe)/norm**2),mean_objective_descent=bool(slopes.mean()<-1e-10),any_cost_harm=bool(max(slopes)>1e-10),**{f'slope_{i}':float(s) for i,s in enumerate(slopes)}))
table('per_repeat.csv',allrows);aggregate=[];states=[]
alloc=[(1,64),(4,16),(16,4),(1,256),(4,64),(16,16),(64,4)]
def summary(rows,meta):
    return dict(**meta,n=len(rows),relative_gradient_RMSE=float(np.sqrt(np.mean([a['relative_MSE'] for a in rows]))),input_relative_MSE=float(np.mean([a['input_relative_MSE'] for a in rows])),output_relative_MSE=float(np.mean([a['output_relative_MSE'] for a in rows])),relative_cross=float(np.mean([a['relative_cross'] for a in rows])),mean_descent=sum(a['mean_objective_descent'] for a in rows),any_harm=sum(a['any_cost_harm'] for a in rows))
for method in ['all_MC','existing_exact3_MC2']:
    for B,K in alloc:
        x=[a for a in allrows if a['method']==method and a['B']==B and a['K']==K];aggregate.append(summary(x,dict(method=method,B=B,K=K,hard_budget=B*K)))
        for seed,step in tm:
            y=[a for a in x if a['seed']==seed and a['update']==step];states.append(summary(y,dict(method=method,seed=seed,update=step,B=B,K=K,hard_budget=B*K)))
table('aggregate.csv',aggregate);table('by_state.csv',states)
exact=[]
for a in truth:
    G=np.array(a['gradient']);g=G.mean(0);s=-G@g/np.linalg.norm(g)
    exact.append(dict(seed=a['seed'],update=a['update'],mean_direction_slopes=s.tolist(),any_harm=bool(max(s)>1e-10),input_variance=a['input_gradient_variance']))
dump('summary.json',dict(aggregate=aggregate,by_state=states,exact_average_direction=exact))
tab='\n'.join('|'+ '|'.join(map(str,[a['method'],a['B'],a['K'],a['hard_budget'],round(a['relative_gradient_RMSE'],4),a['mean_descent'],a['any_harm']]))+'|' for a in aggregate)
report=f'''# 恢复单步干净图结构训练：输入与硬输出抽样预算

本轮遵循最新用户选择：随机t及Xt→一次干净图概率预测→多张硬干净图→原图结构误差。完整链研究保留备选，暂停critic路线。此票没有完整链、参数更新或真实网络调用，不把单步目标换成转移目标。

## 实际复用与新工作

旧单步有限K64训练、三精确两采样矩、真实网络短更新已做过，不重做。新问题是固定硬评分预算下，噪声输入B与每输入硬图K如何分配。三旧微型seed2026110901/902/903的typed_endpoint raw0/100，共6状态；每种128独立重复、7种固定分配。不是6独立训练，亦不是5376独立训练。原4节点6槽、10共享参数、5硬成本和均匀t1..100保持，无CE。原参考图身份和输入全部绑定。

精确遍历64输出只用于计算真值和原有可解析风险的对照；抽样估计器不读取真值来修正剩余两项。真实规模不能枚举全部输出。主面板638976唯一CPU硬微图，5376原估计，无逆链；二次复用5376既有三精确两采样估计，不抽新图。二次方案在主面板结果已查看后冻结，明确为开发复用，不独立确认或生成结果挑方法。

## 结果

|估计器|噪声输入B|每输入硬图K|硬图数|相对梯度RMSE↓|均风险下降/768|至少一项代价/768|
|---|---:|---:|---:|---:|---:|---:|
{tab}

以上“下降”仅为相对于精确单步总体梯度的一阶方向，不是执行更新后的变化，不是生成质量。分状态全部记录在by_state.csv，风险分项斜率在per_repeat.csv。0步精确平均方向本来就有错误011三角代价；100步三个精确平均方向五项均下降。有限抽样仍可使弱小分项方向翻转，不能用总分下降冒充全面收益。

原全采样64预算，1×64→4×16在六状态MSE均下降，汇总.418307→.260478，约37.7%减少；256预算1×256→4×64也六状态改善。16/64份输入不是免费：每份都需网络预测。同硬图数并不等同算力、更新次数或输入曝光量。每输入4图的统计更粗，在部分状态有反向取舍，不按最低汇总挑一个“大batch赢家”。

## 推导与含义

J(θ)=E_Z E_Y[C(Y)]，Z=(t,Xt)按原加噪过程独立抽取，Y各合法边按干净预测独立Bernoulli。对固定Z，score=(Y-p)∇θlogit；估计为每输入K份 `(C_j - 其他K-1份的平均成本)score_j` 再对B输入平均。基线不使用本样本，故梯度估计无偏；多目标协调后的方向若另行计算，不再自动无偏。本票只检查平均梯度，不新引入协调器。

记g(Z)为固定输入下精确梯度，G=E_Z g(Z)，g_hat_K条件无偏，则

`Cov(g_hat_B,K) = Cov_Z(g(Z))/B + E_Z[Cov(g_hat_K|Z)]/B`。

增加同一输入的硬图只能压第二项，不能减少第一项。增加独立输入可以压第一项，但需要更多前向，固定BK时每输入K变少也可能增加第二项。逐次误差的有限样本交叉项已保留；不能强行让经验分解的两项完全相加。

“输入噪声”是正常SGD相对总体目标的随机波动，不是某份输入错误；降低它不自动意味着单位GPU时间进展更快。B份输入一次清梯度/一次更新与B次单输入逐次更新也不等价，参数和Adam历史不同。

## 与已有低方差反馈结合

此前已核验关系边数和两类三角硬平方损失可算精确期望，包含方差与共享边协方差；联合度、分量仍硬抽样。二次复用保持这3行准确、其他2行不变，目标没有变成软指标或加边配额。CPU微型条件梯度可直接给出这3行，生产需用exact_hard_moments解析实现，不能从微型表读取真实答案。

## 成本与下一步

一次平均损失更新大致B(F+Bwd)+BK(S+M)，S为硬图结构评分成本；若分项协调需多个VJP另计。前向特征、三项矩计算亦按输入增加，不能只报硬图总数。旧真实评分记录：64硬图全采样约IMDB .300秒/PubMed .843秒；三精确两采样约.249/.450秒，属于历史CPU分项测量，不能直接乘成新端到端GPU吞吐。

当前会话nvidia-smi仍无法通信，本票CPU已完成，不冒称宿主GPU损坏或真实训练已启动。唯一推荐：沿既有三精确两采样估计器，先在四个旧真实A状态比较1×64与4×16的实际成本和有限匹配更新；两组使用预先固定的IID输入及独立硬抽样流，多输入一次更新必须明确记账。先绑定有限预算/恢复语义，不直接扩大batch或开长训。若没有真实成本优势，保留单输入，不能以微型RMSE降幅直接宣布方案更快。

朴素解释：一份噪声输入多抽图，是让“这道题怎么评分”更准；多份噪声输入，是避免这次更新只围绕一道题。我们现在发现两种波动都存在。下一步应该小幅增加看题范围并实测代价，而不是回到每次训练跑完整生成链。

## 复现与核验

运行sparse_block环境下run.py、audit.py、summarize.py。worker拒绝覆盖结果目录；复现用隔离副本/新结果路径。84独立样本协方差/LOO对照、6全输入期望、12输入RNG/768输出RNG核验与6有限差分全部通过。固定源码及旧资产哈希未变。原始估计、真梯度、样本和协议全部保留；无真实模型、EMA或Adam状态变化。
'''
(OUT/'REPORT_ZH.md').write_text(report)
dump('completion.json',dict(status='CPU_complete_no_real_model',finished=datetime.datetime.now().astimezone().isoformat(),updates=0,full_chains=0,GPU_hours=0,production_calls=0,worker=json.loads((OUT/'receipt.json').read_text()),audit=json.loads((OUT/'audit.json').read_text()),production_sources_changed=False,formal_training_started=False))
files=list(OUT.glob('*'))+list(Path(__file__).parent.glob('*.py'));dump('manifest.json',{str(p.relative_to(ROOT)):sha(p) for p in files if p.is_file() and p.name!='manifest.json'})
print(json.dumps(aggregate,ensure_ascii=False))
