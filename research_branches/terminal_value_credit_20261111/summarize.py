"""CPU report and real-asset readiness audit; never imports a production model."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import json,hashlib,csv,subprocess,datetime,time,resource
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/terminal_value_credit_20261111';BASE=ROOT/'results/terminal_state_baseline_20261111'
def dump(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
rows=json.loads((BASE/'estimates.json').read_text())+json.loads((OUT/'estimates.json').read_text());summ=[];by=[]
for method in ['LOO_reuse','exact_population_constant','exact_conditional_value','exact_value_credit']:
    for K in [32,128,512]:
        x=[v for v in rows if v['method']==method and v['K']==K]
        a=dict(method=method,K=K,n=len(x),relative_gradient_RMSE=float(np.sqrt(np.mean([v['relative_error']**2 for v in x]))),true_all_improve=sum(v['true_all_improve'] for v in x),true_any_harm=sum(v['true_any_harm'] for v in x))
        a['neither']=a['n']-a['true_all_improve']-a['true_any_harm'];summ.append(a)
        for seed in [2026110901,2026110902,2026110903]:
            for step in [0,20]:
                y=[v for v in x if v['seed']==seed and v['update']==step];z=[v for v in rows if v['method']=='LOO_reuse' and v['K']==K and v['seed']==seed and v['update']==step]
                by.append(dict(method=method,K=K,seed=seed,update=step,MSE_ratio_vs_LOO=float(np.mean([v['relative_error']**2 for v in y])/np.mean([v['relative_error']**2 for v in z])),true_all_improve=sum(v['true_all_improve'] for v in y),true_any_harm=sum(v['true_any_harm'] for v in y),n=len(y)))
for filename,data in [('aggregate.csv',summ),('by_state.csv',by)]:
    with (OUT/filename).open('w') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
decision={}
for method in ['exact_conditional_value','exact_value_credit']:
    b=next(x for x in summ if x['method']=='LOO_reuse' and x['K']==128);a=next(x for x in summ if x['method']==method and x['K']==128);ratio=(a['relative_gradient_RMSE']/b['relative_gradient_RMSE'])**2
    each=all(v['MSE_ratio_vs_LOO']<1 for v in by if v['method']==method and v['K']==128)
    decision[method]=dict(pooled_MSE_ratio=ratio,all_states_improve=each,engineering_screen=bool(each and ratio<=.8 and a['true_all_improve']>=b['true_all_improve']),production_ready=False)
dump(OUT/'summary.json',dict(aggregate=summ,by_state=by,decision=decision))
src=ROOT/'results/terminal_structure_real_20261109T233600';proto=json.loads((src/'protocol.json').read_text());assets=[]
for name,expected in proto['bindings'].items():
    p=Path(name);actual=sha(p) if p.exists() else None;assets.append(dict(path=name,expected=expected,actual=actual,match=actual==expected))
chains=[];datasets=[]
for ds,model in proto['models'].items():
    d=np.load(model['data']);ed=d['edges'];ed=ed.T if ed.shape[0]!=2 else ed
    datasets.append(dict(dataset=ds,nodes=int(len(d['types'])),stored_edge_columns=int(ed.shape[1]),unique_undirected_edges=int(len(set(tuple(sorted(map(int,e))) for e in ed.T))),train_seed=model['train_seed'],readout=model['readout'],identity=model['readout_identity'],updates=model['updates']))
    for ss in proto['sample_seeds']:
        p=src/ds/str(ss);identity=json.loads((p/'identity.json').read_text());snaps=sorted(p.glob('step_*.npz'))
        assert identity['readout_sha256']==proto['bindings'][model['readout']] and identity['T']==1000 and identity['updates']==3000 and len(snaps)==8
        chains.append(dict(dataset=ds,sample_seed=ss,snapshots=len(snaps),identity_sha256=sha(p/'identity.json'),graph_sha256=sha(p/'graph.npz'),has_score_gradients=(p/'score_gradients.pt').exists(),score_gradients_bytes=(p/'score_gradients.pt').stat().st_size,terminal_targets=1))
try:
    gpu=subprocess.run(['nvidia-smi'],capture_output=True,text=True,timeout=10);gpu_info=dict(returncode=gpu.returncode,stdout=gpu.stdout,stderr=gpu.stderr)
except Exception as e:gpu_info=dict(error=repr(e))
real=dict(checked_at=datetime.datetime.now().astimezone().isoformat(),host=os.uname().nodename,datasets=datasets,chains=chains,bindings=assets,all_existing_bindings_match=all(x['match'] for x in assets),gpu=gpu_info,device_nodes=[str(p) for p in Path('/dev').glob('nvidia*')],actual_this_ticket_GPU_hours=0,new_model_calls=0,reason_not_run='NVIDIA device unavailable; exact conditional terminal values unavailable on real graph. Four trajectories per model provide four independent terminal targets, not 32 independent snapshot targets. Existing seven real costs differ from the five toy costs; cannot silently substitute.',next='First test a cheap independently fitted value estimator against exact toy values with held-out paths; only then allocate a bounded real-graph frozen-model pilot. No new large training.')
dump(OUT/'real_asset_readiness.json',real)
table='\n'.join('|'+ '|'.join(map(str,[a['method'],a['K'],round(a['relative_gradient_RMSE'],6),a['true_all_improve'],a['true_any_harm'],a['neither']]))+'|' for a in summ)
report=f'''# 终点结构反馈：条件基线与逐步预期贡献

本票完成两项CPU配对研究，没有训练/生产模型调用、新轨迹或GPU占用。保持旧三初态×raw0/20、五终点风险、T100、32重复和K32/128/512，所有方法使用同一批历史硬轨迹。192个结果每档来自6状态×32重复，不是192份独立训练。共同下降按精确梯度的一阶斜率评定，不是有限更新或新生成质量。

## 结果

|方法|K|相对梯度RMSE|五项均下降/192|至少一项受损/192|其余|
|---|---:|---:|---:|---:|---:|
{table}

仅减去状态条件基线，K128均方误差比原LOO为{decision['exact_conditional_value']['pooled_MSE_ratio']:.6f}，六状态中一状态变差；没有通过预先固定的投入筛查，不推进真实实现。

替换为逐步预期贡献，K128均方误差比为{decision['exact_value_credit']['pooled_MSE_ratio']:.6f}（下降约{100*(1-decision['exact_value_credit']['pooled_MSE_ratio']):.1f}%），六状态均改善；178/192共同下降，14/192仍有代价。K512的192/192仅是当前有限面板，不能宣称普遍无误。通过的是“值得研究低成本估计”的门槛，不是生产可用或生成效果验收。

## 数学对象

令反向转移Pθ(y|x,t)，s_t=∇log Pθ(X[t-1]|Xt,t)，终点第i项成本C_i(X0)，V_t(x)=Eθ[C(X0)|Xt=x]。初态分布不依赖θ。固定θ，V0=C，Vt=P_t V[t-1]。

原梯度 G=E[Σ_t C(X0)s_t]。因为E[s_t|Xt]=0，可以减去Vt(Xt)，但路径终点的后续随机误差仍在。

条件期望与Markov性给出 E[C(X0)s_t]=E[V[t-1](X[t-1])s_t]，因此同一个G也等于 E[Σ_t (V[t-1](X[t-1])-Vt(Xt))s_t]。本票采用后式，求导时价值表停止梯度；没有改变终点目标、反向核或图支持，也没有把下一步图与干净图直接比较。该替换逐项条件化，但因跨时间协方差，总和方差不自动保证下降，本票是实际检验。

若用估计值Vhat=V+e，状态基线误差项条件期望为0（要求独立拟合或交叉拟合且本次停止梯度），下一状态误差会留下偏差 E[Σ_t e[t-1](X[t-1])s_t]。可用Cauchy–Schwarz给出每成本偏差范数上界 Σ_t sqrt(E[e[t-1]^2] E[||s_t||²])。这不是任意critic无偏定理；全局回归MSE小也不足以保证共同方向正确，尤其少数风险/近零梯度。

此思路对应政策梯度中使用条件价值作反馈的思想：[Sutton等原论文](https://proceedings.neurips.cc/paper/1999/file/464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf)。原论文的兼容函数逼近条件不能直接套用任意图回归器；这里的精确有限状态恒等式由代码单独核验。

## 审计与失败

Stage A：600独立Bayes价值核验/12标量32路径估计核验通过。Stage B：6完整精确梯度期望恒等式、12独立标量路径score累加、576样本均值核验通过。初版审计脚本把dict按数组索引，KeyError，已原样保留；audit_v2仅修审计索引，未重跑worker或更改结果。绑定保持。

## 真实图边界与决定

IMDB A 0411、PubMed A 0421的3000步EMA及数据/源码绑定重新检查；现有8条T1000旧链每模型4条、每链8快照。它们可以复用，但每模型只有4个独立终点，32快照不等于32条独立终点监督。已有7项真实成本与微型5项不相同，不能直接说已通过真实7项验证。

当前主机NVIDIA驱动通信失败、无可用设备节点，未改驱动或占用其他任务。因此本票没有真实网络前向/训练或新生成。更重要的是精确V在真实图不可枚举；即使GPU恢复，也不能把微型oracle直接装进生产模型。

唯一推荐：先把精确V替换为低成本、独立路径拟合的近似V，并在微型图的留出路径上核查偏差、共同方向可靠性与拟合成本；有可靠收益后，再用冻结真实A做有界检验。不得追加K搜索、正式长训或把现阶段称为最终方法。

朴素解释：直接把最终图的好坏反馈给所有步，容易把后面随机发生的好坏也算到前面。新办法试着判断“这一步之后，最终结果的预期有没有变好”。微型图能算准，反馈确实更清楚；真实图能否便宜地估准还没有证明。

## 复现

使用sparse_block环境，依次运行terminal_state_baseline_20261111/run.py、audit.py，以及terminal_value_credit_20261111/run.py、audit_v2.py、summarize.py。worker拒绝覆盖已存在输出目录；复现请在独立工作副本中保留绑定输入、为结果目录使用新位置，不删除原结果。运行耗时见receipt.json及audit.json；正式资产清单见real_asset_readiness.json，逐状态结果见by_state.csv，逐重复原始梯度见estimates.json。
'''
(OUT/'REPORT_ZH.md').write_text(report)
(BASE/'REPORT_ZH.md').write_text('# 时间状态条件基线配对结果\n\n条件基线本身未通过冻结投入门槛：K128相对均方误差下降约0.95%，一个状态恶化；不接真实图训练。\n\n完整配对结果、方法区别、局限与下一项检验见 ../terminal_value_credit_20261111/REPORT_ZH.md。原协议、全部1728逐次估计、600独立Bayes检查与失败边界保留。\n')
for directory in [OUT,BASE]:
    files=[p for p in directory.iterdir() if p.is_file() and p.name!='manifest.json'];dump(directory/'manifest.json',{str(p.relative_to(ROOT)):sha(p) for p in files})
print(json.dumps(dict(decision=decision,real_bindings_ok=real['all_existing_bindings_match'],chains=len(chains),gpu=gpu_info),ensure_ascii=False))
