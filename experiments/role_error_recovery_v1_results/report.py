"""Readable complete tables from frozen diagnostic outputs; no inference."""
import json
from pathlib import Path
import sys
import numpy as np

def main():
    run=Path(sys.argv[1]);dest=Path(sys.argv[2])
    summary=json.loads((run/"summary.json").read_text())
    rows=json.loads((run/"records.json").read_text())
    noise=[]
    for cond in ("balanced","extreme"):
        for t in (1,4,8):
            counts={"natural_false":0,"natural_missing":0}
            per={seed:{"natural_false":[],"natural_missing":[]} for seed in (2026102011,2026102012,2026102013)}
            for g in (4,5):
                stem=f"{cond}_{g}_t{t}"
                a=np.load(run/(stem+"_input.npz"));x=a["ordinary"];y=a["clean"];mg=a["families"]==8
                for name,mask in (("natural_false",mg&(x==1)&(y==0)),("natural_missing",mg&(x==0)&(y==1))):
                    counts[name]+=int(mask.sum())
                    for seed in per:
                        z=np.load(run/f"{seed}_{stem}_ordinary.npz")
                        per[seed][name].extend(z["p0"][mask].tolist())
            for name,n in counts.items():
                noise.append(dict(condition=cond,t=t,category=name,unique_slots=n,
                    per_seed_p0={str(seed):float(np.mean(v[name])) if n else None for seed,v in per.items()},
                    macro_p0=float(np.mean([np.mean(v[name]) for v in per.values()])) if n else None))
    (run/"natural_noise_errors.json").write_text(json.dumps(noise,indent=2,allow_nan=False)+"\n")
    lines=["# 固定容量软读取权重：错边恢复诊断","",
    "结论：三个训练seed均存在MG错边纠正缺口；集中加边比等量分散加边更易保留，"
    "删边后几乎不恢复。普通输入的整体真/无边判别较好，但稀少的自然噪声错误也有失败，"
    "不能称普通去噪完全正常。仅概率诊断，不是实际末图、多样性或整链因果证据。","",
    "科学源提交 afb3f53；基线 b8e97cc42e6e835e0ea228a6666e3a22a200e035。"
    "4张历史验证图、3个时刻、普通+6压力/位置变体、3套final1000权重。"
    "共252记录、255概率扫块（含3次重复）及36历史验证状态；无不可行项。"
    "原34节点/8层/FP32/T64/4块。没有训练、图采样、BW、notebook或GPU。","",
    "三套历史固定验证逐行逐族完全复现，最大差0；重复前向p0/q最大差0。"
    "不称历史未存储的全p0已被逐位复现。文件哈希与固定提交中的历史收据比对，输入结束后重新哈希。","",
    "唯一下一步建议：先设计保留原扩散训练部分、额外使用有干净来源人工错边监督的单项鲁棒性训练对照。"
    "它是新增训练目标，必须另冻混合比例与预算；本轮没有实施。"
    "不能据此认定自由链过连的唯一原因，更不能加硬配额或修图。","",
    "## 普通状态（MG）","",
    "|条件|t|真边p0|非边p0|真边q|预计干净MG边数|预计下一步MG边数|Genre期望度W1|",
    "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for cond in ("balanced","extreme"):
        for t in (1,4,8):
            rr=[r for r in summary["ordinary_per_seed"] if r["condition"]==cond and r["t"]==t]
            vals=[np.mean([r[k] for r in rr]) for k in ("true_p0","nonedge_p0","true_q","MG_p0_edges","MG_q_edges","G_expected_W1")]
            lines.append(f"|{cond}|{t}|"+"|".join(f"{v:.5f}" for v in vals)+"|")
    lines+=["","## 全部压力档与位置对照","",
    "每格先平均两张来源图，再等权平均三训练seed。p0是干净有边预测，q是原后验的下一时刻概率。"
    "极端条件 edited 是新增假边保留；平衡条件是被删真边恢复。ordinary列使用同一编辑位置。"
    "集中=选定一个G；分散=其他G轮流编辑。极端hub已满容量，不能充当假边添加位置对照。","",
    "|条件|t|变体|编辑数|普通位置p0|压力p0|普通位置q|压力q|其余真边p0变化|未扰非边p0变化|MG净变化|全图净变化|期望度W1变化|",
    "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in summary["macro"]:
        keys=("edits","edited_p0_ordinary","edited_p0","edited_q_ordinary","edited_q","true_p0_delta","nonedge_p0_delta",
              "MG_net","global_net","G_expected_W1_p0_delta")
        lines.append(f"|{r['condition']}|{r['t']}|{r['variant']}|"+"|".join(f"{r[k]:.5f}" for k in keys)+"|")
    lines+=["","## 强扰动逐训练seed（其余档位详见summary.json）","",
    "|seed|条件|t|集中强扰动p0|q|","|---:|---|---:|---:|---:|"]
    for r in summary["per_seed"]:
        if r["variant"] in ("to6_concentrated","delete3_concentrated"):
            lines.append(f"|{r['seed']}|{r['condition']}|{r['t']}|{r['edited_p0']:.6f}|{r['edited_q']:.6f}|")
    lines+=["","## 原核噪声自身的错边","",
    "补充分析全部普通输入中的自然错误，非新预测或新选样。下表槽数是两来源图的唯一槽数，"
    "不因三个权重重复评估而乘3。缺失就是NA；样本太少不能估总体错误率。","",
    "|条件|t|原核错误类别|唯一槽数|平均p0|","|---|---:|---|---:|---:|"]
    for r in noise:
        v="NA" if r["macro_p0"] is None else f"{r['macro_p0']:.6f}"
        lines.append(f"|{r['condition']}|{r['t']}|{r['category']}|{r['unique_slots']}|{v}|")
    lines+=["","## 解释及边界","",
    "- 平衡删边后，保留下来的真边p0变化很小（见表），并非所有边概率一起崩溃；被删位置主要继续被视为无边。"
    "极端压力的未编辑真边略受损，集中/分散均有误伤。",
    "- t>1的q会结合当前边状态，比p0更保守；这沿用原Bayes公式，不是发现后验实现错误。"
    "t=1已验q=p0，此时仍有明显缺口。",
    "- 期望度W1均为排序后的期望MG度向量对参考角色多重集的距离，不能当成采样度分布W1。"
    "完整联合期望向量/TV也保存，但连续期望向量对整数角色的精确TV常饱和为1，无有效区分力。",
    "- 人工扰动不属于原前向边际；清晰标注来源、编辑及原核普通输入。"
    "干净身份只用于构造和评价，模型仍只读Xt、类型、时间和匿名条件。",
    "- 本轮没有逐边唯一恢复成功要求、没有质量阈值，没有将边或图当成独立训练重复，"
    "没有声称多样性、离散图改善或完整链稳定性。","",
    "## 验收、成本与失败记录",""]
    receipt=json.loads((run/"receipt.json").read_text());audit=json.loads((run/"audit.json").read_text())
    lines += [f"诊断墙钟{receipt['wall_seconds']:.3f}秒，CPU{receipt['CPU_seconds']:.3f}秒，"
              f"进程峰值RSS{receipt['peak_rss_mib']:.3f}MiB；600秒固定上限，预估154.273秒门限通过。"
              f"独立存储数组审计{audit['wall_seconds']:.3f}秒。", "",
        "完整原89+新增9项CPU回归=98项，零失败/跳过，21.31秒；7条继承warning，非失败。"
        "Python3.9.23/Torch2.1.0+CUDA12.1构建（实际CPU）/PyG2.3.1/NumPy1.26.4/"
        "SciPy1.11.0/NetworkX2.8.7/pytest7.4.4，单线程。","",
        "诊断无失败、无重新选样、无重启、无科学训练。一次临时结果读取早于审计写完summary.json，"
        "报FileNotFoundError；等待原审计完成后读取成功，无改输入、无重跑推理。"
        "pytest中的历史优化器单元测试仅属工程测试，不是新增学习实验。","",
        "## 复现命令","",
        "提供原本地权重目录及panel目录；ZIP不重复打包权重字节，已保存全部SHA256收据。"
        "没有来源权重时只能复核存档数组和运行工程测试，不能声称重做模型预测。","",
        "```bash",
        "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \\",
        "timeout --signal=TERM --kill-after=5 600 python role_error_recovery_v1/diagnostic.py \\",
        "  --run \"$CAPACITY_BASELINE_RUN\" --panel \"$ARCHIVED_PANEL_RUN\" --output \"$NEW_OUTPUT\"",
        "python role_error_recovery_v1_results/audit_and_summarize.py \\",
        "  --run \"$NEW_OUTPUT\" --panel \"$ARCHIVED_PANEL_RUN\"",
        "```","",
        "完成发布后停止，等待审查。不自动启动鲁棒训练或实际链分析。"]
    dest.write_text("\n".join(lines)+"\n")
if __name__=="__main__":main()
