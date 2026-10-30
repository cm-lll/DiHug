# 下午汇报交付

先看[一页中文摘要](SUMMARY_ZH.md)、[分seed完整报告](REPORT.md)和[指标目录](METRIC_DIRECTORY.md)。全部生成图均为对应更新数EMA0.99。

完整数据为48张原图和48份LCC派生评价：IMDB3000/6000各8图；PubMed1000/2000/3000/6000各8图。早期PubMed16图全部完成，IMDB没有1000/2000生成。480条固定验证与48000更新记录另列，不以p0替代生成图。

下载目录：`releases/count_aux_results_closeout_20261030/`。完整ZIP超过GitHub单文件限制，按字节分为两卷；没有删指标或训练日志。下载两个part、SHA256SUMS.txt，然后在同一目录执行：

```bash
cat COUNT_AUX_full_results_48graphs_final.zip.part01 COUNT_AUX_full_results_48graphs_final.zip.part02 > COUNT_AUX_full_results_48graphs_final.zip
sha256sum -c SHA256SUMS.txt
unzip -t COUNT_AUX_full_results_48graphs_final.zip
```

如果只下载PubMed修复子包，单独核对SHA256SUMS.txt中对应的`PubMed_1000_2000_3000_6000_repaired_final.zip`行即可；它可直接解压，不需要分卷。该子包包含32原图/32LCC、四阶段全指标；全局审计及摘要中的IMDB项是关联参照。

完整ZIP内有逐图CSV/JSON、逐seed均值、同期配对差、同轨迹阶段差、类别闭合/元路径/度谱/联合度细表、原图与LCC、映射和来源绑定。每张图先算参考误差再平均，不混用两个训练seed，不平均成功子集，null有原因。

旧IMDB独立归档仍保留在原路径，SHA见report_repair_receipt.json。新结果中的科学完成、报告修复与发布分别记时；旧PubMed字段错误与所有账本不改写。CPU合并时数个失败尝试及极小的重复报表求和尾差均有记录，没有更改模型验收容差或指标数值。

固定科学实现未改变。本次新增训练、模型前向、生成均为0；没有新的审批或后继任务入口。大型权重不公开，只登记文件路径、身份和哈希。当前没有跨指标统一优胜者，建议完整图验证保留CE匹配对照和数量辅助候选，仍需另行授权。
