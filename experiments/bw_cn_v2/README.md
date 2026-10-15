# BW CN v2：仅修复审查发现的工程问题

本目录是对 CN v1 审查意见的两项修复。原 `experiments/bw_cn_v1/`、提交 `4783258` 和原 ZIP 全部保留；不改变模型结构、训练目标或采样分布，不启动正式训练／科学采样。

## 两项修复

1. **最终 logits 与概率一致**：CN 的 legacy API 直接返回 canonical upstream 读出的最终 logits/p0，不再用 backbone hook 截取遗漏 CN 增量的 logits。同类辅助诊断接口一并修正。非零 CN 权重反例可复现 v1 的 softmax/p0 误差 0.00927098，v2 为 0；返回概率变化为 0。CN 的 legacy/upstream 是同一实现的两个入口，日志明确标注，不能作为独立实现的正确性证明。
2. **幂等收尾恢复**：最后一个反向步已提交后，允许从未导出、已导出但无 terminal summary、待评价，以及评价失败状态恢复。已有 graph.npz 必须逐字段匹配最终持久状态、权重指纹、seed、绑定及最终检查点哈希；不符立即失败、不覆盖。终态恢复不调用 reverse_step、不更换随机数、不重新采样。完整且评价完成的结果拒绝重复执行。

## 审查入口

- `package/cn_v1_to_v2.diff`：相对已交付 v1 的完整改动，含版本与测试。
- `package/hsgd/runtime.py`、`perf_checks.py`：诊断读出修复。
- `package/hsgd/cn_sampling.py`：最终状态／图证据核对及幂等导出。
- `package/hsgd/cn_entry.py`：收尾状态恢复与可重试评价。
- `package/tests/test_review_regressions.py`、`finalization_worker.py`：非零分支、真实进程硬退出、评价失败及篡改拒绝测试。
- `acceptance/`：CPU 测试、旧版本最小反例、资源与冻结内容核对。
- `release/`：ZIP 及 SHA256；`review_provenance.json` 绑定原始源码提交与逐文件哈希。

## 方法与证据边界

8 层／8 头，隐藏宽 256/64/128、FF 256/64/256，AdamW lr2e-4、weight_decay1e-12、AMSGrad=False，microbatch1，无累积，FP32 网络／float64 概率核，T1000／20 块不变。新增 CN 参数仍为 10116（0.155%），不能用参数比例代替计算开销。

源码完整 55 项 CPU 测试通过（228.10 秒），含原有 41 项和新增 14 项；墙钟229.28秒，CPU219.49秒，峰值RSS约1044.81MiB。新 ZIP 展开后64项哈希通过，部署副本另执行非零CN读出（含空查询）及真实硬退出恢复的3项回归，通过。展开源码与已测源码逐字节一致。

这些仅为 CPU 工程检查。BW 算子、真实 IMDB 训练／一步20块的速度和显存尚未实测；正式训练和生成质量没有新结论。不复用旧 v1 检查点绕过 v2 的源码／版本绑定。

## 下一步仅运行 BW check

下载 `release/dihug_sparse_bw_cn_v2.zip` 及旁边 SHA256SUMS，上传 notebook 持久目录，在终端运行：

```bash
set -euo pipefail
cd /root/private_data/gnn_projects
sha256sum -c SHA256SUMS
unzip -n dihug_sparse_bw_cn_v2.zip
cd dihug_sparse_bw_cn_v2
sha256sum -c SHA256SUMS
bash run.sh check --output /root/private_data/gnn_projects/bw_cn_v2/check01
```

外层 SHA256SUMS 对应 release 的 ZIP；内层对应包内文件。check 子进程限时570秒，输出两臂成本证据。先检查真实耗时与显存，再决定正式训练；本次上传不自动训练或启动完整生成链。训练／停止／恢复／手动采样接口说明见 `package/说明.txt`，此次不执行。
