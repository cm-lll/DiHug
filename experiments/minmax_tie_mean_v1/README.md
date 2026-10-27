# 三组公共 min/max 并列梯度工程修订（不授权正式训练）

基线：cb75b37b116a69d45aa6e78cab55787308a009db 的 v2。
唯一生产代码修改为 package/hsgd/vendor/layers.py 的六个极值调用以及新 tie_extrema.py；相对补丁 relative_to_v2.diff。
不改 mean/方差、注意力、重复边、初始化、输入、损失、网络规模、合成字典、噪声或后验。
SparseXtoy/SparseEtoy 的所有实例（八层及 CN 复用路径）共用此函数。

## 数学约定

对组 b、通道 j，M[b,j]=max{x[i,j]:batch[i]=b}，K[b,j] 为真实输入中精确等于 M 的个数。
导数为 g[b,j]/K[b,j]（极值位置），其他位置为0。min 使用 -max(-x)，两次负号按链式法则处理。
唯一极值 K=1；不使用 epsilon，也没有节点编号式赢家。复制边表示保留，反向通过 repeat 累加回原始边变量。
空组沿用原 PyG 零输出，K=0但没有参与输入，不给初始化填充值分配梯度。仅公共模块使用的二维浮点输入/一维同设备 int64 batch 接口。
前向委托原 PyG global_max_pool，在自定义 Function 的无梯度 forward 内使用原生 scatter_reduce；后向使用布尔精确相等和 int64 计数。
输入、输出和上游梯度非有限时显式失败。没有回退。
这保留前向数学定义，但改变并列点的子梯度选择，不能称为完全无变化的优化。

## 已执行与边界

冻结启动2026-10-28 01:00:19 +08:00，截止01:30:19；计算阶段01:11:47结束，最后交付时间见 delivery_receipt.json。
实际环境 Python3.9.23、Torch2.1.0+cu121、PyG2.3.1、torch-scatter2.1.2+pt21cu121；CPU单线程，一张RTX2080Ti。
确定性True/warn_onlyFalse，TF32关闭；记录实际环境，不据此假定所有第三方算子确定。

1. 上票真实张量逐张量哈希核验；min内部max和max各20次同GPU重复，前向与旧值逐位一致，梯度逐位稳定并符合平均分配式；守恒、非极值零梯度和min符号通过。
2. 合成检查包括唯一/并列、负值、多组、空组、节点/边顺序置换、重复表示回传、非有限拒绝；光滑点double gradcheck通过。CPU对照独立记录。
3. 两数据集×A/B/C全部完成相同完整查询、两份半损失 normal2 的CPU/GPU对照与两次同GPU重复。六格同GPU所有参数梯度逐位相同。
4. 六格前置门全部通过后，每格恰好3次临时更新：1、连续2、恢复后重做2。所有加载字段精确；六格续算的参数、梯度、优化器浮点状态、RNG及输入逐位一致。
5. CPU/GPU完整梯度门六格仍未通过。最大绝对梯度差分别为IMDB A 3.0293e-5，B/C 1.04684e-4；PubMed A 2.95534e-5，B/C 1.87499e-4。未扩大诊断或放宽2e-6/1e-5门槛。不能用这些不同参数尺度直接判定某特征机制更不稳定。
6. 原22+3项CPU回归及新增6项均通过，零跳过。40次真实算子重复是单列证据，不拼成夸大的测试数量。
7. optimizer.step实际18次，正式训练0、新生成图0、正式字典拟合0。临时权重不得进入正式实验。

精确性口径：完整网络报告中的 exact/bitwise 字段由 torch.equal 得到，表示逐元素完全相等、最大绝对差0，而非仅在容差内；没有另存两侧全部网络梯度以独立检查正负零的位模式。真实算子40份原始数组则另做逐字节重复核验。不得把这一固定面板结果推广为任意状态/后端的位级确定性保证。

## 成本

本次临时 normal2（包含诊断数组导出、梯度范数与同步）IMDB约2.47–2.58秒，PubMed约3.29–3.39秒；不是稳定吞吐估计。
同权重同输入完整查询 eval/no_grad（无CPU取证hook，不含反向）旧/新分别约IMDB0.32–0.33/0.45–0.46秒，PubMed0.43–0.44/0.61–0.615秒。每侧两次，未随机化测量顺序；只是工程组件测量，约37%–42%开销不回避。
梯度/前向成本阶段峰值allocated：IMDB2293.69MiB，PubMed3142.17MiB；reserved最高3350MiB；进程累计RSS在该阶段最高8347.32MiB。
这些是明确采集阶段的峰值；未独立记录恢复阶段的进程/显存全程峰值，不能冒称整个票的绝对峰值。大量CPU诊断副本在内。
完整查询是IMDB13120槽、PubMed18336槽，一次前向；normal2分母均2×18336，两份上下文独立，顺序反向后一次step。
C仅既有每类型4角色、16列合成工程基，NNLS成本不能代表正式字典。
没有T1000链、并发竞争或正式拟合实测，不能可靠承诺三小时完整实验预算。

## 审计和载荷

evidence/含全部参数误差表、真实算子40次原始输出/梯度、输入/源码绑定、初始化逐张量哈希、优化器恢复结果、成本和日志。
prior_evidence/仅原封复制上票捕获张量、哈希及来源回执；inputs/原封复制v2两份统一输入。
initial_weights.tar.xz包含六份完整 initial.pt（首次前向前保存，含RNG、配置及构造回执），解到evidence下对应格目录；归档只压缩，不修改任何权重。
after1.pt六份临时完整检查点保留本地results/minmax_tie_mean_v1/各格，不纳入发布ZIP；文件SHA及加载/续算逐参数证据均在包内。它们不是科学训练产物。
IMDB A与上票获准重建初态逐张量相同，仍不声称找回旧v2从未落盘的原始文件。其余按原seed/anchor/重置seed/model/copy_common流程重建，共同张量实际复制并逐项检查。

只读解包复核（不运行 acceptance.py、不使用GPU、不更新参数）：
```bash
sha256sum -c SHA256SUMS
tar -xJf initial_weights.tar.xz -C evidence
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=package python -m unittest discover -s package -p test_engineering.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=package python test_revision_extra.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python test_tie_extrema.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python audit.py --evidence evidence --capture prior_evidence/captured.pt --output audit_local.json
```
使用既有sparse_block环境，不安装或升级。acceptance.py是绑定本票绝对截止及本地路径的执行存档，不是可再次启动新预算的训练入口。

## 下一决定

已在本次固定真实张量上消除随机并列赢家，在六格指定normal2输入与18次工程更新范围内实现同GPU逐位重复与恢复。
不推广为所有GPU状态完全确定，也不声称解决CPU/GPU差异或旧v2全部误差。
正式实验仍需审查这一子梯度变更及跨后端门的剩余差异，另行批准正式每类型字典拟合材料/收敛验收、实验预算与资源；本票停止，不自动继续。
