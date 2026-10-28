# PubMed O：固定EMA读出的最小实现（仅供审查）

本包只增加参数EMA及其原循环接入函数、冻结协议和CPU小张量验收。**不是可直接启动正式实验的运行包。** 没有正式审批、活动账本或已生效的130分钟窗口。没有加载原模型、GPU测试、扩散训练或完整生成。

## 唯一假设与解释边界

检验同一训练轨迹的参数平滑读出，能否提高PubMed O的生成数量准确性；跨训练seed差异缩小仅为次要描述。既有四格诊断证明了同状态下预测倾向不同，没有证明差异由末期参数噪声造成。

固定公式：`ema_0 = theta_0`；每次完整normal2更新后，`ema_k = 0.99*ema_(k-1) + 0.01*theta_k`。第一次更新也执行这条公式，不直接复制theta1。服务器Torch为2.1.0+cu121，原`AveragedModel.update_parameters`在`n_averaged==0`时直接复制，与本合同不同，因此采用显式实现。

0.99对整个历史指数衰减，并非截取最后100步。长期加权年龄为99更新；1000更新时约98.996更新，theta0仍有约4.32e-5权重。效果可能同时包含平滑和滞后；若各seed长期形成不同倾向，EMA不保证消除它。成功不能唯一解释为消除了末期噪声，失败也不直接否定所有EMA设置；本轮不搜索其他设置。

只平均参数，不平均buffer。EMA不进入原模型的forward、CE、梯度、AdamW或训练随机流。先验、整数状态和其他buffer直接从同检查点原模型复制；不改变噪声核、后验、样本或边数。不是两模型概率平均，不是两seed权重混合，不是角色机制贡献。

## 固定实验提案

- 数据：旧192节点PubMed、926条边；仅基础O，不加CN/ReFeX/RolX。
- 两条训练轨迹：2026102812、2026102833，分别使用对应已封存原始科学初态，新的AdamW；不使用旧final权重或旧优化器续训。
- 每条1000次normal2、完整18336槽、原CE/FP32/优化器/学习率；原与EMA两种读出共享同一条轨迹，没有额外训练样本。
- 保存0/500/1000的原模型、EMA、优化器、计数、RNG及绑定；EMA在首次前向前保存。原固定0/500/1000验证分别评价两读出并保持训练RNG。
- 仅最终1000更新读出用于科学采样。每训练seed×raw/ema×采样seed2026102821/2026102822，共8条全新T1000链。每个readout使用自己的权重哈希、相同配对采样初态/随机安排；不复用旧图充数，不挑checkpoint或补抽。
- 无衰减率搜索、预热、学习率变化、额外损失、校准、约束或修图。

## 判断标准

对每训练seed，先逐图计算`abs(edges-926)`，再平均两图。**只有两个训练seed的主要误差都下降，才能称本面板有一致准确性收益。** 只有合并均值下降而一seed变差，报告取舍；两seed一起生成相近但错误的数量不是改善。逐关系误差同时列出，不能以关系间抵消掩盖结构偏差。

两条人为“不推荐采用”红线分别检查每训练seed的两图均值：incident4 W1相对原始读出增加超过10%，或LCC下降超过19.2节点。原始W1为0时，EMA变为任何正值触线。不将红线称为有证据的安全界限，也不将未触线称为结构合格；LCC少约19节点本身是明显代价。尾部/三角等严重退化须明确说明，不能自动宣布整体改善。`assessment.py`即使主要收益一致、红线未触发，也不会自动批准采用。

完整保留joint-degree TV、逐关系数量/误差、类型三角与闭合机会、聚类、LCC/分量/孤立点、度尾部、MP2/MP3和样本差异。W1 incident4主口径、all10兼容口径与原评价相同。缺图或未定义主要量保留不完整，不只平均成功图。两个训练seed只能支持探索判断。

## 已实现的最小接线

`ema.py`：初始化theta0、完整更新后维护、状态保存/恢复、导出EMA权重；参数别名同步覆盖，buffer保持原值。

`integration.py`：围绕既有normal2和验证函数的调用，不另写科学训练/采样算法：

```python
# 原模型构造、公共参数复制、上GPU和训练随机流重置保持原顺序。
ema = ParameterEMA(model)       # first forward前创建/保存
# 继续使用原normal2，两次半损失反向后只step一次。
rec = complete_normal2(normal2, model, opt, case, inputs, 1000, device, ema, completed)
completed += 1
# 原atomic_checkpoint保存checkpoint_payload(...)，含原权重、EMA和原RNG。
# 原validation本身保存恢复训练RNG，并关闭每个FrozenTypedContext。
ema_row = ema_validation(validation, model, ema, case, cfg, seed, rng)
# 最终生成入口仍使用原reverse_step；仅加载选定readout的完整state_dict。
readouts = final_readouts(model, ema, 1000)
```

EMA验证采用临时state_dict切换并在finally恢复原模型，参数对象不替换，优化器指向不变；不跨权重切换保留学习输出缓存。原checkpoint路径/哈希/原子落盘继续复用既有实现；新EMA状态增加格式、0.99、参数布局和更新计数验证。正式RNG恢复仍调用原`rng_set`，不改变GPU随机流管理。

完整更新边界包括原normal2和随后的EMA维护。若原参数已经更新而EMA失败，该内存状态不能导出成完整检查点或科学读出；保留异常，使用此前完整检查点的恢复规则，不能把旧计数配上新参数。这里不授权自动恢复。

`review.py plan`仅打印2训练/8采样唯一身份、目录及依赖。`review.py formal`始终非零退出，不导入任何旧worker。

**完整正式launcher、审批绑定和独立采样进程分发尚未接线，也未做实际GPU集成测试。** 本次交付的是EMA内核与明确调用接口，不把小张量通过称为完整运行包已验收。代码审查通过并获正式预算授权后，按既有O运行入口接线；只检查新增EMA调用和身份/保存路径，不重做旧科学模块验收。原始/EMA权重需要独立文件和模型哈希，不能沿用raw的哈希标记EMA采样。

## 成本与队列

历史PubMed O每1000更新约55分钟，每条链约10.3分钟。默认两张获授权的空闲2080Ti各负责一条训练轨迹及四条采样链；包含旧初始化/验证/保存的历史外推约97.8分钟，加EMA/双读出验证、竞争及评价交付余量，130分钟仍只是待确认规划。EMA成本尚未GPU实测，不能保证截止。新增存储约一份FP32参数副本4P字节，验证切换/导出另有临时CPU副本；没有新增Adam状态或网络计算图。

若两条训练完成后有额外获授权空闲卡，可将同一八任务清单分发采样；不增加图数。具体设备及队列应在正式启动前固定并据此核算，不在本轮查询/占用GPU，不创建活动预算起止。

## 复核与停止

使用当前环境，不安装升级：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTEST_ADDOPTS='' /data2/lyh/miniconda3/envs/sparse_block/bin/python -m pytest -q -o addopts='' test_ema.py
python review.py plan
```

测试仅Tiny张量、合成优化器状态、模拟normal2调用及评估记录。没有真实optimizer.step、反向训练或模型forward；完整网络/GPU检查不在本次范围。CPU Torch/Python/全局NumPy/显式Generator均验证EMA维护不改变随机流；CUDA随机状态未新实测，旧工程证据保留。

测试与源码绑定见`evidence/`、`SOURCE_BINDING.json`及`SHA256SUMS`。完成发布后停止，等待独立审查，不自动进入正式实验。
