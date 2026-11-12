# 高性能服务器迁移与验证

本目录是新接线适配器；`../current/`保存实际完成2026-11-12实验的源码。适配器复用相同模型、CE、反向核、EMA和事务函数，但它不是历史任务的逐位重放入口。当前仅CPU合成接线检查；尚未在完整大图或新服务器GPU上验证。不会自动启动任何计算。

## 安装与数据

Python 3.9，历史环境的确切版本见根目录requirements.txt。先安装与你服务器驱动匹配的PyTorch和torch-scatter CUDA wheel，再安装其他依赖。不要把CUDA12.1 wheel强行用于不兼容驱动。保存`pip freeze`、GPU型号、驱动/CUDA、TF32、线程配置；迁移入口保持原FP32及关闭TF32，不默认AMP。继承源码归属见根目录THIRD_PARTY.md。

数据由用户在服务器单独准备；仓库不含数据、权重、生成图或账号凭据。NPZ字段：

- `types`: int64[N]，连续的0起始类型编号，每种声明类型至少一个节点。
- `edges`: int64[2,E]，无向简单图，只存u<v，不重复、不含自环。
- `schema`: int64[R,2]，固定有序关系表，每行a<=b；当前同一端点类型对只支持一种关系，不能静默合并多关系。
- 可选`families`: int64[E]零起始关系；或`relations`: int64[E]一始关系。提供时必须与schema一致。

类型名称及预处理/抽图规则另存元数据。先验由本参考图逐关系边数/合法槽数计算；边数为0的声明关系为固定零支持。节点数量/类型固定，边数不固定；不能把未知类型转换为已有类型冒充适配。不能把192节点检查点直接接入大图：旧prior、schema和数据绑定必须一致，否则应在完整参考图上训练新基底。

## 先检查，后运行

下面命令中的`/scratch/dihug`是示例工作目录，须替换；不要上传实际数据。

```bash
export PYTHON=python
export TMPDIR=/scratch/dihug/tmp
mkdir -p "$TMPDIR"
$PYTHON server/inspect_data.py --data /scratch/dihug/full.npz --rounds 20 --out /scratch/dihug/inventory.json
$PYTHON server/guard.py --gpu 0 --minutes 10 --gpu-hours 0.16 --receipt /scratch/dihug/smoke_resource.json -- \
  "$PYTHON" server/runtime.py smoke --data /scratch/dihug/full.npz --config server/configs/IMDB_A.json \
  --arm A --seed 2026110411 --rounds 20 --device cuda --minutes 9 --out /scratch/dihug/smoke
```

`smoke`执行一次完整分块CE更新、完整保存/加载核验及一个原反向步骤；产物是临时模型，不是正式训练或完整生成效果。**文件载入一致不代表CUDA重新执行轨迹逐位相同。** 暂不自动重复历史GPU恢复容差判定，不能称所有CUDA异步故障都覆盖。

训练/采样每步均遍历全部活动块；固定Xt和模型直至块结束才提交，CE是所有槽损失求和除以整个schema槽数，不额外除块数。与历史R1的网络查询上下文并不完全等价；更多块可能改变效果。输入仍使用既有前向核，未加入结构损失或裁剪到CE阶段。

## 显式启动CE基底

以下预算仅是命令格式示例，**不是大图成本承诺或自动授权**。先用smoke实测更新、反向步、显存和CPU内存，核算3000更新及T1000整链能否完成。`guard`拒绝已占显存设备，只终止本次子进程组；占卡从启动到结束全计入。集群请同时申请调度器独占资源，避免空闲检查与他人启动之间的竞态。

```bash
$PYTHON server/guard.py --gpu 0 --minutes 240 --gpu-hours 4 --receipt /scratch/dihug/ce_resource.json -- \
  "$PYTHON" server/runtime.py train-ce --data /scratch/dihug/full.npz --config server/configs/IMDB_A.json \
  --arm A --seed 2026110411 --rounds 20 --updates 3000 --minutes 239 --out /scratch/dihug/ce
```

O对照用`--arm O`；其余数据/seed/预算/块合同保持一致。预设模型8层、width256、CE学习率2e-4、EMA0.99和1000噪声步；PubMed使用对应配置。配置中历史ReFeX/RolX字段为继承元数据，O/A不实例化角色分支。不得根据测试结果挑seed或终点。

完整状态含raw、AdamW、EMA、RNG和计数。恢复时使用新输出目录，`--resume --checkpoint .../latest_complete.pt --checkpoint-sha256 <sha256> --updates 3000`；3000是总目标，不是再加3000。恢复要求该适配器的模式、数据、模型、seed、配置和块数相同。优化器部分失败时拒绝伪造可恢复状态，使用上一个完整检查点。

## 完整采样与结构反馈

```bash
sha256sum /scratch/dihug/ce/final_complete.pt
# 把下面HASH替换为上一步完整哈希；只加载可信的自己生成的checkpoint。
$PYTHON server/guard.py --gpu 0 --minutes 120 --gpu-hours 2 --receipt /scratch/dihug/sample_resource.json -- \
  "$PYTHON" server/runtime.py sample --data /scratch/dihug/full.npz --config server/configs/IMDB_A.json \
  --arm A --seed 2026110411 --rounds 20 --minutes 119 --checkpoint /scratch/dihug/ce/final_complete.pt \
  --checkpoint-sha256 HASH --readout ema --sample-seeds 2026102821 2026102822 2026110231 2026110232 --out /scratch/dihug/samples
```

每条链固定T1000，保存初态身份、逐步随机数/查询哈希及最终原图。LCC是单独评价，不裁剪生成图冒充原图。raw与EMA必须显式选择，默认raw；反馈实验主读出raw，CE主对照EMA。

`feedback`模式接收相同数据的CE完整检查点/EMA；从该权重新建AdamW和EMA，lr1e-5，每次更新2条全新T1000链，各取8个预定反馈时间，全块反馈、外层系数0.01、梯度裁剪1。固定4更新对应当前试点。用`--mode`是不支持的；模式是位置参数，例如将上面`sample`替换为`feedback --updates 4`并指定新输出目录。反馈结束后须另用`sample --readout raw`评价，不能取训练轨迹作为测试图。

反馈使用**有偏clean-head终态代理**，不是精确无偏反向策略梯度，也不是官方GDPO复现。新数据的热/关系度尺度只由参考数据推导并冻结，不能从生成结果调尺度。独立采样seed不得用于选择训练图。

## 大图必须单独验证的成本

查询块只限制一次网络查询量，并不消除：

1. O(S)完整合法槽状态、coverage、uniform等CPU数组；S可能是O(N²)。
2. typed-CN的wedge/path枚举和预设max_wedges/max_query_paths容量。超限先报告，不自动放宽或裁掉邻居。
3. 当前热签名的dense EVD：O(N³)时间、O(N²)CPU内存，默认单CPU线程；64GiB GPU不解决CPU EVD。
4. 每类联合核MMD的O(sum n_type²)开销，以及完整结构评价的最短路/高阶枚举。

`inspect_data`列的是单个数组下界，**不是峰值内存保证**。反馈默认拒绝超过2000节点；这只是防止未经核算的dense评分，不是科学目标或模型容量界。必须明确评估CPU内存和耗时后才能用`--allow-dense-score-nodes N`提高执行阈值；不改变评分公式。不在本次偷偷换近似谱算法。大图反馈与完整评价尚未通过成本验收，CE/采样通过也不能替代该检查。

建议服务器顺序：数据审计→O/A smoke→完整CE成本→完整采样成本→独立评分成本→再固定正式面板。不必一次跑满八卡。当前仓库只提供这条验证路径，不宣称整张IMDB/PubMed已训练成功。
