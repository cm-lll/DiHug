# IMDB基础模型／匿名角色软读取：v1.1修复与交接包

本轮只准备和CPU工程验收，未启动正式训练、GPU任务或完整大图链。原始方法来自固定提交f4d410cc37a76b065634eadb27a5421d53f638a0。

本版修复基线bb32bfaa5b63f049cbcba4408e2184a39e5c8b6a；新源码manifest与配置绑定，旧预检不能授权本版。
恢复latest使用经过路径/哈希/步数/内容检查的相对路径（包括recovery/）；同一更新两位置内容冲突即拒绝，不忽略冲突。
从滚动点再停止、再次恢复和滚动轮换均保持指针可读；原永久千步文件不删除。

## 比较对象

base与role都使用相同Transformer、类型/关系处理、CN、完整Xt当前度分支、公共全局统计、联合预测头、原CE/原Bayes/原20块规则。
base不接收匿名目标，角色编码/读取不计算，联合头角色输入明确为零；role使用当前CPU Newton容量熵软读取与原角色融合。
两臂保留相同形状的完整参数库存，初始化逐张量复制并检验；base角色专属参数未使用，不称为相同有效参数量。
这不是纯SparseDiff与多模块模型的对比，也不是旧v9对比。目标角色仅为训练参考的匿名完整联合向量/整数频数，无原节点对应关系或边身份。
单图条件固定，是同参考条件生成，不能声称跨图或未见条件泛化。容量软行列约束不等于最终图硬度配额。

## 方法与配置

8层/8头，节点256/边64/全局128，FF256/64/256，CN32，联合头128；T1000、原余弦核、20块，FP32网络/CE，float64 CPU似然、Newton及Bayes。
AdamW lr2e-4/wd1e-12/betas(.9,.999)/eps1e-8/amsgradFalse，无裁剪/调度器/类提权/AMP/TF32；配置dropout.1但原vendor没有实际dropout。
沿用normal2：每个有效更新两份原核普通加噪输入，两个R/S CE的普通均值，只执行一次独立优化器更新；时间均匀抽取1..1000。
配对臂共享实际输入、t、query、标签，均保存摘要，两个独立新优化器。学习/推理使用同一条件代码；缓存只在同一冻结Xt/时间/目标/模型版本内使用。
当前IMDB1653节点/4079物理边；完整schema1,365,378槽，活跃997,313槽，确定性空族368,065槽。没有裁图、删角色或改查询数。

## 两个限定小修

1. capacity_readout.forward读ctx.prior，只复用FrozenTypedContext已有只读CPU prior；模型prior版本/设备、参数、时间、target变化仍由原上下文合同拒绝，并显式拒绝整个prior张量被替换，避免旧对象版本未变时误用缓存。
2. SparseEtoy由调用者传入已知global_state行数，在edge_attr.device/dtype建立零张量。生产FP32的形状、输出/损失/梯度/更新对照逐位相等；不宣称旧半精度路径已认证。
两个补丁分别见发行包patches。两臂使用相同公共修改。没有跨块Transformer隐藏缓存、批量查询搬运或后验重写。

## Notebook预检（先执行，前台可见进度）

保留厂商Torch与已有PyG/NumPy/SciPy，不运行pip、不安装编译扩展。不修改CUDA/HIP/ROCR可见范围；仅使用当前可见逻辑卡0，RNG也只管理该卡。
默认Python=/usr/local/bin/python3.11，可用PYTHON_BIN指定已有解释器。本地Python3.9与notebook Python3.11可分别预检，但正式恢复严格绑定实际Python/Torch/依赖/设备与普通执行配置。

```bash
cd /root/private_data/gnn_projects/imdb_role_pair_gpu_v1_1/package
df -h /root/private_data
findmnt -T /root/private_data
bash run.sh preflight --output /root/private_data/gnn_projects/imdb_role_pair_gpu_v1_1_runs/preflight_01
```

另开终端查看（不启动另一份任务）：

```bash
tail -f /root/private_data/gnn_projects/imdb_role_pair_gpu_v1_1_runs/preflight_01/launcher.log
bash run.sh progress --output /root/private_data/gnn_projects/imdb_role_pair_gpu_v1_1_runs/preflight_01
```

preflight父监督器在Torch导入/数据加载/GPU初始化前计时，总上限570秒，强制清理额外预算最多10秒并单独记账。支持STOP、Ctrl+C、SIGTERM、SIGHUP；SIGKILL不可捕获。
预算耗尽或STOP记为incomplete并非算法失败；始终非零退出，不进入formal。最终摘要preflight.json；完整回传清单及环境/导出命令见包根HANDOFF_zh.md。
先在9节点小夹具上严格模式检查CPU/GPU logits、p0、loss、参与梯度、一次更新、同设备重复与保存/恢复续算，ATOL2e-6/RTOL1e-5沿用旧门。
跨CPU/GPU Adam参数差异只描述：原门不要求跨设备第一步参数完全相同；严格同设备第一步重复必须逐位相同，严格保存恢复后按同容差验收，并核对优化器内容。
通过后改回普通模式，在完整IMDB每臂做5次临时normal2更新：1预热、3稳态、1详细trace；每臂同一固定t500状态做两次完整20块采样单步，不串接链。
检查Newton支持/容量/KKT/尾部误差和跨设备角色表示。失败不回退DP、不改精度/块数/宽度、不自动装依赖。
临时权重只留在预检目录，永不作为正式初始化或续训点。CPU --fixture-only只产生工程通过，不能授权GPU正式训练。

## 正式方案：等待用户核验后端并确认总预算

提议3个新配对训练seed2026102151/52/53，两臂各12000次有效更新，共72000更新、144000训练前向。12000是既有大图入口的预定观察终点，不称充分收敛。
0/每1000步固定验证(1/250/500/750/1000，每个全20块)；初始与全部千步模型/优化器/RNG永久保存，约30分钟另存最多两份滚动恢复点。
6000/12000两个预定点，每臂每训练seed用8个配对生成seed2026102161..68，96条全新完整T1000链，无最佳检查点选择/筛图/补抽。
训练和采样顺序占用同一可见卡，不并发争抢。固定验证保存/恢复训练RNG，采样另用独立流；所有查询块读同一冻结Xt，完成后统一提交。

预算必须先依据目标后端preflight.json：训练=36000×(base更新中位+role更新中位)；采样=48000×(base单步中位+role单步中位)；验证另计。加25%与600秒准备/保存/审计余量，仅在估计不超过用户确认预算80%时允许启动。
完整验证成本是5个单步的保守估计，未称实测；单步外推不覆盖自由链变密、CN路径与读取难度变化，运行仍逐次验收并失败停止。
固定更新长度/样本数量与科学设置都在config.json里，不因预算不足缩成单seed。不够时先返回阶段方案供用户决定。
初始与千步检查点预计约6.24GB十进制。首次启动仍至少16GiB空闲；恢复改查实际输出卷上的剩余需求。
保守单个双臂检查点上界取192MiB与已见最大文件的较大者；剩余39个计划永久位置按已有文件扣减、补足每seed两份滚动位，另预留原子替换空间及每seed一份额外STOP保存。
这些剩余检查点写入量乘1.25，另留2GiB日志/评价与512MiB图/状态储备；同时要求16GiB初始门减去已留存计划/滚动文件大小的余额，两者取大。
运行中每次永久/滚动保存前重查。估计不是磁盘配额保证，异常大量日志/反复STOP额外文件仍可消耗余量；不足即停止、不自动清理。storage.json记录公式结果、实际st_dev及/proc挂载信息。
挂载信息不证明平台持久性，必须由操作者核实；private_data目录名本身不是证明。

先获取冻结方案（不会训练）：

```bash
bash run.sh plan > /root/private_data/gnn_projects/imdb_role_pair_gpu_v1_1_runs/frozen_plan.json
```

以下formal启动与恢复模板均尚未授权执行。只有用户确认方案哈希、总秒数和持久存储之后，才执行。占位符未提供会拒绝启动，包里没有默认无限训练或另一个小时上限：

```bash
bash run.sh formal \
  --preflight /root/private_data/gnn_projects/imdb_role_pair_gpu_v1_1_runs/preflight_01 \
  --output /root/private_data/gnn_projects/imdb_role_pair_gpu_v1_1_runs/formal_01 \
  --confirmed-budget-seconds USER_CONFIRMED_SECONDS \
  --approved-plan-sha256 USER_CONFIRMED_PLAN_HASH \
  --persistent-output-confirmed
```

安全停止与恢复：

```bash
bash run.sh stop --output /root/private_data/gnn_projects/imdb_role_pair_gpu_v1_1_runs/formal_01
bash run.sh progress --output /root/private_data/gnn_projects/imdb_role_pair_gpu_v1_1_runs/formal_01
# 确认旧进程已退出后，显式归档本次STOP；不要删除旧证据或重置activation。
mv /root/private_data/gnn_projects/imdb_role_pair_gpu_v1_1_runs/formal_01/STOP \
   /root/private_data/gnn_projects/imdb_role_pair_gpu_v1_1_runs/formal_01/STOP.acknowledged
bash run.sh formal --resume --persistent-output-confirmed \
  --preflight /root/private_data/gnn_projects/imdb_role_pair_gpu_v1_1_runs/preflight_01 \
  --output /root/private_data/gnn_projects/imdb_role_pair_gpu_v1_1_runs/formal_01
```

恢复包含模型/优化器/RNG/更新数、原截止时间及源码/配置/数据/backend/实际依赖绑定；停机计入总预算，不重新授时。普通后端不承诺之后逐步轨迹完全复现。
手动STOP在配对优化器边界保存；硬错误中途的半对更新不发布为有效配对检查点，保留最近有效检查点及失败输入。部分反向步会回滚RNG，保存上一个完整提交状态。
最终图先绑定完整终态，已有图必须核对状态/权重/seed一致后才允许补完评价；不把中间图当完成输出。

## 评价与限制

使用原raw metrics：各关系边数有符号/绝对误差、各类型各关系度W1/完整联合频数与TV、最大度、低度/零度与p90/p99、MP1/2/3、类型三角形、DMA路径闭合与去重对闭合分别报告、连通分量/LCC/孤立点。
每张原始图均保留。多样性报告标号边差、标号重复与WL可区分类数下界；相同WL不冒充同构，8张不同也不证明总体多样性充分。先在训练seed内汇总，再等权比较3个配对差异。
高连接尾部阈值在评价前固定为各类型/关系参考P99向上取整且至少1，另报完整排序top10总度谱、低度与零度计数；不根据生成效果挑阈值。末图的边族数及完整联合度频数从原始图独立复核。
没有硬角色匹配、概率引导、度配额、删孤立点或后处理；遗留role_degree仅为原回归依赖，没有正式入口开关。
CPU工程通过不认证BW/CUDA梯度，不认证完整IMDB训练容量或生成质量；旧GPU梯度失败和历史设备记录见发行包evidence，不删除也不外推到未测notebook后端。

结果中每seed有training.jsonl、validation_*.json、永久千步checkpoints和最多两份rolling recovery。samples_u*/arm/seed*/保存完整状态/RNG、原始graph.npz及评价；all_graph_metrics.json、per_train_seed.json和paired_summary.json统一汇总，权重留在原持久目录。
导出训练/验证CSV只读取本地日志，不加载模型，输入输出须分开：

```bash
/usr/local/bin/python3.11 export_metrics.py \
  --run /root/private_data/gnn_projects/imdb_role_pair_gpu_v1_1_runs/formal_01 \
  --output /root/private_data/gnn_projects/imdb_role_pair_gpu_v1_1_runs/formal_01_csv
```
