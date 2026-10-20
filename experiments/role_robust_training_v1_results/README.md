# 纠错训练方案与工程门：待审查，不启动学习

科学源固定提交：38a4f437d00f8066c7123dd0bbb5c9cfc6f674ed。
容量软读取基线b8e97cc，诊断基线72c7648。模型、优化器、原噪声和生成器不变。

|项目|normal2公平基线|robust唯一候选|
|---|---|---|
|每次更新输入0|普通前向加噪|相同输入|
|输入1|独立普通前向加噪|同一输入，再加/删合法MG边|
|损失|两份原CE平均|普通与受扰CE平均|
|前后向与优化器|2前向/2反向/1更新|相同|

两套模型每seed从相同的已保存随机初始张量开始，独立新AdamW，不能拿旧单输入1000步当公平主对照。
三原训练seed、每臂1000有效更新，未来共6000更新/12000查询前向；96条T64/R4实际末图。
时间仍独立均匀U{1,...,64}，没有重启时间比例实验。
两条件均以相同规则抽加/删方向；加[1,4,5]、删[1,2,3]。在能精确实现所抽强度的G节点中均匀选择。
不可行则原样输入并记录，不改方向/强度。两条件的可行节点/角色位置混合不保证相同，必须统计。
仅模型输入Xt发生变化，标签/查询/CE权重不变；编辑标记不输入模型、不加权训练。
这是新鲁棒目标，不能声称保持原扩散期望目标。

## 本轮实际完成

- 原98+新增28=126项CPU测试通过，零失败/跳过，最终源码回归26.15秒；7条继承warning。
- 4张新评估拓扑封存，类型保持同构检查排除全部12张旧训练/验证拓扑；同匿名联合角色统计。
- 冻结12个低t噪声输入、48个加删压力变体，全部精确编辑可行；标签不进入网络。
- 实际父入口freeze完成4.218秒；probe完成18.042秒，均从启动计入300秒工程上限。
- 临时初始副本每臂5次双输入工程更新，总10次更新/20次查询前向；另各3次完整4块概率扫过及固定验证。
  没有科学训练、完整生成链、GPU、notebook/BW访问；没有改变历史权重。
- 所有6658481个参数的架构不变，配对初始化1446个state_dict张量完全相等、存储独立，独立优化器。
  FF实际256/64/256，配置dropout.1但运行未启用dropout。完整实例化维度在probe.json。

|CPU单线程测量|normal2|robust|
|---|---:|---:|
|首次双输入更新(s)|0.52593|0.50716|
|稳定更新中位数(s)|0.48626|0.49193|
|稳定更新P90(s)|0.48686|0.49273|
|完整4块概率扫过P90(s)|0.20327|0.20366|
|12状态固定验证(s)|2.65164|2.66310|

稳定更新仅3次、扫过仅2次热测；不能当作可靠长期吞吐分布或加速结论。
扫过使用同一人工全活跃槽为1的固定稠密输入，计算p0和q但不抽样、不推进状态，
不冒充实际链耗时。首次初始化/构图缓存和稳定部分分列于原始记录。
两臂日志策略、精度和概率检查相同。worker墙钟14.844秒/CPU15.173秒，峰RSS716.785MiB；
父入口18.042秒包含导入与初始化。GPU显存不适用。

## 预算方案（未激活）

拟议单次累计90分钟；不是复用历史预算，也不是已经执行/批准的学习窗口。
短测外推：训练48.979分钟、96链20.835分钟、正常验证1.063分钟、末期压力验证1.221分钟，
另预留独立审计60秒；加15%余量和固定120秒后共5163.759秒=86.063分钟。
粗略余量3.94分钟，还需扣正式启动与成本门耗时，安全余量不宽裕。
该估计没有实测完整链、学习曲线或样本质量；实际链随状态变化，不能保证按时完成。
未来激活时必须重测成本，预测超过剩余预算则拒绝开始，不缩组/seed/模型/块数。
超时保存全部已完成证据并报告未完成，不重新授时或自动续跑。

## 判据与解释

主要证据为实际末图：MG有符号和绝对偏差、Genre W1和完整联合角色TV、hub最大度/≥12数量、
低度/零度、其他关系边数、连通/LCC/孤立点、类型三角形、DMA路径/去重对闭合、MP2/MP3、
同条件类型保持拓扑频率。八张图先在训练seed内汇总，再列三个seed配对差异。
96图不是96次独立训练；不可只看总loss或纠错概率。改善与结构损失并存就报告取舍。
新4张仅为开发holdout，目标条件仍已见；不能称任意角色分布泛化。
最终仅用1000步权重，保留0/250/500/1000正常验证，不能事后选择最好检查点。
正常验证、生成器与新增鲁棒CE分开解释。未定义闭合率保留NA，不填0。

## 失败与未验证

首次新测试收集误导入旧baseline/run.py，同名模块遮蔽导致ImportError；
已改为按绝对文件位置加载本版父入口，保留失败日志，随后28项及完整126项通过。
一次apply_patch上下文未匹配，原子拒绝后重读并修正，没有产生部分科学修改。
没有科学运行失败、没有质量结果、没有完整面板端到端运行证据。
未来完整学习/96链/最终大汇总仍待执行，不能以单元测试宣称这些都通过。

## 本地复现与下一次入口

本轮CPU：Python3.9.23、Torch2.1.0、PyG2.3.1、NumPy1.26.4、SciPy1.11.0、NetworkX2.8.7、pytest7.4.4。
源码/数据/权重字节/数值依赖均检查；ZIP不重复包含旧初始权重字节，只附历史SHA256及本轮只读绑定。
需由本地已有的容量基线与原面板目录显式提供，不能伪造或自动重训来补权重。

在解包目录内，先设置实际本地路径（本次使用下列已存在路径）：

```bash
PY=/data2/lyh/miniconda3/envs/dihug/bin/python
BASELINE_RUN=/data2/lyh/gnn_project/DiHug/results/role_specialization_20261020_v1/learning_run01
PANEL_RUN=/data2/lyh/gnn_project/DiHug/results/role_late_paired_20261020_run01
NEW_ROOT=/data2/lyh/gnn_project/DiHug/results/role_robust_training_review_run02

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= "$PY" \
  role_robust_training_v1/run.py freeze --source "$BASELINE_RUN" --panel "$PANEL_RUN" \
  --output "$NEW_ROOT/frozen"

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= "$PY" \
  role_robust_training_v1/run.py probe --source "$BASELINE_RUN" --panel "$PANEL_RUN" \
  --frozen "$NEW_ROOT/frozen" --output "$NEW_ROOT/probe"
```

下面仅展示审查后另行激活的完整面板命令，本次没有执行：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= "$PY" \
  role_robust_training_v1/run.py panel --execute-learning-panel \
  --source "$BASELINE_RUN" --panel "$PANEL_RUN" \
  --frozen "$NEW_ROOT/frozen" --output "$NEW_ROOT/learning_run01"

tail -f "$NEW_ROOT/learning_run01/worker.log"
# 安全停止：在完整更新/检查点边界留存，失败/停止不冒充完成
touch "$NEW_ROOT/learning_run01/STOP"
```

独立验证入口检查参数、--help和完整126测试集合；超时/STOP/SIGINT/SIGTERM/SIGHUP的
监督器测试使用小子进程，均核对退出码与子进程回收。不能捕获SIGKILL或断电。
本版不自动恢复中断学习面板，保留状态等待审查。完成发布后停止。
