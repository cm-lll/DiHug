# PubMed O：raw／EMA完整运行接线（等待正式授权）

本版接通两次训练、八条采样链和统一评价。原EMA方案不变；不是新的角色机制。**目前没有正式审批文件、正式起点或活动账本。不得因为本包存在或工程测试通过自行开训。**

## 实际数据流与复用边界

`package/`逐文件复用首轮已封存OABC执行包（58文件），包括模型、normal2、CE、min/max、前向噪声、原后验、`reverse_step`和图评价。历史未用入口保留来源，不是本版正式入口；唯一正式入口为`run.py formal`。

`ema.py`、`integration.py`、`assessment.py`、`protocol.json`与审查提交`bda747ebb18604d69040ec33ffa36fdbaa0cd46a`逐字节一致。θ̄₀=θ₀，每完整normal2后θ̄ₖ=0.99θ̄ₖ₋₁+0.01θₖ；buffer只复制，EMA不反馈训练或优化器。不搜索衰减率，不修改模型/损失/采样。

`model_io.fresh`核验并加载两轮已封存的**原始0更新科学权重**，只取原model，不取旧优化器或训练后状态。模型构造后重置原配对训练随机流，新建空AdamW，然后在首次前向前保存本轮initial（raw、EMA、空优化器、RNG、计数及绑定）。两份历史初态的实际字节本轮已重新核验。O不读取字典、ReFeX、CN或角色库存。

`run_worker.train`调用原`normal2`，仍为一次zero_grad、两份完整查询各sumCE/(2×18336)顺序反向、一次step；随后调用EMA。保存0/500/1000，0/500/1000各用原固定t1/500/1000验证两种读出并恢复训练RNG。完整更新失败不生成虚假的可恢复检查点。

1000更新后从同一完整checkpoint导出`readout_raw.pt`和`readout_ema.pt`，两文件分别保存读出类型、训练seed、原final哈希及各自独立哈希。两次训练全部完成后才启动链。采样worker核对指定文件及元数据、严格加载后比对张量，原`reverse_step`逐步完整覆盖并统一提交；不复用工程权重、不将raw文件误标为EMA、不新增后处理。

每条链目录包含训练seed/读出/采样seed。保存原初态哈希与采样Generator状态、11个原规则状态点、最终图、完整指标及绑定。汇总核对配对初态与随机状态。训练随机流和各链随机流分离；不自动恢复、补抽或重启。

## 冻结任务与实际队列

仅PubMed192节点、926边的O，训练seed2026102812、2026102833，各1000更新；每个最终raw/EMA使用采样seed2026102821、2026102822。共两次训练、八张全新T1000图。

本版固定**两张获授权空闲2080Ti**：slot0负责第一个训练seed，slot1负责第二个。两个训练任务完成后，每卡按raw/2821、raw/2822、ema/2821、ema/2822顺序执行四条链。CPU线程1；不DDP。成本文件和launcher调用同一`jobs()`清单。此版本未实现额外卡分发，避免成本与实际队列不一致；如以后改变队列，须明确重新绑定成本和资源，不能运行中临时扩图。

formal入口第一时刻开始累计7800秒，包括启动检查；固定1200秒交付预留。起止用Asia/Shanghai记录。正常或失败退出均尝试统一汇总，保留原始错误和缺项，不用汇总失败覆盖worker失败。STOP、SIGINT/SIGTERM/SIGHUP清理本任务进程组；SIGKILL不可捕获。输出必须新建且不与来源/输入重叠。实际输出卷至少4GiB空闲；当前不是持久性保证。

资源授权、调度预留和其他进程单独检查；1%利用率不是拒绝理由，也不能用无进程推断有权限。本轮工程只使用GPU0一次，正式设备UUID须在新授权时指定。

## 本轮实际检查与限制

- 原22项CPU小张量检查，加9项运行接线测试，最终31项零失败/跳过。首次模拟失败路径的测试替身缺少`wait`方法（30通过/1失败），补全替身后完整31通过；保留旧失败XML，没有修改科学实现或容差。
- 同一真实PubMed O初态、实际GPU路径，**5次临时normal2 step**：原流程2次、EMA流程2次、恢复第1步后重复第2步1次。正式训练0，完整链0，新图0，未做GPU跨后端或公共算子复验。
- 两次更新原参数、梯度、完整优化器状态及输入/RNG均逐位一致；EMA验证后原模型/优化器/RNG/模式恢复一致；checkpoint的raw/EMA/优化器/RNG/计数载入精确，恢复续算也逐位一致。这仅覆盖本次短段，不保证任意长期轨迹逐位一致。
- 原FP32比较容差仍为atol=2e-6、rtol=1e-5；精确载入和后续计算的判据分开。本面板结果同时满足更强的逐位一致，不增加为普遍训练门槛。
- 真实工程用时约42.77秒（进程内部），峰值allocated3217.03MiB、reserved3472MiB、进程RSS4638.95MiB。RSS包含多份CPU取证快照，不能当正式稳定占用；CUDA峰值也仅是本次状态。
- 暖态原normal2约3.224秒，EMA版本内部原normal2约3.261秒，EMA维护约0.0978秒，总计3.358秒。首步含冷成本，单独保存，不将总时间差全归因EMA。约3%是本次组件观测，不是完整训练吞吐保证。
- 3时刻raw验证约1.857秒、EMA约1.833秒，连同切换3.863秒；checkpoint保存1.924秒、恢复0.281秒。未新跑采样成本短链，不声称测过EMA自由生成状态。

## 预算建议

历史最慢训练3322.52秒/1000更新、单链621.82秒。用历史与本次暖态更新的较大者，再加EMA维护；两卡并行关键路径含1000更新、每卡四条链、旧初始化/保存成本和新增双读出验证/保存约5980.79秒（99.68分钟）。另加598.08秒**未实测规划余量**，覆盖新的进程初始化/读出加载、竞争和状态变化风险，再加1200秒评价交付预留，合计**7778.87秒＝129.65分钟**。

130分钟可作为这份队列的规划上限，但仅比含余量估算多约21秒；不能承诺完成。真正compute预留门在启动后110分钟触发，达到就保存缺项，不延期。两卡并发与EMA后期自由状态成本尚未实测；若希望更保守预算，应在正式授权时决定，不能本轮自行延长或换队列。

磁盘最低4GiB是检查点/双读出/状态/日志的保守工程门，不包含公开包及历史权重。EMA常驻参数副本约25.30MiB；验证、导出和取证还有临时副本。

## 评价

复用原`evaluate`及`summary_metrics`。逐图先计算相对固定参考的误差，再按训练seed分别比较EMA−raw；保留incident4/all10、逐关系、jointTV、类型三角、闭合机会、聚类、LCC/孤立/尾部、MP2/MP3和两图差异，null/缺图不只平均成功项。

只有两个seed各自的平均绝对边数误差都降低，才称本面板一致数量准确性收益。跨seed差异缩小只是次要结果。每seed的W1+10%或LCC下降19.2是人为“不推荐采用”红线，不是安全证明；严重尾部/三角代价仍须说明，不自动批准采用。EMA平滑同时带来滞后，不保证消除持久偏差。两个训练seed不能证明统计稳定性或角色机制有效。

## 命令（正式命令尚未授权执行）

运行环境保持`/data2/lyh/miniconda3/envs/sparse_block/bin/python`、Torch2.1.0+cu121，不安装升级。首次使用先核验发布清单。两份0更新初态不打入公开ZIP；`initial_assets.json`给出路径及已封存预期SHA256，搬迁时只能显式修改外部approval的`initial_paths`映射，哈希仍严格核对。

```bash
# 仅打印队列，无GPU探测/worker/账本
/data2/lyh/miniconda3/envs/sparse_block/bin/python run.py dry-run --output /tmp/ema_future_plan

# 仅CPU接线回归，无真实更新
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTEST_ADDOPTS='' /data2/lyh/miniconda3/envs/sparse_block/bin/python -m pytest -q -o addopts='' test_ema.py test_runtime.py

# 以下须用户正式授权后，填写外部approval模板的真实审查/授权/资源证据。
# 不自动创建许可；示例不是本轮执行授权。新目录不得已存在。
/data2/lyh/miniconda3/envs/sparse_block/bin/python run.py formal \
  --approval /path/to/authorized_approval.json --cost /path/to/bound_cost.json \
  --output /path/to/new_run --gpu-uuids GPU-APPROVED-0 GPU-APPROVED-1

# 安全停止；不自动恢复或重抽
touch /path/to/new_run/STOP
```

外部approval须绑定当前源码、protocol、cost、preflight、完整环境、初态及两卡队列；`approval.template.json`明确false，仅是待填格式，不可直接激活。cost中的preflight路径可显式迁移到同哈希文件，重新生成cost哈希后再由审批绑定；不能用新算的未知文件哈希替代预期来源。

结果位置：`activation.json`、`queue.json`、`launcher_summary.json`、`unified_evaluation.json`；各`PubMed_s…_O/`内为initial/midpoint/final、readout_raw/ema、validation/training；链在`raw|ema/sample_…/`内。`launcher_summary`保留主失败原因，`unified_evaluation`保留逐seed及缺项。正式批准前不执行这些任务。

本轮结束后停止等待运行审查及正式预算授权；不再拆分EMA方案，不新增科学候选。
