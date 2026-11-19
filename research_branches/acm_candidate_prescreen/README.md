# DiHug 的学习式候选预筛

## 当前方案

DiHug 面向单参考异质图重采样。第一阶段通过合法迁移逼近匿名联合关系度目标。第二阶段以已经分配好的关系度为约束，用合法双边交换调整连接结构。这次验证集中在第二阶段的候选搜索。

每步先产生一份合法交换名单。学习器用关系类型、端点关系度、端点间邻接、图的当前结构残差和每个相关节点最多四个邻居的抽样，选出最值得精算的少量候选。训练标签来自实际的结构误差降幅。精算阶段使用等权的分类型三角和跨关系四环归一化 L1 误差，并检查整图连通。入选候选如果未通过检查，仍消耗一个精算名额。没有合法正收益动作时，保留原图。这样可以直接检验学习器是否让少量精算机会更常落在有用的动作上。

## 已有结果

主读数是每步在入选候选中找到的最佳可执行结构收益。以下单位为误差下降量 ×10⁻³，越高越好。候选、精算目标和名额在同一行内完全共享。

| ACM 评估范围 | 每步精算数 | 学习预筛 | 最强已比较的手写规则 |
| --- | ---: | ---: | ---: |
| 38 条扰动轨迹整档互留 | 4 | 2.402 | 2.233（抽样闭合直排） |
| 同一互留评估中的 312 个连通状态 | 4 | 0.995 | 0.967（残差分槽） |
| 正式轨迹，三个 C 起点等权 | 2 | 0.684 | 0.180（残差分槽） |
| 正式轨迹，三个 C 起点等权 | 4 | 0.759 | 0.298（残差分槽） |

第一组互留评估以 3,328 个已存状态拟合和检验两个小树模型。精算 4 个时，学习比闭合近似直排平均高 7.6%，但其中连通子集的逐轨迹结果有明显取舍。精算 8 个时，整体上残差分槽略好。

随后用完全留出的 24 条真实连通后缀复核。七张起点图均来自同一 ACM 参考，共 49,152 次编辑机会，三个 C 起点是主要面板。每种预算下，两个冻结模型都在三个 C 起点超过残差分槽，20 条主要路径的平均配对结果均为正。精算 2 个时，C 面板正收益命中率由 34.0% 提至 66.5%。四个 D 起点的方向也相同。正式轨迹的结论是相对于残差分槽的局部反事实证据。

存档轨迹上的单步优势已经通过预设离线门槛。合理的后续生成验证固定每步精算 2 个候选、相同的连通起点、候选机制、结构目标和总编辑次数，比较完整终图的综合结构误差与连通性。当前证据覆盖已有候选名单内的筛选。最终生成质量、跨参考表现和端到端运行时间仍待验证。

## 代码和复现

- `offline/run.py` 拟合候选预筛器，按整条轨迹留出。`offline/verify.py` 独立重算候选收益、随机期望和预算。`offline/summarize.py` 汇总预算与连通子集。
- `recheck/run.py` 确定性复原原拟合，再沿完整存档正式后缀做配对评估。`recheck/summarize.py` 输出逐路径、逐拟合、逐起点和通过条件。
- `expected/` 提供汇总结果，可用于核对复现。两项协议保存在各自的 `PROTOCOL_ZH.md`。

用 Linux、Python 3.9 和以下依赖版本复现。程序只执行 CPU 拟合和存档评估。

```bash
python -m pip install -r requirements.txt
export DIHUG_ACM_ARCHIVE_ROOT=/path/to/results
export DIHUG_ACM_OUTPUT_ROOT=/path/to/new/output
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

python offline/run.py
python offline/verify.py
python offline/summarize.py
python recheck/run.py
python recheck/summarize.py
```

在 DiHug 项目内，输入默认取仓库的 `results/`，输出默认写到本目录的 `runs/`。建议为每次复现指定独立输出目录。`ACM_OFFLINE_WORKERS` 可设为 1 到 6。复现从本地读取以下原始资产，按原哈希核验。GitHub 分支包含源码、协议和汇总表，复现所用的原图、原始候选列表和轨迹从现有本地档案读取。

- `typed_edit_complexity_20261116/data/acm/graph.json`
- `acm_frozen_participation_replay_20261118/{pilot,bank4}` 及其清单指向的 `typed_edit_autonomous_20261118` 下的原始拟合档案
- `typed_edit_autonomous_20261118/acm_solver_repair/deploy` 和 `additional_f_pilot/prefix/acm`
- `typed_edit_autonomous_20261118/additional_confirmation_v1/{deploy/acm,prefix/acm}`
