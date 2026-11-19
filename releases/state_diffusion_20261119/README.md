# DiHug 异质图状态扩散实验源码

这份归档覆盖 2026 年 11 月 6 日至 16 日的结构反馈实验。模型从带噪异质图预测各关系槽位的干净边概率，再结合离散后验进行反向采样。这里收录了软概率上的结构目标、对抽样硬图打分、以及完整反向链的终图反馈。

| 路线 | 代表源码 | 反馈含义 |
| --- | --- | --- |
| 可微软结构与输出校正 | [度分布与闭合校正](../../research_branches/joint_output_formal_review_20261108/)、[投影对照](../../research_branches/joint_output_projected_review_20261108/)、[直接结构目标](../../research_branches/direct_structure_cpu_20261108/) | 在干净边概率或其结构代理上计算可微损失。目录保留含逐边 CE 的对照与不同候选。 |
| 解析结构期望及异质联合目标 | [联合关系度解析风险](../../research_branches/analytic_joint_risk_20261113/core.py)、[类型化联合结构目标](../../research_branches/hetero_micro_macro_20261116/objective.py) | 解析风险显式计算独立边假设下的矩和协方差。后续联合度、类型化闭合与传播描述使用不同口径，部分为软代理。11 月 16 日有一组 CE 加结构的已停用开发对照。 |
| 一步预测后抽硬图 | [初期硬图评分](../../research_branches/hard_clean_pilot_v2_20261110/)、[B1K64 与 B4K16](../../research_branches/single_step_formal_20261111/)、[实际续跑与修复](../../research_branches/single_step_formal_resume_20261111/) | 在同一去噪预测下抽样硬图，评价结构代价并估计梯度。B/K 分别记录去噪输入数和每输入硬图数。 |
| 一步单抽样、多抽样与整链配对 | [S1／S16／T-eager 三臂](../../research_branches/one_step_vs_terminal_plan_20261111/pilot_worker.py)、[S16 补齐](../../research_branches/one_step_vs_terminal_plan_20261111/resume_s16_worker.py)、[整链续跑](../../research_branches/one_step_vs_terminal_plan_20261111/terminal_continuation_worker.py) | S1 用一个硬图，S16 按每张图的留一基线计算反馈。T-eager 用完整链终图的结构评分来加权干净图头的训练项，它是终点代理。 |
| 完整链评分与异质消融 | [终图转移梯度诊断](../../research_branches/terminal_structure_real_20261109/)、[类型化热签名](../../research_branches/typed_heat_audit_20261111/)、[联合度及热签名](../../research_branches/typed_heat_degree_audit_20261111/)、[OT 语义与终点反馈](../../research_branches/joint_transport_ablation_20261113/)、[输入与评分消融](../../research_branches/paper_ablation_20261112/) | 包含实际反向转移的梯度诊断、终点干净图头代理、类型感知评分，以及分别匹配输入和评分的对照。两种梯度的口径应分别解读。 |
| 分块支持 | [分块采样](../../research_branches/block_generation_panel_20261112/)、[匹配分块训练](../../research_branches/matched_block_training_20261112/)、[服务器入口](../../server/README.md) | 已有服务器交接提供分块训练、反向采样与终图反馈入口。 |

这些实验保留了结构指标间的取舍与失败。一步反馈出现过骤然过密、欠边或空图。完整链终点代理出现过部分结构收益，跨种子方向和完整生成质量仍有差异。可微目标中的局部改善也不等价于终图结构改善。没有把这些开发结果合并成一个已通过验证的最终模型。

`SOURCE_MANIFEST.json` 记录原文件与本次发布文件的 SHA-256，以及与历史实验清单匹配的版本。原项目中用于依赖绑定的个人绝对路径在发布副本中改为仓库相对路径。部分启动器需用环境变量 `DIHUG_SPARSE_BLOCK_PYTHON` 指定兼容解释器。历史脚本默认读取同名 `results/` 下的状态、样本、旧权重和冻结清单。跨机器执行需要先准备这些资产并核对源与数据绑定。原始训练数据、生成图、检查点及原始运行日志需另行提供。公开服务器交接已有的配置和汇总仍可在父提交中查阅。
