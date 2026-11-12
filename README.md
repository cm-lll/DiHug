# DiHug current experiments and server handoff

2026-11-12固定交付，分支`codex/server-handoff-20261112`。保留旧公开审查分支历史，未修改main。

- [中文完整结果报告](CURRENT_RESULTS_ZH.md)：O/A/输出反馈消融、分块匹配、DBLP/ACM试点，含收益和代价。
- [服务器操作](server/README.md)：依赖、数据、预检、训练、T1000采样与终态反馈。
- [逐图/逐seed结果](reports/paper_ablation_20261112T190000/)：原图/LCC分开，训练seed与采样波动分开。
- [原始来源和哈希](provenance/current_export.json)、[迁移测试](provenance/server_validation.json)。

`current/`是此次实际运行源码及依赖，保留相对目录供审查。根目录旧`experiments/`属于历史版本，不要用旧日期launcher启动新票。服务器从`server/runtime.py`进入，它是新迁移接线，不冒充历史实验原入口。

## 实现索引

科学核心目录：`current/research_branches/cn_multiseed_20261104/research_branches/typed_structure_early_cn_expansion_review/experiments/typed_structure_early_cn_expansion_v1/`。

| 内容 | 核心目录内位置 |
|---|---|
| O/A/D/E结构 | `package/model_variants.py`：O无CN，A晚CN，D为ReFeX加晚CN，E再加角色；本次仅O/A |
| 类型/schema/槽 | `package/hsgd/slots.py`、`package/hsgd/bw_preflight/common.py` |
| 前向加噪/先验/后验 | `package/hsgd/noising.py`、`package/hsgd/kernel.py` |
| 时间/查询 | `package/hsgd/bw_preflight/common.py: train_input/plan_for` |
| CE/固定支持 | `package/hsgd/losses.py`；单输入更新在`current/research_branches/cn_multiseed_20261104/experiments/late_final_single_input_v1/single_input.py` |
| 主干/度分支/预测头 | `package/hsgd/model.py`、`package/hsgd/vendor/`、`package/model_variants.py` |
| 晚CN/冻结上下文 | `package/hsgd/common_neighbor.py`、`package/hsgd/computational.py` |
| 历史特征/角色 | `package/features.py`、`package/role_dictionary.py`；O/A不实例化角色分支 |
| EMA/完整恢复 | `ema.py`、`integration.py`、`boundary_transactions.py` |
| 反向生成 | `package/hsgd/runtime.py: reverse_step`、`package/hsgd/kernel.py: full_sweep` |

当前实验入口位于`current/research_branches/`：

- `paper_ablation_20261112/{worker,scoring,evaluate}.py`：此次论文消融。
- `matched_block_training_20261112/{worker,sweep}.py`：训练/采样全块一致试点。
- `one_step_vs_terminal_plan_20261111/`：终态代理与事务支持；`heat_solver_candidate.py`是已接入的evd热核实现，文件最初的“not wired”注释属于历史状态。
- `new_datasets_20261112/{support,metrics}.py`：可变类型数据、联合评分及指标。

## 192节点查询与CE

全无向无自环节点对是18336；两张历史图声明全部类型对，schema槽数也为18336。R1覆盖活动关系的全部槽；先验0关系固定无边，正确支持CE为0，仍计入全schema分母。不是18336个槽都做可学习正负分类。无向边只计一次，不含自环；新图S按声明schema计算。

服务器全块CE是`sum(CE)/S`，不再除块数或加R倍权重。参数和Xt在全部块结束前冻结，生成也完成全覆盖才提交。查询分块降低单次激活，不消除CPU全槽或热评分成本；不同R不保证预测等价。

## 结论与使用边界

CN在IMDB有重复收益；异质反馈有局部增益，但PubMed两seed度和高阶数量代价明显，不能称通用稳定修复。完整大图尚未验证。反馈为有偏clean-head终态代理，类型盲消融不是官方GDPO复现。来源与原创边界见[THIRD_PARTY.md](THIRD_PARTY.md)。

迁移入口仅通过CPU合成接线，服务器GPU和完整大图仍需验收。保存载入一致不等于CUDA重算逐位一致。数据、checkpoint、生成图、账号信息及大型日志不上传；下载后单独准备合法数据，并建立与该数据匹配的基底。64GiB GPU不自动解决CPU dense热核与MMD瓶颈。
