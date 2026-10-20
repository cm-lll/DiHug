# 固定权重新目标验证结果

先读RESULTS.md。科学源码封存f538c746，协议bdd30ef；三固定normal2@6000模型全部120条
T64/R4 CPU链重新生成，保留全部原图和步骤。原执行父验收因浮点字典相等/布尔P90出现失败，
旧回执及STOP没有改写；纯评价修复在同一45分钟账本内完成，未重复任何链。

目录索引：

* evidence/run01：唯一activation、source/config/environment、120图/逐步记录、权重哈希绑定、3工作进程及原失败父回执。
* evidence/audit_repair_v1：整数频数/浮点运算误差界的修复回执、初次失败、完整generation/per_seed/topology_classes及审计。
* evidence/post_audit.json：120图与最终步链接、48新控制图事后复现历史的核对。
* evidence/per_seed_summary.csv / macro_summary.json：所有seed与五条件统计，原始8个值保存在per_seed.json。
* evidence/genre_degree_ranks.png：实际离散图排序Genre度谱，每个模型8图均值。
* evidence/repair_audit.py / test_repair.py：独立纯后置评价修复，数学与采样实现未改。
* source_vs_bdd30ef.patch：执行适配器对冻结协议的补丁。
* review ZIP：完整绑定源码+全部无权重机器证据；不含私有模型文件。

审查用纯CPU回归（不会启动科学链）：
```
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTEST_ADDOPTS= \
 /data2/lyh/miniconda3/envs/dihug/bin/python -m pytest -q \
 experiments/role_unseen_target_validation_v1/test_execution.py \
 experiments/role_target_scalability_v1/test_planning.py \
 --import-mode=importlib
```

这些包含三套本地权重只读严格加载门；ZIP不含权重，在没有原本地权重的机器上不能声称
该门通过。review_bundle保留相同experiments布局，后置审计测试设置ROLE_REVIEW_SOURCE_ROOT
为解包根目录。本票只发布审查，不建议另激活科学入口重跑120链，更不自动训练或大图。

原失败目录不可用作恢复入口。repair_audit明确只审计已有文件：每个TV用整数计数的准确
有理值验证正项浮点求和误差；不改变原目标、质量判据或资源/GPU容差。不可把原父状态
改为passed，也不能据此声称新目标控制已经有效。
