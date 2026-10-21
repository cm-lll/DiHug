# 冻结权重新目标验证执行适配器

授权：bdd30ef的冻结协议已获用户批准。原方案中“尚未授权/只准备”的文字是历史记录，
本版配置另外保存本次授权，不改写旧方案。三normal2@6000 × 五目标 × 八配对seed，
全部120条重新运行。三个单线程CPU进程共享唯一2700秒账本；不是每进程45分钟。

不改变科学实现：直接调用原hsgd.runtime.reverse_step（默认upstream），T64、R4，
每t冻Xt，四块完成后提交。完整561槽先验及每步561个uniform与原采样安排一致；
仅312活跃槽被查询，其余249槽精确先验0。相同sample_seed的先验/查询/uniform/RNG
摘要跨五条件与三模型核对。无训练、无优化器、无概率调整、无修图。

输入为固定权重、冻结匿名角色目标、原类型库存/schema及原族先验。推理case完全不含
clean/clean_ids；族先验从已核验匿名角色的桩数恢复并逐位核对原参考计数与权重buffer。
panel只读取schema及类型库存，不读取其边身份。证明图目录不在白名单内，入口没有
np.load或旧图读取。每个t重新建FrozenTypedContext，保留原精确DP200000/路径2m门。

启动前回归/封存源码，启动账本早于重依赖导入。父监督器沿用既有信号/进程组清理；
三个子作业属于同组，STOP/截止在查询块/完整步边界检查。保留每8步的诊断partial
状态与RNG（不是认证恢复入口），各完整链立即保存图/64步记录/独立结果。任何失败
非零退出，已完成图保留，不自动补抽。硬截止后仅必要清理时间另列，不扩大计算窗。

评价沿用冻结MP1/2/3与闭合定义，不引入四环。角色误差逐图对匿名目标计算，无拓扑
真值的新条件只描述结构分布。独立NX/整数度统计核对全部图、角色/度W1、三角形、
连接、路径总量与DMA闭合；MP模式枚举器不另写第二套，仅复用已验收实现并核对总量。
每格8图先汇总，再描述3训练seed。类频数使用类型/关系保持精确同构，WL只作筛分。
孤立点描述而非否决。所有状态必须有非空完整证据才能记完成。

运行（只激活一次，输出必须不存在）：
```
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
/data2/lyh/miniconda3/envs/dihug/bin/python -u experiments/role_unseen_target_validation_v1/execute.py launch \
  --run /data2/lyh/gnn_project/DiHug/results/role_robust_training_6000_20261021_v1/learning_run01 \
  --panel /data2/lyh/gnn_project/DiHug/results/role_late_paired_20261020_run01 \
  --output /data2/lyh/gnn_project/DiHug/results/role_unseen_target_validation_20261021_v1/run01
```

输出：activation/source_manifest/resolved_config/environment；每seed绑定/日志/40图及逐步
记录；generation/per_seed/topology_classes/audit/资源回执和父controller。
计算完成后发布全部结果和审查提交并停止，不启动任何后继。
