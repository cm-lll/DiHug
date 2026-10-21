# 条件覆盖 CPU 配对面板交付

先读RESULTS.md；全部36000更新/528新T64R4链及独立审计已完成。科学源795b4baf，
后置报告代码和机器证据单独保存；无其他训练候选、GPU或大图。

索引：

* evidence/activation.json：第一次实际准备就开始的唯一3h账本；initial formal_learning_started=False仅表示激活时尚未学习，实际开学见learning_authorization。
* conditions/panel/construction/split_binding/freeze：目标规则、训练/验证清单、图构造及完整哈希绑定。
* correctness：三旧6000权重固定验证差0及60个条件/权重读取检查，解析PMF/置换证据。
* cost、cost_gate：六全模型临时短测、80%剩余预算门；工程权重不进入正式训练。
* jobs：六作业完整逐更新日志、逐条件次数、0/250/500/1000/3000/6000固定验证及全部原始图/64步记录。
* sealed_final_weights/checkpoint_inventory：六最终权重统一封存、18完整检查点和初始权重的实际哈希/大小。PT完整留本地，不包含在公共ZIP。
* generation/per_seed/macro_summary/training_condition_counts：全部528结果，66个8图格、3配对训练seed；不筛图或先去重算质量。
* training_curves/validation_curves/genre_degree_ranks.png：训练、匹配验证、实际离散排序度谱。
* audit：独立重算全部图的整数度/完整联合频数/W1/连通/三角形/闭合/简单MP1-3模式，覆盖/RNG及末图链接。
* preparation_attempt01/02、test_attempt01：正式学习前工程失败完整保留，无预算重置；正式面板无失败。

原运行命令（已执行一次，**不是邀请再次运行**）：
```
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
 /data2/lyh/miniconda3/envs/dihug/bin/python -u experiments/role_condition_coverage_v1/run.py --execute
```
该入口绑定实际本地路径results/role_coverage_20261021_v1，formal_launch存在会拒绝重启；
不自动新建窗口。后续任何新学习/生成需要另行授权。纯审查请检查SHA256SUMS及verify_release.py，
它只解包/哈希/同33项工程测试，不开启科学面板。测试需要本地原初始化文件，公共ZIP不携带私有权重。

完整配置和来源在freeze.json/resolved_config.json；类频数的8个不同图只说明本批未复制，
不等于总体多样性充分。构造图非均匀抽样；条件+对应图覆盖不是只改数值的纯因果实验。
所有后继停止，等待审查。

独立解包复验已实际完成：1839载荷SHA、冻结源码/输入哈希及入口帮助通过；
与封存集合完全相同的33项测试零失败/错误/跳过，30.59秒。
原始测试输出、JUnit及环境成本见unpacked_tests.log/xml和unpacked_acceptance.json。
这是本地工程复验，使用只读本地初始权重，不表示公共包携带私有权重。
ZIP SHA256：0b3dda1e636e8af7c4ab08be4af94bd6f2113c6a3f40b2e0167ea4999ecf3e56。
ZIP内README为打包时快照；本段及独立解包回执在固定Git提交中另行提供，ZIP没有事后覆盖。
