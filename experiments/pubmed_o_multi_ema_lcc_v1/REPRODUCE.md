# 复核与文件说明

本票2026-10-29 01:32:08开始，固定截止02:17:08，Asia/Shanghai；新票CPU分析预算与已结束的185分钟训练票独立，旧账本未改。正式训练、模型前向、重新采样和GPU使用均为0。

数据源为a4a87a19ca17ba866300ea8bcbe7b49e27302a9e对应的PubMed O多EMA64图。input_manifest.json记录来源路径、原图封存回执SHA256及派生图SHA256，reference.npz来自原95b1ee8包，实际SHA256应为f6ea8ec461052094dd02c6b07d8ebb69e85fb3e595195007563593b9568914ef。没有重新加载任何模型或权重。

本包自包含64原始图副本、64派生图、映射、参考、完整统计表与纯CPU代码。复制的原图字节与封存源一致。主分析目录是新建目录，旧图、日志和报告未写入。

## 只读复核

使用现有Python／NumPy／SciPy／NetworkX／pytest环境，不安装或升级依赖：

```bash
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTEST_ADDOPTS=''
python -m pytest -q test_lcc.py
python audit.py --root "$EXTRACTED_PACKAGE" --offline-review
python audit_reporting.py
```

audit.py的offline-review只核对现有公开文件、分量及统计，最多120秒，不重新生成派生图、不启动任何科学账本，也不加载Torch／模型。结果写入带新时间戳的offline_audit文件，不覆盖封存audit.json。原本票执行模式仍受ledger.json固定截止约束。

audit_reporting.py属于本票执行审计，保留本票截止；将来只读复核主要使用offline-review和现有表格。测试和offline-review可在原科学票到期后运行，但不代表训练、采样或新分析窗口授权。

## 本轮分析命令（历史回执，不自动重复执行）

```bash
python analyze.py --run /原封存运行目录/run --reference /原95b1ee8源码/package/assets/PubMed.npz --output /新建且含本票ledger的目录
python report.py
```

analyze.py需要提前显式记录的45分钟ledger和全新图输出目录；已有目录会拒绝覆盖派生图。不将运行失败作为换目录重授时的理由。默认审计同样服从原截止。重放以上分析需要后续明确范围，本包不会自动启动。

## 数值与边界

- 所有参考比较使用同一完整192节点参考，LCC是生成图的使用版本，不悄悄截取参考来适配结果。
- 固定参考类型权重的incident4为变节点数主适配口径；实际类型权重兼容项同时保留。两者差异与定义见METRICS.md。
- 类型条件经验PMF按实际保留类型节点数归一，度数本身不缩放。容量按实际节点数计算。缺类型、零机会和不可用实现保留null及原因。
- 计数项每图先算参考绝对误差再汇总，分布距离直接比较。原数量差不自动叫改善。cohort_pairs.csv和cleaning_cohorts.csv分别对应EMA−raw和LCC−original。
- HGEN legacy字段保留源码数学含义：square实际K4、gini零基、clustering为3T/claws。原生powerlaw缺依赖、可选社区Z_obs缺失、完整orbit谱不在冻结实现内，不能把本包称所有原生HGEN指标零缺项复现。
- 13项CPU测试、独立结构／算术审计及报告审计分别保存；源包校验清单SHA256SUMS不包含自身。独立解包的复核回执作为发布附件保存，不修改源图或既有审计。

完成后只等待结果审查，不追加训练、生成或修图。
