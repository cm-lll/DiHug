# 有界数量项八任务500步配对试验

本票八任务已完成500步并停止，正式更新4000、临时更新6、生成0。唯一正式入口为run.py，当前protocol.json只允许500更新、无生成任务；后续续训未授权。

损失来自已审3be64f7：sqrt(max(m,1))尺度、sqrt(1+u²)-1惩罚，lambda=.001；精确函数与formula.json未改。运行源码提交974e5dca8cbdd05ffddc53fd599ae815828d8b56，运行源码哈希5e81a49ae1d6afebbae7fed54e4898f4f29a05e02e9cb67637a802af89b82636；发布后的新增报告与文档不改变运行代码。

请先读STAGE500.md与仓库results/count_aux_stage500_20261029T205144/REPORT.md。当前CPU测试为test_stage500.py的6项；当前GPU回执为结果目录engineering_IMDB/summary.json及engineering_PubMed/summary.json。源目录evidence/、旧check_prepare.py、旧test_count_aux.py、build_cost.py及historical_metadata均为父版本证据/工具，不是本票测试或启动入口，禁止据其旧9000方案启动。

独立结果目录位于/data2/lyh/gnn_project/DiHug/results/count_aux_stage500_20261029T205144。32份完整科学检查点保存在其中run目录；轻量审查包保留路径/字节数/SHA256，不重复打包约3.08GB检查点。模型、字典、数据、旧源码和旧失败记录未覆盖。
