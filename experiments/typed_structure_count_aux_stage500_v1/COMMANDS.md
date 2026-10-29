# 本票运行与交接

本票已完成并停止。环境为/data2/lyh/miniconda3/envs/sparse_block/bin/python，CPU线程1，RTX2080Ti；不升级环境。

只读检查队列：python run.py dry-run --output /data2/lyh/gnn_project/DiHug/results/count_aux_stage500_20261029T205144/run。应列8个500步任务、0条链；dry-run不启动worker。

实际审批、成本、设备、启动账本在结果目录approval.json、cost.json、run/activation.json和run/resources.json。原输出目录拒绝覆盖，旧审批只绑定本票目录和既定截止。后续不得直接调用worker、重开账本或继续训练。

本票停止机制为run/STOP及既定22:16:44科学停止线；现在所有worker已经自然结束，GPU已释放，无需发送停止信号。已有检查点包括raw、AdamW、EMA、更新计数、完整RNG和来源绑定。是否整体继续至3000须新授权，不在本票自动执行。

CPU复算入口在结果目录analyze_stage500.py及audit_checkpoint_payloads.py；仅读取本票已有日志和保存载荷。审计不构造模型，不前向，不优化，不生成。
