# 单步硬结构训练的故障后续跑

本目录隔离修复 `single_step_formal_20261111`，保留原源码、全部失败日志及八份u2500状态。修复里程碑保存漏参、汇总重复task字段及独立输出路径。模型、输入、损失、AdamW更新、验证和采样函数通过AST一致性检查；没有科学配置变更。

每任务从原 `stopped_complete.pt` 完整恢复模型、AdamW、EMA、RNG及完成计数，复用u2001—2500日志，只继续u2501—3000。GPU预检临时运行2501/2502并丢弃，正式再次从原u2500恢复。序列化与载入状态一致不代表普通CUDA重复计算逐位一致。

四个父模型的B1K64/B4K16两臂共八任务。各最终EMA生成四条R1/T1000链，共32条，不增加训练seed、不换损失、不缩短链。普通CE旧图只作背景。

2026-11-11用户确认的新窗口：10:21:57开始，12:21:57停止科学计算，12:51:57交付。累计16 GPU小时包含前次3.5634818106小时。实际分配至进程退出均计占卡；任务结束立即释放，CPU评价不占GPU。监督器和独立看门狗只停止本票进程。

```bash
python research_branches/single_step_formal_resume_20261111/test_repair_cpu.py
# 以下只能在获准窗口执行一次，launch.lock禁止重复启动。
python research_branches/single_step_formal_resume_20261111/preflight.py
python research_branches/single_step_formal_resume_20261111/launch.py
```

机器结果位于 `results/single_step_formal_resume_20261111/`。`window.json`是生效截止；`remaining_budget.json`是先前预算草案，实际启动看`launch_decision.json`及`budget_calibration.json`。运行状态看`supervisor_progress.json`，失败看`failure_notice_*.json`及各任务`receipt.json`。最终检查`supervisor_receipt.json`、`finalization_receipt.json`和`completion.json`；进程退出不等于实验成功，缺图不能报完整均值。
