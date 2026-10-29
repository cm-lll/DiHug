# 历史命令：父准备版本，非本票入口

环境复用 /data2/lyh/miniconda3/envs/sparse_block/bin/python，不安装升级。
本目录独立于旧实验；临时路径显式位于/data2。

CPU干运行（不会创建科学输出或启动worker）：

    python run.py dry-run --output /data2/lyh/gnn_project/DiHug/results/count_aux_NEW/run

新增CPU回归：

    CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TMPDIR=<数据盘临时目录> python -m pytest -q test_count_aux.py --basetemp=<新的数据盘测试目录>

不应再次运行工程GPU脚本：本票临时optimizer.step额度8次已经用完。check_prepare.py保存机制测试流程，历史结果对应被否决的归一化；check_final_scale.py为最终公式不更新检查。真实GPU现有验收范围见README。

正式启动命令仅作交接，不授权执行：

    python run.py formal --approval <经用户批准的真实JSON> --cost cost.json --output <全新数据盘目录> --gpu-uuids <按cost.json顺序8个UUID>

审批须包含：user_formal_authorized=true、independent_review=passed、authorization_reference、source_sha256、protocol_sha256、asset_bindings_sha256、initial_bindings_sha256、cost_sha256、updates、budget_seconds、output、gpu_uuids、environment（python_version/torch_runtime_version/packages/device_name）、resource_authorization（allocation_check_complete/allowed_uuids/reservation_conflicts/minimum_free_mib）。不得自行从模板生成真实授权。cost.json通过仅表示规划算术通过，不代表科学授权。

当前默认9000/48链，780分钟仅建议。6000/32链是待决定备选，不能通过私改worker或删队列自动降档；选择后更新protocol/cost并重绑审查。

输出：
- 各任务 initial.pt、checkpoint_u1000...u9000.pt、validation.json、training.jsonl。
- readout_u3000/6000/9000_ema_0.99.pt及readouts_u*.json。
- 各任务/u3000|6000|9000/ema_0.99/sample_<seed>/graph.npz、steps.jsonl、binding.json、summary.json。
- derived_lcc_evaluation、unified_evaluation.json、stage_status.json、stage_u*_brief.json。
- launcher_summary.json、execution_receipt.json保留不完整和失败。
- 创建run/STOP安全停止；无自动重启/恢复入口。保存只在完整normal2边界，链失败保存状态但不自动补抽。

源码绑定包含launcher、worker、汇总、协议、字典/数据与新损失。已有模型/初态外部文件只读核验，压缩包不携带大体积权重；initial_bindings.json列本地路径和SHA256。
