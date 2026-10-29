# 新增六任务：入口与停止边界

使用原 sparse_block 环境。CPU干运行不会创建实验目录、审批或worker：

```bash
python run.py dry-run --output /data2/lyh/gnn_project/DiHug/results/early_cn_six_NOT_ACTIVATED
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
TMPDIR=/data2/lyh/gnn_project/DiHug/results/early_cn_six_review_tmp \
python -m pytest -q test_extension.py --basetemp /data2/lyh/gnn_project/DiHug/results/early_cn_six_review_tmp/test
```

检查脚本check_extension.py、check_pairing.py只作本票证据；本票已结束后不要重跑GPU工程更新。historical_test_reference.py是原票测试源码，不是新队列的测试入口。

**正式命令尚未授权，审批文件未创建：**

```bash
python run.py formal --output /data2/lyh/gnn_project/DiHug/results/NEW_SIX_RUN \
  --approval /path/to/FUTURE_APPROVAL.json --cost cost.json \
  --gpu-uuids GPU-slot0 GPU-slot1 GPU-slot2 GPU-slot3 GPU-slot4 GPU-slot5
```

未来审批须绑定本次独立审查与用户明确授权、源码/协议/资产/成本哈希、3000更新、新输出目录、六卡UUID及资源权限。不得引用原两任务审批。正式窗口从新formal入口开始，最后20分钟交付；本票没有正式起止时间。GPU0/1禁止选用。六卡不可用须报告，不删任务或延长。

任务目录 DATASET_sTRAINSEED_VARIANT，采样子目录 ema_0.99/sample_SAMPLESEED。每卡独立训练完成后接自己的两条链。STOP只放新增run根目录；不向原run写STOP，不调用原监督器。失败停止仅处理本launcher的子进程，不重启、不补抽、不恢复。

新批汇总run/unified_evaluation.json先独立交付，包含完整/缺失身份。原两任务仍在运行时，不导入它们的未封存指标；cross_batch_missing明确保留。原两任务封存后，可只读合并，不重新评价图：

```bash
CUDA_VISIBLE_DEVICES='' python cross_batch.py \
 --new-summary /path/to/NEW_SIX_RUN/unified_evaluation.json \
 --output /data2/lyh/gnn_project/DiHug/results/NEW_UNION_REPORT.json
```

合并入口验证新报告源码、原两任务审批/激活与完成回执、旧结果SHA，拒绝重复或未知身份。输出必须新文件，不覆盖任一批结果。原seed D/E×早晚四格、新seed E早晚分别报告；原图和LCC分开。原C只作PubMed原seed参照。

大初态与工程检查点不夹带ZIP；外部资产路径/哈希见baseline_bindings.json及evidence/artifact_index.json。正式初态重新按冻结规则构造，禁止使用after1或工程更新权重。
