# 入口及复核

使用现有sparse_block环境；不安装升级依赖。只支持本目录run.py正式入口，package下历史入口不属于本候选。

安全查看队列（不创建运行目录、不启动worker）：

```bash
python run.py dry-run --output /data2/lyh/gnn_project/DiHug/results/early_cn_formal_NOT_ACTIVATED
```

CPU回归；准备票结束后不要重跑GPU工程更新：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
TMPDIR=/data2/lyh/gnn_project/DiHug/results/early_cn_review_tmp \
python -m pytest -q test_early.py --basetemp /data2/lyh/gnn_project/DiHug/results/early_cn_review_tmp/pytest
```

baseline_bindings.json引用本地封存原E的0步初态和旧评价文件，必须可读且哈希相符。包不包含大权重；初态/工程权重本地路径及哈希见evidence/artifact_index.json。不能把工程after1.pt作为正式初态。

**以下正式命令尚未授权：**

```bash
python run.py formal --output /data2/lyh/gnn_project/DiHug/results/NEW_EARLY_CN_RUN \
  --approval /path/to/future_user_approved_binding.json --cost cost.json \
  --gpu-uuids GPU-PubMed-slot0 GPU-IMDB-slot1
```

审批需要未来真实用户授权和独立审查，绑定source_sha256、protocol/asset_bindings/cost哈希、3000更新、预算、两张设备UUID及环境。source绑定包含baseline_bindings.json、训练/采样/汇总脚本；实际旧资产也验哈希。不提供可激活审批样例。

STOP：在实际run根目录创建STOP。保存仅在完整normal2边界；沿原监督停止机制，不自动重启/补抽/延期。最后20分钟留作评价交付。输出目录必须全新。

模型输出DATASET_sTRAINSEED_E，采样子目录ema_0.99/sample_SAMPLESEED；统一评价unified_evaluation.json明确E_earlyCN身份。派生图/映射/CSV在derived_lcc_evaluation。汇总失败保留原工作失败原因，缺图仍输出不完整状态。旧原图及对照不覆盖。
