# 入口与复现

使用现有`sparse_block`环境，不安装升级依赖。下列相对路径均从本包根目录执行。准备票已结束后不重跑工程更新。

安全查看队列（不创建输出，不调用worker）：

```bash
python run.py dry-run --output /path/to/new_final_screen_run
```

CPU回归（选择有空间的本任务临时目录，不使用已满系统/tmp）：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTEST_ADDOPTS='' python -m pytest -q test_screen.py --basetemp /path/to/new_review_tmp
```

**尚未授权执行**的正式命令：

```bash
python run.py formal --output /path/to/new_final_screen_run \
  --approval /path/to/future_user_approved_binding.json --cost cost.json \
  --gpu-uuids GPU-slot0 GPU-slot1 GPU-slot2 GPU-slot3 GPU-slot4 GPU-slot5 GPU-slot6 GPU-slot7
```

审批须由新审查及用户预算授权产生，绑定最终source_sha256、protocol/asset_bindings/cost哈希、3000更新、预算秒数、实际8卡UUID及完整环境。没有本次授权不创建true审批。launcher记录首次进入formal的时间，所有启动检查计入；源码绑定覆盖训练、采样、汇总及LCC指标源码。旧package/launch.py和worker.py不是本轮入口。

停止：在实际run目录创建`STOP`。监督器在已审安全边界保存和停止，只处理自己的子进程，不自动恢复、重启、补抽或延时。SIGKILL/断电不可捕获，不能保证该瞬间完成保存。

原始输出：`DATASET_sTRAINSEED_ARM/ema_0.99/sample_SAMPLESEED/graph.npz`。
训练：对应模型目录的initial.pt、checkpoint_u1000.pt、checkpoint_u2000.pt、final.pt、validation.json、training.jsonl及binding.json。
统一评价：run/unified_evaluation.json；派生图及CSV：run/derived_lcc_evaluation/。监督器在失败时也尝试输出不完整汇总，并保留主失败原因。

独立CPU汇总可用：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python summarize.py --run /path/to/run --output /path/to/run/unified_evaluation.json \
  --deadline APPROVED_ABSOLUTE_EPOCH
```

该入口拒绝覆盖已有derived_lcc_evaluation目录。需要独立复核时，在独立副本上运行，不删除旧派生结果；运行截止仍受原授权约束，不借独立评价延长科学计算。
