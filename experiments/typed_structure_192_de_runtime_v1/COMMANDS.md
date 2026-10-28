# 本地复核与未来运行命令

进入解包后的`typed_structure_192_de_runtime_v1`。使用现有环境，不安装或升级依赖。

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/data2/lyh/miniconda3/envs/sparse_block/bin/python
sha256sum -c SHA256SUMS
env -u PYTEST_ADDOPTS CUDA_VISIBLE_DEVICES='' "$PY" -m pytest -q tests/test_runtime.py
"$PY" package/launch.py dry-run \
  --protocol proposal.json --cost evidence/execution/cost_proposal.json \
  --output /tmp/DE_PROPOSED_NOT_STARTED \
  --manifest-output /tmp/de_dry_manifest_NEW.json
```

`dry-run`只写指定清单，不创建任务目录、不启动worker、不访问GPU、不建立正式账本。清单文件必须不存在。

下面是**未来获批后**的命令结构，本轮未执行。审批文件必须由真实审查及用户继续授权产生，当前包不提供可激活的审批文件。GPU UUID顺序对应冻结槽0至7，不用GPU编号代替UUID。不能为通过检查擅自解除平台设备隔离。

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
"$PY" package/launch.py formal \
  --protocol proposal.json --dictionary-dir dictionaries \
  --cost evidence/execution/cost_proposal.json \
  --approval "$REVIEWED_APPROVAL" --output "$NEW_RUN" \
  --gpu-uuids "$GPU_SLOT0" "$GPU_SLOT1" "$GPU_SLOT2" "$GPU_SLOT3" \
              "$GPU_SLOT4" "$GPU_SLOT5" "$GPU_SLOT6" "$GPU_SLOT7"
```

将来输出目录必须新建且在源包/输入之外。launcher是唯一正式入口，worker需要父进程管道票据；不要直接运行worker，也不要用旧OABC命令。没有自动恢复或失败重抽入口。

未来运行中查看`$NEW_RUN/schedule.json`、各任务`progress.json`、任务`.log`及`sample_<seed>/steps.jsonl`；安全停止可创建`$NEW_RUN/STOP`。完成/失败后保存整个目录，重点返回：`activation.json`、`resource_admission.json`、`execution_receipt.json`、`launcher_summary.json`、`unified_evaluation.json`、`summary_process.log`及各任务`summary.json`。

对已有运行离线重汇总可以使用下面的CPU入口。输出文件不得已存在，也不能写入不可变源包。返回0表示完整，2表示不完整；指标或输入损坏会保留错误，不伪造成功。

```bash
"$PY" summarize.py --run "$EXISTING_RUN" --output "$NEW_SUMMARY_JSON"
```

未来审批所需字段（说明，不是审批实例）：`independent_review=passed`、`user_continue_confirmed=true`及两个证据引用；`RELEASE_BINDING.json`中的source/protocol/cost/summarize/dictionaries绑定；`updates=1000`、`budget_seconds=8700`、`main_W1_definition_confirmed=type_weighted_incident4`、新的绝对output；冻结软件environment；实际resource_authorization含八卡UUID、明确权限依据、allocation_check_complete、reservation_conflicts、minimum_free_mib。不得仅复制这些字段伪造授权。
