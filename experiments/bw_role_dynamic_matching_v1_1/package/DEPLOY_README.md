# v1.2 三项运行保护修复，交付后等待复审

基线4740416，保留v1.1及更早版本。数学目标、科学判据、900秒总上限未改。
不连接BW、不训练、不跑真实六状态或完整生成链。
GUARD_FIXES.md和INPUT_CONTRACT.md说明新边界，evidence_guard为本轮证据；
继承evidence_repairs等目录属于旧版本。

## 复审后手动运行

上传新ZIP和同名.zip.sha256校验文件，不覆盖旧包：

```bash
set -euo pipefail
cd /root/private_data/gnn_projects
sha256sum -c dihug_sparse_bw_role_multiscale_tail_v1_2.zip.sha256
unzip -n dihug_sparse_bw_role_multiscale_tail_v1_2.zip
cd dihug_sparse_bw_role_multiscale_tail_v1_2
export PYTHON_BIN=/usr/local/bin/python3.11
export CUDA_VISIBLE_DEVICES=''
bash multiscale.sh test --output /root/private_data/gnn_projects/role_mt_v1_2_cpu_tests --seconds 300
```

真实转换仍需要完整旧产物与来源文件。下面使用用户从旧封存报告ZIP核对的预期ledger哈希，
先比对，再传给转换器；不得把当前文件自算值替换为预期值，也不得覆盖旧ledger。
OLD_ZIP需要指向实际上传位置。任何不一致立即停止，原目录不能被“修复”。

```bash
set -euo pipefail
OLD_RUN=/root/private_data/gnn_projects/joint_role_frozen_diag_v1_run
OLD_ZIP=/root/private_data/gnn_projects/joint_role_frozen_diag_v1.zip
EXPECTED_LEDGER_SHA=961e6d93c8194396223ac9062b1658ac87232e82710b1ab3d36e722ed2c00196
printf '%s  %s\n' "$EXPECTED_LEDGER_SHA" "$OLD_RUN/evidence_hashes.json" | sha256sum -c -
EXPORT=/root/private_data/gnn_projects/role_mt_v1_2_export01
bash multiscale.sh export-legacy --legacy-zip "$OLD_ZIP" --legacy-run "$OLD_RUN" \
  --legacy-ledger-sha256 "$EXPECTED_LEDGER_SHA" --output "$EXPORT" --seconds 900
```

只有复审后获准实际短测再手动执行（转换产物的现算哈希仅作本次输入完整性绑定）：

```bash
set -euo pipefail
MANIFEST_SHA=$(sha256sum "$EXPORT/six_states.json" | cut -d' ' -f1)
EVALUATION_SHA=$(sha256sum "$EXPORT/original_evaluation.json" | cut -d' ' -f1)
bash multiscale.sh short --manifest "$EXPORT/six_states.json" --manifest-sha256 "$MANIFEST_SHA" \
  --evaluation "$EXPORT/original_evaluation.json" --evaluation-sha256 "$EVALUATION_SHA" \
  --output /root/private_data/gnn_projects/role_mt_v1_2_short01 --seconds 900
```

每次使用新输出目录；输出不能落在被保护输入树内或与它重叠。
touch 输出目录/STOP，或向activation记录的本次控制器发送INT/TERM/HUP，可安全停止。
SIGKILL不可捕获。状态和真实兼容性边界沿用输入合同，任何未完成均不得标为成功。
