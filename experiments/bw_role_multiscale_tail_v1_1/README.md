# bw_role_multiscale_tail_v1_1 审查交付

基于固定d1e1d9a，仅工程修复；原版本不改。
package/REPAIR_NOTES.md说明修复边界，package/INPUT_CONTRACT.md定义真实旧产物适配。
package/evidence_repairs/是本轮CPU工程证据；继承的evidence目录属于旧版本。
package/v1_to_v1_1_repair.diff为相对基线包的修复补丁。
build_repair.py核验81个原文件和数学/控制器/科学判据AST不变，制作独立ZIP。

## 手动命令（交付后等待复审，不自动运行）

先上传新ZIP与外部SHA256SUMS；不要覆盖原包。以下不安装依赖，不启用GPU：

```bash
cd /root/private_data/gnn_projects
sha256sum -c dihug_sparse_bw_role_multiscale_tail_v1_1.zip.sha256
unzip -n dihug_sparse_bw_role_multiscale_tail_v1_1.zip
cd dihug_sparse_bw_role_multiscale_tail_v1_1
export PYTHON_BIN=/usr/local/bin/python3.11
export CUDA_VISIBLE_DEVICES=''
bash multiscale.sh test --output /root/private_data/gnn_projects/role_mt_v1_1_cpu_tests --seconds 300
bash multiscale.sh cost --output /root/private_data/gnn_projects/role_mt_v1_1_synthetic_cost --seconds 120
```

下面转换命令核对真实旧目录，不改变概率或调用模型；缺原轨迹/权重就拒绝。
OLD_ZIP应指向实际已上传的原诊断ZIP。ledger的SHA应与旧封存清单核对，
现场计算值只是完整性基准，不是独立来源认证。

```bash
OLD_RUN=/root/private_data/gnn_projects/joint_role_frozen_diag_v1_run
OLD_ZIP=/root/private_data/gnn_projects/joint_role_frozen_diag_v1.zip
sha256sum "$OLD_ZIP" "$OLD_RUN/evidence_hashes.json"
LEDGER_SHA=$(sha256sum "$OLD_RUN/evidence_hashes.json" | cut -d' ' -f1)
EXPORT=/root/private_data/gnn_projects/role_mt_v1_1_export01
bash multiscale.sh export-legacy --legacy-zip "$OLD_ZIP" --legacy-run "$OLD_RUN" \
  --legacy-ledger-sha256 "$LEDGER_SHA" --output "$EXPORT" --seconds 900
```

复审后如用户批准实际短测，再单独手动执行：

```bash
MANIFEST_SHA=$(sha256sum "$EXPORT/six_states.json" | cut -d' ' -f1)
EVALUATION_SHA=$(sha256sum "$EXPORT/original_evaluation.json" | cut -d' ' -f1)
bash multiscale.sh short --manifest "$EXPORT/six_states.json" --manifest-sha256 "$MANIFEST_SHA" \
  --evaluation "$EXPORT/original_evaluation.json" --evaluation-sha256 "$EVALUATION_SHA" \
  --output /root/private_data/gnn_projects/role_mt_v1_1_short01 --seconds 900
```

每次使用新输出目录。STOP：touch 输出目录/STOP。
也可向本次控制器PID发送INT/TERM/HUP（见activation.json），不可全局pkill。
状态见controller.json、summary.json、progress.json；完成需要文件/计数/最终哈希齐全，
不接受只有status标签。Ctrl+C和STOP保留未完成证据，不算科学失败。
本版CPU参考；BW性能、真实旧数组兼容性及六状态科学结果均待核验。
