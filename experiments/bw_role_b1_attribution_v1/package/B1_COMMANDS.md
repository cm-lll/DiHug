# 手动命令；本轮只交审查，不自动运行真实B1

新ZIP/校验文件上传至private_data后解压新目录，不覆盖任何旧包或原结果。

```bash
set -euo pipefail
cd /root/private_data/gnn_projects
sha256sum -c dihug_sparse_bw_role_b1_attribution_v1.zip.sha256
unzip -n dihug_sparse_bw_role_b1_attribution_v1.zip
cd dihug_sparse_bw_role_b1_attribution_v1
export PYTHON_BIN=/usr/local/bin/python3.11
bash b1.sh test --output /root/private_data/gnn_projects/B1_attribution_engineering01 --seconds 120
bash b1.sh inspect
```

inspect只读并打印文件当前指纹，不调用模型、不创建输出。将其与**已封存副本**核对。
以下两值必须来自封存记录，不能仅现场自算后当作来源证明；当前对话未提供这两值。

```bash
set -euo pipefail
: "${OLD_SUMMARY_SHA:?请设置封存summary.json的SHA256}"
: "${OLD_CASE_SHA:?请设置封存case_05.json的SHA256}"
bash b1.sh diagnose \
  --short-run /root/private_data/gnn_projects/role_mt_v1_2_short_20261018T091510Z \
  --export /root/private_data/gnn_projects/role_mt_v1_2_export_20261018T091055Z \
  --summary-sha256 "$OLD_SUMMARY_SHA" --case-sha256 "$OLD_CASE_SHA" \
  --output /root/private_data/gnn_projects/B1_attribution_run01 --seconds 900
```

不提供新步长、频率、权重、状态或模型选择开关。只处理B1旧记录。
安全停止：touch /root/private_data/gnn_projects/B1_attribution_run01/STOP。
或向activation.json记录的本次控制器发送INT/TERM/HUP，不可全局pkill。
此诊断不提供自动重试或从部分状态续算；再次手动重放需新输出目录，仍是同一旧接受步。

主要产物：
- attribution.json：梯度分量求和证据、类型CF/tail损失变化、内积/余弦、一阶贡献/实际变化、原指标复核。
- B1_gradient_evidence.npz：完整分量/原总/独立指标logit梯度、原名义变化、前后p0/q和canonical槽。
- genre_pmf_cdf.npz：完整节点/均值/匿名目标角色PMF与CDF、各W1求和项、度支持及目标行/频数。
- summary.json/controller.json：完成、失败、未完成、成本和输入输出哈希。超时不报成功。

本轮本地小图通过不代表上述真实文件绑定、旧case_05复现、真实梯度冲突或实际W1分段已得到结果。
