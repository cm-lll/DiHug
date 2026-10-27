# 192节点 CN / ReFeX / RolX v2 工程交付

这是修订与数值诊断包，**不是正式训练版本**。先读REPORT_zh.md和REVISION.md。
保留v1旧版本，拒绝把本包工程权重/合成字典作为正式实验初始化。

活动代码在package；historical_v1仅是本票隔离的旧入口/字典记录，不要执行。
package/prepare.py旧拟合入口已明确禁用；实际config使用合成字典。
没有formal、T1000生成、监督器或自动续训入口。

以下仅运行CPU回归，不更新真实模型：
```bash
cd /你的独立解包目录/typed_refex_rolx_192_v2
env PYTEST_ADDOPTS= PYTHONPATH="$PWD/package" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /data2/lyh/miniconda3/envs/sparse_block/bin/python -m pytest -q \
  package/test_engineering.py test_revision_extra.py
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /data2/lyh/miniconda3/envs/sparse_block/bin/python audit_v2.py --evidence evidence
sha256sum -c SHA256SUMS
```
实际环境路径不同请仅选择既有兼容Python，不自动安装/升级。证据环境见evidence/probe/environment.json。

原本次设备补测入口为package/probe.py，已经用完每格3次临时更新额度。
**不得直接用新目录重新运行以重置本票账本**；需要再次执行须另行授权和独立预算。
STOP文件放在probe输出目录中，在阶段边界停止；Ctrl+C/SIGTERM保存部分summary。
本包不包含临时工程checkpoint字节，但逐个路径/大小/哈希登记在evidence/local_engineering_checkpoints.json；它们仍保留本地。
离线审计不认证checkpoint载入字节，仅复核原回执与哈希记录；若实际pt存在则额外重哈希。
