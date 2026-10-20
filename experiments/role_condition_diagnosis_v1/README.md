# 角色条件失效诊断与单项后移融合候选

基线 c1163a885322c9dffc2919c8968c6a98adf375c7。
请先读 DESIGN.md；本包不是已经证明有效的生成模型，不含正式训练入口。
`baseline/`逐字节保留原审查源码与 manifest；`late_fusion.py`是独立新结构。
旧 C 权重仅供原结构诊断，不能作为新结构的原生恢复权重。

已执行：三个既有 C 权重的全验证面板目标切换／同条件重复、九状态逐层追踪、
60项旧回归＋15项新回归、本地一张2080Ti完整IMDB资源短测。
未执行：新学习面板、完整生成链、BW迁移或私有权重读取、生成质量验证。
原源码／权重／结果不改。`evidence/`含机器证据，ZIP不包含旧训练权重。

本地依赖实际为Python3.9.23/PyTorch2.1.0 CUDA12.1/PyG2.3.1/
NumPy1.26.4/SciPy1.11.0/NetworkX2.8.7/pytest7.4.4；不安装新依赖。
BW兼容性不由这次本地结果认证。普通模式不承诺跨设备逐位轨迹一致。

从本目录运行（替换路径为实际存在的本地归档；没有网络查找/补训逻辑）：

```bash
PY=/data2/lyh/miniconda3/envs/dihug/bin/python
sha256sum -c SHA256SUMS
env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTEST_ADDOPTS= PYTHONPATH="$PWD/baseline" \
  "$PY" -m pytest -q baseline/tests test_late_fusion.py

"$PY" diagnose.py --run /path/to/original_small_panel \
  --output /path/to/new_diagnosis
"$PY" layer_trace.py --run /path/to/original_small_panel \
  --output /path/to/new_layer_trace
"$PY" cpu_cost.py --run /path/to/original_small_panel \
  --output /path/to/new_cpu_cost
```

diagnose读取原panel.json、resolved_config.json、analysis/weight_hashes.json、
三个seed目录的C_final_1000.pt和C_validation_0000.json。全部缺失/哈希错误即失败。
初始权重由原seed重建且核对初始完整验证收据；不会将重建冒称独立保存的初始字节。

资源入口仅用于已获授权且空闲的本地卡；本票已执行一次，不自动重复。
完整IMDB文件按包内冻结哈希校验。所有输出目录必须不存在。

```bash
"$PY" resource_probe.py --data /path/to/IMDB.npz \
  --gpu-uuid GPU-EXPLICIT-AUTHORIZED-ID \
  --output /path/to/new_resource_probe
# 安全停止本次短测：
touch /path/to/new_resource_probe/STOP
```

监督器在启动和设备检查前计时，570秒计算总上限，终止清理最多额外5秒。
不解除已有CUDA可见性限制；若平台以索引设置可见性，需先人工明确对应UUID，
本入口不猜测、不重写该限制。SIGKILL不可捕获；普通可捕获中断清理本进程组。
该入口两模型各5次临时工程更新和一个t500反向步，不保存训练权重，不输出科学图。
没有OOM后降配置、截断CN或扩大预算的回退。

下一步仅提案：原C与后移C，3配对seed×1000更新、T64/4块、8采样seed×2条件；
保留全部输出，按实际图评价极端尾部、多个hub、边数、闭合、连通与拓扑重复。
预计本地CPU约41分钟核心工作，建议另批一个累计60分钟窗口；当前不启动。
