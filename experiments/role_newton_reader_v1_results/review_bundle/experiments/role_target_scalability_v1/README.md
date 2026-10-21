# 新角色目标与精确软读取可扩展性准备

基线33e2c28b6ab1919fcec56f1deb58b1210b63ffce。只准备，不训练/调用模型/运行完整链。
主要结论：小图五目标均可实现且精确读取成本可承受；两张完整大图的原容量DP均被
原资源门拒绝，不能直接扩展。详见`PROTOCOL.md`和`SCALABILITY.md`。

* `config.json`：预先固定五目标、seed、生成数量和未授权标志。
* `algorithms.py`：整数成本核算、度核验、微小二部流/Gale–Ryser证明工具。
* `prepare.py`：只读核对本地权重/原完整检查点及绑定、训练记录、数据；生成匿名目标、
  证明图和成本证据。没有模型构造/前向或采样入口。
* `evidence/`：全部实际机器结果。`proof_only/*.npz`只用于存在性证明，严禁作为
  模型输入、先验初始化、逐节点角色安排、修图或科学输出。
* `test_planning.py`：本票14项纯算法/绑定检查；旧135模型测试只作为历史证据，不冒充
  本票重新执行。此票不需要额外神经前向。

复验纯算法：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTEST_ADDOPTS= \
  /data2/lyh/miniconda3/envs/dihug/bin/python -m pytest -q \
  experiments/role_target_scalability_v1/test_planning.py
```

重做准备只使用调用者明确提供的本地路径，输出必须新建且不与输入/源码重叠：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  /data2/lyh/miniconda3/envs/dihug/bin/python \
  experiments/role_target_scalability_v1/prepare.py \
  --run /data2/lyh/gnn_project/DiHug/results/role_robust_training_6000_20261021_v1/learning_run01 \
  --panel /data2/lyh/gnn_project/DiHug/results/role_late_paired_20261020_run01 \
  --output /tmp/role_target_scalability_review_new
```

这不是正式验证启动命令。不提供自动进入训练/完整生成的脚本；120条小图链及可选
120秒CPU前向短测均等待用户决定范围与新预算。GPU数值门失败事实继续保留，不访问
notebook/BW。所有旧代码、权重与结果不变，交付后停止等待审查。
