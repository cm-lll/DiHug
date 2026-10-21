# 独立CPU软读取候选审查包

阅读RESULTS.md：候选数值合同检查通过，但55/132真实状态拒绝，不建议直接替代DP。

源码在../role_logscale_reader_v1；source_vs_7b45d502.patch包含唯一候选和诊断入口。
evidence保留唯一60分钟账本、冻结数值合同、三广6000权重绑定和原固定验证、
132真实Xt及完整LL/分配/角色上下文/p0/q（含55失败载荷）、逐关系出生/消失/净变化、
16完整大图角色库存读取测量、19项原回归和21项最终数值防御回归。
原模型/角色编码/似然/噪声/优化及生成器全部未改；候选只在CPU评估，不重新训练。

数值防御补丁的原scaling.py及旧SHA保留，528实际矩阵新旧概率逐位相同；
guard_cost为交付版完整似然+缩放的重测，原成本记录不覆盖。计时口径及局限见RESULTS。
review_bundle/history含三模型各11条旧链的原图、完整64步哈希和来源回执，
用于核对重放来源；这些均不是本票新增独立样本。PT不上传，哈希/配置/来源完整保留。

ZIP独立解包后的纯算法复验：
```
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTEST_ADDOPTS= \
 python -m pytest -q experiments/role_logscale_reader_v1/test_scaling.py \
 experiments/role_logscale_reader_v1/test_additional.py \
 experiments/role_logscale_reader_v1/test_numeric_guard.py
```
不需要私有权重；需要已记录的本地NumPy/SciPy/Torch/PyG环境，不自动安装或更新依赖。
本票已停止，不应再次执行run.py --execute，入口会拒绝重复launch；没有自动生成或训练命令。
