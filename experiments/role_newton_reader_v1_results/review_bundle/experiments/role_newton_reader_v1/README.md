# 角色偏置约化对偶＋线搜索阻尼Newton

唯一科学目标仍为原熵系数1的完整联合角色软分配，源码/数值合同见PROTOCOL.md和config.json。
solve返回普通P、完整logP、独立验收回执；required只接纳new_contract_pass。
538个实际矩阵满足原普通概率接口，6个大图矩阵仅满足有遗漏质量证书的新尾部输出合同，必须区分。
数学零/约束强制零/有限logP但普通输出舍入零分别记录，不按概率删支持。
原DP和行列缩放均保留，没有回退、温度变化、训练或完整新链。

完整本地工程复验（不需要私有权重，不运行面板）：
```
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTEST_ADDOPTS= \
 python -m pytest -q experiments/role_newton_reader_v1/test_newton.py \
 experiments/role_newton_reader_v1/test_receipt.py \
 experiments/role_newton_reader_v1/test_imports.py \
 experiments/role_logscale_reader_v1/test_scaling.py \
 experiments/role_logscale_reader_v1/test_additional.py \
 experiments/role_logscale_reader_v1/test_numeric_guard.py
```
实际环境为Python3.9.23/NumPy1.26.4/SciPy1.11.0/mpmath1.3.0，旧回归还需要已存在的Torch/PyG。
不自动安装依赖。run.py --execute为本票已执行一次的诊断入口，绑定唯一已有60分钟账本，
不得为复验重建预算或自动重开面板。原始初次JSON故障及辅助导入故障日志均保留。

输入边界：528个小图LL来自已封存132状态，逐载荷核对9b010的预期哈希；
16个大图LL旧版本未保存数组字节，本票只按旧源码/旧种子确定性重建，先核对旧状态哈希，
再核对旧缩放状态/迭代/残差完全一致。不能声称核对了不存在的历史LL哈希。
新生成的544矩阵各自哈希封存；不抽取新状态。

predict.py只在原132个Xt上使用已验收的Newton分配，逐一核对重新计算的LL与封存输入一致；
原角色编码、主干、预测头和后验均复用，记录普通FP32算术界与尾部舍入界，绝不抽样。
全部结果及源码补丁见../role_newton_reader_v1_results。完成发布后停止等待审查。
