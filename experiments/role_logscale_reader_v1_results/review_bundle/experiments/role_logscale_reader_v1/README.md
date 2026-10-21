# 对数域容量软读取候选（CPU，独立版本）

先读PROTOCOL.md及../role_logscale_reader_v1_results/RESULTS.md。
只改完整联合角色似然到软归属的计算，不改变任何模型权重或参数形状。
默认推理适配器adapter.reader_context使用required：未收敛/不可行/下溢均抛错，绝不回退。
scaling.scale是诊断低层接口，允许返回带失败状态的末次迭代，**不能将其失败输出送进网络**。
adapter为单进程单线程显式上下文，非线程安全；没有宣称已经接入生产采样器。

纯算法复验（不会加载私有权重或运行链）：
```
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTEST_ADDOPTS= \
 python -m pytest -q experiments/role_logscale_reader_v1/test_scaling.py \
 experiments/role_logscale_reader_v1/test_additional.py
```

本票run.py --execute已执行一次，仅在唯一已有3600秒账本上运行；会拒绝重复launch。
它读取三广覆盖final6000及原完整检查点做绑定核验，不初始化优化器，不训练。
仅原CPU生成器只读重放旧路径以恢复Xt，全部旧64步状态/查询/uniform/RNG哈希核对；
候选仅在相同Xt上计算概率，没有按新概率推进任何完整链。
failure_evidence.py补存拒绝状态的完整LL和末次未接受矩阵；不将其送入网络。
这些补载荷的额外旧链重放也不是新的独立生成样本。

large.py只使用已核对的完整IMDB/PubMed参考、schema和匿名联合角色，构造原前向噪声输入，
不实例化大图网络。未通过缩放数值门的项目不能宣称读取成功。
small_cost_full.py覆盖44个seed11状态；输入解压在tracemalloc开始前完成，
保留初版27状态成本记录，但采用完整版本作主要成本证据。

发布材料包含全部132状态（含55拒绝状态）、完整似然/分配/p0/q、来源及失败回执，
21项最终回归及独立解包证据。旧源码/权重/输出不变，私有PT不在ZIP中。
交付前仅补强非有限迭代/目标值的立即拒绝；原scaling.py字节保留并核对原封存SHA，
528个真实LL矩阵的新旧概率逐位相同，状态及迭代数不变，未放宽残差或修改任何科学设置。
CPU普通模式，原GPU数值门未修改；本票没有GPU/BW/notebook测试。
不应将该候选当作已认证的无损DP替代；任何完整生成或数值修订须另获授权。
