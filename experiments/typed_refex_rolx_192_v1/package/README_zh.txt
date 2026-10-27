这是实现/工程短测审查包，不是正式训练运行包。没有formal或chain入口。
本轮不调用旧BASE权重、不导入旧E07 worker/supervisor，不执行采样或生成后处理。

目录：
  config.json：本轮完整展开配置及两数据集/冻结基哈希。
  assets/：净化的192节点图(删除original_ids特征入口)、固定8角色基。
  data_audit.json：原附件文件清单核验、精确诱导、schema/prior/槽容量与分量。
  hsgd/：E07接收快照的历史纯核心源码原字节；旧大图helper未被新数据入口使用。
  model_variants.py：三个从头初始化的模型；A保留旧BASE有效当前度分支与CN；B/C移除CN。
  features.py：固定类型化递归特征及冻结基非负编码。
  audit_reference/：只给回归调用的旧纯模型公式，无旧调度入口。
  DESIGN_zh.txt：精确方法、信息边界、原论文差异、已知数值失败。

本地已使用环境：Python3.9.23，Torch2.1.0+cu121，PyG2.3.1，NumPy1.23.0，SciPy1.11.0。
不自动安装依赖。CPU线程均1；GPU仅一张RTX2080Ti，现已释放。

安全复核（不更新权重/不生成图）：
  cd package
  env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    /data2/lyh/miniconda3/envs/sparse_block/bin/python -B -m unittest discover -s . -p test_engineering.py -v

benchmark.py用于有限工程更新，不是正式训练入口。当前票已用完各数据集每组5次额度，
不要再执行该入口。executed_update_counts.json明确记录30次实际临时更新；新授权才可另做工程更新。
forward_probe.py是零优化器更新/零抽样的成本诊断；当前票已经完成，也不自动重跑。
STOP通过在运行时指定的独立输出目录创建STOP文件，循环边界停止；SIGINT/SIGTERM也保存失败/停止摘要。
不声称SIGKILL可捕获，不删除任何历史输出。

下一步只审阅：数据与方法定义、GPU数值门失败、PubMed基未收敛、成本估计。
正式更新数、seed数、T/查询合同是否沿用、学习面板总预算需另行冻结；本包不自行决定或执行。
