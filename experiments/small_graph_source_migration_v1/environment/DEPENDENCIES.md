# 依赖：记录与迁移边界

真实A/C筛选与数量项续训的审批环境记录见本目录两个`*_recorded_environment.json`，只提取环境字段，未复制有效审批。

共同记录：Python3.9.23、Torch2.1.0+cu121、CUDA12.1、torch-geometric2.3.1、NumPy1.23.0、SciPy1.11.0、NetworkX2.8.7、pytest8.4.2；RTX2080Ti、FP32、TF32关闭、线程1、确定性配置按原代码保留。配置并不自动保证第三方算子确定性；历史CPU/GPU逐元素等价未通过的结论不改。

现有sparse_block环境中读取安装元数据：scikit-learn1.6.1、matplotlib3.7.1、pandas1.4.0、torch-scatter2.1.2+pt21cu121、omegaconf2.3.0；这些是打包时元数据补充，不是历史运行回执未记录字段的追溯证明。torch-scatter有dist元数据不等于扩展可以导入，也不证明PyG该次实际分派路径；本次没有导入Torch/PyG/scatter来重验后端。

模型/特征主要需要Torch、PyG、NumPy、SciPy、scikit-learn；完整CPU评价主要需要NumPy、SciPy、NetworkX，绘图需要matplotlib。`features.py`/`role_dictionary.py`即使本轮不拟合也有sklearn导入。测试需要pytest。没有自动安装脚本；不同Notebook环境请自行核对兼容性，不直接升级原环境。

vendor目录保留HGEN等出处源码。当前评价器通过AST提取所需纯函数，避免整模块导入缺失依赖。旧原生HGEN入口所需powerlaw、igraph、numba、torch_two_sample，及历史四环审计依赖完整旧仓库，**不等于当前stats/compare入口的必需依赖**；本包不保证这些独立旧脚本可单独执行。既有缺项不补造数字。

`static_import_names.json`为所有历史源文件的静态import名称并集，混有本地模块/标准库/未执行历史入口，不是可盲目pip安装的requirements清单。
