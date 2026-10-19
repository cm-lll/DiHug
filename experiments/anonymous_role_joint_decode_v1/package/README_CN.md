# 匿名联合角色约束解码／采样：独立CPU工程原型

源码为本轮独立实现。旧版本、权重和结果不改；不连接BW，不训练，不调用模型，
不跑完整扩散链。所有概率都明确是工程输入，没有真实模型输出。

已装依赖：Python、NumPy、SciPy；测试还需pytest。入口不会pip安装或启用GPU。
具体实测版本见evidence/dependencies.json。所有运行输出必须是新目录且与包分离。

```bash
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  python verify.py --output /tmp/anonymous_decode_tests_01
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  python run.py small --output /tmp/anonymous_decode_small_01 --max-seconds 120
```

small完整枚举6节点32768张图（另有异质6节点枚举），并计算五种分布的精确有限和；
之后每方法4096次工程抽样。保留原独立分布的全部失败，不筛选补抽。
分布熵/有效图数是带编号图指标；另报类型保持同构类别，不能混称拓扑数量。

```bash
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  python run.py imdb --output /tmp/anonymous_decode_imdb_engineering_01 --max-seconds 300
```

imdb仅是一次固定完整角色匹配下的工程MAP／成本测试；随机工程评分与参考边身份无关。
不代表真实模型生成质量，不代表匿名全局MAP，更不是IMDB条件随机采样器。
300秒包括启动、核验、预处理、求解、保存和清理；超时不自动续跑或改配置。
此命令用于复现已经授权的单次规模检查，不自动在small后串行追加。

```bash
touch /tmp/anonymous_decode_imdb_engineering_01/STOP
```

安全停止由父入口清理自己的子进程组，退出码与controller.json一致。SIGKILL或
机器故障无法捕获。部分输出/超时可行解不能标为最优。工程图文件明确命名
engineering_graph.npz，不能冒充科学生成图。

公式、证明、反例和尺度边界见DESIGN.md；参数/seed/温度/容差在frozen_config.json，
冻结提交在发行manifest中。固定匹配解码与匿名精确条件采样是两个不同API。

## 本轮实际结果与验收边界

27项CPU测试完整通过、零跳过。6节点70个度2解中，60个六环、10个双三角形，
匿名拓扑仅2类。异质例完整联合角色约束54解，各维边际约束却有162解。
六环偏好下T1最大单图质量0.950489；双三角形偏好下T1连通率0.0661645。
T2提高熵但降低原logq评分；不能把满足角色或带编号图数增加当作免费拓扑收益。
异质例均匀角色分配混合与真正条件分布TV=0.411865；固定匹配MAP评分
-11.166113，匿名全局枚举MAP为-8.320450，不能混称。

完整IMDB工程检查仅执行一次，1653节点/1365378槽/10关系族，固定匹配输出4079边；
逐节点关系度、各类型完整联合频数与支持精确验收。包含启动/清理墙钟12.935秒，
worker CPU12.450秒，峰RSS481.23MiB。求解器浮点最优性界仅针对该固定匹配；
最大块目标间隙2.28e-13，全图logq界累加出现-2.33e-10舍入差，未隐藏。
与参考相同的边数由角色约束构造保证，不是模型学习结果。

独立audit_results.py不导入生产模块，另以整数权重/直接概率乘积和Python邻接集合
重算全部四组概率、主要精确指标、抽样重放、固定匹配穷举最优值，以及IMDB输出度数。
该独立复核未重做完整拓扑分类和优化器内部对偶证明；这些由生产穷举/专项测试及
已记录的浮点上下界支持。耗时约6.41秒CPU、峰RSS99.52MiB。可只读复核已封存证据：

```bash
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  python audit_results.py --small evidence/small --imdb evidence/imdb_engineering \
  --reference data/IMDB_movie_reference.npz --output /tmp/anonymous_decode_audit_01.json
```

所有主要指标来自枚举有限和（FP64），只有4096次观测不同图数及观测通过数来自抽样。
主面板81920次小图工程抽样；为补充固定匹配图证据，用同一配置、seed做过一次完全相同
小图重放，合计执行163840次抽样但只有同一个81920输出面板，原有全部数值数组逐位
一致，见evidence/trace_replay.json。不是新增科学样本。
没有真实模型logits、BW设备或完整科学链结果，没有训练、模型调用或GPU使用。
后续应先审查离散条件分布与固定匹配边界；本包不自动启动后继实验。
