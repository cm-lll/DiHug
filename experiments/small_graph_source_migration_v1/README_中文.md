# 已完成小图实验：源码与完整数据迁移包

本包只封装已有源码及数据，未改模型、未加载权重、未前向、未训练、未生成，也未连接Notebook。它是供迁移和审查的材料包，不是已经适配完整图的Notebook正式运行包。

## 从哪里开始

1. 用ZIP旁的SHA256核对下载，解压到一个新的数据盘目录。
2. 在解压根目录运行 `python -B verify_bundle.py`。仅需Python标准库，校验文件、大小和语法，不导入模型、不启动worker。
3. 先读本文件、`ENTRYPOINTS.md`、`environment/DEPENDENCIES.md`及`PORTABILITY_LOCATIONS.csv`，然后由Notebook容量短测任务按需要引用源码。此包没有自动安装、自动训练或有效审批文件。

## 实际执行来源

|目录|固定源码提交|用途|
|---|---|---|
|snapshots/ac_screen|472e533fd28cfc067702b1b75082ac5105140a2d|已完成A/C等组3000步EMA筛选；A=CN，C=ReFeX+RolX、无CN|
|snapshots/early_count_stage500|974e5dca8cbdd05ffddc53fd599ae815828d8b56|E-earlyCN的CE/CE_COUNT从0到500，当前有界数量项接线|
|snapshots/early_count_length|c3f9ed007167484e4d336b2768ea2376b92fcb6e|同轨迹从500继续，3000阶段生成、PubMed6000终点生成|
|snapshots/imdb_continue|7f095e11af2c1441d57012d86d2b7f4ae9bf8058|IMDB同轨迹从3000继续6000的历史适配层|
|evaluation|69ca80556320c1deef7c37eb640981397019d841|最终CPU整理及dataset字段发布修复；纯指标源码另冻结保存|

A/C封存结果提交为aa4227b9727ca6853ba7e26412ce6d438a0fa971；500步结果为68a93bd2d6f39a71e6d525dac8647d9435df25ec；最终合并发布为b656f75c412b8fa067f800a00eb886cab924486d。本包不夹带这些结果集。

`provenance/*actual_execution_manifest.json`来自真实启动时的文件清单；所包含的清单内源码/NPZ逐项与当时SHA256核验。代码从上述固定Git对象提取；未入Git的NPZ只在其实际哈希与运行清单相等时纳入。`MANIFEST.json`与`SHA256SUMS`绑定本包全部载荷。原执行清单中的组合摘要算法与本包SHA清单不同，二者不混称。

各快照保留原文件字节；它们不是一套后来拼装并假称已经运行的统一launcher。旧文档或参考文件中的O/B/D、多EMA、Newton等历史定义不表示本次新增候选；正式目标源码入口以下表为准。`package/audit_reference`只作历史参考，不接入本次A/C/E-earlyCN路径。`*_synthetic_basis.npz`是旧接口测试夹具，随原绑定留存；**真实C/E必须读取dictionaries/dictionary_*/final_basis.npz，不能使用合成字典。**

## 完整数据与小图资产

`data/full_original/IMDB.npz`是1653节点、4079边原件；`PubMed.npz`是1565节点、12284边的Gene/Disease/Chemical/Species图，不是PubMed23论文检索语料。

- IMDB SHA256：`10f1dbbbe55c6a069594b2dcb590d1b8295ca5436320ef4597cd30e6b7f73e42`
- PubMed SHA256：`dacff5a392e23fbcfb695076a14a6db5317f4a4382b69af2a7596a1a217163f8`

保留原NPZ全部数组和原编号，未裁剪、未重抽、未修边。`*_historical_config.json`及provenance记录类型、schema、关系映射、先验和来源；这是历史完整图配置，**不能当作新架构正式完整图配置**。NPZ的edges为E×2、relations从1开始；网络内部通常用2×E及从0开始的family。只在接口边界转换一次。

各快照package/assets/含最终attempt02的192节点小图：IMDB323边、PubMed926边。dictionaries/含已冻结真实小图字典及selection/panel/frozen_splits。IMDB K=5/4/3/2（14列），PubMed K=8/8/8/6（30列）；PubMed四类型命中搜索上界。系数是非负强度，不作行概率归一化。

这些字典仅绑定192节点训练面板，**不是完整图正式字典**。完整图字典材料、选K、拟合、成本及准入未在本包执行。数据原件没有另行声称新的再分发许可证；现有代码LICENSE按各快照保留，历史vendor文件的原出处继续保留。

## 当前损失及EMA

CE对照：每份输入 `sumCE/18336`。CE_COUNT：在此基础上加`0.001 * mean_r(sqrt(1+u_r²)-1)`，其中μ是完整无向槽的p0有边概率和（包含固定支持贡献）、m是X0关系边数、u=(μ-m)/sqrt(max(m,1))，仅平均可动关系。以`early_count_length/formula.json`和`package/count_aux.py`为精确定义；无旧K分母，不是已否决的λ=0.01公式。固定关系支持、零目标和空可动集合处理均在源码及测试中保留。

normal2一次清梯度，两份独立Xt各用上述单输入损失乘1/2反向，最后一次AdamW更新。EMA0.99从θ0独立副本开始，每个完整更新之后递推，首步不直接复制θ1；训练仍使用raw。原概率核和采样不设边数配额、不校准、不修图。

E-earlyCN保留ReFeX/RolX节点输入；37维CN经零初始化、无偏置37→64投影在主干第一层前加入边表示；末端37维CN已移除。CN只依赖完整Xt真实边，正反边对应共享，不混入暂停的CN Dropout候选。

## 不能直接迁移运行的部分

- 小图入口的全部可动槽一次查询、S=18336、192节点资产校验、训练seed/阶段白名单仍在。完整图IMDB活跃槽997313、全schema1365378；PubMed活跃/全schema1223830。不能沿用小图分母或把一次全查询成本当完整图容量保证。
- 历史完整图20块查询代码在kernel/runtime中保留，但本次小图入口使用rounds=1；本包没有把A/C/E-earlyCN在完整图20块下重新接线或验收。CN及ReFeX必须从完整Xt取上下文，不能只在查询块诱导子图算；不同查询计划也不能宣称主干输出天然相同。
- model_io.configure仍要求单张RTX2080Ti；历史environment记录Torch2.1/CUDA12.1。未证明任意Notebook后端数值相同，未更换环境或算子。
- E-earlyCN初始入口读取initial_bindings.json中的封存0步权重绝对路径；续训入口还读取500/3000步父检查点。**本包遵照要求不含任何.pt或模型权重，因此不能原样启动这两类入口。** 初态迁移/构造合同与源码保留，Notebook若另建可运行初态，应另行明确并核验。
- launcher保留旧审批、成本、UUID、绝对路径和已过期账本校验。运行源码字节未改；本包不提供激活审批，不应绕过worker唯一入口校验。不能复用旧审批开新任务。
- evaluation/source/build.py、report.py和IMDB历史adapter仍绑定node1旧结果路径及身份面板；属于最终报告接线，不是任意输入的通用CLI。纯指标函数支持实际节点集合，但完整图上昂贵motif/路径成本和资源上限未测试。

上述命中位置见PORTABILITY_LOCATIONS.csv。它是静态检索索引，不等价于已经完成全部迁移修改；不要对所有数字192或18336盲目替换。

## 评价范围

最终入口/定义见ENTRYPOINTS.md及evaluation/definitions/。包含原图和LCC、实际节点容量、固定参考类型权重incident4 W1、all10兼容值、联合度TV、度谱/尾部、类型三角、闭合机会与比例、真正四环与legacy K4、MP1/2/3数量与组成、路径/连通及样本差异。不能把LCC保证的连通性当模型收益。

最终dataset发布修复以单独文件保存，未覆写原执行时finalize.py；真实运行清单仍绑定当时原件。原生HGEN powerlaw缺依赖、社区标签/未实现项等仍以原报告缺项定义为准，不把“当前全部已实现指标”说成全部原生HGEN。null、缺图和失败必须保留。

本次只做源码哈希、语法、数据原件及ZIP独立解包校验；不是Notebook预检通过，也未重复历史GPU验收。没有训练或生成授权被本包激活。
