# 单参考异质图离散扩散：源码审查快照

**2026-11-06 输入增强候选审查分支：** 新增A晚CN上的类型化三角/分量节点输入及同分量候选对输入，保持CE、后验和采样算法。请先阅读 [调整目的、公式、完整路径、CPU检查、限制及待批准预算](review/structure_input_v1/REVIEW_ZH.md)。候选类位于 [StructureInputA](experiments/structure_input_v1/model.py)，**未接入正式launcher、未做生成效果验证**。[GPU工程检查部分完成但未通过完整验收](review/structure_input_v1/GPU_ENGINEERING_ZH.md)：精确重载通过，重复更新差异超出沿用浮点容差，保留失败与未完成项。下文为继承的历史正式入口说明，不能把历史结果当新候选结果。

**2026-11-04边界修复：** 本分支在固定审查版本2f2cb300之上修复异常事务，并继续核查4b676417的两个遗漏：O/A/D无角色buffer时的回滚、验证临时权重未完整恢复时的快照拒绝。见 [异常边界、测试及历史影响](ABORT_BOUNDARIES.md)；没有更改正常科学公式或启动正式实验。原十项及新增八项CPU测试由同一runner执行，GPU故障与开销未测。`provenance/source_files.json`仍保存原执行来源，补丁清单单独记录当前文件变化，不把修复后的代码冒称为旧实验已执行版本。

这是实际小图实验的**源码审查分支**，不是新的模型实现或完整数据适配声明。分支以独立快照提交，不携带工作仓库的旧数据、结果及历史；不改写任何原有分支。代码来源及逐文件原始／发布 SHA256 见 [来源清单](provenance/source_files.json)。工作仓库 HEAD 仅作定位，不能代替这些实际执行文件的哈希。

当前主线：固定节点数量和节点类型、schema 约束的单参考异质图生成；普通单输入 CE、EMA0.99、R1、3000 次更新，最终 EMA 做 T1000 生成。**E 指晚CN**。小图是实际实验材料，不是完整图容量证明。A 并非官方 SparseDiff；本分支不声称 E 在所有指标／seed 上优于 A，也不声称已解决训练 seed 差异。

## 正式入口和版本

| 实验／模型 | 已实际使用的入口 | 网络实现／说明 |
|---|---|---|
| 当前 O→A 匹配补证 | [formal_entry.py](experiments/o_cn_matched_execute_v1/formal_entry.py) → [launch.py](experiments/o_cn_matched_execute_v1/launch.py) → [worker_run.py](experiments/o_cn_matched_execute_v1/worker_run.py) | [O/A 工厂](experiments/o_cn_matched_execute_v1/model_factory.py)，O 正式运行；A 只读复用已完成对照 |
| A／E 晚CN匹配主面板 | [launch.py](experiments/late_final_single_input_v1/launch.py) → [worker_run.py](experiments/late_final_single_input_v1/worker_run.py) | [A/E 工厂](experiments/late_final_single_input_v1/model_factory.py) |
| PubMed D 晚CN补齐 | [generation_launch.py](experiments/d_late_ard_runtime_v1/generation_launch.py) → [generation_worker.py](experiments/d_late_ard_runtime_v1/generation_worker.py) | 虽名为 generation，实际含从初态训练；[D 工厂](experiments/d_late_ard_runtime_v1/model_factory.py)；该历史票也含 ARD，**不能将整个票等同正式 E** |
| 完整 T1000 生成 | 各运行目录的 [sampling.py](experiments/late_final_single_input_v1/sampling.py)：`sample_one` | 绑定训练步数、EMA 文件哈希和采样身份；调用 [reverse_step][reverse] |
| 原图／LCC完整评价 | [A/E evaluate_all.py](experiments/late_final_single_input_v1/evaluate_all.py)、[O/A evaluate_all.py](experiments/o_cn_matched_execute_v1/evaluate_all.py)、[D 历史评价](experiments/d_late_ard_runtime_v1/generation_evaluate.py) | [可变节点数指标][metrics]、[附加字段][fields]；空分母和不可用指标保留 null／原因 |

上述是**实际运行入口源码**，不是可直接激活的新审批：它们依赖原票的配置、初态、输入合同、资源账本及审批文件。本快照不含这些私有运行资产；不能跳过绑定检查或把旧审批用于新运行。原服务器的 `common_run.py` 输出位置、任务数量、看门狗和来源清单仍有票据绑定。详见 [依赖与复现边界](REPRODUCIBILITY.md)。本次仅做静态检查，未在快照上重新训练／生成。

所有 O–E 的共享科学构造位于 [StructuralDenoiser][variants]：

| 组别 | 当前 Xt 度分支 | 显式学习式CN | 630维结构特征 | 冻结字典、动态角色编码 | 联合头输入 |
|---|---|---|---|---|---|
| O | 有 | 无 | 无 | 无 | 739 |
| A | 有 | 晚入口 | 无 | 无 | 776 |
| D | 有 | 晚入口 | 有 | 无 | 776 |
| E | 有 | 晚入口 | 有 | 有 | 776 |

代码还保留 B（结构特征，无CN无角色）／C（结构特征＋角色，无CN）构造供历史追溯。早CN的 [early_cn.py][early] 仅为历史实验依赖，**不是本表的正式入口**；其37→64边残差和739维头不能与晚CN混用。`normal2.py`、底层 `worker.py` 和 `bw_preflight/config.json` 保留为共享函数／历史来源，不能用其历史默认值代替当前 [执行配置](configs)。COUNT、Dropout、扰动辅助、学习率修补和新的自适应角色方法不属于当前正式方案。D 历史调度所依赖的 ARD 适配器作为审查依赖保留，不代表 ARD 被采用。

## 从输入到生成：逐行审查索引

以下链接均指本快照内实际文件；[机器可读符号索引](provenance/code_symbols.json) 给出函数／类的行号。

| 环节 | 代码位置及应核对的契约 |
|---|---|
| 节点类型、关系和合法无向槽 | [TypedSlots / UniformQueryPlan][slots]；固定类型，不预测节点数量或类型；每个无序端点类型对至多一个二状态关系，不支持同一对的平行多关系；无自环 |
| 数据绑定、经验关系先验 | [prepare.load_case][prepare]、[common.make_case][common]；数据哈希核对，先验为该关系参考边数／合法容量 |
| 前向加噪 | [global_noisy_edges][noising]、[MarginalKernel][kernel]；关系特定边际混合，保留精确 prior=0/1 支持，不用 epsilon 伪造支持 |
| 时间与训练输入 | [common.schedule / train_input][common]；cosine offset=.008，端点 alpha[0]=1、alpha[T]=0；训练 t 均匀取整数1…1000 |
| 查询覆盖和CE掩码 | [UniformQueryPlan][slots]、[one_input / update][single]、[supported_binary_ce][loss]；下面单列18,336口径 |
| 冻结 Xt 与计算图 | [FrozenTypedContext][context]；物理边与临时查询边分开，候选可进入主干上下文，不能声称查询分块不影响输出 |
| 网络主干／公共数值修正 | [TypedSparseDenoiser][model]、[GraphTransformerConv][backbone]、[Transformer层][transformer]、[并列极值梯度][tie]；8层，节点256／边64／全局128，FP32 |
| 当前 Xt 度分支 | [DegreeBranch][variants]；关系度容量归一化和log1p双路＋类型＋时间，MLP后加到节点初始表示，输出层零初始化 |
| 晚CN | [CommonNeighborFeatures][cn]、[joint_predict][model]；只用完整当前 Xt 真实边；共享首层计算邻居上下文，类型化计数＋学习均值，候选正反方向对应 |
| ReFeX风格结构特征 | [features.extract][features]；630维，关系度／闭合egonet内部和边界＋一轮逐关系邻居mean与sum；不是完整原ReFeX递归剪枝算法 |
| 角色字典及当前Xt编码 | [select_dictionary][select]、[role_dictionary][roles]、[StructuralDenoiser][variants]；每类型字典用全部630列，字典冻结，系数由当前Xt重新NNLS，非固定逐节点干净角色 |
| 预测头 | [joint_predict][model]；边64＋端点均值256＋绝对差256＋全局128＋时间1＋Xt统计20＋关系10＋当前状态2＋主干logits2＋晚CN37＝776；O移除CN列与编码器 |
| EMA及保存恢复 | [ParameterEMA][ema]、[checkpoint_payload / restore_payload][integration]；从theta0副本开始，每次完整optimizer更新后递推.99/.01；不进入训练反向；buffer按合同复制 |
| 后验与反向状态提交 | [MarginalKernel][kernel]、[reverse_step][reverse]、[sample_one][sample]；p0是干净图概率，q是反向一步概率；同一步覆盖使用同一Xt，完成后统一提交X(t−1)，不是固定边数采样 |

晚CN的37维＝32维学习式共同邻居均值＋4维log1p类型计数＋1维存在标记；无共同邻居时为零。上下文与计数来自当前 Xt，不能把查询边算成物理边。IMDB MG 在冻结的非零关系支持下普通共同邻居始终为空；不应声称CN直接为所有关系提供闭合证据。O 的消息传递仍可能间接利用邻域，O→A检验的是显式节点对CN及相关头参数的增量。

E 同时将完整630维结构特征和角色系数分别经可训练投影输入节点表示；**RolX不是所有结构信息必经的压缩瓶颈**。正式字典K：IMDB 5/4/3/2，PubMed 8/8/8/6；PubMed前三类型命中当时搜索上界，不能写成自动选阶已充分验证。拟合状态来自同一参考图的干净状态和t=100/500/1000各两份扰动，生成时只读取当前Xt和冻结字典，不读取未知干净边身份。拟合来源仍是单参考图，不能宣称跨图泛化。

## 192节点 R1：查询什么，CE如何计算

四种固定节点类型的10个无序类型对覆盖所有不同节点对，因此完整schema容量为 `S = 192×191/2 = 18,336`。节点编号只作存储／映射，不作网络特征。一次更新先构造**完整图 Xt**，然后只调用一次主预测，查询每个可动的无向槽一次；该集合同时包含当前有边和无边，标签均取同一X0。不是只预测“剩余40%边”，也不是只监督正边。计算图为双向存储时，监督仍按一个无向候选槽计一次。

| 数据集 | 非零先验关系 | 实际查询／监督槽 | 固定无边槽 | CE分母 |
|---|---|---:|---:|---:|
| IMDB，192节点／323边 | DA、DM、AA、AM、MG | 13,120 | 5,216 | 18,336 |
| PubMed，192节点／926边 | GG、GD、GC、GS、DD、DC、DS、CC、CS、SS | 18,336 | 0 | 18,336 |

查询计划跳过 prior=0 的关系族，这些槽状态精确固定为无边，概率与CE贡献为0，**仍保留在18,336的分母内**。如果对精确prior=0/1槽直接调用 `supported_binary_ce`，它检查标签兼容后返回精确零贡献；可动槽使用二logit CE。当前更新为 `sum(slot_CE)/18,336`，一次清梯度、一次反向、一次AdamW、一次EMA；无normal2遗留的½。历史normal2为两份独立输入各乘½，不能混作同曝光的结果。分关系容量与边数见 [支持配置](configs/schema_support.json)。

## 配置、证据和限制

- [configs](configs) 提取实际任务配置，保留每票来源清单哈希；这些是历史计算配置，不是运行审批。完整原始执行源码哈希及发布时路径替换见 [source_files.json](provenance/source_files.json)。历史来源包含科学基底 `472e533fd28cfc067702b1b75082ac5105140a2d`；本次工作仓库定位为 `5f2c92790fd5f928cfa6317286a14db11d220d58`。详细[版本链](provenance/version_lineage.json)与逐文件绑定共同解释版本，不能把旧earlyCN审查commit当作当前晚CN训练版本。
- 普通单输入CE＋EMA.99、AdamW lr=2e−4、weight_decay=1e−12；不新增COUNT、硬配额或推理校准。模型 `dropout=.1` 是历史配置字段，实际随机算子应以主干源码为准，不能由字段推断存在额外dropout实验。
- 本快照保留官方SparseDiff衍生主干／查询工具及许可证，来源见 [THIRD_PARTY.md](THIRD_PARTY.md)。查询／分块、ReFeX、RolX、EMA不单独宣称原创。
- 原图为主，LCC单列；LCC连通是选择结果，不是模型额外学会连通。旧HGEN K4命名与真正四环、旧聚类与标准聚类在评价中分开；依赖缺失项不得填0或冒称原生HGEN全套。
- 不包含原始数据、字典二进制、科学初态／检查点、审批、训练日志、生成图或大型结果。外部完整复现需另行取得绑定资产；本分支只提供源码审查，不宣称开箱即跑或已适配完整IMDB／PubMed。未重新验证大图20块／10块、COUNT分块或多图任务。
- 现有实验有训练seed差异与结构取舍；源码可读不等于方法效果已全面证实。O补证提交时仍在运行，不提前写成已完成。

[variants]: research_branches/typed_structure_early_cn_expansion_review/experiments/typed_structure_early_cn_expansion_v1/package/model_variants.py
[slots]: research_branches/typed_structure_early_cn_expansion_review/experiments/typed_structure_early_cn_expansion_v1/package/hsgd/slots.py
[prepare]: research_branches/typed_structure_early_cn_expansion_review/experiments/typed_structure_early_cn_expansion_v1/package/prepare.py
[common]: research_branches/typed_structure_early_cn_expansion_review/experiments/typed_structure_early_cn_expansion_v1/package/hsgd/bw_preflight/common.py
[noising]: research_branches/typed_structure_early_cn_expansion_review/experiments/typed_structure_early_cn_expansion_v1/package/hsgd/noising.py
[kernel]: research_branches/typed_structure_early_cn_expansion_review/experiments/typed_structure_early_cn_expansion_v1/package/hsgd/kernel.py
[single]: experiments/late_final_single_input_v1/single_input.py
[loss]: research_branches/typed_structure_early_cn_expansion_review/experiments/typed_structure_early_cn_expansion_v1/package/hsgd/losses.py
[context]: research_branches/typed_structure_early_cn_expansion_review/experiments/typed_structure_early_cn_expansion_v1/package/hsgd/computational.py
[model]: research_branches/typed_structure_early_cn_expansion_review/experiments/typed_structure_early_cn_expansion_v1/package/hsgd/model.py
[backbone]: research_branches/typed_structure_early_cn_expansion_review/experiments/typed_structure_early_cn_expansion_v1/package/hsgd/vendor/conv_transformer_model.py
[transformer]: research_branches/typed_structure_early_cn_expansion_review/experiments/typed_structure_early_cn_expansion_v1/package/hsgd/vendor/transconv_layer.py
[tie]: research_branches/typed_structure_early_cn_expansion_review/experiments/typed_structure_early_cn_expansion_v1/package/hsgd/vendor/tie_extrema.py
[cn]: research_branches/typed_structure_early_cn_expansion_review/experiments/typed_structure_early_cn_expansion_v1/package/hsgd/common_neighbor.py
[features]: research_branches/typed_structure_early_cn_expansion_review/experiments/typed_structure_early_cn_expansion_v1/package/features.py
[roles]: research_branches/typed_structure_early_cn_expansion_review/experiments/typed_structure_early_cn_expansion_v1/package/role_dictionary.py
[select]: research_branches/typed_structure_early_cn_expansion_review/experiments/typed_structure_early_cn_expansion_v1/package/select_dictionary.py
[ema]: research_branches/typed_structure_early_cn_expansion_review/experiments/typed_structure_early_cn_expansion_v1/ema.py
[integration]: research_branches/typed_structure_early_cn_expansion_review/experiments/typed_structure_early_cn_expansion_v1/integration.py
[reverse]: research_branches/typed_structure_early_cn_expansion_review/experiments/typed_structure_early_cn_expansion_v1/package/hsgd/runtime.py
[sample]: experiments/late_final_single_input_v1/sampling.py
[metrics]: research_branches/typed_structure_early_cn_expansion_review/experiments/typed_structure_early_cn_expansion_v1/lcc_eval/metrics.py
[fields]: research_branches/typed_structure_early_cn_expansion_review/experiments/typed_structure_early_cn_expansion_v1/evaluation_fields.py
[early]: research_branches/typed_structure_early_cn_expansion_review/experiments/typed_structure_early_cn_expansion_v1/package/early_cn.py
