# 投稿表2的实际评测定义与版本差异

本文件描述找到的源码及本次逐图复核，不依据指标名称猜测。复现默认使用原仓库提交71dedb3d456ee6bbdfeab15ace42b4ef0e169180中原样提取的heterogeneous_metrics.py，SHA256为7d4c3c20992d83e642d667c08876eaec2c504507a93fa34ab5d627ff9223289b。其余实际调用函数的原文件也逐字节保留，清单见source_inventory.json。

## Degree MMD

源码：original_git_source/sparse_diffusion/structure_energy/heterogeneous_metrics.py中的degree_mmd_rbf与degree_mmd_report。

每张生成图分别与同一参考图比较。输入为全体节点的无类型总度标量，包括度0节点；不是先归一化的度直方图向量，也不是LCC上的度。重复度值只为精确加速而压缩成取值/整数频数，不改变节点样本的权重。

核为k(x,y)=exp(-4(x-y)^2)，不调带宽、不使用median heuristic；按exp(-(x-y)^2/(2sigma^2))约定，sigma=1/sqrt(8)。度数不除以最大度、均值或节点数。

原始量是无偏MMD平方U统计：

    sum_{i!=j} k(x_i,x_j)/(n(n-1))
  + sum_{i!=j} k(y_i,y_j)/(m(m-1))
  - 2 sum_{i,j} k(x_i,y_j)/(nm)

不是MMD的平方根。n或m小于2时返回0。保留可能为负的raw值；表格对每张图先取max(raw,0)，再对8张的结果取普通平均。不是把8图节点先池化，也不是比较8个图级统计向量。

## 无类型标量结构指标与孤立点

源码：original_source/sparse_diffusion/structure_energy/switch_chain.py的graph_metrics；HGEN使用evaluate_external_graphs_canonical.py的simple_metrics，全部数值已交叉复现。

- 基础对象是完整库存上的简单、无自环、无向图。三角形、聚类、LCC、同配性都不使用节点/关系类型。
- 三角形用u<v<w逐个计数，每个三角只计一次；不是特定D-A-M三角数量。
- 平均聚类为所有节点local clustering的均值；度0或1的节点local clustering为0。孤立点保留。
- LCC是完整图中最大连通分量的节点数；评价不先裁为LCC，孤立点作为单节点分量参与搜索。
- 同配性是所有物理边两方向的端点总度Pearson相关；每个无向边贡献(u,v)及(v,u)，没有关系分层加权。方差为0时返回0。
- transitivity为3*三角形数/sum_v C(d_v,2)，与平均local clustering分开，表2Clust.使用后者。

## Typed-triangle TV

源码：_triangle_signature、typed_triangle_counts、_normalize、distribution_distances。

每个u<v<w三角只计一次。对三个节点的6种排列构造：三个节点类型名称，加(a,b)、(b,c)、(a,c)三个关系家族名称，以竖线连接；取字典序最小签名。因此同时保留类型和关系，不依赖原节点编号。

各图按本图全部三角数量归一化，再在参考/生成模式键的并集上计算0.5*sum|p-q|。不按参考三角总数归一化，不先池化8图。

重要历史口径：提交71dedb3的实现对空计数返回零向量；一侧有三角而另一侧没有时返回TV=0.5，双空时为0。当前旧工作树另外添加了单空TV=1的处理，修改并未包含在该Git提交中。本次首次用工作树版复核，在IMDB NetGAN sample6实际出现1.0对旧报告0.5的差异，失败回执完整保留。

这不是通过调参修复数值，而是找回了真实的历史定义。两份源码与historical_vs_retained_metrics.diff均交付。单空零向量不是质量为1的概率分布，历史约定有局限；以后统一评测可另选定义，但必须全方法重算，不能把新版空分布规则与原表数值直接拼接。

## 长度2/3 Typed-walk TV

源码：typed_metapath_counts。

这是walk，不是要求各节点不同的simple path；允许立即回溯和重复节点/边。由每个节点作为起点，沿完整无向邻接延伸；签名按“节点类型、关系、节点类型、关系、...”交替编码。

每条有向遍历实例计数。签名与其完整反向字符串序列取字典序最小者并池化，但不除以2、不把回文回溯实例删掉。比如往返路径可以多次贡献。按本图该长度的全部walk实例总数归一化，再算TV；表格分别对8个逐图TV取均值。不是按候选路径抽样估计，也不是只统计某一元路径。

## Edge overlap

实际源码分母为参考物理边数：100*|E_gen交E_ref|/max(|E_ref|,1)。比较无向端点编号，不是图同构匹配，也没有做节点重排优化；不是生成边数分母、Jaccard或两者平均。论文文字提到生成边占比，与实际实现存在措辞差异，以源码及本次重现为准。

DiHug/经典基线固定节点库存。HGEN旧导出仅存num_nodes、edges、run_seed，没有node_types和关系标签；不能声称原图已有这些字段。本次依原评测器从附带new_node_dict.p、adj_matrix.p恢复HGEN LCC顺序的类型，并重新验证其类型数组及参考物理边集合与规范参考完全相同。IMDB/PubMed均通过，因此其EO可比较。

HGEN typed评价将所有无序类型对定义为typepair:i:j家族，同时保留不在参考活跃schema中的边并报告schema-valid比例；不删边来修复schema。参考和HGEN都映射到同一类型对名称空间。当前数据一类型对只一种关系，故此映射可与DiHug原家族标签的模式对应；不能无条件推广到同类型对多关系数据。

## 汇总与来源边界

表2每种方法8张图，标量是均值±总体标准差（ddof=0）；MMD、Tri TV、MP2、MP3、EO是逐图数值的普通均值。原aggregate_comparisons内部用样本标准差，build_consistent_paper_evidence.py转换为总体标准差后输出论文表。两种std不得混用。

GraphRNN/NetGAN/VGAE各8张通常来自一个训练seed0模型的8次采样；HGEN为8个独立训练/run seed；DiHug为同一留存模型的8个生成seed。8张不等于所有方法均有8次独立训练。VGAE在论文标为重构参考，不能称严格target-free生成基线。

本次历史指标复现只证明旧图、旧口径与表格数值对上，不证明旧生成方法、模型权重出处或统计实验设计已完全认证。详见README_zh.md的缺口。
