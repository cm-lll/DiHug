# 全指标目录与口径

完整逐字段目录为`tables/metric_catalog.csv`，包含dataset、group、定义、单位、参考值、优劣方向与实现来源。原图与LCC全部字段分别评价；raw组名指指标原值，不是raw权重读出。

各数据集635个原值字段、对应635个逐图绝对误差、73个分布字段；同名字段也保留dataset避免参考值混淆。未定义值及原因见`tables/undefined_values.csv`，不填0、不删除。

详细定义见`by_dataset/IMDB/METRIC_DIRECTORY.md`与`by_dataset/PubMed/METRIC_DIRECTORY.md`。W1主口径固定参考类型权重incident4，all10兼容=0.4×incident4；TV为归一频数半L1。精确整数联合度TV不能直接解释成网络学会了角色机制。

MP1/2/3对应1/2/3条边的简单路径，顶点互异、反向去重；walk允许重复节点/回退，另列。类别三角按排序类型计一次；参考为0不是schema禁止。闭合path与unique_pair分开，分母0为null。真正四环与HGEN legacy square(K4)分开；标准聚类与legacy 3T/claws分开。

度MMD采用冻结核exp(-4*(x-y)^2)，组内去对角除n(n-1)、交叉除nm的U-stat形式；raw允许负值，另列max(raw,0)，不宣称节点独立。旧源码中若有“biased”初稿文字，以本目录及vendor实现为准，数值从未改变。

HGEN原生powerlaw：缺依赖；原生社区/NMI：无可比标签；通用graphlet/orbit：未实现。项目fixed-xmin1估计不是原生powerlaw替代认证，现有四环并非完整orbit谱。对这些缺项不造数字。无机会闭合、无类型、非连通全局直径等按数学定义保留null。

同条件两图样本差异完整见`tables/sample_diversity.json`；编号对齐边差/Jaccard不是同构不变的结构多样性证据。LCC按原存储节点编号规则选最大分量，映射保留；统计分母用实际节点集合，固定参考权重不变。IMDB固定参考自身有188/2/2三分量，PubMed参考连通；不能偷偷更换参考。

数量/闭合/度谱明细为all_pattern_counts_compositions、all_typed_closure_details、complete_degree_spectra、all_degree_histograms、complete_joint_degree_histograms长表。完整嵌套原值见evaluation.json；报告并不局限摘要选中的字段。
