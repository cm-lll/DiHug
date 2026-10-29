# 当前评价合同

底层定义逐字节复用LCC审查提交1edb9bc的metrics.py及vendor。完整旧定义原文保存在metric_definition_reference.md，作为来源说明，不将其中旧64图实测数字作为本轮结果。本票正式生成0。

summarize.py通过evaluation_fields.py复用原逐图标量/距离字段提取，只把固定PubMed类型名字改为从各数据集绑定配置读取。主W1、TV、simple path、typed walk、三角、真正四环、legacyK4、闭合、距离及遗留HGEN公式没有改写。

MP1/2/3分别是一、二、三条边；simple paths的节点不重复、反向合并计一次。typed walks允许重复和回溯，按原方向计数约定另列，不能互称。缺失模式计数为0，缺失图或未定义指标为null，两者不同。

W1是每类型所关联四schema列的均值，类型权重固定为192节点参考中该类型比例。LCC经验分布的每类型分母使用实际保留节点。all10兼容值=incident4×0.4，另报实际保留类型权重值。缺少类型时该条件分布及主聚合null，不丢弃类别后重新平均。

三角类别覆盖4类型的全部20个无序多重集。IMDB DAM=(0,1,2)、DAA=(0,1,1)、AAM=(1,1,2)从同一表读取；Genre排序度谱在typed_degree.Genre.sorted_degrees。参考0的类别是观测事实，不是禁边规则。PubMed使用Gene/Disease/Chemical/Species，不改名。

类型化二跳机会有路径实例和去重端点对两种分母，分别保留分子/分母/比例。零分母比例null。LCC=节点数、零孤立点是派生规则后果，不能算训练增益。有限节点对平均距离、LCC条件距离、断开全局直径null分别保留。

每个数值标量保存raw/reference/absolute_error；误差先逐图取绝对值。distribution_error包含W1、jointTV、typed triangleTV、fourcycleTV、MP1/2/3及walk组成TV等，直接比较。raw差、绝对误差差和分布距离差分别输出；严格要求两样本都有定义才计算两图均值。

HGEN原生powerlaw依赖仍缺；现有固定xmin1投影统计另名，不代替原生结果。社区Z_obs无真实社区标签；完整orbit未实现。legacy squares实际K4与真正四环、legacy3T/claws与标准聚类/transitivity都保持不同名字。不安装依赖，不把“适用的全部现有指标”写成无缺项的原生HGEN复现。

同模型两图保留节点/边标签对齐Jaccard、边对称差、度W1、三角/MP2/MP3组成TV，及节点/边数不同导致的非同构证据。节点ID相同的边差不是同构不变多样性；同样大小不自动判定同构。两张图不足以认证总体多样性。
