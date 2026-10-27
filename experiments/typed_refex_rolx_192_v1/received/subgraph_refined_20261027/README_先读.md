# 两张192节点诱导子图：选择及约束内调整（最终 attempt02）

## 结论
这次是重新选择部分节点的结构分布调整，不是修改边，也不是少量边的补丁。IMDB基线为2026102701，PubMed基线为2026102706；均由预先冻结的同一综合目标对三个候选统一评分后选出，不依据模型训练结果挑图。
两张最终图都保持192节点、原定类型配额、零孤立点及原图所有活跃关系族，所有内部边均原样保留。新增连通性及路径类型不退化门槛后，推荐根目录 samples/ 中的 attempt02；attempt01 仅供追溯，不是最终推荐。

## 结果（前→后；距离越低越接近原图）
|指标|IMDB|PubMed|
|---|---:|---:|
|关系比例TV|0.10598 → 0.01709|0.09039 → 0.01642|
|分类型×关系 log度CDF距离|0.20165 → 0.12022|0.05774 → 0.05017|
|平均聚类系数绝对误差|0.31831 → 0.01034|0.06739 → 0.00150|
|类型化三角组成TV|0.02710 → 0.00068|0.12386 → 0.01584|
|MP2组成TV|0.12746 → 0.09950|0.13753 → 0.02018|
|综合目标|0.15610 → 0.04956|0.09788 → 0.02332|
|MP3组成TV（非优化目标）|0.05782 → 0.02687|0.15058 → 0.02676|
|边数|291 → 323|975 → 926|
|三角总数|20 → 45|1996 → 1937|
|最大连通分量|180 → 188|192 → 192|
|孤立点|0 → 0|0 → 0|

综合目标=五项距离各权重0.20，再加0.05×“缺失正计数三角类型比例”。平均聚类按全体节点计算，度<2记零。度距离是固定原图尺度下log(1+d)的经验CDF积分/一维Wasserstein距离，按存在的类型×关系单元平均；只是目标的数值标准化，保存图的度数没有缩放、修饰或重采样。

## 仍然存在的偏差与适用范围
- **IMDB**：替换了82个不同节点（不是把接受次数当作换掉的节点数）；129/2000次提案接受。节点类型配额为[48, 56, 78, 10]；最终边数323，三角45。
  原图平均度/最大度=4.935/497；基线=3.031/58；最终=3.365/65。绝对长尾仍不能复现。
  MP2类型覆盖15/15；MP3类型覆盖26/27；连通分量数3→3。
- **PubMed**：替换了77个不同节点（不是把接受次数当作换掉的节点数）；141/2000次提案接受。节点类型配额为[40, 74, 66, 12]；最终边数926，三角1937。
  原图平均度/最大度=15.698/674；基线=10.156/116；最终=9.646/92。绝对长尾仍不能复现。
  MP2类型覆盖40/40；MP3类型覆盖136/136；连通分量数1→1。
- IMDB保留稀有Actor–Actor–Movie三角，但仍缺少基线就未覆盖的MP3类型 `1,4,1,5,2,5,1`（Actor–Actor–Movie–Actor，关系族从0起计数）；并未声称覆盖原图全部机制。
- PubMed保留全部136种MP3，但仍缺少基线就缺失的Species–Species–Species三角（原图仅3个）。20种正三角类型保留19种；该缺失显式计入目标，未隐瞒或伪造。
- 类型配额本身改变原图类型比例：例如IMDB Genre为10/192=5.21%，原图23/1653=1.39%；PubMed Species为12/192=6.25%，原图45/1565=2.88%。这些是小图类型覆盖约束，不是无偏抽样。
- 192节点简单图最大度最多191，同类型/异类型邻接还受各类型配额限制；原图外部邻边必须被截断。绝对度尾、三角数和路径总数不应期待与大图一致。原图全节点、所选节点在原图、最终诱导图三个口径都保存在metrics及逐节点NPZ，不能混用。
- PubMed未做log变换的原始分类型×关系度Wasserstein距离（独立宏平均）从1.88637升至1.98723，略有变差；平均度从10.156降到9.646，与原图15.698的差距也变大。它的log-CDF目标虽改善，不能概括为所有度分布指标都更好。详见root_independent_audit/v2_final_audit.json。
- 最终五项目标及MP3组成距离均比所选基线好，但没有保证每一个类型、每个分位数或任意未优化指标都改善；更不等价于保留原图联合分布、全局最优、无偏样本或大图泛化。适用于快速、同一小图上受控的模型/机制对照；尚未运行任何模型训练。

## 冻结、失败和纠正记录
1. attempt01先冻结目标、种子及每图2000提案后执行。结果目标显著改善，但IMDB分量3→11、最大分量180→165，并丢失部分MP3类型；明确弃用为最终推荐。
2. 独立核验还发现既有三角计数器要求局部边u<v；未排序节点槽导致PubMed三角显示值与独立计数不一致。将输出节点按原ID排序并完整重映射（纯序列化、原始标记边集完全相同）后重算，修复该显示/评价错误；优化器自身三角统计没有此问题。
3. 新的attempt02协议在运行前重新冻结，保留同一目标/基线，明确加上LCC不低于基线、分量数不高于基线、基线全部MP3类型不丢失；固定新种子2026102713/14，各2000提案。没有继续追加搜索。attempt01完整保留。
4. 两次共8000候选提案（每次每数据集2000）；重放只是验证同一固定搜索，不是增加选择预算。候选只作同类型节点替换；不满足门槛直接拒绝并保留上一个有效图，从未删除孤立点“修补”图。

## 文件与复用
- 最终图：samples/IMDB/graph.npz；samples/PubMed/graph.npz。
- graph.npz字段：types（0起类型ID）、edges（E×2、0起局部节点ID、u<v且字典序）、relations（1起关系族ID）、schema（10×2类型端点）、original_ids（局部ID→原图ID）。
- 每图metadata.json、original_node_ids.npy、node_map.csv、2000条完整optimization_trace.json；baseline_graph.npz为原基线加显式节点映射的副本。源文件字节原样保存在source_preview/samples。
- evaluation/metrics.json有完整原图/基线/最终指标，final_induced_metrics.json供独立比对，summary.csv汇总；五张PNG提供结构图、原始度CDF及组成对比。
- protocol.json与protocol_freeze_receipt.json记录目标/门槛/种子/预算及SHA256；serialization_receipt.json记录纯重编号；independent_audit.json及portable_replay_audit.json为核验证据。
- code/refine.py为冻结搜索代码；code/replay.py在新/tmp目录完整重跑并比较所有提案记录及规范化最终数组；无需GPU。依赖Python3、NumPy、Matplotlib（独立审计另需SciPy，如已安装）。
- 必须针对这两张新图重新计算训练所需的reference统计、度/关系/路径先验、划分及图绑定哈希；不能复用原图/旧小图的缓存或先验。此交付没有开始或批准训练。

最小加载示例：
```python
from pathlib import Path
import numpy as np
root = Path("subgraph_refined_20261027")
g = np.load(root / "samples/IMDB/graph.npz")
types, edges = g["types"], g["edges"]
relations_1based, schema = g["relations"], g["schema"]
original_ids = g["original_ids"]
assert len(types) == 192 and (edges[:, 0] < edges[:, 1]).all()
# 仅在下游要求0起关系ID时转换，不要重复减1：
# relations_0based = relations_1based.astype(int) - 1
```

重新核验：
```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python code/evaluate_refined.py
python code/independent_audit.py --root . --source source_preview --metrics evaluation/final_induced_metrics.json
python code/replay.py
```