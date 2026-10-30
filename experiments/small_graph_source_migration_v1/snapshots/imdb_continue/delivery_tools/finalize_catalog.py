"""Make field units/definitions explicit; never changes any metric value."""
import pathlib,json,csv,re
OUT=pathlib.Path(__file__).parent
def main():
 p=OUT/'tables/metric_catalog.csv'
 rows=list(csv.DictReader(p.open()));unknown=[]
 reference=json.loads((OUT/'run/derived_lcc_evaluation/IMDB_reference.json').read_text())
 for row in rows:
  row['reference_self_value']=reference['distribution_error'].get(row['field']) if row['group']=='distribution_error' else row['reference']
  row['reference_interpretation']='ideal discrepancy0; empirical reference self-estimate shown separately (U-stat MMD can be negative)' if row['group']=='distribution_error' else 'fixed reference raw metric; absolute-error ideal0'
 counts={'nodes','edges','components','lcc','second_component','isolates','triangles','four_cycles','HGEN_square_legacy_K4','claw_count','wedge_count','cpl_pair_count','edge_overlap_count'}
 degrees={'maximum_degree','minimum_degree','mean_degree','cpl_finite_ordered_pairs','cpl_lcc','diameter_global','diameter_max_component_finite','diameter_lcc'}
 for row in rows:
  k=row['field'];group=row['group']
  if group=='distribution_error':
   row['unit']='degree' if 'W1' in k or k=='incident4_actual_typeweights' else 'dimensionless'
   if k=='incident4_actual_typeweights':row['definition']='mean of4 incident relation-degree W1 within each type,weighted by generated retained type frequencies; diagnostic separate from fixed-reference-weight main W1'
   if 'MMD' in k:
    row['definition']='per-node degree Gaussian MMD^2 U-statistic form: kernel exp(-4*(degree_i-degree_j)^2); within-sample diagonal excluded and divided by n*(n-1), cross divided by n*m. raw may legitimately be negative; nonnegative=max(raw,0). Graph-node observations are not asserted independent.'
    row['direction']='distribution discrepancy diagnostic; negative raw U-stat does not mean better than perfect agreement'
   continue
  suffix=k
  if k in counts:unit='count'
  elif k in degrees:unit='hops' if k.startswith(('cpl_','diameter')) else 'degree'
  elif k.startswith('family'):unit='fraction' if k.endswith('density') else 'count'
  elif k.startswith(('triangle_','walk')):unit='count'
  elif k.startswith('MP'):unit='paths/node' if k.endswith('per_node') else 'paths'
  elif k.startswith('closure_'):unit='fraction' if k.endswith('ratio') else 'count'
  elif k.startswith('type_'):
   if k.endswith(('zero_fraction','low_le1_fraction','share','gini_standard','top5_degree_share')):unit='fraction'
   elif k.endswith(('_count','_zero','_low_le1')):unit='nodes'
   else:unit='degree'
   typ=next((t for t in ('Director','Actor','Movie','Genre') if k.startswith('type_'+t+'_')),None)
   sub=k[len('type_'+typ+'_'):] if typ else k
   rel=re.match(r'relation(\d+)_(.*)',sub)
   subset=f'{typ} nodes';measure=sub
   if rel:subset+=f', schema relation column {rel[1]}';measure=rel[2]
   definitions={'count':'number of nodes','maximum':'largest degree','second':'second-largest degree (null if fewer than2 nodes)',
    'max':'largest degree','mean':'arithmetic mean degree','zero':'number with degree0','low_le1':'number with degree<=1',
    'zero_fraction':'count(degree0)/type-node-count','low_le1_fraction':'count(degree<=1)/type-node-count',
    'share':'type-node-count/actual graph nodes','p90':'NumPy linear-interpolated0.9 degree quantile','p99':'NumPy linear-interpolated0.99 degree quantile',
    'gini_standard':'2*sum(i*sorted_degree_i)/(n*sum_degree)-(n+1)/n; i starts1; null for zero sum',
    'top5_degree_share':'sum5 largest degrees/sum degrees (fewer than5 uses all); null for zero sum',
    'fixed_ref_P99_ge_count':'number of nodes with degree>=max(1,ceil(reference type p99))',
    'fixed_ref_P99_threshold':'fixed degree threshold max(1,ceil(reference type p99))'}
   row['definition']=f'{subset}: '+definitions.get(measure,measure)
  else:unit='dimensionless'
  if group=='absolute_error':row['definition']='abs(generated raw value - fixed reference raw value), computed per graph before averaging; '+row['definition']
  row['unit']=unit
  row['source']='frozen_evaluation_source/evaluation_fields.py; frozen_evaluation_source/lcc_eval/metrics.py; metric_definition_reference.md'
 with p.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 (OUT/'METRIC_DIRECTORY.md').write_text('''# 完整指标目录与缺项

`tables/metric_catalog.csv`逐字段给出原值/逐图绝对误差/分布距离的定义、单位、固定参考及方向。
635个原值字段、对应635个误差字段、73个分布误差字段均来自冻结评价器，不增加新评分。
raw组名表示原始指标数值，不表示raw网络读出；全部实际生成使用EMA0.99。

`tables/per_graph_all_detail_fields.csv`逐叶导出全部统计与比较；`evaluation.json`保留完整嵌套对象，
不限于标量表。以下是变长字段的编码与单位，参考明细见`run/derived_lcc_evaluation/IMDB_reference.json`：

|字段族|定义与单位|
|---|---|
|total_degree / typed_degree.*.sorted_degrees|实际图按类型/全图的完整降序度谱，单位degree|
|*.histogram[i]|度恰为i的节点数量，单位nodes；非累计、不去掉0度|
|typed_degree.*.by_relation[r]|schema第r列关系度分布，非关联列全0；分位数沿用NumPy线性插值|
|joint_relation_degree[type][v]|完整10列整数度向量v的节点频数，单位nodes；不压缩成均值|
|typed_triangles[pattern]|排序的三个类型编号，无向三角计一次，单位triangles；合法0参考类型不硬禁|
|typed_twohop_closure[(a,m,c)]|端点类型a/c、中间类型m；path按每个中间节点计数，unique_pair将端点对去重；分子是已闭合的对应实例，分母是全部机会；比值0分母null并保留reason|
|typed_four_cycles[signature]|简单四环，无重复顶点；旋转/反向规范化，单位cycles；与legacy K4分开|
|mp[i].counts[pattern]|i=0/1/2对应长度1/2/3简单路径；顶点互异、反向去重；模式保留节点类型和关系|
|mp[i].total / per_node|全部模式之和 / 实际保留节点数，单位paths与paths/node|
|typed_walks[i]|允许回退、重复节点的长度1/2/3walk；正反模式同键，但有向实例不除2；与简单路径分开|
|component_sizes|全关系无向图分量节点数降序，单位nodes|
|distance_histogram[i]|有限有序不同节点对距离恰为i的数量，单位pairs；不可达不混为距离0|
|comparison.*W1*|离散度经验分布W1，单位degree；类型权重主口径固定参考频率；all10=0.4*incident4|
|comparison.*TV*|归一化频数分布半L1；双方无模式0，仅一方无模式1；缺失类型主TV为null|
|comparison.type_total_degree_MMD2|与总度MMD相同核exp(-4*(x-y)^2)，按类型分别计算；组内去对角/n(n-1)、交叉/nm的U-stat形式；raw可合法为负，不只因舍入，另列max(raw,0)；不宣称图中节点独立|

所有数量/率/距离原值同时保留参考与逐图绝对误差；数量大小本身不等于优劣。
度谱/模式频数的原始参考及生成数组保留在完整JSON与逐叶CSV；其比较距离沿用冻结实现。
目录初稿的MMD文字说明经vendor源码核对后更正为上述精确公式；数值实现从未改动，
不是Gaussian带宽4或有偏MMD，raw负值也不能一概解释为舍入。以本目录和最终CSV为准。

## 样本差异

每条件两图的编号对齐边对称差、边/节点Jaccard、exact_labeled_duplicate、总度W1、
三角组成TV、MP2/MP3组成TV全部保留在`tables/sample_diversity.json`。
编号对齐差异不证明非同构多样性；节点/边数不同时可证明非同构，相同时未额外求同构。
节点数不同按实际集合算Jaccard，W1按各自经验分布，LCC派生单列。

## HGEN/既有评价范围与缺项

已实现且逐图保留：总边数、最大度、度分布/MMD、三角、聚类、legacy聚类、
Gini(标准及legacy)、legacy熵、legacy square(K4)、真正四环、特征路径长度、
同配性、项目fixed-xmin1幂律估计，以及异质度/联合度/类型三角/闭合/元路径全项。
HGEN原生powerlaw依赖不可用，`HGEN_native_powerlaw`保持null；项目估计不冒充原生值。
原生HGEN社区标注/NMI等未有可比参考标签、通用graphlet/orbit谱未实现：明确缺项，
不安装依赖、不造社区标签、不把本表称为完整原生HGEN全套复现。
无机会闭合率、无类型/单节点次大度、无边同配性、非连通全局直径等数学未定义值保留null与原因。

## 时间与后处理

1000/2000是后续补充探索，3000/6000仍是原先预定正式生成点；不选最好检查点替换原结论。
实际生成图、固定验证p0和一步q不是同一统计对象，本报告趋势只连接实际生成图指标。
原图是主结果；LCC的连通和零孤立点是选择规则后果，不能作为模型新增能力。
6000若尚未封存完成保持pending/null，不只平均成功样本，也不等待它延迟早期结果交付。
''')
 print('catalog rows',len(rows))
if __name__=='__main__':main()

