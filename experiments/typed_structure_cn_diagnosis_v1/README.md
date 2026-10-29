# PubMed C/E固定权重退化定位

先读[一页结论](RESULTS.md)、[CN实际数据流](CN_DATAFLOW.md)、[唯一待审候选](ONE_CANDIDATE.md)。原始结果来源aa4227b9727ca6853ba7e26412ce6d438a0fa971。未修改科学模型；原45分钟账本保留，所有新临时材料位于data2。

本包包含已完成8次完整查询的输入、逐槽p0/q/raw logits、模型/源码/输入SHA、CPU汇总及必要旧评价/验证副本。不包含大体积权重；路径和哈希在frozen_panel.json。没有新图、反向、训练、optimizer.step或分支遮蔽。

## 无模型的CPU复算

在独立解包目录，使用现有含NumPy的环境，限定CPU线程：

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python analyze_existing.py
python analyze_panel.py
python audit_and_cn.py
```

以上三脚本只读随包证据并重新生成诊断表，不导入模型、不访问GPU。原实际执行脚本完整保存在execution_sources，路径适配只发生于上述便携复算入口。prepare_inputs.py和forward_panel.py是已执行取证脚本，不是继续运行授权；本票8次额度已用完，禁止再次运行。其原账本已结束时会拒绝执行。

## 文件口径

- graph_totals/graph_relations/typed_degrees_and_joint/degree_spectra/typed_closure：直接整理已审计图记录，original和LCC分别列示，不重新评价所有图。
- fixed_validation：raw与EMA、0/1000/2000/3000、t1/500/1000；总CE为sum/S（S=18336），正负CE为该族该类别样本均值，样本数同时保存。不是normal2半份损失。历史只存聚合p0，没有q/logits/中间向量。
- cross_relations/cross_totals：8格新前向，同一输入/完整查询，原q的出生、消失及净值。
- cross_slot_strata/strata_all_relations：分组规则在frozen_panel.json先冻结；每轴内分箱互斥，不同轴不可相加。空分组均值/分位数为空，即null含义，不填0；和为0只是空集合求和。
- forward/*.npz：完整槽ID布局，p0/q和state/query/family对应；paired_*只作CPU差值和CN/端点度描述。logit差用有边减无边，不从概率反推。
- closure_decomposition：闭合机会和闭合率的对称精确代数分解，非因果归因；平均为先逐图计算后平均，未定义保留reason。三元标签(a,middle,c)中间项为共同邻居类型，端点类型无序。
- cn_count_scales、CN_DATAFLOW：源码/拓扑证据；未捕获32维学习向量尺度，不冒称已测。
- evidence：只复制判断所需已封存JSON/配置/五个源码文件/成本表，manifest标SHA，不覆盖原件。

## 仍缺的证据

每条链仅一个X499，不知道真实偏差起点；C/E权重分别训练，差异含共同参数变化，不能单独归因即时CN。最终8前向已结束才收到CN尺度补充要求，学习CN和头各块实际激活尺度未测，不增加前向。现有统计不证明CN需要新编码器或幅度归一化。候选仅为待审训练正则化假设，未实现/未运行。

原图是主要结论对象。LCC不是纠错或模型收益，既有原始结论不改。本票结束后停止，完整图实验暂停。
