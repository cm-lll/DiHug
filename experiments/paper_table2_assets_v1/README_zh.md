# 投稿表2旧实验资产交接（只读查找与CPU复核）

本次不训练、不调用生成模型、不重新生成、不修图、不筛图，没有修改任何旧结果。原始载荷复制时及复核后重新哈希，read_only_audit.json记录检查数量与零变更。

## 找到什么

投稿PDF为paper/submitted.pdf（原temp/35541_DiHug_Typed_Edit_Diffusi (2).pdf），SHA256 c8134c9ddfdf6f4fd097b58e2404cc35eb73b28f8b7f95b69e76b16f0a1376bb，第6页表2。

|方法|IMDB原图|PubMed原图|版本与绑定|
|---|---:|---:|---|
|旧DiHug|8/8|8/8|20260729 current-base main；已复现投稿表2|
|GraphRNN|8/8|8/8|classic_baseline_real_followup；已复现投稿表2|
|NetGAN|8/8|8/8|classic_baseline_extension；已复现投稿表2|
|VGAE|8/8|8/8|classic_baseline_extension，重构基线；已复现投稿表2|
|HGEN|8/8|8/8|fullgraph_canonical_8run_0727导出＋0729 EO审计；已复现投稿表2|

共80张投稿原图，保持原JSON字节、原文件名和节点编号；目录仅作独立整理。asset_index.json列出每张的原绝对路径、新相对路径、SHA256、采样/训练seed语义及证据。
64张原TypedGraph文件的边指纹还与旧逐图报告完全一致；HGEN原导出缺类型字段，恢复规则及局限独立注明，没有把恢复字段写回原图。

另附96张候选历史图，来自paper_final_context_binomial_xiaob、同名_s8_15及paper_final_context_xiaob三个完整快照，每套IMDB/PubMed×legacy/context各8张。它们放在candidate_history，明确不是投稿样本；597.75和595的旧均值属于这些历史分支，不用来替换601.75。其他旧实验/消融目录未宣称全盘穷尽。

## 投稿绑定链路

原DiHug_clean_joint/paper/results_20260729/README.md明确指向results/diffusion_ablation_current_base_0729，main/final_graph.json为主结果。
逐图extended_structure报告→paper/results_20260729/fullgraph_dihug_extended.tsv，以及paper/results_20260727/provenance.json→fullgraph_core_8run.tsv/fullgraph_distribution_8run.tsv→aaai2027_dihug.tex→投稿PDF表2，数值一致。
本包保留这些原表、原出处文件与实际汇总脚本；不是只根据某张汇总数值挑出8图。

IMDB旧DiHug均值精确为：边4051.75、LCC1651.75、三角601.75、Clust.0.4126436971823853、Assort.-0.1563428002870525；MMD0.012184835517401828、Tri TV0.018825118859057514、MP2 0.004924104374800773、MP3 0.01363726028525026、EO6.876685462123069%。
参考1653节点、4079物理边、655三角。PubMed参考1565节点、12284物理边、61137三角；旧DiHug均值边12332.75、LCC1562.375、三角63916.25。

## 本地实际复核

reproduction/submission_git_80_graphs含80份新逐图复核、10份按方法汇总、环境、完整summary及日志；reproduced_table2.tsv为便于读取的汇总。
80/80完成，表2全部指标及标量总体标准差复现；固定ATOL1e-10、RTOL0，最大差1.4210854715202004e-14。用时289.319秒，峰RSS425.566MiB，CPU单线程；模型调用/训练/新生成图均0，未初始化GPU。
实际Python3.9.23、Torch2.1.0、NumPy1.26.4、SciPy1.11.0。旧环境声明的NumPy pin为1.23.0，本次没有改变环境，不把实际环境说成逐项等于原声明。

首次复核使用当前工作树留存评测文件，因空三角TV约定与历史报告不同而停止；失败结果保留在reproduction/retained_worktree_first_discrepancy。随后从Git直接提取提交71dedb3的原版评测代码，以真实历史口径完成复核。两版都保留，没有编辑原数学定义或为凑表改变参数。
详细指标定义、公式、方向/去重/孤立点/汇总规则和此版本差异见METRICS_zh.md。

## 参考、映射与当前实验

reference/imdb和reference/pubmed保留当时nodes.pt、edges.pt、meta.json；这些.pt是数据，不是模型权重。hgen_reference保留恢复HGEN顺序所需的原pickle参考。预处理器与旧源码在original_source；source_inventory.json列出原路径和哈希。
reference的原始节点ID与规范连续ID映射、关系族与类型名称见reproduction/submission_git_80_graphs/node_and_relation_mapping.json；并保留两个规范参考JSON，所有派生项明确标注。

current_IMDB_reference_comparison.json确认当前IMDB配对包数据SHA256 10f1dbbbe55c6a069594b2dcb590d1b8295ca5436320ef4597cd30e6b7f73e42与投稿参考的节点类型数组及物理边集合完全一致。此结论不认证当前模型权重，也没有评价当前基础/软角色生成结果。

后续共同评价要统一类型/关系映射、节点库存和指标版本。若用新版单空TV=1，必须全部方法重新计算并与历史复现表分开；不能只换新模型指标再接旧表。HGEN在参考schema外的边仍须保留并另报合法性，不修图后冒充原始输出。

## 尚缺或仅有历史来源的材料

- 80张投稿图和参考/核心评测/逐图表齐全；不是所有方法的运行时源码及权重绑定都完整。
- HGEN每run.log记录源码SHA256 81ac099d5bef7da08dedc2c22d2626b113eede9e5dd08abb3acac70add9e23b5，当前HGEN/hgen.py为另一个哈希；本地Git及所查副本未恢复该精确历史版本。不能用当前文件冒充。
- HGEN这16个导出目录未找到已保存模型检查点；run seed、5 epochs及实际参数列表可由日志第一行读取。
- DiHug两份基模型、6份经典基线模型都只登记当前路径/大小/SHA256，不打包权重。旧DiHug逐图报告没有逐输出认证的模型哈希；旧启动脚本指向的模型路径及留存PubMed独立权重收据提供部分出处，不能据新算哈希宣称全部历史权重字节已认证。
- 启动器、默认配置、部分训练日志与train_report保留；部分经典基线没有完整原始argv封存，不能猜测环境变量覆盖。method_protocols.json区分原存、来源规则与未认证部分。
- 旧工作树评测和经典基线源码有未提交修改；同时保留Git版本、工作树版本及文件哈希，不凭仓库HEAD推断每个文件干净。

## 只读复核命令

不要执行original_source中的训练/生成脚本。它们只是历史证据，可能包含旧机器路径和写入逻辑。
解包后先校验SHA256SUMS；用现有兼容CPU环境运行，不安装依赖：

```bash
cd DiHug_table2_assets_20261022_v1
sha256sum -c SHA256SUMS
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /path/to/existing/python reproduce.py \
  --metric-version submission-git --max-seconds 300 \
  --output /path/to/new/separate/table2_recheck_01
```

输出须新建，禁止与包/输入/源码重叠。300秒含核验、导入、统计和保存，超时/缺证据非零退出，部分文件保留；其他CPU可能无法在相同时间内完成，不自动扩时或换算法。
包装器仅导入原评测定义，使用隔离namespace避免旧__init__导入无关生产模型；原函数源码未修改，无神经模型调用。

本次交接完成后停止。下一步可以把已有新实验原始图交给同口径评测，但本票未重训、未重新生成，也没有把历史DiHug模型当作当前状态扩散模型。
