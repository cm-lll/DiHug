# 完整IMDB/PubMed：固定20块本地有界短测

本票独立起点2026-10-30 15:27:38，固定最终截止16:27:38 Asia/Shanghai，主动预留最后10分钟用于交付。仅工程短测，不含12,000更新正式训练、T1000链或新角色拟合。旧票与封存材料不改。

## 当前结论

- 完整IMDB A+CE：3次临时normal2（更新1、连续更新2、从1恢复重做2），一个普通t500→499的完整20块反向步通过。完整加载模型/AdamW/EMA/RNG正确；后续模型/优化器/EMA逐位相同，输入与RNG一致。没有验证完整T1000链或所有生成状态的显存。
- 完整PubMed A+CE：第一份t500输入的grad-enabled主干前向OOM，0次反向完成、0次optimizer.step。61,190个查询槽、12,261条完整Xt真实边。按合同停止该配置，没有增加块数、减宽度、换精度或再试一遍。
- C/E-earlyCN：科学源码保留，完整图字典未准备；没有加载192节点/合成字典、没有构造C/E模型或跑GPU。仅CPU实测完整Xt的630维ReFeX。
- COUNT：本票CE入口明确禁止非零数量项。COUNT_REVIEW.md只提出一个需确认的双遍全关系方案，未执行。

详细数值/预算见交付目录REPORT_ZH.md、cost.json、各dataset_A/status.json及cpu_feature_cost.json。

## 来源与最小修改

基于迁移包固定6a9b20c8a73100d7001af981e666b9d7c5966ddc，实际执行科学父版c3f9ed007167484e4d336b2768ea2376b92fcb6e。package/及ema.py直接复制冻结源码，backbone、CN、公共min/max、早CN、ReFeX、RolX、核均未修改。source_parent.json记录原文件哈希。E-earlyCN仍由package/early_cn.py安装，末端CN不恢复，无Dropout。

新增block20.py只适配完整数据、历史抽块输入、块合同检查和CE normal2；probe_local.py提供有界工程更新、完整单步及保存恢复。test_protocol.py/cpu_cost.py为新增合同的CPU检查。历史package/prepare.py、normal2.py、worker.py等仍是192节点材料，**不作为本票入口**；有效配置为configs/与输出resolved_config.json。没有formal训练入口或可激活审批。

## 核对的20块协议

1. TypedSlots按无向端点类型对划分10个schema关系。只监督有非零参考支持的活跃关系，固定零关系留在全schema分母S里；不对它们猜边。
2. UniformQueryPlan对每个关系用独立SeedSequence子流随机排列局部槽，均分20份，余数由随机块分配。每个活跃槽在一轮完整计划中恰出现一次，各训练块以1/20概率均匀抽取，不是对每个关系取等量边。
3. 原train_input实际顺序为抽t、生成查询计划、抽round_id、以同一显式rng构造完整前向噪声Xt；本票直接调用其rounds=20路径，未自行调整随机调用顺序。两份normal2各自有计划及完整Xt。块不按正/负标签筛选。
4. 每份损失为`0.5 * 20 * sumCE(block) / S`，两份顺序反向后更新一次AdamW/EMA。IMDB S=1,365,378，PubMed S=1,223,830；不再用18,336。该比例修正抽块包含概率，不意味着查询上下文与全查询一致。
5. FrozenTypedContext始终持有完整Xt真实邻接。度、全局统计及CN都用完整Xt；CN共同邻居不包含临时查询边。主Transformer输入是完整Xt真实边与该块查询边的并集（正反方向），因此查询块会影响消息传递上下文。
6. reverse_step用`[seed,t,61]`建立20块计划，一次抽全schema随机均匀量，各块读取同一冻结Xt。原full_sweep检查重复与完整覆盖、计算原Bayes后验，全部成功后返回新状态；本票再原子保存一次完整状态。不把20块当20个时间步，不在块间更新Xt。

## 本地命令

只用于**新的有明确起止时间的工程票**，不直接重用已过期ledger或旧输出；每个配置最多3次更新。本票运行已结束，不应重跑已知OOM配置。

```sh
# 使用已授权且空闲的一张卡；两数据集分别运行到独立新目录。
CUDA_VISIBLE_DEVICES=GPU_UUID CUBLAS_WORKSPACE_CONFIG=:4096:8 \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONDONTWRITEBYTECODE=1 /path/to/existing/python -u -B probe_local.py \
  --dataset PubMed --output /data/new_ticket/PubMed_A --ledger /data/new_ticket/ledger.json
```

ledger需包含`science_stop_Asia_Shanghai`（ISO8601+08:00），本票完整示例在交付材料中；它是工程绝对截止记录，不是科学训练审批。未提供新授权时不要自行改时间激活。程序检查单卡可见，不安装依赖，也不限定只有2080Ti可调用，因此同一固定源码可以给BW/Notebook做后续有界20块检查。实际后端兼容性须在该环境重新核实，不宣称BW已通过。

当前库按原FP32、Torch确定性设置和TF32关闭运行。宿主GPU可见而本工具沙箱nvidia-smi失败，已通过只读宿主核实及受限工程执行区分；未改驱动。运行资源为GPU0 PubMed、GPU1 IMDB，结束后已释放。

首次probe对异常写failed状态但进程返回0；交付前只补了失败退出码1（通过返回0），新增CPU用例验证。GPU实际执行的原脚本及哈希在executed_adapter_source/中保留，没有为了这个退出码修复重跑OOM。科学前后向/更新/采样代码没有改动，不能凭旧进程exit0把PubMed写成通过。

## 完整图字典缺项

仅有小图字典，本目录不装载它们。完整图仍需每数据集干净图+100/500/1000各2份噪声、固定两折节点/两侧15组特征及原状态权重，K=1..min(8,两训练折节点数)，原CD/NNDSVDa/Frobenius/tol1e-5/max_iter50000及固定NNLS验收。本次完整两图四类型折节点均足够，候选上界各8，即每数据集64次候选拟合+4次最终拟合；两图共128候选+8最终。没有因此自动选择K=8。

七状态630维FP64面板约IMDB55.61MiB、PubMed52.66MiB（不含NMF工作区、折复制等）。当前实测的单状态ReFeX耗时不能代表NMF或NNLS；求解次数/迭代决定总成本，尚不能可靠给出正式字典墙钟。不得拿小图字典/合成字典或未测编码耗时宣称C/E可开训。

## SparseDiff参照

已核对官方实现与论文：核心确实是保留完整稀疏噪声图、训练抽候选查询、推理分块覆盖。官方Protein配置的8层、dx256/de64/dy128、8头、FF256/64/256和T1000与本票冻结主干一致；官方该配置edge_fraction=.1，不是我们固定的20块=.05。没有据此修改当前参数。

arXiv v2附录D.1还报告Facebook单图1045节点/27755边，说明有接近规模的同质图实例，但它不是本项目11GB显卡可运行的保证，也不是异质图质量证据。当前同质参照只是配置依据，不替代本票实测。

- https://github.com/qym7/SparseDiff
- https://raw.githubusercontent.com/qym7/SparseDiff/main/configs/experiment/protein.yaml
- https://arxiv.org/pdf/2311.02142 （核对版本v2；§3.2/§3.3/附录D.1）

正式选本地还是BW，应按本票具体配置判定：IMDB A有本地短测可行性；PubMed A当前11GB/20块失败，应先用固定适配包在BW做同协议20块测试。没有连接或操作Notebook，也没有把BW全查询OOM外推为20块失败。完整图正式任务数、seed和预算仍须后续明确。
