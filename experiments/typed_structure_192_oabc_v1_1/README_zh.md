# O/A/B/C准备修订v1.1：等待独立审查，正式训练/完整生成均0

基线4123d8f7da6dada9513f592aabdcc81b053bc4a9。沿用原北京时间02:06:09起点、02:51:09准备截止、05:06:09总截止；本修订开始02:35:44，不增加预算。旧118次拟合、源码和ZIP保留。没有审批文件；dictionary_bindings_PROPOSAL_NOT_APPROVAL.json仅提供将来审批所需的实际哈希，绝不构成授权。

## 唯一字典求解变更与结果

所有候选/最终拟合max_iter统一5000→50000。CD/Frobenius/NNDSVDa/tol1e-5/shuffle=False、无正则、原seed、节点折、特征掩码、630列尺度、状态权重、K范围、NNLS/选K规则均不变。
refit_dictionary.py按旧发行manifest的固定预期哈希2a977f8fc55cae52c93ed697cb6284afa56b6cce13f8f9feb3c8dc7e559232dc核验实际载荷，复制原14份Xt/特征，不重新抽样。旧frozen_splits.json字节不变，内含历史5000元数据；实际统一覆盖仅见refit_config.json的50000设置。叫重新拟合，不叫续算。

全部118次候选折拟合和8次最终拟合收敛。每次原始W/H、规范化基、输入矩阵内容哈希/形状、实际迭代、重建误差、警告及停止状态均保存。没有删K、换seed、放宽阈值或新增候选。

| 数据集 | 按原类型序的K | 列偏移 | 上界命中 | 整个重新拟合墙钟/峰值RSS |
|---|---|---|---|---|
| IMDB Director/Actor/Movie/Genre | 5/4/3/2 | 0/5/9/12/14 | 无 | 77.42秒 / 475.37MiB |
| PubMed Gene/Disease/Chemical/Species | 8/8/8/6 | 0/8/16/24/30 | 全部四类型 | 29.93秒 / 473.16MiB |

原105次已收敛候选拟合的新旧规范化基和评分最大差均0。IMDB候选最大实际迭代25946，PubMed6585；最终8次分别160/814/58/284、2058/602/1208/570。重新计算全部留出方向评分和原始W/H重建误差见revision_audit.json，不以小误差替代收敛判据。

## 运行保护与随机流

launcher及worker都核对外部审批指定的final_basis.npz、selection.json、panel.json、frozen_splits.json SHA256，以及K、offsets、总维数。仅内部字典自证不够。
唯一支持的正式入口是launcher：worker要求从当前launcher继承的只读pipe FD，绑定父PID、臂、数据集、设备、输出、审批/成本哈希及新鲜启动时间；还独立重查完整成本/审批/字典门。它防止误用独立worker绕过入口，不是针对能够伪造审批文件的恶意本地用户的安全沙箱。没有自动生成授权或回退路径。

模型构造与copy_common后，按数据集统一重置Python、全局NumPy、显式Generator、TorchCPU及所用CUDA设备随机流，独立记录派生seed和状态哈希。各臂初始化流与训练流分离。八格工程初态在首次前向前已保存；同数据集四臂实际公共张量与重置RNG已独立核对一致。正式初态将来必须另建，不能用工程权重。

预算估计与执行共享admission.assignment的固定每卡FIFO队列，不再使用不同的排程规则。worker不能单独选更新档。正式审批还必须明确确认main_W1_definition_confirmed=type_weighted_incident4；本票未批准此最终指标选择。

## 实际八格工程验收

engineering_real.py读取真实最终字典，遍历两数据集×O/A/B/C，使用原账本绝对截止。实际以GPU0跑IMDB、GPU1跑PubMed，各内部四组串行，CPU线程各1。使用前二卡均空闲；GPU5持续1%使用率，未列作确认空闲设备。未解除设备隔离、安装或升级依赖。

八格同GPU完整normal2前后向重复、实际复制/优化器覆盖、真实字典t=1/500/1000有限非负NNLS、B/C无CN、完整可动槽、模型/优化器/RNG精确载入和后续更新续算均通过。每格3次临时step：连续1、2与恢复后重做2，共24次，原40额度还剩16但未使用。无CPU/GPU等价复验，旧失败不改写。
首次前向前initial.pt及逐张量哈希/随机绑定见eight_engineering_initials.tar.xz；恢复用after1.pt在本地完整保留，发行清单登记其哈希，未再复制大体积临时优化器文件。恢复比较原始报告完整包含参数/梯度/优化器误差，不能仅凭载入成功声称续算通过。

| 数据集 | normal2组件范围 | 全查询前向+原后验+一次工程提交 | 峰值allocated / reserved |
|---|---:|---:|---:|
| IMDB四组 | 2.476–2.595秒 | 0.444–0.467秒 | 2393–2398 / 2510–2550MiB |
| PubMed四组 | 3.352–3.530秒 | 0.605–0.642秒 | 3243–3248 / 3408–3508MiB |

normal2包含工程取证开销，取三个临时更新最大值；不是稳定正式吞吐。采样组件覆盖全查询、q、uniform和一次离散状态提交，仅工程探针，不是T1000链；未给科学末图或质量结论。真实链更密状态可能有不同CN/NNLS开销。

## 成本建议与未完成项

两卡实测并不等于八卡并发。当前确认空闲7张，固定排程下2000/1000/500更新的保守总剩余成本约329.45/205.18/143.04分钟；含25%未实测并发余量、验证保存组件估算、20分钟评价交付与15分钟故障余量。在计算预算时剩余时间已不足143.04分钟，**目前七卡没有可放行档位**。
若以后实际确认8卡空闲，对应估计208.60/135.05/98.44分钟；只是条件方案，不能把尚未确认的第八张卡自动占用。等待审查继续减少余量，开训前必须重新核算，不能延长05:06:09或删组/图。实际formal validation/save全流程与八卡CPU/IO竞争未测；具体成分/队列及每格数值在cost_proposal.json。

本版提供2000/1000/500三个固定档位建议，没有选定正式更新数，没有生成审批文件，不自动启动。训练/调度源码只验证拒绝入口、绑定及排程逻辑，未实际执行正式长流程；断电/强杀等完整监督器故障覆盖尚未完成，不伪称全运行系统验收。

## 统一评价与W1口径

summarize.py独立读取八格及16张原始图，核对seed/最终权重/图哈希/类型/schema，调用同一逐图评价，保留全部绝对指标；给出5种对照的逐seed差和两图均值、结构代价、边差异及有编号重复情况。缺失/失败格保留，任一配对缺图时均值为null；不只平均成功图，返回非零。两样本的边差异不等于充分拓扑多样性，未宣称完成图同构分类。

本版W1按每类型关联4个schema关系列平均，再按类型节点数加权。旧E07按全部10列；在本次同一完整schema且非关联6列都为0的条件下，旧all10值=0.4×本版incident4值。不可直接混用；真实主口径需正式开训前确认，不能看结果挑口径。

安全复验（不拟合、不训练、不生成）：
```bash
sha256sum -c SHA256SUMS
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /data2/lyh/miniconda3/envs/sparse_block/bin/python -m unittest test_revision -v
cd package
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /data2/lyh/miniconda3/envs/sparse_block/bin/python -m unittest test_oabc -v
```

独立汇总入口（仅对将来已存在的结果；缺图输出incomplete）：
```bash
python summarize.py --run /调用者明确提供的新实验目录 --output /新的汇总文件.json
```
不提供默认执行正式任务的命令。完成本票交付后停止，等待独立审查和明确继续确认。
